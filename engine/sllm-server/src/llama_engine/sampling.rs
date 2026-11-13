//! Greedy sampling: the argmax over a row, the grammar sampler, and the text of a token.

use super::*;

/// The best token that is not an end-of-generation token (`ignore_eos`); the plain argmax when that is not one.
pub(crate) fn argmax_excluding(row: &[f32], eog: &[i32]) -> i32 {
    let t = argmax(row);
    if !eog.contains(&t) {
        return t;
    }
    let (mut best, mut best_v) = (0usize, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > best_v && !eog.contains(&(i as i32)) {
            best_v = v;
            best = i;
        }
    }
    best as i32
}

/// The first index of the largest value (the lowest id on a tie; NaN never wins; 0 when nothing beats -inf). Two passes:
/// a branch-free maximum over 16 independent lanes (vectorises to NEON/AVX), then the first position that holds it.
pub(crate) fn argmax(row: &[f32]) -> i32 {
    let mut lanes = [f32::NEG_INFINITY; 16];
    let mut chunks = row.chunks_exact(16);
    for c in &mut chunks {
        for i in 0..16 {
            lanes[i] = lanes[i].max(c[i]);
        }
    }
    for &v in chunks.remainder() {
        lanes[0] = lanes[0].max(v);
    }
    let m = lanes.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    row.iter().position(|&v| v == m).unwrap_or(0) as i32
}

/// A request's grammar (`grammar`, GBNF): a libllama grammar sampler and the candidate array of the masked path.
///
/// Greedy sampling with the rejection shortcut of llama.cpp's `common_sampler_sample`: take the argmax, ask the grammar about
/// that one token, and only when it refuses build the full candidate array, let the grammar mask it and take the argmax of
/// what is left (the masked argmax equals the plain one whenever the plain one is allowed, so the shortcut changes no token).
/// No token reaches the grammar's state unchecked: `accept` on a token no stack takes aborts the process (a C++ exception, or
/// `GGML_ABORT` for an end-of-generation token while no stack is empty), so [`LlamaEngine::on_token`] runs [`Gram::allows`]
/// on every token before [`Gram::accept`].
pub(crate) struct Gram {
    smpl: *mut ffi::Sampler,
    cand: Vec<ffi::TokenData>,
}

/// What [`Gram::pick`] returns when the grammar leaves no token: [`LlamaEngine::on_token`] ends the request.
pub(crate) const GRAMMAR_DONE: i32 = -1;

impl Gram {
    /// None when libllama refuses the text (syntax error, no `root` rule, undefined rule, left recursion).
    pub(crate) fn new(vocab: *const sys::llama_vocab, text: &str) -> Option<Gram> {
        let text = CString::new(text).ok()?;
        let smpl = unsafe { ffi::llama_sampler_init_grammar(vocab, text.as_ptr(), c"root".as_ptr()) };
        (!smpl.is_null()).then(|| Gram { smpl, cand: Vec::new() })
    }

    /// The grammar takes token `t` now (a one-candidate `apply`: -inf means refused; an end-of-generation token is taken only
    /// when the grammar can end here).
    pub(crate) fn allows(&mut self, t: i32) -> bool {
        if t < 0 {
            return false;
        }
        let mut d = ffi::TokenData { id: t, logit: 0.0, p: 0.0 };
        let mut arr = ffi::TokenDataArray { data: &mut d, size: 1, selected: -1, sorted: false };
        unsafe { ffi::llama_sampler_apply(self.smpl, &mut arr) };
        d.logit > f32::NEG_INFINITY
    }

    /// Advance the grammar over a token [`Gram::allows`] took (never an end-of-generation token: the request ends there).
    pub(crate) fn accept(&mut self, t: i32) {
        unsafe { ffi::llama_sampler_accept(self.smpl, t) }
    }

    /// The grammar takes token `t` if it allows it (false when it refuses; then nothing changes).
    pub(crate) fn takes(&mut self, t: i32) -> bool {
        let allowed = self.allows(t);
        if allowed {
            self.accept(t);
        }
        allowed
    }

    /// The greedy token that follows the grammar (`ignore_eos`: end-of-generation tokens are masked first), or
    /// [`GRAMMAR_DONE`] when nothing is left: the grammar is complete and cannot continue, and `ignore_eos` forbids the
    /// end-of-generation token that would otherwise end it.
    pub(crate) fn pick(&mut self, row: &[f32], ignore_eos: bool, eog: &[i32]) -> i32 {
        let t = if ignore_eos { argmax_excluding(row, eog) } else { argmax(row) };
        if self.allows(t) {
            return t;
        }
        self.cand.clear();
        self.cand.extend(row.iter().enumerate().map(|(i, &logit)| ffi::TokenData { id: i as i32, logit, p: 0.0 }));
        if ignore_eos {
            for &e in eog {
                if let Some(c) = self.cand.get_mut(e as usize) {
                    c.logit = f32::NEG_INFINITY;
                }
            }
        }
        let mut arr = ffi::TokenDataArray { data: self.cand.as_mut_ptr(), size: self.cand.len(), selected: -1, sorted: false };
        unsafe { ffi::llama_sampler_apply(self.smpl, &mut arr) };
        let (mut best, mut best_v) = (GRAMMAR_DONE, f32::NEG_INFINITY);
        for c in &self.cand {
            if c.logit > best_v {
                (best, best_v) = (c.id, c.logit);
            }
        }
        best
    }
}

impl Drop for Gram {
    fn drop(&mut self) {
        unsafe { ffi::llama_sampler_free(self.smpl) }
    }
}

/// Greedy token of every picked row, in order; rows are spread over a few threads once there are enough of them (each row is
/// n_vocab floats, 0.5 to 1 MB: 64 rows is a memory-bound scan of tens of MB on one core).
pub(super) fn sample_rows(picks: &[(ReqId, bool, &[f32])], eog: &[i32], out: &mut Vec<(ReqId, i32)>) {
    let one = |&(req, ignore_eos, row): &(ReqId, bool, &[f32])| (req, if ignore_eos { argmax_excluding(row, eog) } else { argmax(row) });
    const MIN_PARALLEL: usize = 12;
    if picks.len() < MIN_PARALLEL {
        out.extend(picks.iter().map(one));
        return;
    }
    let parts = 4.min(picks.len() / 4);
    let per = picks.len().div_ceil(parts);
    let mut res: Vec<Vec<(ReqId, i32)>> = Vec::new();
    std::thread::scope(|sc| {
        let hs: Vec<_> = picks.chunks(per).map(|ch| sc.spawn(move || ch.iter().map(one).collect::<Vec<_>>())).collect();
        res = hs.into_iter().map(|h| h.join().expect("argmax thread")).collect();
    });
    for r in res {
        out.extend(r);
    }
}

/// Text of a token: its bytes appended to `carry`, minus any incomplete UTF-8 tail that stays in `carry`.
pub(crate) fn text_of(vocab: &Vocab, id: i32, carry: &mut Vec<u8>) -> String {
    carry.extend_from_slice(&vocab.piece(id));
    let mut out = String::new();
    loop {
        match std::str::from_utf8(carry) {
            Ok(s) => {
                out.push_str(s);
                carry.clear();
                return out;
            }
            Err(e) => {
                let good = e.valid_up_to();
                out.push_str(std::str::from_utf8(&carry[..good]).unwrap());
                match e.error_len() {
                    None => {
                        carry.drain(..good);
                        return out;
                    }
                    Some(bad) => {
                        out.push('\u{FFFD}');
                        carry.drain(..good + bad);
                    }
                }
            }
        }
    }
}
