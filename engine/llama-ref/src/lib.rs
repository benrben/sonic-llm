//! llama-ref: the llama.cpp b11669 reference side of the SLLM parity harness (phase 1 task 4a).
//!
//! It loads a GGUF with the vendored official `libllama` and offers three things:
//!
//! * [`reference_logits`]: the f32 logits of every position of a token sequence (teacher forced);
//! * [`reference_greedy`]: greedy generation with the top-2 logits of every step;
//! * [`tokenize`]: llama.cpp's own tokenizer for a GGUF (vocab only, no GPU);
//! * [`Vocab`]: that vocabulary loaded once and kept (tokenize, token text, end-of-generation ids).
//!
//! Context settings are fixed and returned with every result ([`Settings`]): flash attention on, f16 KV cache (the
//! b11669 default), every layer offloaded (Metal), `n_ubatch` as asked (512 for the parity run). Prompts are decoded
//! in chunks of `n_ubatch` tokens, one `llama_decode` per chunk, which is the same physical split llama.cpp makes
//! for a larger batch and the same chunking the SLLM side uses (batch 1, prefill chunks matched to the ubatch).
//!
//! Everything GPU runs in the calling process; run it under `engine/scripts/with_gpu_lock.py`.

pub mod inputs;
pub mod npy;
pub mod sys;

use serde::{Deserialize, Serialize};
use std::ffi::{c_char, c_void, CString};
use std::path::Path;
use std::ptr;
use std::sync::Once;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}

/// The context settings a result was produced with (read back from the context where llama.cpp can change them).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Settings {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub flash_attn: String,
    pub n_gpu_layers: i32,
    pub type_k: String,
    pub type_v: String,
    pub offload_kqv: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
}

/// One greedy step: the chosen token (the argmax, lowest id on a tie) and the runner-up, with their logits.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Top2 {
    pub id1: i32,
    pub logit1: f32,
    pub id2: i32,
    pub logit2: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Greedy {
    /// The generated token ids (exactly `n`; EOS does not stop the run).
    pub ids: Vec<i32>,
    /// `top2[i]` belongs to `ids[i]`: the two largest logits of the distribution that token was drawn from.
    pub top2: Vec<Top2>,
    pub settings: Settings,
}

/// All-position logits, row major `[n_tokens][n_vocab]`.
#[derive(Debug, Clone)]
pub struct Logits {
    pub n_tokens: usize,
    pub n_vocab: usize,
    pub data: Vec<f32>,
    pub settings: Settings,
}

static INIT: Once = Once::new();

/// Drops llama.cpp's and ggml's informational lines but keeps errors on stderr: a failed Metal command buffer (its
/// status and Metal's error text) must not vanish (the sllm-server status -1 investigation).
unsafe extern "C" fn quiet_log(level: sys::ggml_log_level, text: *const c_char, _ud: *mut c_void) {
    if level == sys::GGML_LOG_LEVEL_ERROR && !text.is_null() {
        // SAFETY: ggml passes a NUL-terminated string.
        let s = unsafe { std::ffi::CStr::from_ptr(text) }.to_string_lossy();
        eprint!("ggml: {s}");
    }
}

/// `llama_backend_init` once per process; llama.cpp's own log lines go to stderr only with `LLAMA_REF_VERBOSE=1`
/// (errors always do).
fn backend_init() {
    INIT.call_once(|| unsafe {
        if std::env::var_os("LLAMA_REF_VERBOSE").is_none() {
            sys::llama_log_set(Some(quiet_log), ptr::null_mut());
        }
        sys::llama_backend_init();
    });
}

/// A loaded model, optionally with a context.
struct Loaded {
    model: *mut sys::llama_model,
    ctx: *mut sys::llama_context,
    vocab: *const sys::llama_vocab,
    n_vocab: usize,
    settings: Option<Settings>,
}

impl Drop for Loaded {
    fn drop(&mut self) {
        unsafe { free_context_and_model(self.ctx, self.model) }
    }
}

/// Frees a libllama context, then its model (either may be null).
///
/// # Safety
/// Each pointer is null or was returned by libllama and is not freed elsewhere or used afterwards.
pub unsafe fn free_context_and_model(ctx: *mut sys::llama_context, model: *mut sys::llama_model) {
    if !ctx.is_null() {
        sys::llama_free(ctx);
    }
    if !model.is_null() {
        sys::llama_model_free(model);
    }
}

fn cpath(p: &Path) -> Result<CString> {
    CString::new(p.to_string_lossy().as_bytes()).map_err(|_| Error("path contains NUL".into()))
}

/// Load the model. `ctx = Some((n_ctx, n_ubatch))` also creates the context with the fixed reference settings.
fn load(gguf: &Path, ctx: Option<(u32, u32)>, kv: Option<(sys::ggml_type, sys::ggml_type)>) -> Result<Loaded> {
    backend_init();
    if !gguf.is_file() {
        return err(format!("GGUF not found: {}", gguf.display()));
    }
    let path = cpath(gguf)?;
    unsafe {
        let mut mp = sys::llama_model_default_params();
        // Keep the device list alive until the load returns. An empty (NULL-terminated) list is `-dev none`.
        let mut no_devices: [sys::ggml_backend_dev_t; 1] = [ptr::null_mut()];
        match ctx {
            Some(_) => mp.n_gpu_layers = 999, // every layer on Metal
            None => {
                mp.vocab_only = true; // tokenizer only: no weights, no GPU
                mp.devices = no_devices.as_mut_ptr();
            }
        }
        let model = sys::llama_model_load_from_file(path.as_ptr(), mp);
        if model.is_null() {
            return err(format!("llama_model_load_from_file failed for {}", gguf.display()));
        }
        let vocab = sys::llama_model_get_vocab(model);
        let n_vocab = sys::llama_vocab_n_tokens(vocab) as usize;
        let mut l = Loaded { model, ctx: ptr::null_mut(), vocab, n_vocab, settings: None };
        if let Some((n_ctx, n_ubatch)) = ctx {
            let mut cp = sys::llama_context_default_params();
            cp.n_ctx = n_ctx;
            cp.n_batch = n_ubatch; // prompts are fed in chunks of n_ubatch
            cp.n_ubatch = n_ubatch;
            cp.flash_attn_type = sys::LLAMA_FLASH_ATTN_TYPE_ENABLED;
            // type_k / type_v stay at the b11669 default (f16) unless `kv` asks for another cache type;
            // offload_kqv / op_offload / swa_full / kv_unified stay at the b11669 defaults.
            if let Some((tk, tv)) = kv {
                cp.type_k = tk;
                cp.type_v = tv;
            }
            let c = sys::llama_init_from_model(model, cp);
            if c.is_null() {
                return err("llama_init_from_model failed (n_ctx too large?)");
            }
            l.ctx = c;
            l.settings = Some(Settings {
                n_ctx: sys::llama_n_ctx(c),
                n_batch: sys::llama_n_batch(c),
                n_ubatch: sys::llama_n_ubatch(c),
                flash_attn: "enabled".into(),
                n_gpu_layers: mp.n_gpu_layers,
                type_k: ggml_type_name(cp.type_k),
                type_v: ggml_type_name(cp.type_v),
                offload_kqv: cp.offload_kqv,
                op_offload: cp.op_offload,
                swa_full: cp.swa_full,
                kv_unified: cp.kv_unified,
            });
        }
        Ok(l)
    }
}

fn ggml_type_name(t: sys::ggml_type) -> String {
    match t {
        0 => "f32".into(),
        1 => "f16".into(),
        2 => "q4_0".into(),
        8 => "q8_0".into(),
        30 => "bf16".into(),
        n => format!("ggml_type {n}"),
    }
}

/// An owned `llama_batch` of a fixed capacity, each token tagged with one sequence id.
pub struct Batch {
    pub b: sys::llama_batch,
    pub cap: usize,
}

impl Batch {
    pub fn new(cap: usize) -> Batch {
        Batch { b: unsafe { sys::llama_batch_init(cap as i32, 0, 1) }, cap }
    }

    /// Slot `k` holds `token` at position `pos` of sequence `seq`, with logits when `logits` is set.
    pub fn set(&mut self, k: usize, token: i32, pos: i32, seq: i32, logits: bool) {
        assert!(k < self.cap);
        unsafe {
            *self.b.token.add(k) = token;
            *self.b.pos.add(k) = pos;
            *self.b.n_seq_id.add(k) = 1;
            **self.b.seq_id.add(k) = seq;
            *self.b.logits.add(k) = logits as i8;
        }
    }

    /// Slot `k` (already set) gives logits too.
    pub fn want_logits(&mut self, k: usize) {
        assert!(k < self.cap);
        unsafe { *self.b.logits.add(k) = 1 }
    }

    /// Fill with `ids` at positions `pos0..` of sequence 0; every token produces logits when `all_logits`, else only the last.
    fn fill(&mut self, ids: &[i32], pos0: i32, all_logits: bool) {
        assert!(ids.len() <= self.cap);
        self.b.n_tokens = ids.len() as i32;
        for (i, &t) in ids.iter().enumerate() {
            self.set(i, t, pos0 + i as i32, 0, all_logits || i + 1 == ids.len());
        }
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe { sys::llama_batch_free(self.b) }
    }
}

fn decode(l: &Loaded, batch: &Batch) -> Result<()> {
    let rc = unsafe { sys::llama_decode(l.ctx, batch.b) };
    if rc != 0 {
        return err(format!("llama_decode returned {rc}"));
    }
    Ok(())
}

/// The `[start, end)` ranges of consecutive chunks of at most `step` of `len` tokens.
fn chunk_ranges(len: usize, step: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..len).step_by(step).map(move |start| (start, (start + step).min(len)))
}

/// Decode `ids` at positions `pos0..` as one batch (logits for every row when `all_logits`, else for the last).
fn feed(l: &Loaded, batch: &mut Batch, ids: &[i32], pos0: usize, all_logits: bool) -> Result<()> {
    batch.fill(ids, pos0 as i32, all_logits);
    decode(l, batch)
}

/// Logits of batch row `i` (an index into the last decoded batch); the slice is valid until the next decode.
fn logits_row<'a>(l: &'a Loaded, i: i32) -> Result<&'a [f32]> {
    let p = unsafe { sys::llama_get_logits_ith(l.ctx, i) };
    if p.is_null() {
        return err(format!("llama_get_logits_ith({i}) returned NULL"));
    }
    Ok(unsafe { std::slice::from_raw_parts(p, l.n_vocab) })
}

/// A chunk of consecutive logit rows handed to the sink of [`reference_logits_stream`].
pub struct Chunk<'a> {
    /// Index of the first row in the whole sequence.
    pub start: usize,
    pub n_vocab: usize,
    /// `rows * n_vocab` floats.
    pub data: &'a [f32],
}

/// Teacher-forced logits for every position of `ids`, streamed chunk by chunk (`n_ubatch` rows at a time) to `sink`.
/// Returns the settings used. `n_ctx` must be at least `ids.len()`.
pub fn reference_logits_stream(
    gguf: &Path,
    ids: &[i32],
    n_ctx: u32,
    n_ubatch: u32,
    sink: impl FnMut(Chunk<'_>) -> Result<()>,
) -> Result<Settings> {
    reference_logits_stream_kv(gguf, ids, n_ctx, n_ubatch, None, sink)
}

/// Parse a `--cache-type-k/v` value (`f16`, `q8_0`, `q4_0`) into a ggml type id.
pub fn parse_cache_type(s: &str) -> Result<sys::ggml_type> {
    match s {
        "f16" => Ok(sys::GGML_TYPE_F16),
        "q8_0" => Ok(sys::GGML_TYPE_Q8_0),
        "q4_0" => Ok(sys::GGML_TYPE_Q4_0),
        other => err(format!("unsupported cache type {other:?}; expected f16, q8_0 or q4_0")),
    }
}

/// [`reference_logits_stream`] with the KV cache types `kv = Some((type_k, type_v))` (None: the f16 default).
pub fn reference_logits_stream_kv(
    gguf: &Path,
    ids: &[i32],
    n_ctx: u32,
    n_ubatch: u32,
    kv: Option<(sys::ggml_type, sys::ggml_type)>,
    mut sink: impl FnMut(Chunk<'_>) -> Result<()>,
) -> Result<Settings> {
    if ids.is_empty() {
        return err("empty token list");
    }
    if (n_ctx as usize) < ids.len() {
        return err(format!("n_ctx {n_ctx} < {} tokens", ids.len()));
    }
    let l = load(gguf, Some((n_ctx, n_ubatch)), kv)?;
    let settings = l.settings.clone().unwrap();
    let step = settings.n_ubatch as usize;
    let mut batch = Batch::new(step);
    for (start, end) in chunk_ranges(ids.len(), step) {
        feed(&l, &mut batch, &ids[start..end], start, true)?;
        // Output rows are numbered by position in the batch; the rows of one decode are contiguous.
        let p = unsafe { sys::llama_get_logits_ith(l.ctx, 0) };
        if p.is_null() {
            return err("llama_get_logits_ith(0) returned NULL");
        }
        let rows = end - start;
        let data = unsafe { std::slice::from_raw_parts(p, rows * l.n_vocab) };
        sink(Chunk { start, n_vocab: l.n_vocab, data })?;
    }
    Ok(settings)
}

/// All-position logits of `ids` as one `[n, vocab]` f32 matrix (2.1 GB for 2,000 Gemma tokens: prefer the stream).
pub fn reference_logits(gguf: &Path, ids: &[i32], n_ctx: u32, n_ubatch: u32) -> Result<Logits> {
    let mut data: Vec<f32> = Vec::new();
    let mut n_vocab = 0;
    let settings = reference_logits_stream(gguf, ids, n_ctx, n_ubatch, |c| {
        n_vocab = c.n_vocab;
        data.extend_from_slice(c.data);
        Ok(())
    })?;
    Ok(Logits { n_tokens: ids.len(), n_vocab, data, settings })
}

/// Largest and second largest logit with their ids. Ties go to the lower id (first maximum), which is what the
/// llama.cpp greedy sampler returns.
fn top2(row: &[f32]) -> Top2 {
    let (mut i1, mut i2) = (0usize, usize::MAX);
    for i in 1..row.len() {
        if row[i] > row[i1] {
            i2 = i1;
            i1 = i;
        } else if i2 == usize::MAX || row[i] > row[i2] {
            i2 = i;
        }
    }
    Top2 { id1: i1 as i32, logit1: row[i1], id2: i2 as i32, logit2: row[i2] }
}

/// Greedy generation of exactly `n` tokens after `prompt_ids` (EOS is ignored, as with `ignore_eos`). The prompt is
/// decoded in chunks of 512 tokens (`n_ubatch`); each generated token is a batch of one.
pub fn reference_greedy(gguf: &Path, prompt_ids: &[i32], n: usize) -> Result<Greedy> {
    let mut v = reference_greedy_many(gguf, &[prompt_ids.to_vec()], n)?;
    Ok(v.remove(0))
}

/// [`reference_greedy`] over several prompts with one model load; the KV cache is cleared between prompts.
pub fn reference_greedy_many(gguf: &Path, prompts: &[Vec<i32>], n: usize) -> Result<Vec<Greedy>> {
    const N_UBATCH: u32 = 512;
    if prompts.is_empty() || prompts.iter().any(|p| p.is_empty()) {
        return err("empty prompt");
    }
    let longest = prompts.iter().map(|p| p.len()).max().unwrap() + n + 1;
    let n_ctx = (longest.div_ceil(256) * 256).max(512) as u32;
    let l = load(gguf, Some((n_ctx, N_UBATCH)), None)?;
    let settings = l.settings.clone().unwrap();
    let step = settings.n_ubatch as usize;
    let mut batch = Batch::new(step);
    let mut out = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        unsafe { sys::llama_memory_clear(sys::llama_get_memory(l.ctx), true) };
        let (ids, tops) = greedy_one(&l, &mut batch, prompt, n)?;
        out.push(Greedy { ids, top2: tops, settings: settings.clone() });
    }
    Ok(out)
}

/// One prompt of [`reference_greedy_many`] on a cleared cache: its `n` greedy ids and their top-2 logits.
fn greedy_one(l: &Loaded, batch: &mut Batch, prompt: &[i32], n: usize) -> Result<(Vec<i32>, Vec<Top2>)> {
    let step = batch.cap;
    // Prefill in chunks; only the last prompt token needs logits.
    for (start, end) in chunk_ranges(prompt.len(), step) {
        feed(l, batch, &prompt[start..end], start, false)?;
    }
    let mut ids = Vec::with_capacity(n);
    let mut tops = Vec::with_capacity(n);
    let mut row_idx = (prompt.len() - 1 - (prompt.len() - 1) / step * step) as i32; // last row of the last chunk
    for k in 0..n {
        let t = top2(logits_row(l, row_idx)?);
        ids.push(t.id1);
        tops.push(t);
        if k + 1 < n {
            feed(l, batch, &[t.id1], prompt.len() + k, true)?;
            row_idx = 0;
        }
    }
    Ok((ids, tops))
}

/// llama.cpp's tokenizer for this GGUF (vocab only: no weights are loaded and the GPU is not touched).
/// `add_special` adds BOS/EOS as the model is configured to (Gemma: BOS; Qwen3: nothing); `parse_special` turns
/// special-token text such as `<|im_start|>` into the control token.
pub fn tokenize(gguf: &Path, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>> {
    let l = load(gguf, None, None)?;
    tokenize_vocab(l.vocab, text, add_special, parse_special)
}

/// llama.cpp's tokenizer on a loaded vocabulary.
fn tokenize_vocab(vocab: *const sys::llama_vocab, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>> {
    if text.len() > i32::MAX as usize - 16 {
        return err("text too long");
    }
    // At most one token per byte, plus the special ones.
    let mut toks = vec![0i32; text.len() + 8];
    let n = unsafe {
        sys::llama_tokenize(
            vocab,
            text.as_ptr() as *const c_char,
            text.len() as i32,
            toks.as_mut_ptr(),
            toks.len() as i32,
            add_special,
            parse_special,
        )
    };
    if n < 0 {
        return err(format!("llama_tokenize failed ({n})"));
    }
    toks.truncate(n as usize);
    Ok(toks)
}

/// The vocabulary size, BOS id (or -1) and whether the model adds BOS by itself; vocab only.
pub fn vocab_info(gguf: &Path) -> Result<(usize, i32, bool)> {
    let l = load(gguf, None, None)?;
    unsafe { Ok((l.n_vocab, sys::llama_vocab_bos(l.vocab), sys::llama_vocab_get_add_bos(l.vocab))) }
}

/// A vocab-only model loaded once and kept: the cached handle behind sllm-server's `/tokenize`, its text endpoints and the
/// real engine's token text and end-of-generation ids. Loading reads only the GGUF's vocabulary (no weights, no GPU).
///
/// Every method takes `&self` and only reads the vocabulary, which llama.cpp does not mutate after load, so a `Vocab` is
/// shared between threads (`Send + Sync`).
pub struct Vocab {
    l: Loaded,
    eog: Vec<i32>,
    bos: i32,
    add_bos: bool,
}

// SAFETY: `llama_vocab` is immutable after `llama_model_load_from_file`; `llama_tokenize`, `llama_token_to_piece`,
// `llama_vocab_is_eog` and the other getters used here take `const llama_vocab *` and keep no state between calls.
unsafe impl Send for Vocab {}
unsafe impl Sync for Vocab {}

impl Vocab {
    /// Load the vocabulary of `gguf` (vocab only).
    pub fn load(gguf: &Path) -> Result<Vocab> {
        let l = load(gguf, None, None)?;
        let (bos, add_bos) = unsafe { (sys::llama_vocab_bos(l.vocab), sys::llama_vocab_get_add_bos(l.vocab)) };
        let eog = (0..l.n_vocab as i32).filter(|&t| unsafe { sys::llama_vocab_is_eog(l.vocab, t) }).collect();
        Ok(Vocab { l, eog, bos, add_bos })
    }

    pub fn n_vocab(&self) -> usize {
        self.l.n_vocab
    }

    /// The libllama handle, for a grammar sampler built on this vocabulary; valid while the `Vocab` lives.
    pub fn as_ptr(&self) -> *const sys::llama_vocab {
        self.l.vocab
    }

    /// BOS id, or -1 when the model has none.
    pub fn bos(&self) -> i32 {
        self.bos
    }

    /// Whether the model adds BOS by itself when `add_special` is set.
    pub fn add_bos(&self) -> bool {
        self.add_bos
    }

    /// Every end-of-generation token of the vocabulary (llama.cpp's `llama_vocab_is_eog`), ascending.
    pub fn eog_ids(&self) -> &[i32] {
        &self.eog
    }

    pub fn is_eog(&self, id: i32) -> bool {
        self.eog.binary_search(&id).is_ok()
    }

    /// As [`tokenize`], on the cached handle.
    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>> {
        tokenize_vocab(self.l.vocab, text, add_special, parse_special)
    }

    /// The bytes of one token as llama-server streams them (`special = false`: control tokens render as nothing). The bytes
    /// are not always valid UTF-8 on their own: a character can be split across tokens.
    pub fn piece(&self, id: i32) -> Vec<u8> {
        self.piece_with(id, false)
    }

    /// The token's text with control tokens printed (`<|tool_call>`); [`Vocab::piece`] prints them as nothing.
    pub fn piece_special(&self, id: i32) -> Vec<u8> {
        self.piece_with(id, true)
    }

    fn piece_with(&self, id: i32, special: bool) -> Vec<u8> {
        let mut buf = vec![0u8; 64];
        loop {
            let n = unsafe { sys::llama_token_to_piece(self.l.vocab, id, buf.as_mut_ptr() as *mut c_char, buf.len() as i32, 0, special) };
            if n >= 0 {
                buf.truncate(n as usize);
                return buf;
            }
            buf = vec![0u8; (-n) as usize];
        }
    }
}
