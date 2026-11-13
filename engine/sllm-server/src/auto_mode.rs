//! `--auto-mode`: size the engine from this machine and this model, so a developer gives the job and no memory flags.
//!
//! The job is `--agents N`, `--ctx N`, both or neither; everything else follows. What fits is always asked of
//! [`plan_engine`], the plan the engine sizes its own cache with, so this cannot promise what the engine then refuses.
//! A flag the developer passes wins: `--kv-type` and `--kv-read` narrow the notes tried here, `--kv-pages` keeps the size.
//!
//! * `--agents N`: `N + 1` requests at once, the extra one so a chat beside the agents does not wait for an agent to end.
//!   The longest context up to the default that holds them, on the first notes that give at least [`MIN_CTX`].
//! * `--ctx N`: as many requests of that context as fit.
//! * both: that shape, or an error that says what fits instead.
//! * neither: the engine's own defaults.

use crate::llama_engine::reader_reserve_bytes;
use crate::profile::{plan_engine, requests_for, Budget, KvType, PlanArgs, Profile, CELL_PAD, MAX_TOTAL_CELLS};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::HashMap;

/// Below this a prompt plus its answer barely fit, so `--agents` moves to smaller notes first. Low on purpose: Qwen3-4B on
/// 4-bit notes with f16 reading differed from solo on 13 of 16 answers at 64 agents, f16 on at most 2 of 16.
pub const MIN_CTX: usize = 512;

/// Most rows a step can hold and stay exact with the library's default settings (`engine/vendor/ggml-rowexact-b11669`:
/// above this, stock kernels). The limit of this process follows `SLLM_ROW_EXACT_MAXN` ([`crate::exactness::row_limit_here`]).
pub const EXACT_ROWS: usize = crate::exactness::MMA_ROWS;

/// `--ssd-budget-bytes` under auto-mode: past it the oldest parked agents drop to their token history, which reads again
/// on the next turn. 256 agents on f16 took 25 GB in 3 rounds.
pub const SSD_BUDGET_BYTES: u64 = 64_000_000_000;

/// What the KV cache holds and how prompts are read into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Notes {
    pub kv: KvType,
    pub read_f16: bool,
}

impl Notes {
    pub fn label(self) -> String {
        format!("{} notes{}", self.kv.name(), if self.read_f16 { ", f16 reading" } else { "" })
    }
}

/// The notes `--auto-mode` tries, in order of preference. `kv_auto`: `--kv-type` is auto; `read_auto`: `--kv-read` was not
/// given; `read_f16`: it is f16; `explicit`: the `--kv-type` given. 4-bit notes only with `allow_4bit` (`--kv-auto-4bit on`):
/// their quality cost depends on the model (Qwen3-0.6B perplexity 20 to 83), so auto takes q8_0, the next option that
/// fits, and the job gets a shorter context or an error that says what fits instead.
pub fn notes_for(kv_auto: bool, read_auto: bool, read_f16: bool, explicit: KvType, can_read: bool, allow_4bit: bool) -> Vec<Notes> {
    let plain = |kv| Notes { kv, read_f16: false };
    match (kv_auto, read_auto, read_f16) {
        (false, _, read) => vec![Notes { kv: explicit, read_f16: read }],
        // --kv-read f16 asks for quantized notes read in f16
        (true, _, true) => vec![Notes { kv: if allow_4bit { KvType::Q4_0 } else { KvType::Q8_0 }, read_f16: true }],
        (true, false, false) => {
            let mut v = vec![plain(KvType::F16), plain(KvType::Q8_0)];
            v.extend(allow_4bit.then(|| plain(KvType::Q4_0)));
            v
        }
        (true, true, false) => {
            let mut v = vec![plain(KvType::F16)];
            v.extend((can_read && allow_4bit).then_some(Notes { kv: KvType::Q4_0, read_f16: true }));
            v.push(plain(KvType::Q8_0));
            v
        }
    }
}

/// The job, and what the machine and the flags leave open.
#[derive(Debug, Clone)]
pub struct Ask {
    pub agents: Option<usize>,
    pub ctx: Option<usize>,
    pub default_ctx: usize,
    /// The longest context of this model (its trained context).
    pub top_ctx: usize,
    /// The notes to try, in order of preference.
    pub notes: Vec<Notes>,
    pub max_streams: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Choice {
    pub ctx: usize,
    pub streams: usize,
    pub notes: Notes,
}

fn commas(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// The longest context, a multiple of a page and at most `top`, that holds `streams` requests on these notes.
fn longest(top: usize, streams: usize, notes: Notes, most: &impl Fn(Notes, usize) -> usize) -> Option<usize> {
    let page = CELL_PAD as usize;
    let (mut lo, mut hi) = (0, top / page);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if most(notes, mid * page) >= streams {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    (lo > 0).then_some(lo * page)
}

/// Pick the shape. `most(notes, ctx)` is how many requests of `ctx` tokens fit on those notes. Asks with no job at all
/// are not for this function: the engine's defaults apply.
pub fn choose(a: &Ask, most: impl Fn(Notes, usize) -> usize) -> Result<Choice, String> {
    let page = CELL_PAD as usize;
    let top = a.top_ctx.max(page) / page * page;
    let ctx = a.ctx.map(|c| c.div_ceil(page).clamp(1, top / page) * page);
    let streams = a.agents.map(|n| (requests_for(n as u64) as usize).min(a.max_streams));
    if let Some(n) = a.agents {
        if n == 0 || n > a.max_streams {
            return Err(format!("--agents: expected 1 to {}, got {n}", a.max_streams));
        }
    }
    match (streams, ctx) {
        (Some(s), Some(ctx)) => choose_both(a, s, ctx, top, &most),
        (Some(s), None) => choose_ctx(a, s, top, &most),
        (None, Some(ctx)) => a
            .notes
            .iter()
            .find_map(|&notes| Some(most(notes, ctx)).filter(|&s| s > 0).map(|s| Choice { ctx, streams: s.min(a.max_streams), notes }))
            .ok_or_else(|| format!("a context of {} tokens does not fit this machine with this model", commas(ctx))),
        (None, None) => Err("nothing asked".into()),
    }
}

/// Both asked (`s` requests of `ctx` tokens): the first notes that hold them, or what would fit instead.
fn choose_both(a: &Ask, s: usize, ctx: usize, top: usize, most: &impl Fn(Notes, usize) -> usize) -> Result<Choice, String> {
    a.notes.iter().find(|&&n| most(n, ctx) >= s).map(|&notes| Choice { ctx, streams: s, notes }).ok_or_else(|| {
        let at_ctx = a.notes.iter().map(|&n| most(n, ctx)).max().unwrap_or(0);
        let at_n = a.notes.iter().filter_map(|&n| longest(top, s, n, most)).max();
        let n = a.agents.unwrap_or(s);
        let mut msg = format!("{n} agents of {} tokens each do not fit this machine with this model. What fits:", commas(ctx));
        if at_ctx > 1 {
            msg += &format!(" --agents {} --ctx {ctx}", at_ctx - 1);
        }
        if let Some(c) = at_n {
            msg += &format!("{} --agents {n} --ctx {c}", if at_ctx > 1 { " or" } else { "" });
        }
        msg
    })
}

/// Only the agents asked (`s` requests): the first notes whose longest context is at least [`MIN_CTX`], else the longest.
fn choose_ctx(a: &Ask, s: usize, top: usize, most: &impl Fn(Notes, usize) -> usize) -> Result<Choice, String> {
    let page = CELL_PAD as usize;
    let top = a.default_ctx.min(top);
    let mut best: Option<(Notes, usize)> = None;
    for &notes in &a.notes {
        let Some(ctx) = longest(top, s, notes, most) else { continue };
        if ctx >= MIN_CTX.min(top) {
            return Ok(Choice { ctx, streams: s, notes });
        }
        if best.is_none_or(|b| ctx > b.1) {
            best = Some((notes, ctx));
        }
    }
    best.map(|(notes, ctx)| Choice { ctx, streams: s, notes })
        .ok_or_else(|| format!("{} agents do not fit this machine with this model, even at {page} tokens each", a.agents.unwrap_or(s)))
}

/// How the engine ended up sized.
#[derive(Debug, Clone)]
pub struct Sized {
    /// `None`: the engine's defaults were left alone (no job given, or `--kv-pages`).
    pub choice: Option<Choice>,
    pub ctx: usize,
    pub streams: usize,
    pub kv_pages: Option<usize>,
    pub notes: Notes,
    pub reserved_bytes: u64,
}

impl Sized {
    /// The cache of this shape, as the engine will plan it.
    fn plan(p: &Profile, base: &PlanArgs, notes: Notes, ctx: usize, pages: Option<usize>) -> Budget {
        // `--kv-read f16`: what the readers allocate at this context (their rings and compute buffers included) comes out first
        let less = if notes.read_f16 { reader_reserve_bytes(p, ctx, base.n_ubatch, base.swa_full) } else { 0 };
        let args = PlanArgs { n_ctx: ctx, pool_pages: pages, kv_type: notes.kv, working_set: base.working_set.map(|w| w.saturating_sub(less)), ..base.clone() };
        plan_engine(p, &args).0
    }

    /// `base` is the engine's plan so far.
    pub fn new(p: &Profile, base: &PlanArgs, a: &Ask) -> Result<Sized, String> {
        let page = CELL_PAD as usize;
        let pages_of = |ctx: usize, streams: usize| ctx * streams / page;
        if a.agents.is_none() && a.ctx.is_none() {
            // --kv-pages can ask for more than the first notes hold: then the engine moves on too
            let want = base.pool_pages.map_or(0, |pg| pg as u64 * CELL_PAD);
            let plan = |n: Notes| Self::plan(p, base, n, base.n_ctx, base.pool_pages);
            let notes = a.notes.iter().copied().find(|&n| plan(n).cells >= want).unwrap_or(a.notes[0]);
            let b = plan(notes);
            return Ok(Sized { choice: None, ctx: base.n_ctx, streams: b.n_seq as usize, kv_pages: base.pool_pages, notes, reserved_bytes: b.kv_bytes(p, b.cells) });
        }
        // each plan reads libllama's compute buffer, so a shape is asked once
        let seen = RefCell::new(HashMap::new());
        let most = |notes: Notes, ctx: usize| {
            *seen.borrow_mut().entry((notes.kv.name(), notes.read_f16, ctx)).or_insert_with(|| {
                let b = Self::plan(p, base, notes, ctx, Some(pages_of(ctx, a.max_streams)));
                (b.max_cells / ctx as u64).min(MAX_TOTAL_CELLS / ctx as u64).min(a.max_streams as u64) as usize
            })
        };
        let c = choose(a, most)?;
        let pages = pages_of(c.ctx, c.streams);
        let b = Self::plan(p, base, c.notes, c.ctx, Some(pages));
        Ok(Sized { choice: Some(c), ctx: c.ctx, streams: b.n_seq as usize, kv_pages: Some(pages), notes: c.notes, reserved_bytes: b.kv_bytes(p, b.cells) })
    }

    /// The one line a developer reads at start.
    pub fn line(&self, model: &str, relief: bool, ssd_budget: Option<u64>) -> String {
        let gb = |b: u64| b as f64 / 1e9;
        let exact_rows = crate::exactness::row_limit_here().effective;
        format!(
            "{model}, {} requests at once, {} tokens each, {}, {:.1} GB reserved, pressure relief {}, SSD limit {}{}",
            self.streams,
            commas(self.ctx),
            self.notes.label(),
            gb(self.reserved_bytes),
            if relief { "on" } else { "off" },
            ssd_budget.map_or("none".to_string(), |b| format!("{:.0} GB", gb(b))),
            if self.streams > exact_rows { format!("; answers are exact only up to {exact_rows} writing at once") } else { String::new() },
        )
    }

    pub fn to_json(&self, line: &str, agents: Option<usize>) -> Value {
        json!({"line": line, "agents": agents, "at_once": self.streams, "ctx": self.ctx, "kv": self.notes.kv.name(), "read_f16": self.notes.read_f16,
               "kv_pages": self.kv_pages, "reserved_bytes": self.reserved_bytes, "sized": self.choice.is_some()})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const F16: Notes = Notes { kv: KvType::F16, read_f16: false };
    const Q4R: Notes = Notes { kv: KvType::Q4_0, read_f16: true };

    /// About Qwen3-4B on this Mac: 217,088 cells of f16, 3.5x that of 4-bit, and the engine's limits.
    fn most(n: Notes, ctx: usize) -> usize {
        let cells = if n.read_f16 { 760_000 } else { 217_088 };
        (cells / ctx).min(MAX_TOTAL_CELLS as usize / ctx).min(256)
    }

    fn ask(agents: Option<usize>, ctx: Option<usize>) -> Ask {
        Ask { agents, ctx, default_ctx: 4096, top_ctx: 40_960, notes: vec![F16, Q4R], max_streams: 256 }
    }

    #[test]
    fn agents_get_the_longest_context_and_room_for_a_chat() {
        assert_eq!(choose(&ask(Some(200), None), most).unwrap(), Choice { ctx: 1024, streams: 201, notes: F16 });
        assert_eq!(choose(&ask(Some(8), None), most).unwrap(), Choice { ctx: 4096, streams: 9, notes: F16 }, "never past the default");
        assert_eq!(choose(&ask(Some(50), None), most).unwrap().ctx, 4096);
        assert_eq!(choose(&ask(Some(100), None), most).unwrap().ctx, 2048);
    }

    #[test]
    fn smaller_notes_only_when_f16_would_leave_too_short_a_context() {
        // f16 holds 256 streams of 768 tokens: short, but over the floor, and f16 stays closer to solo answers
        assert_eq!(choose(&ask(Some(256), None), most).unwrap(), Choice { ctx: 768, streams: 256, notes: F16 });
        // a smaller machine: f16 would leave 256 tokens each, so 4-bit it is
        let small = |n: Notes, ctx: usize| if n.read_f16 { most(n, ctx) } else { (100_000 / ctx).min(256) };
        assert_eq!(choose(&ask(Some(256), None), small).unwrap(), Choice { ctx: 2816, streams: 256, notes: Q4R });
        // only f16 allowed (--kv-type f16): a short context rather than other notes
        let a = Ask { notes: vec![F16], ..ask(Some(256), None) };
        assert_eq!(choose(&a, small).unwrap(), Choice { ctx: 256, streams: 256, notes: F16 });
    }

    #[test]
    fn a_context_gets_as_many_requests_as_fit() {
        assert_eq!(choose(&ask(None, Some(32_768)), most).unwrap(), Choice { ctx: 32_768, streams: 6, notes: F16 });
        assert_eq!(choose(&ask(None, Some(1000)), most).unwrap().ctx, 1024, "rounded up to a page");
        assert_eq!(choose(&ask(None, Some(1 << 20)), most).unwrap().ctx, 40_960, "never past the model");
    }

    #[test]
    fn a_shape_that_does_not_fit_says_what_does() {
        assert_eq!(choose(&ask(Some(40), Some(4096)), most).unwrap(), Choice { ctx: 4096, streams: 41, notes: F16 });
        assert_eq!(choose(&ask(Some(60), Some(4096)), most).unwrap().notes, Q4R);
        let e = choose(&ask(Some(256), Some(4096)), most).unwrap_err();
        assert!(e.contains("--agents 184 --ctx 4096") && e.contains("--agents 256 --ctx 2816"), "{e}");
        let e = choose(&ask(Some(300), None), most).unwrap_err();
        assert!(e.contains("1 to 256"), "{e}");
        let tiny = |_: Notes, _: usize| 0;
        assert!(choose(&ask(Some(4), None), tiny).unwrap_err().contains("do not fit"));
        assert!(choose(&ask(None, Some(4096)), tiny).unwrap_err().contains("does not fit"));
    }

    #[test]
    fn four_bit_notes_only_when_allowed() {
        let (f16, q8, q4, q4r) = (F16, Notes { kv: KvType::Q8_0, read_f16: false }, Notes { kv: KvType::Q4_0, read_f16: false }, Q4R);
        let q8r = Notes { kv: KvType::Q8_0, read_f16: true };
        // --kv-type auto, --kv-read not given: f16, then q8_0; 4-bit read in f16 between them only when allowed
        assert_eq!(notes_for(true, true, false, KvType::F16, true, false), vec![f16, q8]);
        assert_eq!(notes_for(true, true, false, KvType::F16, true, true), vec![f16, q4r, q8]);
        assert_eq!(notes_for(true, true, false, KvType::F16, false, true), vec![f16, q8], "a model that cannot be read in f16");
        // --kv-read kv: plain notes
        assert_eq!(notes_for(true, false, false, KvType::F16, true, false), vec![f16, q8]);
        assert_eq!(notes_for(true, false, false, KvType::F16, true, true), vec![f16, q8, q4]);
        // --kv-read f16 with the type on auto: quantized notes read in f16, q8_0 unless 4-bit is allowed
        assert_eq!(notes_for(true, false, true, KvType::F16, true, false), vec![q8r]);
        assert_eq!(notes_for(true, false, true, KvType::F16, true, true), vec![q4r]);
        // an explicit --kv-type is never overridden by the flag
        assert_eq!(notes_for(false, true, false, KvType::Q4_0, true, false), vec![q4]);
        assert_eq!(notes_for(false, false, true, KvType::Q4_0, true, false), vec![q4r]);
    }

    #[test]
    fn a_job_that_needed_four_bit_gets_a_shorter_context_or_an_error() {
        // a small machine: f16 and q8_0 hold 100,000 and 190,000 cells, 4-bit read in f16 holds 760,000
        let small = |n: Notes, ctx: usize| {
            let cells = match (n.kv, n.read_f16) {
                (KvType::F16, _) => 100_000,
                (KvType::Q8_0, _) => 190_000,
                _ => 760_000,
            };
            (cells / ctx).min(256)
        };
        let q8 = Notes { kv: KvType::Q8_0, read_f16: false };
        let off = Ask { notes: notes_for(true, true, false, KvType::F16, true, false), ..ask(Some(256), None) };
        let on = Ask { notes: notes_for(true, true, false, KvType::F16, true, true), ..ask(Some(256), None) };
        assert_eq!(choose(&on, small).unwrap(), Choice { ctx: 2816, streams: 256, notes: Q4R });
        assert_eq!(choose(&off, small).unwrap(), Choice { ctx: 512, streams: 256, notes: q8 }, "q8_0, the next option that fits");
        // a context the off notes cannot hold says what fits
        let off = Ask { notes: notes_for(true, true, false, KvType::F16, true, false), ..ask(Some(256), Some(4096)) };
        let e = choose(&off, small).unwrap_err();
        assert!(e.contains("do not fit") && e.contains("--agents 256 --ctx 512"), "{e}");
    }
}
