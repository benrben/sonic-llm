//! The helper thread: the helper's model, its KV cache, the step loop and the four operations as async jobs.
//!
//! A job is plain async code over [`Seq`]s (sequences of the helper's libllama executor, [`Exec`]): `prefill` and `decode` submit a
//! request and resolve when the loop has run it. The loop never blocks a job on another: every iteration it polls the
//! jobs, gathers what they asked for into one batched step (decode rows first, then prefill chunks up to the step's
//! token budget, compaction calls first), runs it and polls again. Dropping a job (finished, failed, cancelled by its
//! client, past its deadline) drops its sequences, which frees their pages at once.

use super::chunk::{chunk_text, select_within_budget};
use super::spans::{choose, locate, Choice, Span, SpanState};
use super::*;
use futures_util::future::join_all;
use super::exec::{Exec, ExecOpts, ItemOut, OutputPolicy, Sampler, SeqId as PoolAgent, StepItem};
use crate::engine::PAGE_CELLS;
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

mod jobs;
mod prepare;
mod step_loop;

use step_loop::Loop;

type Res<T> = Result<T, EngineError>;

/// Decode rows one helper step can carry.
const MAX_ROWS: usize = 64;
/// Chunks a compress call scores at once (bounds its pages).
const COMPRESS_WAVE: usize = 16;

/// Prefill tokens of ordinary calls in one helper step (compaction calls may use the whole step budget).
const ORDINARY_STEP_TOKENS: usize = 512;

/// The chat framing around a user turn and the start of the model's answer, found from the model's own vocabulary (any GGUF
/// can be the helper): the special tokens of Gemma's, ChatML's (Qwen3 non-thinking, `enable_thinking=false`, when the
/// vocabulary has `<think>`) and Llama-3's turn formats; a model with none of them gets plain text.
fn template(vocab: &Vocab) -> (&'static str, &'static str) {
    let has = |tok: &str| vocab.tokenize(tok, false, true).is_ok_and(|v| v.len() == 1);
    if has("<start_of_turn>") {
        ("<bos><start_of_turn>user\n", "<end_of_turn>\n<start_of_turn>model\n")
    } else if has("<|im_start|>") {
        if has("<think>") {
            (USER, ASSIST)
        } else {
            (USER, "<|im_end|>\n<|im_start|>assistant\n")
        }
    } else if has("<|start_header_id|>") {
        ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    } else {
        ("", "\n\nAnswer: ")
    }
}

const USER: &str = "<|im_start|>user\n";
/// The end of the user turn and the start of a non-thinking assistant turn (Qwen3's template, `enable_thinking=false`).
const ASSIST: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

fn invalid(m: impl Into<String>) -> EngineError {
    EngineError::new(422, "invalid_request_error", m)
}
fn server_err(m: impl Into<String>) -> EngineError {
    EngineError::new(500, "server_error", m)
}
fn pages(cells: usize) -> usize {
    cells.div_ceil(PAGE_CELLS)
}

// -- requests the jobs hand to the loop ---------------------------------------------------------------------------

struct Req {
    agent: PoolAgent,
    tokens: Vec<i32>,
    decode: bool,
    out: OutputPolicy,
    sampler: Sampler,
    prio: bool,
    order: u64,
    done: Cell<usize>,
    result: RefCell<Option<Result<ItemOut, String>>>,
}

struct ReqFut(Rc<Req>);

impl Future for ReqFut {
    type Output = Result<ItemOut, String>;
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        match self.0.result.borrow_mut().take() {
            Some(r) => Poll::Ready(r),
            None => Poll::Pending,
        }
    }
}

#[derive(Default)]
struct Stats {
    steps: u64,
    step_us: u64,
    prefill_tokens: u64,
    decode_rows: u64,
    failed_steps: u64,
    completed: [u64; 4],
    failed: [u64; 4],
    cancelled: u64,
    expired: u64,
    deferred: u64,
    run_ms: [f64; 4],
    queued_ms: [f64; 4],
    max_queued: usize,
}

struct Ctx {
    bf: RefCell<Exec>,
    vocab: Arc<Vocab>,
    n_vocab: usize,
    eog: Vec<i32>,
    /// Tokens that end a span: whitespace with a newline in it.
    seps: Vec<i32>,
    /// The bytes of every token (special tokens render as nothing).
    pieces: Vec<Vec<u8>>,
    /// Bullet and numbering tokens ("-", "*", "1.") a model puts in front of a listed span: allowed before a span
    /// starts, never part of it.
    bullets: Vec<i32>,
    /// The chat template: the start of the user turn, and its end plus the start of the answer.
    user: Vec<i32>,
    assist: Vec<i32>,
    cfg: HelperConfig,
    submitted: RefCell<Vec<Rc<Req>>>,
    next_order: Cell<u64>,
    stats: RefCell<Stats>,
}

impl Ctx {
    /// Tokens of caller text (special-token text stays text).
    fn tx(&self, s: &str) -> Vec<i32> {
        self.vocab.tokenize(s, false, false).unwrap_or_default()
    }
    fn first_tokens(&self, words: &[String]) -> Vec<i32> {
        let mut v: Vec<i32> = words.iter().filter_map(|w| self.tx(w).first().copied()).collect();
        v.sort();
        v.dedup();
        v
    }
}

/// One sequence of the helper's executor. Dropped, it frees its pages.
struct Seq<'c> {
    ctx: &'c Ctx,
    id: PoolAgent,
    prio: bool,
}

impl<'c> Seq<'c> {
    fn new(ctx: &'c Ctx, capacity: usize, prio: bool) -> Res<Seq<'c>> {
        let id = ctx.bf.borrow_mut().add_agent(capacity).map_err(|e| server_err(format!("helper sequence: {e}")))?;
        Ok(Seq { ctx, id, prio })
    }

    /// A sequence sharing every cell of this one (copy-on-write).
    fn fork(&self) -> Res<Seq<'c>> {
        let id = self.ctx.bf.borrow_mut().fork_agent(self.id).map_err(|e| server_err(format!("helper fork: {e}")))?;
        Ok(Seq { ctx: self.ctx, id, prio: self.prio })
    }

    fn submit(&self, tokens: Vec<i32>, decode: bool, out: OutputPolicy, sampler: Sampler) -> ReqFut {
        let order = self.ctx.next_order.get();
        self.ctx.next_order.set(order + 1);
        let r = Rc::new(Req {
            agent: self.id,
            tokens,
            decode,
            out,
            sampler,
            prio: self.prio,
            order,
            done: Cell::new(0),
            result: RefCell::new(None),
        });
        self.ctx.submitted.borrow_mut().push(r.clone());
        ReqFut(r)
    }

    /// Prefill `tokens`; `out` says which rows come back (logits with `HostLogits`, tokens with `Greedy`). The loop cuts a
    /// long prefill into chunks; an `All` prefill runs in one step, so it must fit the step's token budget.
    async fn prefill(&self, tokens: Vec<i32>, out: OutputPolicy, sampler: Sampler) -> Res<ItemOut> {
        if tokens.is_empty() {
            return Ok(ItemOut::default());
        }
        if out == OutputPolicy::All && tokens.len() > self.ctx.cfg.step_tokens {
            return Err(invalid(format!("{} scored tokens are more than a helper step ({})", tokens.len(), self.ctx.cfg.step_tokens)));
        }
        self.submit(tokens, false, out, sampler).await.map_err(server_err)
    }

    async fn decode(&self, tok: i32, sampler: Sampler) -> Res<ItemOut> {
        self.submit(vec![tok], true, OutputPolicy::Last, sampler).await.map_err(server_err)
    }

    fn len(&self) -> Res<usize> {
        self.ctx.bf.borrow().agent_len(self.id).map_err(|e| server_err(e.to_string()))
    }

    fn truncate(&self, len: usize) -> Res<()> {
        self.ctx.bf.borrow_mut().truncate(self.id, len).map_err(|e| server_err(e.to_string()))
    }

    async fn logits(&self, tokens: Vec<i32>) -> Res<Vec<f32>> {
        self.prefill(tokens, OutputPolicy::Last, Sampler::HostLogits).await?.logits.ok_or_else(|| server_err("no logits"))
    }

    /// Decode `tok` and return the logits of the next token.
    async fn decode_logits(&self, tok: i32) -> Res<Vec<f32>> {
        self.decode(tok, Sampler::HostLogits).await?.logits.ok_or_else(|| server_err("no logits"))
    }
}

impl Drop for Seq<'_> {
    fn drop(&mut self) {
        let _ = self.ctx.bf.borrow_mut().remove_agent(self.id);
    }
}

fn logsumexp(xs: impl Iterator<Item = f32>) -> f32 {
    let v: Vec<f32> = xs.collect();
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f32>().ln()
}

fn text_of(pieces: &[Vec<u8>], ids: &[i32]) -> String {
    let bytes: Vec<u8> = ids.iter().flat_map(|&t| pieces[t as usize].iter().copied()).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Output rows (full logits) one scored-prefill item may carry.
const MAX_OUT_ROWS: usize = 256;
/// Output rows (decode rows, final prefill rows and scored rows) one helper step may carry: the context is built for no
/// more, which keeps its logits buffer small.
const MAX_STEP_OUTPUTS: usize = MAX_OUT_ROWS + MAX_ROWS;
// -- the thread ---------------------------------------------------------------------------------------------------

/// A token that ends a span: whitespace with a newline in it.
fn is_sep(p: &[u8]) -> bool {
    !p.is_empty() && p.contains(&b'\n') && p.iter().all(|b| b.is_ascii_whitespace())
}

/// A bullet or numbering token ("-", "*", "1.") a model puts in front of a listed span.
fn is_bullet(piece: &[u8]) -> bool {
    let p = String::from_utf8_lossy(piece);
    let w = p.trim_matches(|c: char| c == ' ' || c == '\t');
    let numbered = w.len() > 1 && w.len() <= 4 && w.ends_with(['.', ')']) && w[..w.len() - 1].chars().all(|c| c.is_ascii_digit());
    !w.is_empty() && !p.contains('\n') && (["-", "*", "\u{2022}", "+"].contains(&w) || numbered)
}

fn exec_opts(cfg: &HelperConfig) -> ExecOpts {
    ExecOpts {
        model: cfg.model.clone(),
        cells: cfg.pool_pages.max(1) * PAGE_CELLS,
        // every call's sequences each reserve at least a page, so the sequences alive at once are bounded by the pool and the slots
        max_seqs: cfg.pool_pages + cfg.slots + 8,
        max_tokens: cfg.step_tokens.max(1) + MAX_ROWS,
        max_outputs: MAX_STEP_OUTPUTS,
        kv_type: cfg.kv_type,
        cpu_threads: cfg.cpu_threads,
    }
}

pub(super) fn run(cfg: HelperConfig, rx: mpsc::Receiver<Msg>, shared: Arc<HelperShared>) {
    let fail = |m: String| {
        *shared.init_error.lock().unwrap() = Some(m);
    };
    let _ = sllm_core::backend::registry();
    let vocab = match load_vocab(&cfg.model) {
        Ok(v) => v,
        Err(e) => return fail(format!("vocabulary: {e}")),
    };
    let exec = match Exec::load(&exec_opts(&cfg)) {
        Ok(e) => e,
        Err(e) => return fail(format!("loading {}: {e}", cfg.model.display())),
    };
    if vocab.n_vocab() != exec.n_vocab() {
        return fail(format!("vocabulary of {} tokens does not match the model's {}", vocab.n_vocab(), exec.n_vocab()));
    }
    let facts = exec.facts.clone();
    let n_vocab = exec.n_vocab();
    let bf = exec;
    let pieces: Vec<Vec<u8>> = (0..vocab.n_vocab() as i32).map(|t| vocab.piece(t)).collect();
    let seps: Vec<i32> = (0..pieces.len()).filter(|&t| is_sep(&pieces[t])).map(|t| t as i32).collect();
    let bullets: Vec<i32> = (0..pieces.len()).filter(|&t| is_bullet(&pieces[t])).map(|t| t as i32).collect();
    let tpl = template(&vocab);
    let ctx = Ctx {
        bf: RefCell::new(bf),
        n_vocab,
        eog: vocab.eog_ids().to_vec(),
        seps,
        bullets,
        user: vocab.tokenize(tpl.0, false, true).unwrap_or_default(),
        assist: vocab.tokenize(tpl.1, false, true).unwrap_or_default(),
        pieces,
        vocab,
        cfg: cfg.clone(),
        submitted: RefCell::new(Vec::new()),
        next_order: Cell::new(1),
        stats: RefCell::new(Stats::default()),
    };
    let _ = shared.info.set(json!({
        "config": cfg.to_json(),
        "model": serde_json::to_value(&facts).ok(),
        "weights_bytes": facts.weights_bytes,
        "thread": "sllm-helper: its own libllama model, context and KV cache (one thread per model)",
        "endpoints": OPS.iter().map(|o| format!("/v1/helper/{o}")).collect::<Vec<_>>(),
    }));
    let mut lp = Loop { ctx: &ctx, rx, shared: shared.clone(), queue: VecDeque::new(), active: Vec::new(), inflight: Vec::new(), next: 0 };
    lp.publish();
    shared.ready.store(true, Relaxed);
    lp.run();
    // `lp` (and every job and sequence in it) goes before `ctx`, which frees the model and its Metal buffers
}

/// Chunks a qlik wave scores at once: their question rows fit one step's output rows.
fn qlik_wave(q_len: usize) -> usize {
    (MAX_OUT_ROWS / q_len.max(1)).clamp(1, COMPRESS_WAVE)
}

/// The default compress scoring.
const DEFAULT_COMPRESS: &str = "yesno";

fn deadline_err() -> EngineError {
    EngineError::new(504, "deadline_error", "the helper call passed its deadline_ms")
}
