//! Compaction and the extracted-fields record (phase 3).
//!
//! # The window
//!
//! An agent's context is `[system prompt] + [summary] + [recent tail]`. The system prompt is the agent's **prefix**: its
//! pages (shared with every agent that has the same system prompt) are never touched. A compaction replaces everything
//! between the prefix and the tail (the previous summary and the older turns) with a new summary. Cache entries depend
//! on their position, so the tail cannot be kept as it is: the engine cuts the agent back to its prefix
//! ([`crate::engine::Engine::agent_cut`], which frees the pages of everything after it) and then prefills the summary
//! and re-prefills the tail. That prefill runs as an ordinary request of the agent on the **helper lane** (the reserved
//! helper slot, rule 073, with its guaranteed share of every step's token budget), with `n_predict` 0: it is planned,
//! chunked and batched with everything else, and it never waits behind the background workers whose pages it frees.
//! The result is exactly the cache a from-scratch run over `[prefix + summary + tail]` would build.
//!
//! # Triggers
//!
//! `POST /v1/agents/{id}/compact` compacts now. A [`Policy`] makes the server compact on its own after a turn ends:
//! when the window reaches `window_tokens`, or when `every_tokens` tokens were added since the last compaction (never
//! "before every call"). A turn that arrives while its agent compacts is held and started when the compaction is done.
//!
//! # The summary
//!
//! The client passes the summary (text or ids), or a [`Summarizer`] writes it from the replaced tokens. With a helper
//! model (`--helper-model`) that is `helper-summarize` (`crate::helper::HelperSummarizer`, asynchronous: the helper
//! lane's compaction slot writes it while the agent stays as it is, and the cut starts when it is there). Without one,
//! [`FirstTokensStub`], a **stub**: the first `summary_tokens` tokens of the replaced range. Every report names its
//! `summary_source`.
//!
//! # Rollover
//!
//! A compaction that a turn does not wait for ([`crate::executor`], `rollover.rs`). A turn that would run past the end of
//! its window is stopped just short of it; the agent writes its own notes in its own stream (no second model, nothing is
//! read again), the window is compacted to those notes, the facts record and the tail, and the same request goes on. A
//! prompt longer than the window is read in slices the same way. A turn that fits never sees any of this.
//!
//! Nothing the window loses is lost: [`Archive`] keeps the lines the agent has seen, and a new turn gets back, word for word, the few that
//! share its rare words. No model, no GPU, nothing a summary could have got wrong.
//!
//! # The extracted-fields record
//!
//! [`Fields`]: a small per-agent key/value record (IDs, emails, state) kept outside the window, returned with the agent
//! (`GET /v1/agents/{id}`) and read or written with `/v1/agents/{id}/fields`. Values are JSON scalars. The helper's
//! extract endpoint will fill it; compaction does not touch it.

use crate::engine::{AgentId, EngineError, ReqId, PAGE_CELLS};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

mod facts;
mod state;

pub use facts::{facts_cap, line_has_fact, select_fact_lines, COMPACTION_FACTS_CAP, FACTS_HEADER};
pub use state::{compacting, AgentState, CompactionPhase, PhaseEv, Fields, FieldsOp, Report, MAX_FIELDS, MAX_KEY_BYTES, MAX_VALUE_BYTES};

/// Request ids of the executor's own compaction requests (the HTTP side counts up from 1).
pub const COMPACT_REQ_BASE: ReqId = 1 << 60;

pub fn is_compaction(req: ReqId) -> bool {
    req >= COMPACT_REQ_BASE
}

pub const DEFAULT_TAIL_TOKENS: usize = 1024;
pub const DEFAULT_SUMMARY_TOKENS: usize = 300;

fn default_tail() -> usize {
    DEFAULT_TAIL_TOKENS
}
fn default_summary() -> usize {
    DEFAULT_SUMMARY_TOKENS
}

/// When the server compacts an agent on its own, and to what. Checked after each of the agent's turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    /// Compact once the window holds at least this many tokens.
    #[serde(default)]
    pub window_tokens: Option<usize>,
    /// Compact once this many tokens were added since the last compaction (or since the agent was made).
    #[serde(default)]
    pub every_tokens: Option<usize>,
    /// The recent tail that is kept (re-prefilled after the summary).
    #[serde(default = "default_tail")]
    pub tail_tokens: usize,
    /// The summary length asked of the summarizer.
    #[serde(default = "default_summary")]
    pub summary_tokens: usize,
}

impl Policy {
    /// The default tail and summary, no trigger yet (the CLI fills one in).
    pub fn untriggered() -> Policy {
        Policy { window_tokens: None, every_tokens: None, tail_tokens: DEFAULT_TAIL_TOKENS, summary_tokens: DEFAULT_SUMMARY_TOKENS }
    }

    /// A policy must be able to shrink the window: a window trigger below what a compaction leaves
    /// (prefix + summary + tail + one page) would fire after every turn.
    pub fn validate(&self, prefix_tokens: usize) -> Result<(), String> {
        if self.window_tokens.is_none() && self.every_tokens.is_none() {
            return Err("a compaction policy needs window_tokens or every_tokens".into());
        }
        let floor = prefix_tokens + self.summary_tokens + self.tail_tokens + PAGE_CELLS;
        if let Some(w) = self.window_tokens {
            if w < floor {
                return Err(format!(
                    "window_tokens {w} is below what a compaction leaves plus one page ({prefix_tokens} prefix + {} summary + {} tail + 256 = {floor})",
                    self.summary_tokens, self.tail_tokens
                ));
            }
        }
        if self.every_tokens == Some(0) {
            return Err("every_tokens must be positive".into());
        }
        Ok(())
    }

    /// The trigger that fires for a window of `context` tokens, `since` of them added after the last compaction.
    pub fn due(&self, context: usize, since: usize) -> Option<Trigger> {
        if self.window_tokens.is_some_and(|w| context >= w) {
            return Some(Trigger::Window);
        }
        if self.every_tokens.is_some_and(|e| since >= e) {
            return Some(Trigger::Every);
        }
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Trigger {
    /// `POST /v1/agents/{id}/compact`.
    Api,
    /// The policy's `window_tokens`.
    Window,
    /// The policy's `every_tokens`.
    Every,
    /// A turn reached the end of the window (a rollover).
    Limit,
}

/// What to compact: the HTTP request, or a policy trigger.
#[derive(Debug, Clone)]
pub struct Spec {
    /// The summary's ids; `None` asks the summarizer.
    pub summary: Option<Vec<i32>>,
    /// Keep the last `tail_tokens` tokens (default [`DEFAULT_TAIL_TOKENS`]) ...
    pub tail_tokens: Option<usize>,
    /// ... or the tail from this context position on.
    pub tail_from: Option<usize>,
    /// Tokens kept in front (default: the system prompt's length).
    pub prefix_tokens: Option<usize>,
    /// The summary length asked of the summarizer (default [`DEFAULT_SUMMARY_TOKENS`]).
    pub summary_tokens: Option<usize>,
    pub trigger: Trigger,
    /// Set when an asynchronous summarizer wrote `summary`: its name, how long it took, and when the compaction began.
    pub summarized_by: Option<(String, f64, std::time::Instant)>,
    /// The window to compact, when it is not what the engine holds now (a rollover: the notes it just wrote are not part of
    /// it, and the turn's last token is read again by the next leg).
    pub context: Option<Vec<i32>>,
    /// Text around the summary and the facts, in front of the tail.
    pub frame: Option<(Vec<i32>, Vec<i32>)>,
}

impl Spec {
    pub fn from_policy(p: &Policy, trigger: Trigger) -> Spec {
        Spec { summary: None, tail_tokens: Some(p.tail_tokens), tail_from: None, prefix_tokens: None, summary_tokens: Some(p.summary_tokens), trigger, summarized_by: None, context: None, frame: None }
    }
}

/// Where a compaction cuts an agent's context of `len` tokens: `(prefix, tail_start)`, with `prefix <= tail_start <=
/// len`. Everything in `prefix..tail_start` is replaced by the summary.
pub fn cut_points(len: usize, prefix: usize, tail_tokens: Option<usize>, tail_from: Option<usize>) -> Result<(usize, usize), String> {
    if prefix > len {
        return Err(format!("prefix_tokens {prefix} is longer than the context ({len} tokens)"));
    }
    let tail_start = match tail_from {
        Some(t) if t > len => return Err(format!("tail_from {t} is past the end of the context ({len} tokens)")),
        Some(t) if t < prefix => return Err(format!("tail_from {t} is inside the prefix ({prefix} tokens)")),
        Some(t) => t,
        None => len.saturating_sub(tail_tokens.unwrap_or(DEFAULT_TAIL_TOKENS)).max(prefix),
    };
    if tail_start == prefix {
        return Err(format!("nothing to compact: the context ({len} tokens) is the prefix ({prefix}) and the tail"));
    }
    Ok((prefix, tail_start))
}

/// The new window's content after the prefix: the summary, then the tail.
pub fn feed(context: &[i32], tail_start: usize, summary: &[i32]) -> Vec<i32> {
    summary.iter().chain(&context[tail_start..]).copied().collect()
}

// ---- the summary hook ----------------------------------------------------------------------------------------------

/// Writes the summary of the replaced tokens. [`Summarizer::summarize`] is called on the executor thread, so it must be
/// quick. A summarizer that takes long (the helper model, phase3-helper) implements [`Summarizer::start`] instead: the
/// executor then leaves the agent as it is, polls the [`PendingSummary`] at every step boundary, and starts the cut when
/// the summary is there (turns that ran meanwhile only lengthen the tail: the cut keeps the tail from the same position).
pub trait Summarizer: Send + Sync {
    /// What reports call it (`summary_source`).
    fn name(&self) -> &'static str;
    /// The summary now. The default is for a summarizer that only works asynchronously ([`Summarizer::start`], the helper's).
    fn summarize(&self, _agent: &str, _replaced: &[i32], _target_tokens: usize) -> Result<Vec<i32>, String> {
        Err("the helper summarizer is asynchronous (start)".into())
    }
    /// The asynchronous form: `None` (the default) means "use `summarize`".
    fn start(&self, _agent: &str, _replaced: &[i32], _target_tokens: usize) -> Option<Box<dyn PendingSummary>> {
        None
    }
    /// The facts record ([`select_fact_lines`]) of the replaced tokens, as brain token ids to place after the summary
    /// (and how many lines it keeps). `None` (the default): this summarizer keeps no facts record.
    fn facts(&self, _replaced: &[i32], _tail: &[i32]) -> Option<(Vec<i32>, usize)> {
        None
    }
}

/// A summary being written elsewhere (the helper thread). Polled on the executor thread; must not block.
pub trait PendingSummary: Send {
    /// `None` while it is not done; then the summary's ids (the brain's vocabulary) or an error, once.
    fn poll(&mut self) -> Option<Result<Vec<i32>, String>>;
}

/// STUB (until phase3-helper's summarize endpoint is merged): the "summary" is the first `target_tokens` tokens of the
/// replaced range. It measures the mechanics (pages, positions, cost); it says nothing about summary quality.
pub struct FirstTokensStub;

impl Summarizer for FirstTokensStub {
    fn name(&self) -> &'static str {
        "stub-first-n"
    }
    fn summarize(&self, _agent: &str, replaced: &[i32], target_tokens: usize) -> Result<Vec<i32>, String> {
        Ok(replaced[..target_tokens.min(replaced.len())].to_vec())
    }
}

/// The server's summarizer (a shared handle, so the server config stays `Clone`).
#[derive(Clone)]
pub struct SummarizerHandle(pub Arc<dyn Summarizer>);

impl Default for SummarizerHandle {
    fn default() -> Self {
        SummarizerHandle(Arc::new(FirstTokensStub))
    }
}

impl std::fmt::Debug for SummarizerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Summarizer({})", self.0.name())
    }
}

// ---- rollover ------------------------------------------------------------------------------------------------------

pub const DEFAULT_ROLL_SUMMARY: usize = 128;
pub const DEFAULT_ROLL_MAX: u32 = u32::MAX; // no cap (Ben 2026-11-09; was 64): a turn ends when the model stops, at its n_predict, or when cancelled

/// What follows the prefix must be at least this long for a rollover to leave room to go on; a smaller window keeps the
/// plain limits.
pub const ROLL_MIN_ROOM: usize = 512;

/// What the agent is asked for at the end of its window.
pub const NOTES_ASK: &str = "Pause the work. Everything above is about to be erased, and only what you write now will be left. Write down \
what a colleague needs to carry on: what the user asked for, what is finished, what is left, the next step, and the names, numbers, \
IDs and file paths that still matter. Do not repeat this message. Start with what the user asked for.";
/// In front of the end of a prompt that is read in slices: the model does not know yet what the text is for.
const NOTES_REQUEST: &str = "\n\nYou are reading a long text in parts. It ends with this request:\n";
/// How much of the end of such a prompt the ask quotes.
pub const NOTES_REQUEST_TOKENS: usize = 128;
const RECALL_HEAD: &str = "Lines from earlier in this work that may help:\n";
const NOTES_HEAD_SHORT: &str = "The earlier part of this conversation was removed to free memory.\n";
const NOTES_HEAD: &str = "The earlier part of this conversation was removed to free memory. Your notes on it:\n";
const NOTES_FOOT_WRITING: &str = "\nYour reply so far continues below. Keep writing it from where it stops.";
/// In front of the request a turn is working on, kept word for word through its rollovers: the target of a long job must
/// not depend on what the notes happen to say.
const NOTES_WORKING_ON: &str = "The request you are working on:\n";
const NOTES_YOUR_NOTES: &str = "\n\nYour notes on the work so far:\n";
/// The reply to the notes when real turns follow them.
const NOTES_REPLY: &str = "Understood.";
const NOTES_FOOT_READING: &str = "\nThe latest part follows word for word.\n\n";

/// The rollover settings of the server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Rollover {
    pub on: bool,
    /// The most tokens of notes.
    pub summary_tokens: usize,
    /// The recent tail that is kept.
    pub tail_tokens: usize,
    /// The most rollovers of one turn: past it the plain limits stand, so a turn that never ends does end.
    pub max_per_turn: u32,
}

impl Default for Rollover {
    fn default() -> Rollover {
        Rollover { on: false, summary_tokens: DEFAULT_ROLL_SUMMARY, tail_tokens: DEFAULT_TAIL_TOKENS, max_per_turn: DEFAULT_ROLL_MAX }
    }
}

impl Rollover {
    /// Notes and tail for `room` tokens behind the prefix: `None` when it is too small to roll over.
    pub fn sizes(&self, room: usize) -> Option<(usize, usize)> {
        (room >= ROLL_MIN_ROOM).then(|| (self.summary_tokens.min(room / 16), self.tail_tokens.min(room / 8)))
    }
}

/// A model's turn frames, read off its chat template ([`crate::chat::ChatTemplate::turn_frames`]).
#[derive(Debug, Clone, PartialEq)]
pub struct TurnFrames {
    pub user_open: String,
    /// From the end of a reply to the start of a user turn.
    pub to_user: String,
    /// From the end of a user turn to the start of a reply.
    pub to_reply: String,
}

impl TurnFrames {
    /// For a model without a chat template.
    pub fn plain() -> TurnFrames {
        TurnFrames { user_open: "User: ".into(), to_user: "\n\nUser: ".into(), to_reply: "\n\nAssistant: ".into() }
    }
}

type Encode = Arc<dyn Fn(&str) -> Vec<i32> + Send + Sync>;
type Decode = Arc<dyn Fn(&[i32]) -> String + Send + Sync>;
type FactsOf = Arc<dyn Fn(&[i32], &[i32], usize) -> Option<(Vec<i32>, usize)> + Send + Sync>;

/// The rollover's words as the brain's tokens.
#[derive(Clone)]
pub struct RolloverText {
    /// The ask after a reply in progress, and in the middle of a prompt.
    pub ask_writing: Vec<i32>,
    pub ask_reading: Vec<i32>,
    /// In front of the quoted end of the prompt.
    pub request: Vec<i32>,
    /// From the end of a user turn to the start of a reply: not part of what the ask quotes.
    pub to_reply: Vec<i32>,
    pub head: Vec<i32>,
    /// Around the request a turn is working on, in front of the notes.
    pub working_on: Vec<i32>,
    pub your_notes: Vec<i32>,
    /// The model's start of text: a window that starts with it keeps it, whatever else is cut.
    pub bos: Vec<i32>,
    /// What opens a user turn, and what opens the lines [`Archive::recall`] finds.
    pub user_open: Vec<i32>,
    pub recall_head: Vec<i32>,
    /// What goes between the notes and the tail, after a reply in progress and in the middle of a prompt.
    pub foot_writing: Vec<i32>,
    pub foot_reading: Vec<i32>,
    /// What goes between the notes and a tail that starts with a turn: they are a finished exchange of their own.
    pub foot_turns: Vec<i32>,
    /// The facts record of the replaced tokens ([`select_fact_lines`]), when the brain's vocabulary is there.
    pub facts: Option<FactsOf>,
    /// Text to tokens and back. Text is never read as special tokens: what a model or a document wrote stays text.
    pub encode: Encode,
    pub decode: Decode,
}

impl RolloverText {
    /// `frames` reads special tokens in its text, `encode` does not.
    pub fn new(f: &TurnFrames, bos: &str, frames: Encode, encode: Encode, decode: Decode, facts: Option<FactsOf>) -> RolloverText {
        let e = |t: String| frames(&t);
        RolloverText {
            bos: frames(bos),
            working_on: frames(&format!("{}{NOTES_HEAD_SHORT}{NOTES_WORKING_ON}", f.user_open)),
            your_notes: encode(NOTES_YOUR_NOTES),
            user_open: e(f.user_open.clone()),
            recall_head: encode(RECALL_HEAD),
            ask_writing: e(format!("{}{NOTES_ASK}{}", f.to_user, f.to_reply)),
            ask_reading: e(format!("\n\n{NOTES_ASK}{}", f.to_reply)),
            request: e(NOTES_REQUEST.to_string()),
            to_reply: e(f.to_reply.clone()),
            head: e(format!("{}{NOTES_HEAD}", f.user_open)),
            foot_writing: e(format!("{NOTES_FOOT_WRITING}{}", f.to_reply)),
            foot_reading: e(NOTES_FOOT_READING.to_string()),
            foot_turns: e(format!("{}{NOTES_REPLY}{}", f.to_reply, f.to_user.strip_suffix(f.user_open.as_str()).unwrap_or(""))),
            facts,
            encode,
            decode,
        }
    }

    /// `prompt` with `lines` in it, at the end of its user turn: right after what it asks, where the reply starts. A prompt
    /// read in slices has a long text in front of that, and a line at its start would be forgotten by the time it is needed.
    /// A prompt without a frame gets them in front of it.
    pub fn with_recall(&self, prompt: &[i32], lines: &str, room: usize) -> Vec<i32> {
        let mut text: Vec<i32> = self.recall_head.iter().copied().chain((self.encode)(lines)).collect();
        text.truncate(room);
        match prompt.strip_suffix(self.to_reply.as_slice()) {
            Some(body) if !self.to_reply.is_empty() => body.iter().chain(&(self.encode)("\n\n")).chain(&text).chain(&self.to_reply).copied().collect(),
            _ => text.iter().chain(&(self.encode)("\n\n")).chain(prompt).copied().collect(),
        }
    }
}

/// How much of the end of a prompt says what it is about: the ask, not the document in front of it.
pub const RECALL_QUERY_TOKENS: usize = 48;
/// A line with fewer words is a frame or a fragment, not something an agent could need back.
const ARCHIVE_MIN_WORDS: usize = 3;
/// The most lines a turn gets back.
const RECALL_LINES: usize = 8;
const ARCHIVE_BYTES: usize = 8 << 20;

fn words(line: &str) -> impl Iterator<Item = String> + '_ {
    line.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() >= 3).map(str::to_lowercase)
}

/// The words of `query` that name something: a number or an ID, a file path, a capital inside a sentence. Plain words say
/// what is asked, not what about, and in a list of similar lines they point at the wrong one.
fn names(query: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = true;
    for tok in query.split_whitespace() {
        let t = tok.trim_matches(|c: char| !c.is_alphanumeric());
        let inner_mark = t.contains(['_', '/', '.', ':']);
        let named = t.chars().any(|c| c.is_ascii_digit()) || inner_mark || (!start && t.chars().any(char::is_uppercase)) || t.chars().skip(1).any(char::is_uppercase);
        if named {
            out.extend(words(t));
        }
        start = tok.ends_with(['.', '?', '!', ':']);
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// The lines an agent has seen, each once, with a word index. A word that many lines have says nothing about any of them; one
/// that few have, a name or an ID, finds them.
#[derive(Debug, Clone, Default)]
pub struct Archive {
    lines: Vec<String>,
    /// Word to the lines that have it, in order.
    index: std::collections::HashMap<String, Vec<u32>>,
    seen: std::collections::HashSet<String>,
    bytes: usize,
}

impl Archive {
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    pub fn add(&mut self, text: &str) {
        for l in text.lines().map(str::trim).filter(|l| l.split_whitespace().count() >= ARCHIVE_MIN_WORDS) {
            if self.seen.contains(l) {
                continue;
            }
            let at = self.lines.len() as u32;
            let mut ws: Vec<String> = words(l).collect();
            ws.sort_unstable();
            ws.dedup();
            for w in ws {
                self.index.entry(w).or_default().push(at);
            }
            self.seen.insert(l.to_string());
            self.lines.push(l.to_string());
            self.bytes += l.len();
        }
        if self.bytes > ARCHIVE_BYTES {
            // the older half goes
            let keep = self.lines.split_off(self.lines.len() / 2);
            *self = Archive::default();
            self.add(&keep.join("\n"));
        }
    }

    /// The lines that share the rarest names in `query` ([`names`]), in their original order, one per line. A name in more
    /// lines than a turn gets back is not specific to any of them, and a question is not an answer: a question about
    /// something the agent never saw finds nothing, not a neighbour's line that the model would repeat as the answer.
    pub fn recall(&self, query: &str) -> Option<String> {
        let mut score: std::collections::HashMap<u32, f64> = Default::default();
        for w in names(query) {
            let Some(p) = self.index.get(&w).filter(|p| p.len() <= RECALL_LINES) else { continue };
            for &l in p {
                if !self.lines[l as usize].contains('?') {
                    *score.entry(l).or_default() += 1.0 / p.len() as f64;
                }
            }
        }
        let best = score.values().copied().fold(0.0, f64::max);
        let mut hits: Vec<(u32, f64)> = score.into_iter().filter(|&(_, s)| s * 2.0 >= best).collect();
        hits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        hits.truncate(RECALL_LINES);
        hits.sort_by_key(|h| h.0);
        (!hits.is_empty()).then(|| hits.iter().map(|h| self.lines[h.0 as usize].as_str()).collect::<Vec<_>>().join("\n"))
    }
}

impl std::fmt::Debug for RolloverText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RolloverText(ask {} tokens)", self.ask_writing.len())
    }
}

/// What a prompt asks: its end, at most `cap` tokens, without its frame.
pub fn request_of<'a>(text: &RolloverText, prompt: &'a [i32], cap: usize) -> &'a [i32] {
    let body = prompt.strip_suffix(text.to_reply.as_slice()).unwrap_or(prompt);
    let body = body.strip_prefix(text.user_open.as_slice()).unwrap_or(body);
    &body[body.len().saturating_sub(cap)..]
}

/// The ask in the middle of a prompt: what the prompt asks, then the ask.
pub fn ask_with_request(text: &RolloverText, prompt: &[i32], cap: usize) -> Vec<i32> {
    text.request.iter().chain(request_of(text, prompt, cap)).chain(&text.ask_reading).copied().collect()
}

/// The first turn that starts at or after `from`: a tail cut there is a conversation, not the middle of one.
pub fn turn_start(ctx: &[i32], from: usize, user_open: &[i32]) -> Option<usize> {
    if user_open.is_empty() {
        return None;
    }
    (from..ctx.len().saturating_sub(user_open.len())).find(|&i| ctx[i..].starts_with(user_open))
}

/// The notes as they will sit in the window: without a thinking block, if the model wrote one.
pub fn notes_text(reply: &str) -> &str {
    reply.rsplit_once("</think>").map_or(reply, |(_, after)| after).trim()
}

/// Compaction settings of the server.
#[derive(Debug, Clone, Default)]
pub struct CompactionConfig {
    /// The policy of agents created without one (`None`: they compact only when asked).
    pub default_policy: Option<Policy>,
    pub summarizer: SummarizerHandle,
    pub rollover: Rollover,
    /// Set when the server starts, for `rollover.on`.
    pub text: Option<RolloverText>,
}

#[cfg(test)]
mod tests;
