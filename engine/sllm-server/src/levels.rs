//! Answer levels for `POST /v1/chat/completions` (`"level": "sonic" | "fast" | "normal" | "deep"`): the pure pieces, no I/O.
//!
//! A port of `playground/src/sllm_playground/levels.py` (the answer-level half: `stage_system`, `tie_share`, `boxed_pmin`,
//! `needs_second`, `answer_key`, `answer_core`, `answers_agree` and the constants). The orchestration (one generation per
//! stage) is in `api/chat_api.rs`; the chat template's thinking is off in every level.
//!
//! The mapping since 2026-11-13:
//!
//! * sonic   one stage, [`ONE`] ("one short line"), at most 1024 tokens.
//! * fast    one stage, [`CHAT`] (an ordinary chat answer, no word on its length), at most 4096 tokens.
//! * normal  [`ONE`], then [`TIGHT`]; when [`answers_agree`] the TIGHT answer is the answer, else [`LONG`] (step by step, at
//!           most 6144 tokens). An empty LONG answer keeps the TIGHT one. No token probabilities.
//! * deep    for now exactly the same as normal (a long-work level is being built).
//!
//! The token-probability gate ([`needs_second`], [`boxed_pmin`], [`tie_share`], [`P_SURE`], [`TIE_MAX`]; normal until
//! 2026-11-12) is kept as a measured record with its tests; no level calls it any more.
//!
//! The python regular expressions are written out by hand (the crate has no `regex`); where Python's `\d` is Unicode this
//! port reads ASCII digits only, and `Decimal.normalize` is done on the digit string (no 28-digit rounding).

use crate::engine::TokProbs;

mod text;

pub use text::{canon_box, final_part, last_boxed, normalize_text, split_thinking};
use text::{box_span, bold_parts, numbers, splitlines};

/// The stage system prompts (measured as written; equal to the python strings byte for byte).
pub const ONE: &str = "Be extremely brief: one short line with only the essential step, then the final answer. \
If asked to write code, give only the complete code in one code block.";
pub const TIGHT: &str = "You are a helpful assistant. Be very brief: at most 3 short lines of key steps, then the final answer.";
pub const BRIEF: &str = "You are a helpful assistant. Be brief: show only the key steps in a few short lines, then the final answer.";
pub const LONG: &str = "You are a careful expert. Work step by step, check each step, then give the final answer.";
/// An ordinary chat answer: no word on its length.
pub const CHAT: &str = "You are a helpful assistant.";

pub const CAP_ONE: i64 = 1024;
pub const CAP_TIGHT: i64 = 2048;
pub const CAP_BRIEF: i64 = 2048;
pub const CAP_LONG: i64 = 6144;
pub const CAP_CHAT: i64 = 4096;

/// normal, with a box: the least probable token inside it must be at least this likely.
pub const P_SURE: f64 = 0.9;
/// The near-tie share above which the answer is not trusted.
pub const TIE_MAX: f64 = 0.072;
/// Two best tokens closer than this (log-probability) are a near tie.
pub const TIE_GAP: f64 = 0.5;
/// What we ask of the engine (it serves at most 2 best tokens).
pub const N_PROBS: i64 = 2;
/// A worded result of at most this many words may agree by being part of the other answer's result.
pub const AGREE_WORDS: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    One,
    Tight,
    Brief,
    Long,
    Chat,
}

// The name the response and the python `PROMPTS` use.
variant_names!(Stage { One => "ONE", Tight => "TIGHT", Brief => "BRIEF", Long => "LONG", Chat => "CHAT" });

impl Stage {
    pub fn prompt(self) -> &'static str {
        match self {
            Stage::One => ONE,
            Stage::Tight => TIGHT,
            Stage::Brief => BRIEF,
            Stage::Long => LONG,
            Stage::Chat => CHAT,
        }
    }

    /// The most tokens the stage may write.
    pub fn cap(self) -> i64 {
        match self {
            Stage::One => CAP_ONE,
            Stage::Tight => CAP_TIGHT,
            Stage::Brief => CAP_BRIEF,
            Stage::Long => CAP_LONG,
            Stage::Chat => CAP_CHAT,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Sonic,
    Fast,
    Normal,
    Deep,
}

impl Level {
    /// `None` for anything that is not one of the four names ("off" is the caller's business).
    pub fn parse(s: &str) -> Option<Level> {
        match s {
            "sonic" => Some(Level::Sonic),
            "fast" => Some(Level::Fast),
            "normal" => Some(Level::Normal),
            "deep" => Some(Level::Deep),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Level::Sonic => "sonic",
            Level::Fast => "fast",
            Level::Normal => "normal",
            Level::Deep => "deep",
        }
    }

    /// The stage the level starts with.
    pub fn first_stage(self) -> Stage {
        match self {
            Level::Sonic | Level::Normal | Level::Deep => Stage::One,
            Level::Fast => Stage::Chat,
        }
    }

    /// The stages the level may run, in order.
    pub fn stages(self) -> &'static [Stage] {
        match self {
            Level::Sonic => &[Stage::One],
            Level::Fast => &[Stage::Chat],
            // deep is, for now, the same as normal (a long-work level is being built)
            Level::Normal | Level::Deep => &[Stage::One, Stage::Tight, Stage::Long],
        }
    }

    /// Whether a token-probability gate may add the LONG stage. No level uses the gate since 2026-11-13 (normal now
    /// compares a brief and a tight answer instead), so this is always false.
    pub fn gated(self) -> bool {
        false
    }
}

/// One generated token as `completion_probabilities` has it: the piece, its log-probability and the log-probabilities of the
/// best tokens (best first).
#[derive(Debug, Clone, PartialEq)]
pub struct ProbEntry {
    pub token: String,
    pub logprob: f64,
    pub top_logprobs: Vec<f64>,
}

impl ProbEntry {
    pub fn from_engine(p: &TokProbs) -> ProbEntry {
        ProbEntry { token: p.chosen.token.clone(), logprob: p.chosen.logprob as f64, top_logprobs: p.top.iter().map(|t| t.logprob as f64).collect() }
    }
}

/// The system prompt of one stage: the stage's own prompt alone when the caller's is the default chat system (or empty), else
/// the stage prompt, a blank line, then the caller's system.
pub fn stage_system(stage_prompt: &str, caller_system: &str, default_system: &str) -> String {
    if caller_system.trim().is_empty() || caller_system == default_system {
        stage_prompt.to_string()
    } else {
        format!("{stage_prompt}\n\n{caller_system}")
    }
}

/// Among the generated tokens that have at least two best tokens, the fraction whose best two log-probabilities differ by less
/// than [`TIE_GAP`]. `None` when no token has two.
pub fn tie_share(probs: Option<&[ProbEntry]>) -> Option<f64> {
    let gaps: Vec<f64> = probs.unwrap_or(&[]).iter().filter(|p| p.top_logprobs.len() >= 2).map(|p| p.top_logprobs[0] - p.top_logprobs[1]).collect();
    if gaps.is_empty() {
        return None;
    }
    Some(gaps.iter().filter(|g| **g < TIE_GAP).count() as f64 / gaps.len() as f64)
}

/// What two answers must share to agree: the content of the last `\boxed{..}` of the answer part, else the number in the last
/// bold part, else the normalized text. Prefixed "box:" / "txt:" so a box never equals prose.
pub fn answer_key(text: &str, eog: &[&str]) -> String {
    let fin = final_part(text, eog);
    if let Some(b) = last_boxed(&fin) {
        return format!("box:{}", canon_box(b));
    }
    for bold in bold_parts(&fin).into_iter().rev() {
        if let Some(n) = numbers(bold).first() {
            return format!("box:{}", canon_box(&n.replace(',', "")));
        }
    }
    format!("txt:{}", normalize_text(&fin))
}

/// A short human form of the answer (the box content, else the first 80 characters).
pub fn answer_shown(text: &str) -> String {
    let fin = final_part(text, &[]);
    match last_boxed(&fin) {
        Some(b) => b.trim().to_string(),
        None => fin.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(80).collect(),
    }
}

/// The part of a chat answer that states its result: the last box, else the last bold part, else the last line with text.
pub fn answer_core(text: &str) -> String {
    let fin = final_part(text, &[]);
    if let Some(b) = last_boxed(&fin) {
        return b.to_string();
    }
    if let Some(b) = bold_parts(&fin).last() {
        return b.to_string();
    }
    match splitlines(&fin).into_iter().rev().find(|l| !l.trim().is_empty()) {
        Some(l) => strip_answer_label(l),
        None => String::new(),
    }
}

/// `re.sub(r"^\s*(?:the\s+)?(?:final\s+)?answer(?:\s+is)?\s*[:\-]?\s*", "", line, flags=re.IGNORECASE)`.
fn strip_answer_label(line: &str) -> String {
    fn ws(s: &str, i: usize) -> usize {
        i + (s[i..].len() - s[i..].trim_start().len())
    }
    fn word(s: &str, i: usize, w: &str) -> bool {
        s.get(i..i + w.len()).is_some_and(|x| x.eq_ignore_ascii_case(w))
    }
    let mut i = ws(line, 0);
    for w in ["the", "final"] {
        if word(line, i, w) {
            let j = ws(line, i + w.len());
            if j > i + w.len() {
                i = j;
            }
        }
    }
    if !word(line, i, "answer") {
        return line.to_string();
    }
    i += "answer".len();
    let k = ws(line, i);
    if k > i && word(line, k, "is") {
        i = k + 2;
    }
    i = ws(line, i);
    if line[i..].starts_with([':', '-']) {
        i += 1;
    }
    i = ws(line, i);
    line[i..].to_string()
}

/// deep: whether the ONE answer and the TIGHT answer give the same result, and why (human text). They agree when their answer
/// keys are equal; else when the last number of each one's core is the same number; else, with no number in either core, when
/// the shorter core (1 .. [`AGREE_WORDS`] words) is, word for word, part of the other. An empty answer never agrees.
pub fn answers_agree(one: &str, tight: &str) -> (bool, String) {
    let yes = || (true, format!("the two short answers agree ({})", answer_shown(tight)));
    let no = || (false, "the two short answers do not agree: answering again step by step".to_string());
    let (k1, k2) = (answer_key(one, &[]), answer_key(tight, &[]));
    if k1 == k2 && k1.split_once(':').is_some_and(|(_, rest)| !rest.is_empty()) {
        return yes();
    }
    let (c1, c2) = (answer_core(one), answer_core(tight));
    let (n1, n2) = (numbers(&c1), numbers(&c2));
    if !n1.is_empty() || !n2.is_empty() {
        let same = match (n1.last(), n2.last()) {
            (Some(a), Some(b)) => canon_box(a.trim_end_matches(',')) == canon_box(b.trim_end_matches(',')),
            _ => false,
        };
        return if same { yes() } else { no() };
    }
    let (t1, t2) = (normalize_text(&c1), normalize_text(&c2));
    let (w1, w2): (Vec<&str>, Vec<&str>) = (t1.split_whitespace().collect(), t2.split_whitespace().collect());
    let (short, long) = if w1.len() <= w2.len() { (w1, w2) } else { (w2, w1) };
    if (1..=AGREE_WORDS).contains(&short.len()) && long.windows(short.len()).any(|w| w == short.as_slice()) {
        return yes();
    }
    no()
}

/// The probability of the least probable token inside the content of the LAST `\boxed{..}` of the answer. The tokens are laid
/// end to end (their pieces) to find which of them cover the box content. `None` when there is no box, no probabilities, or the
/// pieces do not spell the box.
pub fn boxed_pmin(probs: Option<&[ProbEntry]>, text: &str) -> Option<f64> {
    let probs = probs.filter(|p| !p.is_empty())?;
    last_boxed(text)?;
    let joined: String = probs.iter().map(|p| p.token.as_str()).collect();
    let (lo, hi) = box_span(&joined).filter(|(a, b)| b > a)?;
    let (mut pos, mut best): (usize, Option<f64>) = (0, None);
    for p in probs {
        let (a, b) = (pos, pos + p.token.len());
        pos = b;
        if b > lo && a < hi && !p.token.is_empty() {
            let pr = p.logprob.min(0.0).exp();
            best = Some(best.map_or(pr, |x| x.min(pr)));
        }
    }
    best
}

/// Whether the first answer must be answered again with LONG, and why (human text). A measured record of the gate normal
/// used until 2026-11-12; no level calls it any more.
///
/// A box in the answer: the least probable token inside the last box, p < [`P_SURE`]. No box (or a box whose tokens cannot be
/// placed): the near-tie share above [`TIE_MAX`]. No probabilities: never again.
pub fn needs_second(text: &str, probs: Option<&[ProbEntry]>) -> (bool, String) {
    if probs.is_none_or(|p| p.is_empty()) {
        return (false, "no token probabilities on this engine".into());
    }
    if let Some(pmin) = boxed_pmin(probs, text) {
        return if pmin < P_SURE {
            (true, format!("not sure (least sure answer token {pmin:.2} < 0.9): answering again step by step"))
        } else {
            (false, format!("sure (least sure answer token {pmin:.2} >= 0.9)"))
        };
    }
    let Some(ties) = tie_share(probs) else { return (false, "no token had two best tokens: kept the first answer".into()) };
    if ties > TIE_MAX {
        (true, format!("not sure ({:.1}% near-tie words > {:.1}%): answering again step by step", ties * 100.0, TIE_MAX * 100.0))
    } else {
        (false, format!("sure ({:.1}% near-tie words <= {:.1}%)", ties * 100.0, TIE_MAX * 100.0))
    }
}

#[cfg(test)]
mod tests;
