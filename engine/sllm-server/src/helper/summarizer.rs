//! The helper's summarize as compaction's summarizer.

use super::*;

/// The helper's summarize as compaction's [`crate::compaction::Summarizer`] (asynchronous): the replaced tokens are
/// turned back into text with the brain's vocabulary, summarized on the helper lane with `"priority": "compaction"`
/// (the reserved slot), and the summary text is tokenized with the brain's vocabulary for the cut.
pub struct HelperSummarizer {
    helper: Arc<Helper>,
    brain_vocab: Arc<Vocab>,
}

impl HelperSummarizer {
    pub fn new(helper: Arc<Helper>, brain_vocab: Arc<Vocab>) -> HelperSummarizer {
        HelperSummarizer { helper, brain_vocab }
    }
}

/// The facts record ([`crate::compaction::select_fact_lines`]) of the replaced tokens, as brain token ids to place after
/// the summary, and how many lines it keeps.
pub(crate) fn brain_facts(vocab: &Vocab, replaced: &[i32], tail: &[i32], cap: usize) -> Option<(Vec<i32>, usize)> {
    let text_of = |ids: &[i32]| -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&t| vocab.piece(t)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    };
    let count = |t: &str| vocab.tokenize(t, false, false).map(|v| v.len()).unwrap_or(usize::MAX / 4);
    let lines = crate::compaction::select_fact_lines(&text_of(replaced), &text_of(tail), cap, &count);
    if lines.is_empty() {
        return None;
    }
    let record = format!("\n{}{}", crate::compaction::FACTS_HEADER, lines.join("\n"));
    let ids = vocab.tokenize(&record, false, false).ok()?;
    if std::env::var_os("SLLM_LOG_FACTS").is_some() {
        eprintln!("sllm-server: facts record ({} lines, {} tokens):{record}", lines.len(), ids.len());
    }
    Some((ids, lines.len()))
}

/// The most tokens a helper-written compaction summary may have; the API's `summary_tokens` is min(requested, this).
/// 100 (Ben, G3 gate 2026-10-27): with the facts record kept verbatim beside it; a compaction under load may take up to 2 s.
pub const COMPACTION_SUMMARY_CAP: usize = 100;

struct PendingHelperSummary {
    rx: oneshot::Receiver<Result<Value, EngineError>>,
    brain_vocab: Arc<Vocab>,
    target_tokens: usize,
}

impl crate::compaction::PendingSummary for PendingHelperSummary {
    fn poll(&mut self) -> Option<Result<Vec<i32>, String>> {
        match self.rx.try_recv() {
            Err(oneshot::error::TryRecvError::Empty) => None,
            Err(oneshot::error::TryRecvError::Closed) => Some(Err("the helper dropped the call".into())),
            Ok(Err(e)) => Some(Err(e.to_string())),
            Ok(Ok(v)) => {
                let text = v["summary"].as_str().unwrap_or("").to_string();
                if std::env::var_os("SLLM_LOG_FACTS").is_some() {
                    eprintln!("sllm-server: summary: {text}");
                }
                Some(self.brain_vocab.tokenize(&text, false, false).map_err(|e| e.to_string()).map(|mut ids| {
                    // the helper counts its own tokens; the brain's may differ a little: never more than twice the target
                    ids.truncate(self.target_tokens.max(1) * 2);
                    ids
                }))
            }
        }
    }
}

impl crate::compaction::Summarizer for HelperSummarizer {
    fn name(&self) -> &'static str {
        "helper-summarize"
    }
    fn facts(&self, replaced: &[i32], tail: &[i32]) -> Option<(Vec<i32>, usize)> {
        brain_facts(&self.brain_vocab, replaced, tail, crate::compaction::COMPACTION_FACTS_CAP)
    }
    fn start(&self, _agent: &str, replaced: &[i32], target_tokens: usize) -> Option<Box<dyn crate::compaction::PendingSummary>> {
        // the helper's context is much shorter than a long agent's: the oldest and the newest of what fits
        let most = self.helper.cfg.n_ctx * 3 / 4;
        let cut = replaced.len().saturating_sub(most);
        let kept = replaced[..most.min(replaced.len()) / 4].iter().chain(&replaced[(most.min(replaced.len()) / 4 + cut).min(replaced.len())..]);
        let bytes: Vec<u8> = kept.flat_map(|&t| self.brain_vocab.piece(t)).collect();
        let text = String::from_utf8_lossy(&bytes).into_owned();
        // compaction summaries are capped at 48 tokens (G3-4: a longer summary costs decode steps on the critical path)
        let target = target_tokens.clamp(16, COMPACTION_SUMMARY_CAP);
        let op = Op::Summarize(SummarizeReq {
            text,
            max_tokens: Some(target),
            words: Some((target * 3 / 4).max(10)),
            instruction: None,
            ignore_eos: false,
            priority: Priority::Compaction,
            deadline_ms: None,
        });
        let (reply, rx) = oneshot::channel();
        let call = Box::new(Call { op, arrived: Instant::now(), reply });
        if self.helper.tx.try_send(Msg::Call(call)).is_err() {
            // the helper's queue is full: report it through the pending summary
            let (tx, rx) = oneshot::channel();
            let _ = tx.send(Err(EngineError::new(503, "unavailable_error", "the helper queue is full")));
            return Some(Box::new(PendingHelperSummary { rx, brain_vocab: self.brain_vocab.clone(), target_tokens: target }));
        }
        Some(Box::new(PendingHelperSummary { rx, brain_vocab: self.brain_vocab.clone(), target_tokens: target }))
    }
}
