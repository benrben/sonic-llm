//! What compaction keeps per agent: its last report, its fields record and its counters.

use super::*;

// ---- the report ----------------------------------------------------------------------------------------------------

/// What one compaction did (the `POST /v1/agents/{id}/compact` answer, and the agent's `compaction.last`).
#[derive(Debug, Clone, Default, Serialize)]
pub struct Report {
    pub agent_id: AgentId,
    pub trigger: Option<Trigger>,
    /// `client` (the request's summary) or the summarizer's name.
    pub summary_source: String,
    pub prefix_tokens: usize,
    /// Tokens the summary replaced (the old summary and turns between the prefix and the tail).
    pub replaced_tokens: usize,
    pub summary_tokens: usize,
    /// The facts record placed after the summary ([`select_fact_lines`]): its tokens (header included) and lines.
    pub facts_tokens: usize,
    pub facts_lines: usize,
    pub tail_tokens: usize,
    pub context_before: usize,
    pub context_after: usize,
    /// The agent's page-table entries before and after, and the pages that went back to the pool (null for an engine
    /// without a page pool).
    pub pages_before: Option<usize>,
    pub pages_after: Option<usize>,
    pub pages_freed: Option<usize>,
    /// Pages of the prefix, kept as they were, and how many of them are shared with other agents.
    pub prefix_pages: Option<usize>,
    pub prefix_shared_pages: Option<usize>,
    /// The cut (page bookkeeping), the re-prefill of summary and tail (its first chunk to its end, the engine's clock)
    /// and the whole compaction from the command to the end of the re-prefill (summary generation included).
    pub summarize_ms: f64,
    pub cut_ms: f64,
    pub reprefill_tokens: usize,
    pub reprefill_ms: f64,
    pub queued_ms: f64,
    pub total_ms: f64,
}

// ---- the extracted-fields record -----------------------------------------------------------------------------------

pub const MAX_FIELDS: usize = 256;
pub const MAX_KEY_BYTES: usize = 128;
pub const MAX_VALUE_BYTES: usize = 4096;

/// A small per-agent key/value record kept outside the window (IDs, emails, state).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Fields(BTreeMap<String, Value>);

/// An operation on an agent's fields.
#[derive(Debug, Clone)]
pub enum FieldsOp {
    Get,
    /// Replace the whole record.
    Replace(Map<String, Value>),
    /// Set the given keys; a `null` value removes its key.
    Merge(Map<String, Value>),
    Remove(String),
}

fn check_field(k: &str, v: &Value) -> Result<(), String> {
    if k.is_empty() || k.len() > MAX_KEY_BYTES {
        return Err(format!("field names are 1 to {MAX_KEY_BYTES} bytes, got {}", k.len()));
    }
    match v {
        Value::String(s) if s.len() > MAX_VALUE_BYTES => Err(format!("field {k}: values are at most {MAX_VALUE_BYTES} bytes")),
        Value::String(_) | Value::Number(_) | Value::Bool(_) | Value::Null => Ok(()),
        _ => Err(format!("field {k}: values are strings, numbers or booleans (a small record, not a document)")),
    }
}

impl Fields {
    pub fn get(&self, k: &str) -> Option<&Value> {
        self.0.get(k)
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn to_json(&self) -> Value {
        serde_json::to_value(&self.0).unwrap_or(Value::Null)
    }

    /// Apply `op` (all or nothing). Returns false for a `Remove` of a missing key.
    pub fn apply(&mut self, op: FieldsOp) -> Result<bool, String> {
        let mut next = self.0.clone();
        let mut found = true;
        match op {
            FieldsOp::Get => return Ok(true),
            FieldsOp::Replace(m) => {
                next.clear();
                merge_fields(&mut next, m)?;
            }
            FieldsOp::Merge(m) => merge_fields(&mut next, m)?,
            FieldsOp::Remove(k) => found = next.remove(&k).is_some(),
        }
        if next.len() > MAX_FIELDS {
            return Err(format!("an agent holds at most {MAX_FIELDS} fields"));
        }
        self.0 = next;
        Ok(found)
    }
}

/// Set the keys of `m` in `fields`, each checked first; a `null` value removes its key.
fn merge_fields(fields: &mut BTreeMap<String, Value>, m: Map<String, Value>) -> Result<(), String> {
    for (k, v) in m {
        check_field(&k, &v)?;
        if v.is_null() {
            fields.remove(&k);
        } else {
            fields.insert(k, v);
        }
    }
    Ok(())
}

// ---- per-agent state -----------------------------------------------------------------------------------------------

/// Where an agent's compaction is: one per agent, moved only by [`CompactionPhase::next`]. A turn of an agent that is not
/// `Idle` does not run (it is held, or queued behind a rollover), and the policy starts no compaction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CompactionPhase {
    #[default]
    Idle,
    /// An asynchronous summarizer writes the summary of the window as it was when the compaction began.
    Summarizing,
    /// The agent writes its notes for a rollover (an executor request of its own, running or waiting for room).
    WritingNotes,
    /// The window is cut; request `.0` reads its summary and tail back (running, or waiting for room).
    Prefilling(ReqId),
}

/// What moves a [`CompactionPhase`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseEv {
    SummaryStarted,
    NotesStarted,
    PrefillStarted(ReqId),
    /// The compaction ended, failed or was called off.
    Ended,
}

impl CompactionPhase {
    /// The phase after `ev`; None: not a move this phase makes.
    pub fn next(self, ev: PhaseEv) -> Option<CompactionPhase> {
        use CompactionPhase::*;
        Some(match (self, ev) {
            (Idle, PhaseEv::SummaryStarted) => Summarizing,
            (Idle, PhaseEv::NotesStarted) => WritingNotes,
            (Idle | Summarizing | WritingNotes, PhaseEv::PrefillStarted(r)) => Prefilling(r),
            (_, PhaseEv::Ended) => Idle,
            _ => return None,
        })
    }

    pub fn is_idle(self) -> bool {
        self == CompactionPhase::Idle
    }
}

/// What the executor keeps per agent for compaction, next to the agent's record.
#[derive(Debug, Clone, Default)]
pub struct AgentState {
    /// The agent's prefix: its system prompt, never compacted.
    pub prefix_tokens: usize,
    pub policy: Option<Policy>,
    pub fields: Fields,
    pub compactions: u64,
    /// The context right after the last compaction (or the system prompt): `every_tokens` counts from here.
    pub context_at_last: usize,
    pub last: Option<Report>,
    /// Where its compaction is.
    pub phase: CompactionPhase,
    /// Policy triggers that found nothing to replace (the context was all prefix and tail).
    pub skipped: u64,
    /// Compactions that a turn reached the end of the window for.
    pub rollovers: u64,
    /// The lines the agent has seen, and the turns that got some back.
    pub archive: Archive,
    pub recalled: u64,
    /// The archive holds the window up to here.
    pub archived_to: usize,
    /// The notes of the last rollover end here. They are not archived: the archive holds what the agent saw, and a summary
    /// of it can be wrong.
    pub notes_end: usize,
}

impl AgentState {
    pub fn new(prefix_tokens: usize, policy: Option<Policy>) -> AgentState {
        AgentState { prefix_tokens, policy, context_at_last: prefix_tokens, ..Default::default() }
    }

    /// The `compaction` member of the agent object.
    pub fn to_json(&self) -> Value {
        json!({
            "prefix_tokens": self.prefix_tokens,
            "policy": self.policy,
            "compactions": self.compactions,
            "compacting": matches!(self.phase, CompactionPhase::Prefilling(_)),
            "skipped": self.skipped,
            "rollovers": self.rollovers,
            "recalled": self.recalled,
            "context_at_last": self.context_at_last,
            "last": self.last,
        })
    }
}

/// The 409 for a turn or compaction on an agent that is compacting.
pub fn compacting(id: &str) -> EngineError {
    EngineError::new(409, "conflict_error", format!("agent {id} is compacting")).with_retry_after(100)
}
