//! The agent lifecycle (phase 3): where each agent's KV lives, and the policy that moves it.
//!
//! # States
//!
//! ```text
//!            turn                     turn ends
//!   Idle ───────────▶ Active ───────────────────▶ Idle
//!    │  ▲                                          │
//!    │  └──── restore done ◀── Restoring ◀── turn, │ suspend (API, idle timer, memory pressure)
//!    │                            ▲         wake   ▼
//!    │                            └──────────── Suspended (KV on SSD, pages free)
//!    │ drop KV (API "history", SSD budget, failed snapshot)        │
//!    ▼                                                             ▼
//!  History (token history only) ── turn: re-prefill history + turn ──▶ Active
//! ```
//!
//! * **Active**: a turn is queued or running. **Idle**: resident in RAM, no turn. **Suspended**: the private KV pages
//!   are in an SSD snapshot and free in the pool; shared prefix pages stay resident. **Restoring**: a wake started the
//!   read; the turn waits in the agent's queue. **History**: no KV anywhere, only the token history (the durable
//!   registry's copy survives a restart): the next turn re-prefills it.
//! * The conversation is the truth and the KV is a cache: a snapshot that is missing, damaged or was never written
//!   falls back to re-prefill from the history, never to an error.
//!
//! # Policy (the executor runs it at every step boundary, [`LifecycleConfig`])
//!
//! * **Explicit**: `POST /v1/agents/{id}/suspend` (`{"tier": "ssd" | "history"}`; on a busy agent it applies when the
//!   turn ends), `POST /v1/agents/{id}/wake` (start the restore now: a tool result is about to arrive).
//! * **Idle limit**: an agent idle in RAM for `idle_suspend_ms` is suspended.
//! * **Memory pressure**: a restore the page pool cannot hold suspends the coldest idle agents (least recently
//!   active) to cover its shortfall; with `relieve_on_refusal` a turn, completion or new agent the pool would refuse
//!   does the same and is tried again once. Opt-in (`pressure_low` > 0): when a pool's free pages (net of what admitted requests may still take) fall under
//!   `pressure_low` of the pool, the coldest idle agents are suspended until the free share is back over
//!   `pressure_high`.
//! * **SSD budget**: past `ssd_budget_bytes` the longest-suspended agents drop to history.
//! * **Wake queue**: a turn for a busy, suspended or restoring agent waits in the agent's FIFO input queue (at most
//!   `inbox_cap`) instead of a 409. A turn for a suspended agent starts its restore at once; at most `max_restoring`
//!   restores are in flight (interactive agents first), and restored pages are uploaded within `upload_budget_bytes`
//!   per step boundary, so a burst of wakes cannot starve the interactive lane.
//!
//! # Durable registry
//!
//! With `registry_dir`, every agent is a JSON file `<id>.json` (id, lane, created, turns, token history), written
//! atomically at creation and at the end of every turn, removed on delete. A restarted server loads them as History
//! agents.

use crate::engine::{AgentId, Lane};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize)]
pub struct LifecycleConfig {
    /// Suspend an agent idle in RAM this long (`None`: only on demand and under pressure).
    pub idle_suspend_ms: Option<u64>,
    /// Memory pressure: suspend idle agents when a pool's free share drops under `pressure_low`, until it is over
    /// `pressure_high`. `pressure_low` 0 (the default) turns the proactive policy off, so a run that fits keeps every
    /// agent resident; a request the pool would refuse still frees room from the coldest idle agents.
    pub pressure_low: f64,
    pub pressure_high: f64,
    /// A turn, completion or new agent the page pool would refuse first suspends the coldest idle agents to cover its
    /// shortfall (`--pressure-relief on`). Off by default: an idle agent keeps its pages and admission refuses as in
    /// phase 2, which the admission tests and the G2 memory bench rely on. A restore always makes room this way.
    pub relieve_on_refusal: bool,
    /// Restores in flight at once.
    pub max_restoring: usize,
    /// Restored bytes uploaded per step boundary.
    pub upload_budget_bytes: u64,
    /// Turns waiting per agent.
    pub inbox_cap: usize,
    /// SSD bytes the snapshots may take before the oldest suspended agents drop to history (`None`: no limit).
    pub ssd_budget_bytes: Option<u64>,
    /// Durable registry of agents (token history and metadata); `None`: in memory only.
    pub registry_dir: Option<PathBuf>,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        LifecycleConfig {
            idle_suspend_ms: None,
            pressure_low: 0.0,
            pressure_high: 0.20,
            relieve_on_refusal: false,
            max_restoring: 4,
            upload_budget_bytes: 256 << 20,
            inbox_cap: usize::MAX, // no cap (Ben 2026-11-09; was 16)
            ssd_budget_bytes: None,
            registry_dir: None,
        }
    }
}

impl LifecycleConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..1.0).contains(&self.pressure_low) || !(0.0..=1.0).contains(&self.pressure_high) || self.pressure_high < self.pressure_low {
            return Err(format!("pressure thresholds {} / {} must satisfy 0 <= low <= high <= 1", self.pressure_low, self.pressure_high));
        }
        if self.max_restoring == 0 || self.inbox_cap == 0 {
            return Err("max_restoring and inbox_cap must be at least 1".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentState {
    Active,
    Idle,
    Suspended,
    Restoring,
    History,
}

impl AgentState {
    pub const ALL: [AgentState; 5] = [AgentState::Active, AgentState::Idle, AgentState::Suspended, AgentState::Restoring, AgentState::History];

    pub fn name(self) -> &'static str {
        match self {
            AgentState::Active => "active",
            AgentState::Idle => "idle",
            AgentState::Suspended => "suspended",
            AgentState::Restoring => "restoring",
            AgentState::History => "history",
        }
    }

    /// Where the KV is.
    pub fn tier(self) -> &'static str {
        match self {
            AgentState::Active | AgentState::Idle => "ram",
            AgentState::Suspended | AgentState::Restoring => "ssd",
            AgentState::History => "none",
        }
    }
}

/// Where a suspend sends the KV.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Ssd,
    History,
}

/// Why an agent was suspended (for `/metrics`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Why {
    Api,
    Idle,
    Pressure,
    Budget,
}

/// The transitions of the state machine. Every change of [`AgentLife::state`] goes through [`AgentLife::apply`], which
/// refuses any other move (a bug would surface as an error, not as a lost turn).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ev {
    TurnStarted,
    TurnEnded { kv_kept: bool },
    Suspended,
    WakeStarted,
    Restored,
    /// The KV is gone (history tier, failed snapshot or restore).
    KvDropped,
}

impl AgentState {
    pub fn next(self, ev: Ev) -> Option<AgentState> {
        use AgentState::*;
        Some(match (self, ev) {
            (Idle | History, Ev::TurnStarted) => Active,
            (Active, Ev::TurnEnded { kv_kept: true }) => Idle,
            (Active, Ev::TurnEnded { kv_kept: false }) => History,
            (Idle, Ev::Suspended) => Suspended,
            (Suspended, Ev::WakeStarted) => Restoring,
            (Restoring, Ev::Restored) => Idle,
            (Idle | Suspended | Restoring | History, Ev::KvDropped) => History,
            _ => return None,
        })
    }
}

/// How the current turn's KV got back (in the turn's final chunk, `wake`).
#[derive(Debug, Clone, Default, Serialize)]
pub struct WakeInfo {
    /// "ram" (it was resident), "ssd" (restored), "reprefill" (rebuilt from the history).
    pub via: &'static str,
    /// From the wake (the turn's arrival or the wake call) to the KV being back; 0 when resident.
    pub restore_ms: f64,
    /// History tokens re-prefilled in front of the turn.
    pub reprefill_tokens: usize,
}

/// One agent's lifecycle record.
#[derive(Debug)]
pub struct AgentLife {
    pub state: AgentState,
    /// Every token of the agent's context in order (system prompt, turns, generated tokens; the last generated token
    /// is the engine's pending one).
    pub history: Vec<i32>,
    /// `history.len()` before the running turn (a turn stopped before its first token is rolled back to it).
    pub hist_before: usize,
    pub last_active: Instant,
    /// Apply this suspend when the running turn ends.
    pub suspend_after: Option<Tier>,
    /// When the current wake started (a turn or wake call on a suspended agent).
    pub wake_t0: Option<Instant>,
    /// When it was suspended (the SSD budget drops the longest-suspended first).
    pub suspended_at: Option<Instant>,
    /// What the running (or next) turn's wake cost.
    pub wake: Option<WakeInfo>,
    /// The running turn re-prefills `n` history tokens (it rebuilds a History agent).
    pub rebuild: usize,
    /// The running request is a compaction (its window is read back from the engine when it ends).
    pub compacting: bool,
}

impl AgentLife {
    pub fn new(history: Vec<i32>) -> AgentLife {
        AgentLife {
            state: AgentState::Idle,
            history,
            hist_before: 0,
            last_active: Instant::now(),
            suspend_after: None,
            wake_t0: None,
            suspended_at: None,
            wake: None,
            rebuild: 0,
            compacting: false,
        }
    }

    pub fn apply(&mut self, ev: Ev) -> Result<AgentState, String> {
        let to = self.state.next(ev).ok_or_else(|| format!("lifecycle: {:?} is not allowed in state {}", ev, self.state.name()))?;
        self.state = to;
        if matches!(to, AgentState::Idle | AgentState::History) {
            self.last_active = Instant::now();
        }
        self.suspended_at = if to == AgentState::Suspended { Some(Instant::now()) } else if to == AgentState::Restoring { self.suspended_at } else { None };
        Ok(to)
    }

    pub fn idle_for(&self, now: Instant) -> Duration {
        now.saturating_duration_since(self.last_active)
    }
}

/// p50 and p95 of a sample (nearest rank), `None` when empty.
pub fn percentiles(v: &[f64]) -> Option<(f64, f64)> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let at = |q: f64| s[((q * s.len() as f64).ceil() as usize).clamp(1, s.len()) - 1];
    Some((at(0.50), at(0.95)))
}

/// Counters for `/metrics` `lifecycle`.
#[derive(Debug, Default)]
pub struct LifeStats {
    pub suspends_api: u64,
    pub suspends_idle: u64,
    pub suspends_pressure: u64,
    pub suspend_errors: u64,
    pub history_drops: u64,
    pub restores: u64,
    pub restore_failures: u64,
    /// The last 4,096 restore times (wake to KV back), ms.
    pub restore_ms: Vec<f64>,
    pub reprefills: u64,
    pub reprefill_tokens: u64,
    /// Turns served from a restored snapshot instead of a re-prefill of the history, and the tokens not re-prefilled.
    pub reprefills_avoided: u64,
    pub reprefill_tokens_avoided: u64,
    pub writes: u64,
    pub write_failures: u64,
    pub bytes_written: u64,
    /// What full (non-incremental) spills would have written.
    pub bytes_private: u64,
    pub last_write_bytes: u64,
    /// Bytes written by each suspend, the last 4,096.
    pub write_bytes: Vec<u64>,
    pub turns_queued: u64,
    pub inbox_full: u64,
    pub pressure_relief: u64,
    pub registry_writes: u64,
    pub registry_errors: u64,
    pub loaded_from_registry: u64,
}

impl LifeStats {
    pub fn push_restore(&mut self, ms: f64) {
        if self.restore_ms.len() >= 4096 {
            self.restore_ms.remove(0);
        }
        self.restore_ms.push(ms);
    }

    pub fn push_write(&mut self, bytes: u64, private: u64) {
        self.writes += 1;
        self.bytes_written += bytes;
        self.bytes_private += private;
        self.last_write_bytes = bytes;
        if self.write_bytes.len() >= 4096 {
            self.write_bytes.remove(0);
        }
        self.write_bytes.push(bytes);
    }

    pub fn json(&self, per_state: &[(AgentState, usize)], engine: Value, cfg: &LifecycleConfig) -> Value {
        let mut states = serde_json::Map::new();
        for (s, n) in per_state {
            states.insert(s.name().into(), json!(n));
        }
        let (p50, p95) = percentiles(&self.restore_ms).map_or((Value::Null, Value::Null), |(a, b)| (json!(a), json!(b)));
        let avg_write = if self.writes > 0 { json!(self.bytes_written as f64 / self.writes as f64) } else { Value::Null };
        json!({
            "agents": states,
            "ssd_bytes": engine.get("ssd_bytes").cloned().unwrap_or(json!(0)),
            "suspends": {"api": self.suspends_api, "idle": self.suspends_idle, "pressure": self.suspends_pressure, "errors": self.suspend_errors},
            "history_drops": self.history_drops,
            "restores": self.restores,
            "restore_failures": self.restore_failures,
            "restore_ms_p50": p50,
            "restore_ms_p95": p95,
            "reprefills": self.reprefills,
            "reprefill_tokens": self.reprefill_tokens,
            "reprefills_avoided": self.reprefills_avoided,
            "reprefill_tokens_avoided": self.reprefill_tokens_avoided,
            "writes": self.writes,
            "write_failures": self.write_failures,
            "bytes_written": self.bytes_written,
            "bytes_written_per_suspend": avg_write,
            "last_suspend_bytes_written": self.last_write_bytes,
            "bytes_full_spill_equivalent": self.bytes_private,
            "turns_queued": self.turns_queued,
            "inbox_full": self.inbox_full,
            "pressure_relief": self.pressure_relief,
            "registry": {"writes": self.registry_writes, "errors": self.registry_errors, "loaded": self.loaded_from_registry,
                         "dir": cfg.registry_dir.as_ref().map(|d| d.display().to_string())},
            "engine": engine,
        })
    }
}

// ---- durable registry --------------------------------------------------------------------------------------------

/// One agent on disk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentRecord {
    pub id: AgentId,
    pub lane: Lane,
    pub created: u64,
    pub turns: u64,
    pub history: Vec<i32>,
    /// The system prompt's tokens (compaction keeps them as the prefix).
    #[serde(default)]
    pub prefix_tokens: usize,
}

/// The durable agent registry: one `<id>.json` per agent under its directory.
#[derive(Debug, Clone)]
pub struct Registry {
    pub dir: PathBuf,
}

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

impl Registry {
    pub fn open(dir: PathBuf) -> std::io::Result<Registry> {
        for c in dir.components() {
            if matches!(c.as_os_str().to_str(), Some("models") | Some("results")) {
                return Err(std::io::Error::other(format!("registry dir {} is inside models/ or results/: refused", dir.display())));
            }
        }
        std::fs::create_dir_all(&dir)?;
        Ok(Registry { dir })
    }

    fn path(&self, id: &str) -> Option<PathBuf> {
        valid_id(id).then(|| self.dir.join(format!("{id}.json")))
    }

    /// Write the record atomically (a temp file renamed over the old one).
    pub fn save(&self, r: &AgentRecord) -> std::io::Result<()> {
        let path = self.path(&r.id).ok_or_else(|| std::io::Error::other(format!("agent id {:?} is not a file name", r.id)))?;
        let tmp = self.dir.join(format!(".{}.json.tmp", r.id));
        std::fs::write(&tmp, serde_json::to_vec(r).map_err(std::io::Error::other)?)?;
        std::fs::rename(&tmp, &path)
    }

    pub fn remove(&self, id: &str) -> std::io::Result<()> {
        match self.path(id) {
            Some(p) if p.exists() => std::fs::remove_file(p),
            _ => Ok(()),
        }
    }

    /// Every record, in id order (`ag_<n>` numerically); unreadable files are skipped and reported.
    pub fn load(&self) -> (Vec<AgentRecord>, Vec<String>) {
        let mut out = Vec::new();
        let mut bad = Vec::new();
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return (out, bad) };
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if !name.ends_with(".json") || name.starts_with('.') {
                continue;
            }
            match std::fs::read(e.path()).ok().and_then(|b| serde_json::from_slice::<AgentRecord>(&b).ok()) {
                Some(r) if valid_id(&r.id) && format!("{}.json", r.id) == name => out.push(r),
                _ => bad.push(name),
            }
        }
        out.sort_by_key(|r| (r.id.trim_start_matches("ag_").parse::<u64>().unwrap_or(u64::MAX), r.id.clone()));
        (out, bad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_machine_allows_only_its_transitions() {
        use AgentState::*;
        let all_ev = [Ev::TurnStarted, Ev::TurnEnded { kv_kept: true }, Ev::TurnEnded { kv_kept: false }, Ev::Suspended, Ev::WakeStarted, Ev::Restored, Ev::KvDropped];
        let allowed = [
            (Idle, Ev::TurnStarted, Active),
            (History, Ev::TurnStarted, Active),
            (Active, Ev::TurnEnded { kv_kept: true }, Idle),
            (Active, Ev::TurnEnded { kv_kept: false }, History),
            (Idle, Ev::Suspended, Suspended),
            (Suspended, Ev::WakeStarted, Restoring),
            (Restoring, Ev::Restored, Idle),
            (Idle, Ev::KvDropped, History),
            (Suspended, Ev::KvDropped, History),
            (Restoring, Ev::KvDropped, History),
            (History, Ev::KvDropped, History),
        ];
        for s in AgentState::ALL {
            for ev in all_ev {
                let want = allowed.iter().find(|(a, e, _)| *a == s && *e == ev).map(|x| x.2);
                assert_eq!(s.next(ev), want, "{s:?} + {ev:?}");
            }
        }
        // a suspended agent cannot run a turn: it is woken first
        assert_eq!(Suspended.next(Ev::TurnStarted), None);
        assert_eq!(Restoring.next(Ev::TurnStarted), None);
        assert_eq!(Active.next(Ev::Suspended), None, "a busy agent is suspended when its turn ends");
    }

    #[test]
    fn life_tracks_times_and_refuses_bad_moves() {
        let mut l = AgentLife::new(vec![1, 2, 3]);
        assert_eq!(l.state, AgentState::Idle);
        assert!(l.apply(Ev::Restored).is_err());
        assert_eq!(l.state, AgentState::Idle, "a refused move changes nothing");
        l.apply(Ev::Suspended).unwrap();
        assert!(l.suspended_at.is_some());
        l.apply(Ev::WakeStarted).unwrap();
        assert!(l.suspended_at.is_some(), "kept while restoring");
        l.apply(Ev::Restored).unwrap();
        assert!(l.suspended_at.is_none());
        assert_eq!(AgentState::Restoring.tier(), "ssd");
        assert_eq!(AgentState::History.tier(), "none");
    }

    #[test]
    fn percentiles_nearest_rank() {
        assert_eq!(percentiles(&[]), None);
        assert_eq!(percentiles(&[5.0]), Some((5.0, 5.0)));
        let v: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        assert_eq!(percentiles(&v), Some((50.0, 95.0)));
        assert_eq!(percentiles(&[3.0, 1.0, 2.0]), Some((2.0, 3.0)));
    }

    #[test]
    fn registry_round_trip_atomic_and_safe() {
        let dir = std::env::temp_dir().join(format!("sllm-registry-test-{}", std::process::id()));
        let reg = Registry::open(dir.clone()).unwrap();
        let a = AgentRecord { id: "ag_2".into(), lane: Lane::Background, created: 7, turns: 3, history: (0..5000).collect(), prefix_tokens: 9 };
        let b = AgentRecord { id: "ag_10".into(), lane: Lane::Interactive, created: 8, turns: 0, history: vec![], prefix_tokens: 0 };
        reg.save(&a).unwrap();
        reg.save(&b).unwrap();
        let mut a2 = a.clone();
        a2.history.push(42);
        a2.turns = 4;
        reg.save(&a2).unwrap();
        std::fs::write(dir.join("junk.json"), b"{not json").unwrap();
        let (got, bad) = reg.load();
        assert_eq!(got, vec![a2.clone(), b.clone()], "numeric id order, the latest save wins");
        assert_eq!(bad, vec!["junk.json".to_string()]);
        assert!(reg.save(&AgentRecord { id: "../x".into(), ..b.clone() }).is_err(), "ids are file names, never paths");
        reg.remove("ag_2").unwrap();
        reg.remove("ag_404").unwrap();
        assert_eq!(reg.load().0, vec![b]);
        assert!(Registry::open(PathBuf::from("/tmp/x/models/reg")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
