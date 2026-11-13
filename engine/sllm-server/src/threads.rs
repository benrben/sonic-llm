//! One thread budget (phase 1 task 7, rules 040, 067, 069, 147).
//!
//! `--threads N` is the number of threads sllm-server may keep busy on the CPU side. It is shared by
//!
//! * the tokio runtime, which is also the HTTP server (axum runs on it): `http` worker threads, and
//! * ggml's CPU backend, used for the ops Metal lacks and for host work: `cpu` threads. ggml's calling thread is thread 0 of
//!   its own pool, and the caller is the executor thread, so the executor thread is counted inside `cpu`, not on top of it.
//!
//! The split is a pure function of the budget ([`split`]): `http = max(1, N / 4)`, `cpu = N - http`, for `N >= 2`.
//! The budget defaults to the value swept on this machine class and stored in `engine/config/threads.toml` (keyed by the
//! chip brand string and the two core counts), never to `available_parallelism()`. `--cpu-threads` and `--http-threads`
//! override their own component; the effective numbers and the overrides are reported in `/caps`.

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The smallest budget: one HTTP worker and one ggml thread (which is the executor).
pub const MIN_BUDGET: usize = 2;
/// Used only when the machine class has no stored sweep (an unswept machine). Fixed, small, and reported as a fallback.
pub const FALLBACK_BUDGET: usize = 4;
/// The rule `/caps` quotes.
pub const SPLIT_RULE: &str = "http (tokio workers, which also serve HTTP) = max(1, budget / 4); ggml cpu threads = budget - http; \
the executor thread is thread 0 of the ggml pool, so it is counted in the cpu share; tokio's blocking pool is capped at http";

/// The chip, as `machdep.cpu.brand_string` plus `hw.perflevel0.physicalcpu` / `hw.perflevel1.physicalcpu`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineClass {
    pub brand: String,
    pub perf_cores: u32,
    pub eff_cores: u32,
}

impl MachineClass {
    pub fn new(brand: impl Into<String>, perf_cores: u32, eff_cores: u32) -> Self {
        MachineClass { brand: brand.into(), perf_cores, eff_cores }
    }

    /// The key of a table in `threads.toml`, e.g. `Apple M5 Pro/perf0=6/perf1=12`.
    pub fn key(&self) -> String {
        format!("{}/perf0={}/perf1={}", self.brand, self.perf_cores, self.eff_cores)
    }

    /// Read the class from sysctl. `None` when any of the three values is unavailable (not macOS, or an odd chip).
    pub fn detect() -> Option<MachineClass> {
        let brand = sysctl("machdep.cpu.brand_string")?;
        let perf = sysctl("hw.perflevel0.physicalcpu")?.parse().ok()?;
        let eff = sysctl("hw.perflevel1.physicalcpu")?.parse().ok()?;
        Some(MachineClass::new(brand, perf, eff))
    }
}

fn sysctl(name: &str) -> Option<String> {
    let out = Command::new("sysctl").args(["-n", name]).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string()).filter(|s| !s.is_empty())
}

/// How a budget is divided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Split {
    pub budget: usize,
    /// Tokio worker threads (the async runtime and the HTTP server are the same runtime).
    pub http: usize,
    /// ggml CPU backend threads, including the executor thread.
    pub cpu: usize,
}

/// The deterministic split of a budget. `budget < 2` is an error: a smaller budget cannot give both sides a thread.
pub fn split(budget: usize) -> Result<Split, String> {
    if budget < MIN_BUDGET {
        return Err(format!("--threads must be at least {MIN_BUDGET} (got {budget})"));
    }
    let http = (budget / 4).max(1);
    Ok(Split { budget, http, cpu: budget - http })
}

/// Explicit per-component settings; they win over the split and are recorded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Overrides {
    pub cpu_threads: Option<usize>,
    pub http_threads: Option<usize>,
}

/// One machine class in `threads.toml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub class: MachineClass,
    pub threads: usize,
    pub swept: Option<String>,
    pub sweep_file: Option<String>,
    pub evidence_run_ids: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ThreadsFile {
    pub entries: Vec<Entry>,
}

impl ThreadsFile {
    pub fn lookup(&self, class: &MachineClass) -> Option<&Entry> {
        self.entries.iter().find(|e| &e.class == class)
    }

    pub fn load(path: &Path) -> Result<ThreadsFile, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        ThreadsFile::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Parse the small TOML subset the file uses: `schema = 1`, then one `[machine."<key>"]` table per class holding
    /// `brand`, `perf_cores`, `efficiency_cores`, `threads` (required) and `swept`, `sweep_file`, `evidence_run_ids`.
    pub fn parse(text: &str) -> Result<ThreadsFile, String> {
        let mut raws: Vec<RawTable> = Vec::new();
        let mut lines = text.lines().enumerate();
        while let Some((i, raw_line)) = lines.next() {
            let n = i + 1;
            let line = strip_comment(raw_line).trim().to_string();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix('[') {
                raws.push(RawTable { key: table_key(rest, n)?, line: n, ..Default::default() });
                continue;
            }
            let (name, value) = key_value(&line, n, &mut lines)?;
            let Some(cur) = raws.last_mut() else {
                if name == "schema" {
                    continue;
                }
                return Err(format!("line {n}: key {name:?} outside a [machine.\"...\"] table"));
            };
            cur.set(n, &name, &value)?;
        }
        Ok(ThreadsFile { entries: entries_of(raws)? })
    }
}

/// The tables as checked entries, in file order.
fn entries_of(raws: Vec<RawTable>) -> Result<Vec<Entry>, String> {
    let mut entries = Vec::new();
    for r in raws {
        let e = r.into_entry(&entries)?;
        entries.push(e);
    }
    Ok(entries)
}

/// One `[machine."..."]` table as read, before its fields are checked.
#[derive(Default)]
struct RawTable {
    key: String,
    line: usize,
    brand: Option<String>,
    perf: Option<u32>,
    eff: Option<u32>,
    threads: Option<usize>,
    swept: Option<String>,
    sweep_file: Option<String>,
    ids: Vec<String>,
}

impl RawTable {
    /// The value of key `name` (line `n`); unknown keys are notes and future fields.
    fn set(&mut self, n: usize, name: &str, value: &str) -> Result<(), String> {
        match name {
            "brand" | "swept" | "sweep_file" => self.set_string(n, name, value),
            "perf_cores" | "efficiency_cores" | "threads" => self.set_int(n, name, value),
            "evidence_run_ids" => {
                self.ids = parse_string_array(value).ok_or(format!("line {n}: evidence_run_ids must be an array of strings"))?;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn set_string(&mut self, n: usize, name: &str, value: &str) -> Result<(), String> {
        let v = parse_string(value).ok_or(format!("line {n}: {name} must be a quoted string, got {value}"))?;
        match name {
            "brand" => self.brand = Some(v),
            "swept" => self.swept = Some(v),
            _ => self.sweep_file = Some(v),
        }
        Ok(())
    }

    fn set_int(&mut self, n: usize, name: &str, value: &str) -> Result<(), String> {
        let v = value.parse::<usize>().map_err(|_| format!("line {n}: {name} must be a non-negative integer, got {value}"))?;
        match name {
            "perf_cores" => self.perf = Some(v as u32),
            "efficiency_cores" => self.eff = Some(v as u32),
            _ => self.threads = Some(v),
        }
        Ok(())
    }

    /// The checked entry: every required field present, the table name matching them, and no table seen twice.
    fn into_entry(self, entries: &[Entry]) -> Result<Entry, String> {
        let at = self.line;
        let need = |what: &str| format!("line {at}: table [{}] has no {what}", self.key);
        let class = MachineClass::new(
            self.brand.clone().ok_or_else(|| need("brand"))?,
            self.perf.ok_or_else(|| need("perf_cores"))?,
            self.eff.ok_or_else(|| need("efficiency_cores"))?,
        );
        if class.key() != self.key {
            return Err(format!("line {at}: table name {:?} does not match its fields, which give {:?}", self.key, class.key()));
        }
        let threads = self.threads.ok_or_else(|| need("threads"))?;
        if threads < MIN_BUDGET {
            return Err(format!("line {at}: threads {threads} is below the minimum budget {MIN_BUDGET}"));
        }
        if entries.iter().any(|e: &Entry| e.class == class) {
            return Err(format!("line {at}: duplicate table {:?}", self.key));
        }
        Ok(Entry { class, threads, swept: self.swept, sweep_file: self.sweep_file, evidence_run_ids: self.ids })
    }
}

/// The class key of a table header (`rest`: the line after its `[`).
fn table_key(rest: &str, n: usize) -> Result<String, String> {
    let rest = rest.strip_suffix(']').ok_or(format!("line {n}: unterminated table header"))?;
    rest.strip_prefix("machine.")
        .and_then(|k| parse_string(k.trim()))
        .ok_or(format!("line {n}: expected [machine.\"<chip>/perf0=P/perf1=E\"], got [{rest}]"))
}

/// A `key = value` line (line `n`); a multi-line array keeps joining `lines` until its bracket closes.
fn key_value<'a>(line: &str, n: usize, lines: &mut impl Iterator<Item = (usize, &'a str)>) -> Result<(String, String), String> {
    let (name, mut value) = line.split_once('=').map(|(a, b)| (a.trim().to_string(), b.trim().to_string())).ok_or(format!("line {n}: expected key = value"))?;
    while value.starts_with('[') && !value.ends_with(']') {
        let (_, next) = lines.next().ok_or(format!("line {n}: unterminated array"))?;
        value.push(' ');
        value.push_str(strip_comment(next).trim());
    }
    Ok((name, value))
}

/// Drop a `#` comment that is not inside a string.
fn strip_comment(line: &str) -> &str {
    let mut in_str = false;
    let mut prev = '\0';
    for (i, c) in line.char_indices() {
        match c {
            '"' if prev != '\\' => in_str = !in_str,
            '#' if !in_str => return &line[..i],
            _ => {}
        }
        prev = c;
    }
    line
}

fn parse_string(s: &str) -> Option<String> {
    let inner = s.trim().strip_prefix('"')?.strip_suffix('"')?;
    Some(inner.replace("\\\"", "\"").replace("\\\\", "\\"))
}

fn parse_string_array(s: &str) -> Option<Vec<String>> {
    let inner = s.trim().strip_prefix('[')?.strip_suffix(']')?.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    let inner = inner.strip_suffix(',').unwrap_or(inner);
    inner.split(',').map(|p| parse_string(p.trim())).collect()
}

/// Where the budget came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// `--threads N`.
    Flag,
    /// The stored sweep for this machine class.
    Stored { key: String, swept: Option<String>, sweep_file: Option<String>, evidence_run_ids: Vec<String> },
    /// No stored sweep: [`FALLBACK_BUDGET`], and why.
    Fallback { reason: String },
}

/// The resolved plan: the budget, the split it gives, the overrides and the effective numbers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub split: Split,
    pub source: Source,
    pub overrides: Overrides,
    pub machine: Option<MachineClass>,
    pub config_path: Option<PathBuf>,
    /// Tokio worker threads actually used (the override, else the split).
    pub http: usize,
    /// ggml CPU threads actually used (the override, else the split).
    pub cpu: usize,
}

impl Plan {
    pub fn budget(&self) -> usize {
        self.split.budget
    }

    /// True when overrides make the effective threads exceed the budget.
    pub fn over_budget(&self) -> bool {
        self.http + self.cpu > self.split.budget
    }

    pub fn source_label(&self) -> String {
        match &self.source {
            Source::Flag => "--threads".into(),
            Source::Stored { key, .. } => format!("stored sweep for {key}"),
            Source::Fallback { reason } => format!("fallback {FALLBACK_BUDGET} ({reason})"),
        }
    }

    /// The `threads` member of `/caps`.
    pub fn to_json(&self) -> Value {
        let source = match &self.source {
            Source::Flag => json!({"kind": "flag", "detail": "--threads"}),
            Source::Stored { key, swept, sweep_file, evidence_run_ids } => json!({
                "kind": "stored", "machine_class": key, "swept": swept, "sweep_file": sweep_file, "evidence_run_ids": evidence_run_ids,
            }),
            Source::Fallback { reason } => json!({"kind": "fallback", "fallback_budget": FALLBACK_BUDGET, "reason": reason}),
        };
        json!({
            "budget": self.split.budget,
            "source": source,
            "split_rule": SPLIT_RULE,
            "split": {"http": self.split.http, "ggml_cpu": self.split.cpu},
            "overrides": {"cpu_threads": self.overrides.cpu_threads, "http_threads": self.overrides.http_threads},
            "effective": {
                "tokio_workers": self.http, "http": self.http, "ggml_cpu": self.cpu,
                "executor": "thread 0 of the ggml pool (counted in ggml_cpu)", "total": self.http + self.cpu,
            },
            "over_budget": self.over_budget(),
            "machine": self.machine.as_ref().map(|m| json!({
                "brand": m.brand, "perf_cores": m.perf_cores, "efficiency_cores": m.eff_cores, "key": m.key(),
            })),
            "config": self.config_path.as_ref().map(|p| p.display().to_string()),
            "never_defaulted_to": "available_parallelism() or ggml's all-cores default",
        })
    }
}

/// `engine/config/threads.toml`, located from this crate's source tree (the binary is built from it).
pub fn default_config_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("config").join("threads.toml")
}

/// What [`resolve`] needs, so tests can supply everything without touching sysctl or the disk.
pub struct Inputs {
    /// `--threads`.
    pub flag: Option<usize>,
    pub overrides: Overrides,
    pub machine: Option<MachineClass>,
    /// The loaded file and where it came from; `Err` when it could not be read.
    pub file: Option<(PathBuf, Result<ThreadsFile, String>)>,
}

/// Decide the budget, split it, apply the overrides.
///
/// Precedence of the budget: `--threads`, then the stored sweep of this machine class, then [`FALLBACK_BUDGET`].
/// A file that exists but does not parse is an error (a silent fallback would hide a broken config); a missing file or a
/// machine class with no entry is a recorded fallback.
pub fn resolve(inp: Inputs) -> Result<Plan, String> {
    for (flag, v) in [("--cpu-threads", inp.overrides.cpu_threads), ("--http-threads", inp.overrides.http_threads)] {
        if v == Some(0) {
            return Err(format!("{flag} must be at least 1"));
        }
    }
    let config_path = inp.file.as_ref().map(|(p, _)| p.clone());
    let (budget, source) = if let Some(n) = inp.flag {
        (n, Source::Flag)
    } else {
        match (&inp.machine, &inp.file) {
            (None, _) => (FALLBACK_BUDGET, Source::Fallback { reason: "machine class unknown (sysctl failed)".into() }),
            (Some(_), None) => (FALLBACK_BUDGET, Source::Fallback { reason: "no threads.toml".into() }),
            (Some(_), Some((p, Err(e)))) => {
                if p.exists() {
                    return Err(format!("cannot use {}: {e}", p.display()));
                }
                (FALLBACK_BUDGET, Source::Fallback { reason: format!("{} not found", p.display()) })
            }
            (Some(m), Some((p, Ok(f)))) => match f.lookup(m) {
                Some(e) => (
                    e.threads,
                    Source::Stored {
                        key: m.key(),
                        swept: e.swept.clone(),
                        sweep_file: e.sweep_file.clone(),
                        evidence_run_ids: e.evidence_run_ids.clone(),
                    },
                ),
                None => (FALLBACK_BUDGET, Source::Fallback { reason: format!("no entry for {} in {}", m.key(), p.display()) }),
            },
        }
    };
    let s = split(budget)?;
    Ok(Plan {
        split: s,
        source,
        overrides: inp.overrides,
        machine: inp.machine,
        config_path,
        http: inp.overrides.http_threads.unwrap_or(s.http),
        cpu: inp.overrides.cpu_threads.unwrap_or(s.cpu),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m6() -> MachineClass {
        MachineClass::new("Apple M5 Pro", 6, 12)
    }

    fn file(threads: usize) -> ThreadsFile {
        ThreadsFile::parse(&format!(
            "schema = 1\n[machine.\"Apple M5 Pro/perf0=6/perf1=12\"]\nbrand = \"Apple M5 Pro\"\nperf_cores = 6\nefficiency_cores = 12\nthreads = {threads}\n"
        ))
        .unwrap()
    }

    fn inputs(flag: Option<usize>, ov: Overrides, f: Option<ThreadsFile>) -> Inputs {
        Inputs {
            flag,
            overrides: ov,
            machine: Some(m6()),
            file: f.map(|f| (PathBuf::from("/x/threads.toml"), Ok(f))),
        }
    }

    #[test]
    fn split_is_deterministic_and_adds_up() {
        let got: Vec<(usize, usize, usize)> = [2, 3, 4, 6, 8, 12, 16].iter().map(|&n| { let s = split(n).unwrap(); (n, s.http, s.cpu) }).collect();
        assert_eq!(got, vec![(2, 1, 1), (3, 1, 2), (4, 1, 3), (6, 1, 5), (8, 2, 6), (12, 3, 9), (16, 4, 12)]);
        for n in 2..64 {
            let s = split(n).unwrap();
            assert_eq!(s.http + s.cpu, n);
            assert!(s.http >= 1 && s.cpu >= 1);
        }
    }

    #[test]
    fn budget_below_two_is_refused() {
        assert!(split(0).unwrap_err().contains("at least 2"));
        assert!(split(1).is_err());
    }

    #[test]
    fn flag_beats_the_stored_value() {
        let p = resolve(inputs(Some(6), Overrides::default(), Some(file(4)))).unwrap();
        assert_eq!((p.budget(), p.http, p.cpu), (6, 1, 5));
        assert_eq!(p.source, Source::Flag);
    }

    #[test]
    fn stored_value_is_the_default() {
        let p = resolve(inputs(None, Overrides::default(), Some(file(4)))).unwrap();
        assert_eq!((p.budget(), p.http, p.cpu), (4, 1, 3));
        assert!(matches!(p.source, Source::Stored { .. }));
        assert_eq!(p.to_json()["source"]["machine_class"], "Apple M5 Pro/perf0=6/perf1=12");
    }

    #[test]
    fn overrides_take_precedence_and_are_recorded() {
        let ov = Overrides { cpu_threads: Some(2), http_threads: Some(3) };
        let p = resolve(inputs(Some(8), ov, None)).unwrap();
        assert_eq!((p.split.http, p.split.cpu), (2, 6), "the split itself is unchanged");
        assert_eq!((p.http, p.cpu), (3, 2), "the overrides win");
        let j = p.to_json();
        assert_eq!(j["overrides"]["cpu_threads"], 2);
        assert_eq!(j["overrides"]["http_threads"], 3);
        assert_eq!(j["effective"]["ggml_cpu"], 2);
        assert_eq!(j["over_budget"], false);
        let only_cpu = resolve(inputs(Some(8), Overrides { cpu_threads: Some(7), http_threads: None }, None)).unwrap();
        assert_eq!((only_cpu.http, only_cpu.cpu), (2, 7));
        assert!(only_cpu.over_budget());
        assert_eq!(only_cpu.to_json()["overrides"]["http_threads"], Value::Null);
    }

    #[test]
    fn zero_override_is_refused() {
        let e = resolve(inputs(Some(4), Overrides { cpu_threads: Some(0), http_threads: None }, None)).unwrap_err();
        assert!(e.contains("--cpu-threads must be at least 1"), "{e}");
        let e = resolve(inputs(Some(4), Overrides { cpu_threads: None, http_threads: Some(0) }, None)).unwrap_err();
        assert!(e.contains("--http-threads must be at least 1"), "{e}");
    }

    #[test]
    fn unknown_machine_or_missing_file_falls_back_to_a_fixed_small_value() {
        let mut inp = inputs(None, Overrides::default(), Some(file(4)));
        inp.machine = Some(MachineClass::new("Apple M9 Max", 12, 4));
        let p = resolve(inp).unwrap();
        assert_eq!(p.budget(), FALLBACK_BUDGET);
        assert!(matches!(&p.source, Source::Fallback { reason } if reason.contains("no entry for Apple M9 Max")));
        let mut inp = inputs(None, Overrides::default(), None);
        inp.machine = None;
        assert_eq!(resolve(inp).unwrap().budget(), FALLBACK_BUDGET);
        let mut inp = inputs(None, Overrides::default(), None);
        inp.file = Some((PathBuf::from("/definitely/not/here.toml"), Err("not found".into())));
        let p = resolve(inp).unwrap();
        assert!(matches!(&p.source, Source::Fallback { reason } if reason.contains("not found")));
    }

    #[test]
    fn a_broken_file_that_exists_is_an_error() {
        let mut inp = inputs(None, Overrides::default(), None);
        inp.file = Some((PathBuf::from("/"), Err("line 3: nonsense".into())));
        assert!(resolve(inp).unwrap_err().contains("nonsense"));
    }

    #[test]
    fn parses_the_documented_format() {
        let text = r#"
# comment
schema = 1

[machine."Apple M5 Pro/perf0=6/perf1=12"]   # trailing comment
brand = "Apple M5 Pro"
perf_cores = 6
efficiency_cores = 12
threads = 4
swept = "2026-10-26"
sweep_file = "results/threads/sweep-x.jsonl"
evidence_run_ids = [
  "a-1",
  "a-2", # why
]
note = "ignored"

[machine."Apple M4/perf0=4/perf1=6"]
brand = "Apple M4"
perf_cores = 4
efficiency_cores = 6
threads = 2
"#;
        let f = ThreadsFile::parse(text).unwrap();
        assert_eq!(f.entries.len(), 2);
        let e = f.lookup(&m6()).unwrap();
        assert_eq!(e.threads, 4);
        assert_eq!(e.evidence_run_ids, vec!["a-1", "a-2"]);
        assert_eq!(e.swept.as_deref(), Some("2026-10-26"));
        assert_eq!(f.lookup(&MachineClass::new("Apple M4", 4, 6)).unwrap().threads, 2);
        assert!(f.lookup(&MachineClass::new("Apple M4", 4, 7)).is_none());
    }

    #[test]
    fn rejects_malformed_files() {
        let hdr = "[machine.\"Apple M5 Pro/perf0=6/perf1=12\"]\nbrand = \"Apple M5 Pro\"\nperf_cores = 6\nefficiency_cores = 12\n";
        assert!(ThreadsFile::parse(&format!("{hdr}threads = 1\n")).unwrap_err().contains("minimum"));
        assert!(ThreadsFile::parse(hdr).unwrap_err().contains("no threads"));
        assert!(ThreadsFile::parse(&format!("{hdr}threads = four\n")).unwrap_err().contains("integer"));
        assert!(ThreadsFile::parse(&format!("{hdr}threads = 4\n{hdr}threads = 4\n")).unwrap_err().contains("duplicate"));
        assert!(ThreadsFile::parse("threads = 4\n").unwrap_err().contains("outside"));
        assert!(ThreadsFile::parse("[machine.\"wrong key\"]\nbrand = \"Apple M5 Pro\"\nperf_cores = 6\nefficiency_cores = 12\nthreads = 4\n").unwrap_err().contains("does not match"));
        assert!(ThreadsFile::parse("[machine \"x\"]\n").is_err());
    }

    #[test]
    fn the_shipped_config_parses_and_has_this_machine_class_when_run_on_it() {
        let f = ThreadsFile::load(&default_config_path()).expect("engine/config/threads.toml parses");
        assert!(!f.entries.is_empty());
        assert!(f.entries.iter().all(|e| e.threads >= MIN_BUDGET && !e.evidence_run_ids.is_empty()), "every entry carries its evidence run ids");
        if let Some(m) = MachineClass::detect() {
            if m.brand == "Apple M5 Pro" {
                assert!(f.lookup(&m).is_some(), "swept value for {}", m.key());
            }
        }
    }
}
