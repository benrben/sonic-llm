//! What the bits of a token depend on, as far as the engine can know it.
//!
//! * The key of the prompt cache on disk: a saved state only loads into the build and the settings that made it. The
//!   Metal backend and libllama are identified by their bytes (hash of the file the loader mapped), the settings by the
//!   environment variables those two files read. The names come out of the files, so a library with a new knob is
//!   covered without a change here, and no model family is named anywhere.
//! * `/caps` `exact`: the row limit of exact decode and what exactness covers, with the source of every figure. The
//!   vendored library has no getter for its row limits, so they are computed here from the environment with the same
//!   clamp, and the output says so.

use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::OnceLock;

/// Rows of one step the row-exact weight mat-mul takes in `engine/vendor/ggml-rowexact-b11669` (its few-row MMA path
/// stops at 16; above that the stock kernels run, whatever `SLLM_ROW_EXACT_MAXN` says). A constant of the vendored
/// patch: nothing in the library reports it.
pub const MMA_ROWS: usize = 16;
/// `SLLM_ROW_EXACT_MAXN` in the vendored patch: default 16, clamped to 1..=32.
const MAXN_DEFAULT: i64 = 16;
const MAXN_RANGE: (i64, i64) = (1, 32);
const MAXN_ENV: &str = "SLLM_ROW_EXACT_MAXN";

#[repr(C)]
struct DlInfo {
    fname: *const c_char,
    fbase: *mut c_void,
    sname: *const c_char,
    saddr: *mut c_void,
}

unsafe extern "C" {
    fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// A library the engine runs on, identified by its bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lib {
    pub path: String,
    pub bytes: u64,
    pub hash: u64,
}

impl Lib {
    fn id(&self) -> String {
        format!("{:016x}/{}", self.hash, self.bytes)
    }
}

/// The environment variable names a library reads: the identifier-like strings in its file that start with a prefix of
/// the stack (`SLLM_`, `GGML_`, `LLAMA_`). Some are not variables (constants of the same shape); only those set in the
/// environment ever matter.
fn env_names(bytes: &[u8]) -> BTreeSet<String> {
    bytes
        .split(|&b| b == 0)
        .filter(|s| (6..=64).contains(&s.len()) && [&b"SLLM_"[..], b"GGML_", b"LLAMA_"].iter().any(|p| s.starts_with(p)))
        .filter(|s| s.iter().all(|&c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == b'_'))
        .map(|s| String::from_utf8_lossy(s).into_owned())
        .collect()
}

fn read_lib(path: &str) -> Result<(Lib, BTreeSet<String>), String> {
    let real = std::fs::canonicalize(path).unwrap_or_else(|_| path.into());
    let bytes = std::fs::read(&real).map_err(|e| format!("{}: {e}", real.display()))?;
    Ok((Lib { path: real.display().to_string(), bytes: bytes.len() as u64, hash: fnv(&bytes) }, env_names(&bytes)))
}

/// The file the loader took the code at `addr` from.
fn file_of(addr: *const c_void) -> Result<String, String> {
    // SAFETY: dladdr only reads the loader's tables; `addr` is a function of a linked library.
    unsafe {
        let mut dl: DlInfo = std::mem::zeroed();
        if dladdr(addr, &mut dl) == 0 || dl.fname.is_null() {
            return Err("the loader does not say which file the code came from".into());
        }
        Ok(CStr::from_ptr(dl.fname).to_string_lossy().into_owned())
    }
}

/// The compiled stack the math runs on: the Metal backend (its kernels are embedded in the file), libllama (graph
/// construction, the cache rotation) and the settings in the environment that those two read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Backend {
    pub metal: Lib,
    pub llama: Lib,
    pub ggml: String,
    /// `NAME=value` of every variable the two libraries read that is set, sorted.
    pub env: Vec<(String, String)>,
    /// The chip and the OS build: the OS compiles the embedded kernels when the process starts.
    pub host: String,
}

impl Backend {
    /// Identify the stack this process runs on.
    fn detect() -> Result<Backend, String> {
        let metal_file = sllm_core::caps::ggml_metal_info().path;
        if metal_file.is_empty() {
            return Err("the Metal backend file is not known".into());
        }
        let (metal, mut names) = read_lib(&metal_file).map_err(|e| format!("cannot identify the Metal backend: {e}"))?;
        let llama_file = file_of(llama_ref::sys::llama_init_from_model as *const c_void)?;
        let (llama, llama_names) = read_lib(&llama_file).map_err(|e| format!("cannot identify libllama: {e}"))?;
        names.extend(llama_names);
        let env = names.into_iter().filter_map(|n| std::env::var_os(&n).map(|v| (n, v.to_string_lossy().into_owned()))).collect();
        let (v, c) = sllm_core::ggml_version();
        let chip = crate::profile::startup().map(|s| s.machine.brand.clone()).unwrap_or_else(|| crate::profile::Machine::detect().brand);
        let os = crate::profile::sysctl("kern.osversion").unwrap_or_else(|| "?".into());
        Ok(Backend { metal, llama, ggml: format!("{v} {c}"), env, host: format!("{chip}, OS build {os}") })
    }

    /// The part of the prompt cache key that stands for the build and the math settings.
    pub fn key(&self) -> String {
        let env: Vec<String> = self.env.iter().map(|(k, v)| format!("{k}={v}")).collect();
        format!("metal={}|llama={}|ggml={}|host={}|env={}", self.metal.id(), self.llama.id(), self.ggml, self.host, env.join(","))
    }

    pub fn to_json(&self) -> Value {
        let lib = |l: &Lib| json!({"file": l.path, "bytes": l.bytes, "fnv1a64": format!("{:016x}", l.hash)});
        json!({
            "metal": lib(&self.metal), "libllama": lib(&self.llama), "ggml": self.ggml, "host": self.host,
            "env": self.env.iter().map(|(k, v)| json!({k: v})).collect::<Vec<_>>(),
            "note": "identified by the bytes of the files the loader mapped; the prompt cache key contains all of this",
        })
    }
}

static BACKEND: OnceLock<Result<Backend, String>> = OnceLock::new();

/// The stack this process runs on (identified once; the files are hashed, a few ms).
pub fn backend() -> &'static Result<Backend, String> {
    BACKEND.get_or_init(Backend::detect)
}

/// `SLLM_ROW_EXACT_MAXN` as the library reads it, and the row limit that follows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowLimit {
    /// The variable as set (`None`: not set, or empty, which the library treats as not set).
    pub requested: Option<String>,
    /// What the library uses: `atoi`, clamped to 1..=32, 16 when not set.
    pub maxn: usize,
    /// The library clamped the requested value without a word.
    pub clamped: bool,
    /// Rows of one step that still equal a solo decode: `maxn`, and at most [`MMA_ROWS`].
    pub effective: usize,
}

/// `atoi` of C: leading spaces, a sign, digits up to the first other character; 0 when there are none.
fn atoi(s: &str) -> i64 {
    let t = s.trim_start();
    let (neg, rest) = match t.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    let n = if digits.is_empty() { 0 } else { digits.parse::<i64>().unwrap_or(i64::MAX) };
    if neg { -n } else { n }
}

pub fn row_limit(var: Option<&str>) -> RowLimit {
    let requested = var.filter(|v| !v.is_empty()).map(str::to_string);
    let (n, clamped) = match &requested {
        None => (MAXN_DEFAULT, false),
        Some(v) => {
            let n = atoi(v);
            let c = n.clamp(MAXN_RANGE.0, MAXN_RANGE.1);
            (c, c != n)
        }
    };
    RowLimit { requested, maxn: n as usize, clamped, effective: (n as usize).min(MMA_ROWS) }
}

/// The row limit of this process: the environment, read the way the library reads it.
pub fn row_limit_here() -> RowLimit {
    row_limit(std::env::var(MAXN_ENV).ok().as_deref())
}

/// The library's own state of the row-exact switch; `None` when the linked library has no switch (the official build).
pub fn library_switch() -> Option<bool> {
    // RTLD_DEFAULT on macOS
    let h = -2isize as *mut c_void;
    // SAFETY: the symbol, when present, is `int (void)`.
    let p = unsafe { crate::llama_engine::ffi::dlsym(h, c"ggml_backend_metal_sllm_get_row_exact".as_ptr()) };
    if p.is_null() {
        return None;
    }
    let f: extern "C" fn() -> i32 = unsafe { std::mem::transmute(p) };
    Some(f() != 0)
}

/// What the engine knows about exactness for `/caps` `exact`.
pub struct ExactFacts {
    /// `--row-exact` as given.
    pub requested: &'static str,
    /// The engine's own flag: it asked the library for the switch (`share_align` and speculation follow it).
    pub engine_on: bool,
    /// The library's state of the switch, asked of the library; `None`: the linked ggml has no switch (the official one).
    /// The environment (`SLLM_ROW_EXACT`) can turn it on without the engine.
    pub library_state: Option<bool>,
    pub unified: bool,
    pub moe: bool,
    pub recurrent_or_hybrid: bool,
    /// Rows the scheduler can put in one step (lane slots).
    pub scheduler_rows: usize,
    pub limit: RowLimit,
}

pub fn exact_json(f: &ExactFacts) -> Value {
    let mut not_covered: Vec<&str> = vec![];
    let on = f.library_state == Some(true);
    match f.library_state {
        None => not_covered.push("the linked ggml has no row-exact switch"),
        Some(false) => not_covered.push("the row-exact switch is off (--row-exact off, or auto found no guarantee for this model)"),
        Some(true) => {}
    }
    if f.unified {
        not_covered.push("a unified KV pool: other sequences' cell layout changes the attention blocks");
    }
    if f.moe {
        not_covered.push("mixture-of-experts mat-muls have no row-exact path");
    }
    let holds = not_covered.is_empty();
    json!({
        "holds": holds,
        "not_covered": not_covered,
        "means": "a row's logits equal a solo decode of the same sequence while the step has at most `rows.effective_limit` rows; steps with more rows (a prefill chunk, a wide batch) run the stock kernels",
        "switch": {"requested": f.requested, "on": on, "library_state": f.library_state, "engine_flag": f.engine_on, "agree": f.library_state.is_none_or(|l| l == f.engine_on)},
        "kv_streams": if f.unified { "unified pool" } else { "one stream per sequence" },
        "recurrent_or_hybrid": f.recurrent_or_hybrid,
        "rows": {
            "effective_limit": f.limit.effective,
            "scheduler_rows": f.scheduler_rows,
            "scheduler_can_exceed_limit": f.scheduler_rows > f.limit.effective,
            "env": {MAXN_ENV: f.limit.requested, "used_by_library": f.limit.maxn, "clamped_silently": f.limit.clamped, "range": [MAXN_RANGE.0, MAXN_RANGE.1], "default": MAXN_DEFAULT},
            "mma_path_limit": MMA_ROWS,
            "source": "the environment variable read with the library's own default and clamp, and the MMA path's row limit as a constant of the vendored patch; the library reports neither",
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_limit_follows_the_library_clamp() {
        let r = row_limit(None);
        assert_eq!((r.maxn, r.clamped, r.effective, r.requested), (16, false, 16, None));
        assert_eq!(row_limit(Some("")).effective, 16, "empty is not set");
        let r = row_limit(Some("8"));
        assert_eq!((r.maxn, r.clamped, r.effective), (8, false, 8));
        // 32 is allowed by the library, but the MMA path stops at 16
        let r = row_limit(Some("32"));
        assert_eq!((r.maxn, r.clamped, r.effective), (32, false, 16));
        // clamped without a word: 99 to 32, 0 and junk and negatives to 1
        let r = row_limit(Some("99"));
        assert_eq!((r.maxn, r.clamped, r.effective), (32, true, 16));
        for v in ["0", "abc", "-5", "  "] {
            let r = row_limit(Some(v));
            assert_eq!((r.maxn, r.clamped, r.effective), (1, true, 1), "{v:?}");
        }
        // atoi stops at the first other character and skips leading spaces
        assert_eq!(row_limit(Some(" 12abc")).maxn, 12);
        assert_eq!(row_limit(Some("+7")).maxn, 7);
        assert_eq!(row_limit(Some("99999999999999999999")).maxn, 32);
    }

    #[test]
    fn env_names_come_out_of_the_file() {
        let mut f = b"\0junk\0SLLM_ROW_EXACT_MAXN\0lowercase_name\0GGML_METAL_FUSION_DISABLE\0LLAMA_ATTN_ROT_DISABLE\0SLLM_\0Sllm_Mixed\0GGML_METAL_HAS BLANK\0".to_vec();
        f.extend_from_slice(b"SLLM_END_WITHOUT_NUL");
        let n: Vec<String> = env_names(&f).into_iter().collect();
        assert_eq!(n, ["GGML_METAL_FUSION_DISABLE", "LLAMA_ATTN_ROT_DISABLE", "SLLM_END_WITHOUT_NUL", "SLLM_ROW_EXACT_MAXN"]);
    }

    fn backend_fixture() -> Backend {
        let lib = |h| Lib { path: "/x".into(), bytes: 10, hash: h };
        Backend { metal: lib(1), llama: lib(2), ggml: "0.27.0 abc".into(), env: vec![], host: "chip, OS build 1".into() }
    }

    #[test]
    fn the_cache_key_changes_with_the_kernels_and_the_settings() {
        let a = backend_fixture();
        assert_eq!(a.key(), backend_fixture().key());
        let mut b = a.clone();
        b.metal.hash = 3;
        assert_ne!(a.key(), b.key(), "other kernels");
        let mut b = a.clone();
        b.llama.bytes = 11;
        assert_ne!(a.key(), b.key(), "other libllama");
        let mut b = a.clone();
        b.env = vec![("SLLM_ROW_EXACT_MAXN".into(), "8".into())];
        assert_ne!(a.key(), b.key(), "another setting");
        let mut c = b.clone();
        c.env = vec![("SLLM_ROW_EXACT_MAXN".into(), "9".into())];
        assert_ne!(b.key(), c.key(), "another value");
        let mut b = a.clone();
        b.host = "chip, OS build 2".into();
        assert_ne!(a.key(), b.key(), "another OS build compiles the kernels again");
    }

    #[test]
    #[ignore = "starts the Metal backend registry: run through engine/scripts/with_gpu_lock.py"]
    fn this_process_is_identified() {
        // the test binary links libllama and the Metal backend: both files are found and hashed
        let b = backend().as_ref().expect("backend identified");
        assert!(b.metal.path.contains("libggml-metal") && b.metal.bytes > 1_000_000, "{b:?}");
        assert!(b.llama.path.contains("libllama") && b.llama.bytes > 1_000_000, "{b:?}");
        assert_ne!(b.metal.hash, b.llama.hash);
        assert!(b.key().contains(&b.metal.id()));
    }

    fn facts() -> ExactFacts {
        ExactFacts {
            requested: "auto",
            engine_on: true,
            library_state: Some(true),
            unified: false,
            moe: false,
            recurrent_or_hybrid: false,
            scheduler_rows: 64,
            limit: row_limit(Some("99")),
        }
    }

    #[test]
    fn caps_report_what_exactness_covers_and_does_not() {
        let j = exact_json(&facts());
        assert_eq!(j["holds"], true);
        assert_eq!(j["switch"]["agree"], true);
        assert_eq!(j["rows"]["effective_limit"], 16);
        assert_eq!(j["rows"]["env"]["clamped_silently"], true);
        assert_eq!(j["rows"]["env"]["used_by_library"], 32);
        assert_eq!(j["rows"]["scheduler_can_exceed_limit"], true);
        let j = exact_json(&ExactFacts { moe: true, unified: true, engine_on: false, library_state: None, ..facts() });
        assert_eq!(j["holds"], false);
        assert_eq!(j["not_covered"].as_array().unwrap().len(), 3);
        // the environment turned the switch on behind the engine's back: reported as it is
        let j = exact_json(&ExactFacts { engine_on: false, ..facts() });
        assert_eq!((j["holds"].as_bool(), j["switch"]["agree"].as_bool()), (Some(true), Some(false)));
        let j = exact_json(&ExactFacts { limit: row_limit(Some("4")), scheduler_rows: 3, ..facts() });
        assert_eq!(j["rows"]["effective_limit"], 4);
        assert_eq!(j["rows"]["scheduler_can_exceed_limit"], false);
    }
}
