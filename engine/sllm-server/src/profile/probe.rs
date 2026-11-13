//! libllama's own accounting: a `no_alloc` context, and the log lines it prints.

use super::KvType;
use llama_ref::sys;
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::sync::Mutex;

/// Serialises every user of libllama's process-global log callback (probes, calibration, tests).
pub static LLAMA_LOG_GUARD: Mutex<()> = Mutex::new(());
static LOG_BUF: Mutex<Option<String>> = Mutex::new(None);

unsafe extern "C" fn cap_log(_level: sys::ggml_log_level, text: *const c_char, _ud: *mut c_void) {
    if text.is_null() {
        return;
    }
    if let Ok(mut g) = LOG_BUF.lock() {
        if let Some(s) = g.as_mut() {
            s.push_str(&unsafe { CStr::from_ptr(text) }.to_string_lossy());
        }
    }
}

/// Same as llama-ref's: drop the informational lines, keep errors (what the server runs with).
unsafe extern "C" fn quiet_log(level: sys::ggml_log_level, text: *const c_char, _ud: *mut c_void) {
    if level == sys::GGML_LOG_LEVEL_ERROR && !text.is_null() {
        eprint!("ggml: {}", unsafe { CStr::from_ptr(text) }.to_string_lossy());
    }
}

/// Run `f` and return what libllama and ggml logged meanwhile. Afterwards the log goes back to errors-only (or to stderr
/// in full with `LLAMA_REF_VERBOSE`). Holds [`LLAMA_LOG_GUARD`].
pub fn capture_log<R>(f: impl FnOnce() -> R) -> (R, String) {
    let _g = LLAMA_LOG_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    *LOG_BUF.lock().unwrap() = Some(String::new());
    unsafe { sys::llama_log_set(Some(cap_log), std::ptr::null_mut()) };
    let r = f();
    if std::env::var_os("LLAMA_REF_VERBOSE").is_some() {
        unsafe { sys::llama_log_set(None, std::ptr::null_mut()) };
    } else {
        unsafe { sys::llama_log_set(Some(quiet_log), std::ptr::null_mut()) };
    }
    let log = LOG_BUF.lock().unwrap().take().unwrap_or_default();
    (r, log)
}

/// One `llama_kv_cache: size = ...` line.
#[derive(Debug, Clone, PartialEq)]
pub struct KvLine {
    pub cells: u64,
    pub layers: usize,
    pub n_seq: u64,
    pub mib_total: f64,
    pub mib_k: f64,
    pub mib_v: f64,
    pub type_k: String,
    pub type_v: String,
}

impl KvLine {
    pub fn bytes(&self) -> f64 {
        self.mib_total * 1048576.0
    }
}

/// The KV cache lines of a libllama log (one per cache: a model with sliding-window layers has two, full then SWA).
pub fn parse_kv_lines(log: &str) -> Vec<KvLine> {
    let mut out = Vec::new();
    for line in log.lines() {
        // "llama_kv_cache: size =  224.00 MiB (  4096 cells,  28 layers,  1/1 seqs), K (f16):  112.00 MiB, V (f16):  112.00 MiB"
        let Some(i) = line.find("llama_kv_cache: size =") else { continue };
        let rest = &line[i + "llama_kv_cache: size =".len()..];
        let num = |s: &str| s.trim().split_whitespace().next().and_then(|x| x.parse::<f64>().ok());
        let Some((total, rest)) = rest.split_once("MiB (") else { continue };
        let Some((paren, rest)) = rest.split_once("), K (") else { continue };
        let Some((tk, rest)) = rest.split_once("):") else { continue };
        let Some((k_mib, rest)) = rest.split_once("MiB, V (") else { continue };
        let Some((tv, rest)) = rest.split_once("):") else { continue };
        let parts: Vec<&str> = paren.split(',').collect();
        if parts.len() < 3 {
            continue;
        }
        let first = |s: &str| s.trim().split_whitespace().next().and_then(|x| x.parse::<u64>().ok());
        let seqs = parts[2].trim().split_whitespace().next().unwrap_or("1/1");
        out.push(KvLine {
            cells: first(parts[0]).unwrap_or(0),
            layers: first(parts[1]).unwrap_or(0) as usize,
            n_seq: seqs.split('/').next().and_then(|x| x.parse().ok()).unwrap_or(1),
            mib_total: num(total).unwrap_or(0.0),
            mib_k: num(k_mib).unwrap_or(0.0),
            mib_v: num(rest).unwrap_or(0.0),
            type_k: tk.trim().to_string(),
            type_v: tv.trim().to_string(),
        });
    }
    out
}

/// `<buft> compute buffer size = X MiB` lines of the context creation, in bytes, summed.
pub fn parse_compute_bytes(log: &str) -> u64 {
    let mut sum = 0f64;
    for line in log.lines() {
        if let Some(i) = line.find("compute buffer size =") {
            if let Some(v) = line[i + "compute buffer size =".len()..].split_whitespace().next().and_then(|x| x.parse::<f64>().ok()) {
                sum += v;
            }
        }
    }
    (sum * 1048576.0) as u64
}

#[derive(Debug, Clone)]
pub struct ProbeParams {
    /// 999 for the real (Metal) buffer types; 0 keeps the probe on the CPU.
    pub n_gpu_layers: i32,
    pub cells: u32,
    pub n_seq: u32,
    pub n_ubatch: u32,
    pub n_batch: u32,
    pub kv_type: KvType,
    /// libllama's default (true) gives the SWA layers a cache as large as the base cache.
    pub swa_full: bool,
    /// One shared pool (true) or `n_seq` streams of `cells / n_seq` cells.
    pub kv_unified: bool,
}

impl Default for ProbeParams {
    fn default() -> Self {
        ProbeParams { n_gpu_layers: 999, cells: 4096, n_seq: 1, n_ubatch: 512, n_batch: 512, kv_type: KvType::F16, swa_full: true, kv_unified: true }
    }
}

#[derive(Debug, Clone)]
pub struct Probe {
    pub kv: Vec<KvLine>,
    pub compute_bytes: u64,
    pub log: String,
}

/// Build a `no_alloc` model and context with libllama and read its own memory accounting from the log: nothing is
/// allocated (no weights, no KV, no compute buffer), so it costs milliseconds and no GPU memory.
pub fn probe_memory(path: &Path, p: &ProbeParams) -> Result<Probe, String> {
    let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| "path contains NUL".to_string())?;
    let (r, log) = capture_log(|| unsafe {
        let mut mp = sys::llama_model_default_params();
        mp.n_gpu_layers = p.n_gpu_layers;
        mp.no_alloc = true;
        mp.use_extra_bufts = false;
        mp.load_mode = sys::LLAMA_LOAD_MODE_NONE; // the memory-fit pass maps nothing (llama.cpp asserts on mmap + no_alloc)
        mp.lazy_mode = sys::LLAMA_LAZY_MODE_OFF;
        let model = sys::llama_model_load_from_file(c.as_ptr(), mp);
        if model.is_null() {
            return Err("llama_model_load_from_file (no_alloc) failed".to_string());
        }
        let mut cp = sys::llama_context_default_params();
        cp.n_ctx = p.cells;
        cp.n_seq_max = p.n_seq;
        cp.n_batch = p.n_batch;
        cp.n_ubatch = p.n_ubatch;
        cp.flash_attn_type = sys::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = p.kv_type.ggml() as _;
        cp.type_v = p.kv_type.ggml() as _;
        cp.no_perf = true;
        cp.swa_full = p.swa_full;
        cp.kv_unified = p.kv_unified;
        let ctx = sys::llama_init_from_model(model, cp);
        let ok = !ctx.is_null();
        if ok {
            sys::llama_free(ctx);
        }
        sys::llama_model_free(model);
        if ok { Ok(()) } else { Err("llama_init_from_model (no_alloc) failed".to_string()) }
    });
    r?;
    Ok(Probe { kv: parse_kv_lines(&log), compute_bytes: parse_compute_bytes(&log), log })
}
