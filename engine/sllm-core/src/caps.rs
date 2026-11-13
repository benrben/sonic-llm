//! The capability record: what the Metal device offers and how much memory the engine may plan with.
//!
//! Two sources, as D028 and the plan specify:
//!   * ggml's own device log (captured through the shim at registry init): `has tensor`, `has bfloat`,
//!     `use residency sets`, unified memory, GPU name and families, the recommended working set in MB;
//!   * `ggml_backend_dev_memory` (exact bytes) for the working set, free and total.
//!
//! The memory budget is the recommended working set times 0.9 (rule 129: a 10% headroom).
//!
//! [`GgmlMetalInfo`] (D042) says which `libggml-metal` this process loaded (the official b11669 binary or SLLM's
//! patched build, `engine/vendor/sllm-metal-b11669`) and, for the patched one, whether its qmm path is active.

use serde::{Deserialize, Serialize};

use crate::backend;
use crate::error::{Error, Result};
use crate::{ggml_version, mem};

/// Fraction of the recommended working set the engine may plan with (rule 129).
pub const BUDGET_FRACTION_NUM: u64 = 9;
pub const BUDGET_FRACTION_DEN: u64 = 10;

/// Facts parsed out of ggml's Metal device log. `None` means the line was not present.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetalLogFacts {
    pub gpu_name: Option<String>,
    pub gpu_families: Vec<String>,
    pub simdgroup_reduction: Option<bool>,
    pub simdgroup_mm: Option<bool>,
    pub has_unified_memory: Option<bool>,
    pub has_bfloat: Option<bool>,
    pub has_tensor: Option<bool>,
    pub use_residency_sets: Option<bool>,
    pub use_shared_buffers: Option<bool>,
    /// Megabytes of 1e6 bytes, as ggml prints it (two decimals).
    pub recommended_max_working_set_mb: Option<f64>,
}

fn parse_bool(v: &str) -> Option<bool> {
    match v.trim() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

/// Parse the lines `ggml_metal_device_init` prints. Unknown lines are ignored; the format is
/// `<func>: <key> = <value>` (and `GPU name:` / `GPU family:` without `=`).
pub fn parse_metal_log(log: &str) -> MetalLogFacts {
    let mut f = MetalLogFacts::default();
    for line in log.lines() {
        let Some(rest) = line.trim().strip_prefix("ggml_metal_device_init:") else { continue };
        let rest = rest.trim();
        if let Some(v) = rest.strip_prefix("GPU name:") {
            f.gpu_name = Some(v.trim().to_string());
        } else if let Some(v) = rest.strip_prefix("GPU family:") {
            f.gpu_families.push(v.split_whitespace().next().unwrap_or("").to_string());
        } else if let Some((key, val)) = rest.split_once('=') {
            f.set(key.trim(), val);
        }
    }
    f
}

impl MetalLogFacts {
    /// One `<key> = <value>` line of the device init (unknown keys are ignored).
    fn set(&mut self, key: &str, val: &str) {
        match key {
            "simdgroup reduction" => self.simdgroup_reduction = parse_bool(val),
            "simdgroup matrix mul." => self.simdgroup_mm = parse_bool(val),
            "has unified memory" => self.has_unified_memory = parse_bool(val),
            "has bfloat" => self.has_bfloat = parse_bool(val),
            "has tensor" => self.has_tensor = parse_bool(val),
            "use residency sets" => self.use_residency_sets = parse_bool(val),
            "use shared buffers" => self.use_shared_buffers = parse_bool(val),
            "recommendedMaxWorkingSetSize" => {
                self.recommended_max_working_set_mb = val.trim().strip_suffix("MB").and_then(|n| n.trim().parse::<f64>().ok());
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceCaps {
    pub index: usize,
    pub name: String,
    pub description: String,
    /// `enum ggml_backend_dev_type`: 0 CPU, 1 GPU, 2 IGPU, 3 ACCEL, 4 META.
    pub dev_type: i32,
    pub memory_free_bytes: u64,
    pub memory_total_bytes: u64,
    pub buffer_from_host_ptr: bool,
    pub mmap_support: bool,
    pub async_compute: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    pub ggml_version: String,
    pub ggml_commit: String,
    pub gpu_name: String,
    pub gpu_families: Vec<String>,
    pub metal_device: String,
    pub has_tensor: bool,
    pub has_bfloat: bool,
    pub use_residency_sets: bool,
    pub has_unified_memory: bool,
    pub simdgroup_reduction: bool,
    pub simdgroup_mm: bool,
    pub use_shared_buffers: bool,
    pub buffer_from_host_ptr: bool,
    /// Recommended working set in MB (1e6 bytes) as ggml prints it in the device log.
    pub recommended_working_set_mb_log: f64,
    /// Recommended working set in bytes, from `ggml_backend_dev_memory` (total).
    pub recommended_working_set_bytes: u64,
    /// Bytes of the recommended working set not yet allocated by this process's Metal device.
    pub working_set_free_bytes: u64,
    /// The planning budget: recommended working set x 0.9.
    pub memory_budget_bytes: u64,
    pub memory_budget_fraction: f64,
    pub physical_memory_bytes: u64,
    /// True when the log's MB figure and the byte figure agree to the printed precision.
    pub log_matches_dev_memory: bool,
    /// Missing log lines (empty means the log was complete).
    pub missing_log_facts: Vec<String>,
    pub devices: Vec<DeviceCaps>,
    /// Which libggml-metal is loaded and whether SLLM's qmm path is active (D042).
    #[serde(default)]
    pub ggml_metal: GgmlMetalInfo,
}

/// The loaded `libggml-metal` and the state of SLLM's patches (D042): qmm and the shared-prefix cascade.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct GgmlMetalInfo {
    /// File the Metal backend's code was loaded from (`dladdr`), canonicalised.
    pub path: String,
    /// `"sllm-qmm"` when the patched build is loaded (it exports `ggml_backend_metal_sllm_qmm_status`),
    /// `"official"` otherwise.
    pub build: String,
    /// True when the patched build is loaded and its qmm path passed the self-test and is not switched off.
    pub qmm_active: bool,
    /// The patch's own status object (mode, self-test, dispatch table, dispatch count); null for the official build.
    pub qmm: serde_json::Value,
    /// True when the patched build carries SLLM's shared-prefix decode attention (cascade), it passed its self-test and
    /// it is not switched off (`SLLM_CASCADE=0`).
    #[serde(default)]
    pub cascade_active: bool,
    /// The cascade patch's status object (mode, self-test, rule, dispatch count: attention ops it took); null when
    /// the loaded build has no cascade.
    #[serde(default)]
    pub cascade: serde_json::Value,
    /// True when the patched build carries the mixed-precision cascade (patch 0006: f16 shared prefix + q8_0 own pages),
    /// it passed its self-test and it is not switched off (`SLLM_CASCADE=0`). sllm-core builds mixed-precision
    /// attention only when this holds (`KvLayout::OwnQ8`).
    #[serde(default)]
    pub cascade_mixed_active: bool,
    /// The mixed-precision cascade's status object (self-test, dispatch count); null when the build has none.
    #[serde(default)]
    pub cascade_mixed: serde_json::Value,
    /// True when the patched build carries SLLM's GQA-packed prefill attention (patch 0007), it passed its self-test
    /// and it is not switched off (`SLLM_PREFILL_CASCADE=0`).
    #[serde(default)]
    pub prefill_active: bool,
    /// The prefill attention's status object (mode, self-test, rule, dispatch count); null when the build has none.
    #[serde(default)]
    pub prefill: serde_json::Value,
    /// True when the patched build carries SLLM's argmax over long rows (patch 0009: greedy sampling of many rows), it passed
    /// its self-test and it is not switched off (`SLLM_ARGMAX=0`).
    #[serde(default)]
    pub argmax_active: bool,
    /// The argmax patch's status object (self-test, dispatch count); null when the build has none.
    #[serde(default)]
    pub argmax: serde_json::Value,
    /// The fused small ops' status object (patch 0013: q/k norm + rope [+ scale | + cache write], add + norm; on
    /// unless `SLLM_FUSE=0`; dispatch count); null when the build has none.
    #[serde(default)]
    pub fuse: serde_json::Value,
    /// `MANIFEST.json` of the patched build (source commit, patch sha256s, output sha256), when found beside `lib/`.
    pub manifest: serde_json::Value,
}

/// The status object a patch of the loaded libggml-metal reports through its `const char * (void)` function `name` (a
/// string value when the text is not JSON); None when the loaded build has no such function.
fn backend_status(name: &std::ffi::CStr) -> Option<serde_json::Value> {
    // SAFETY: dlsym on the global namespace; the symbol, when present, is `const char * (void)` returning a
    // NUL-terminated JSON string in a thread-local buffer.
    unsafe {
        let sym = libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr());
        if sym.is_null() {
            return None;
        }
        let f: extern "C" fn() -> *const std::ffi::c_char = std::mem::transmute(sym);
        let s = std::ffi::CStr::from_ptr(f()).to_string_lossy().to_string();
        Some(serde_json::from_str(&s).unwrap_or(serde_json::Value::String(s)))
    }
}

/// A status object's `active` flag (false when it has none).
fn is_active(v: &serde_json::Value) -> bool {
    v.get("active").and_then(|v| v.as_bool()).unwrap_or(false)
}

/// Probe the loaded `libggml-metal`. Call after the registry is up (the patch's status is set at device init);
/// the `dispatches` count in `qmm` grows as graphs run, so a run record should take this at its end.
pub fn ggml_metal_info() -> GgmlMetalInfo {
    backend::registry();
    let mut info = GgmlMetalInfo { build: "official".into(), ..Default::default() };
    // SAFETY: dladdr only reads the loader's tables; the address is a function of the linked libggml-metal.
    unsafe {
        let mut dl: libc::Dl_info = std::mem::zeroed();
        if libc::dladdr(crate::sys::ggml_backend_is_metal as *const libc::c_void, &mut dl) != 0 && !dl.dli_fname.is_null() {
            let p = std::ffi::CStr::from_ptr(dl.dli_fname).to_string_lossy().to_string();
            info.path = std::fs::canonicalize(&p).map(|c| c.display().to_string()).unwrap_or(p);
        }
    }
    info.read_patch_status();
    if info.build == "sllm-qmm" {
        info.read_manifest();
    }
    info
}

impl GgmlMetalInfo {
    /// The status objects of the patches the loaded build carries (the qmm patch's makes it the `sllm-qmm` build).
    fn read_patch_status(&mut self) {
        if let Some(v) = backend_status(c"ggml_backend_metal_sllm_qmm_status") {
            self.build = "sllm-qmm".into();
            self.qmm_active = is_active(&v);
            self.qmm = v;
        }
        if let Some(v) = backend_status(c"ggml_backend_metal_sllm_cascade_status") {
            self.cascade_active = is_active(&v);
            self.cascade = v;
        }
        if let Some(v) = backend_status(c"ggml_backend_metal_sllm_cascadeq_status") {
            self.cascade_mixed_active = is_active(&v);
            self.cascade_mixed = v;
        }
        if let Some(v) = backend_status(c"ggml_backend_metal_sllm_prefill_status") {
            self.prefill_active = is_active(&v);
            self.prefill = v;
        }
        if let Some(v) = backend_status(c"ggml_backend_metal_sllm_argmax_status") {
            self.argmax_active = is_active(&v);
            self.argmax = v;
        }
        if let Some(v) = backend_status(c"ggml_backend_metal_sllm_fuse_status") {
            self.fuse = v;
        }
    }

    /// `MANIFEST.json` beside the patched build's `lib/`, when there is one.
    fn read_manifest(&mut self) {
        if let Some(dir) = std::path::Path::new(&self.path).parent().and_then(|d| d.parent()) {
            if let Ok(t) = std::fs::read_to_string(dir.join("MANIFEST.json")) {
                self.manifest = serde_json::from_str(&t).unwrap_or(serde_json::Value::Null);
            }
        }
    }
}

/// The live status objects of the patched libggml-metal (`{"qmm": .., "cascade": ..}`, null where the loaded build
/// lacks one), without the manifest or the path: cheap enough for every `/metrics` scrape. Their `dispatches` counts
/// grow as graphs run (the copy in the capability record is taken at startup).
pub fn ggml_metal_status() -> serde_json::Value {
    let read = |name: &std::ffi::CStr| backend_status(name).unwrap_or(serde_json::Value::Null);
    serde_json::json!({
        "qmm": read(c"ggml_backend_metal_sllm_qmm_status"),
        "cascade": read(c"ggml_backend_metal_sllm_cascade_status"),
        "cascade_mixed": read(c"ggml_backend_metal_sllm_cascadeq_status"),
        "prefill": read(c"ggml_backend_metal_sllm_prefill_status"),
        "argmax": read(c"ggml_backend_metal_sllm_argmax_status"),
        "fuse": read(c"ggml_backend_metal_sllm_fuse_status"),
    })
}

/// Budget = recommended working set x 9/10, in integer arithmetic.
pub fn memory_budget(recommended_bytes: u64) -> u64 {
    (recommended_bytes as u128 * BUDGET_FRACTION_NUM as u128 / BUDGET_FRACTION_DEN as u128) as u64
}

impl Capabilities {
    /// Build the record from the captured registry log and a fresh device query.
    pub fn probe() -> Result<Capabilities> {
        let reg = backend::registry();
        let facts = parse_metal_log(&reg.init_log);
        let mut devices = Vec::new();
        for i in 0..reg.n_devices {
            if let Some(d) = crate::dev_query(i) {
                devices.push(DeviceCaps {
                    index: i,
                    name: crate::cstr_array(&d.name),
                    description: crate::cstr_array(&d.description),
                    dev_type: d.type_,
                    memory_free_bytes: d.memory_free,
                    memory_total_bytes: d.memory_total,
                    buffer_from_host_ptr: d.buffer_from_host_ptr,
                    mmap_support: d.mmap_support,
                    async_compute: d.async_,
                });
            }
        }
        let metal = devices
            .iter()
            .find(|d| (d.dev_type == 1 || d.dev_type == 2) && d.name.starts_with("MTL"))
            .ok_or_else(|| Error::Backend("no Metal device found; cannot fill capabilities".into()))?
            .clone();
        let mut c = Self::from_parts(&facts, &metal, devices);
        c.ggml_metal = ggml_metal_info();
        Ok(c)
    }

    /// Combine parsed log facts with a device query (separate from `probe` so tests can feed fixtures).
    pub fn from_parts(facts: &MetalLogFacts, metal: &DeviceCaps, devices: Vec<DeviceCaps>) -> Capabilities {
        let mut missing = Vec::new();
        let mut flag = |name: &str, v: Option<bool>| -> bool {
            if v.is_none() {
                missing.push(name.to_string());
            }
            v.unwrap_or(false)
        };
        let has_tensor = flag("has tensor", facts.has_tensor);
        let has_bfloat = flag("has bfloat", facts.has_bfloat);
        let use_residency_sets = flag("use residency sets", facts.use_residency_sets);
        let has_unified_memory = flag("has unified memory", facts.has_unified_memory);
        let simdgroup_reduction = flag("simdgroup reduction", facts.simdgroup_reduction);
        let simdgroup_mm = flag("simdgroup matrix mul.", facts.simdgroup_mm);
        let use_shared_buffers = flag("use shared buffers", facts.use_shared_buffers);
        if facts.recommended_max_working_set_mb.is_none() {
            missing.push("recommendedMaxWorkingSetSize".into());
        }
        let mb = facts.recommended_max_working_set_mb.unwrap_or(0.0);
        let rec = metal.memory_total_bytes;
        Capabilities {
            ggml_version: ggml_version().0,
            ggml_commit: ggml_version().1,
            gpu_name: facts.gpu_name.clone().unwrap_or_default(),
            gpu_families: facts.gpu_families.clone(),
            metal_device: metal.name.clone(),
            has_tensor,
            has_bfloat,
            use_residency_sets,
            has_unified_memory,
            simdgroup_reduction,
            simdgroup_mm,
            use_shared_buffers,
            buffer_from_host_ptr: metal.buffer_from_host_ptr,
            recommended_working_set_mb_log: mb,
            recommended_working_set_bytes: rec,
            working_set_free_bytes: metal.memory_free_bytes,
            memory_budget_bytes: memory_budget(rec),
            memory_budget_fraction: BUDGET_FRACTION_NUM as f64 / BUDGET_FRACTION_DEN as f64,
            physical_memory_bytes: mem::physical_memory_bytes(),
            // ggml prints MB with two decimals, so the bytes may differ by up to 0.005 MB.
            log_matches_dev_memory: (mb * 1e6 - rec as f64).abs() <= 0.005e6 + 1.0,
            missing_log_facts: missing,
            devices,
            ggml_metal: GgmlMetalInfo::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Captured from this machine (Apple M5 Pro, b11669) by `sllm-core caps --with-log`.
    const LOG: &str = include_str!("../tests/fixtures/metal_init.log");

    #[test]
    fn parses_captured_device_log() {
        let f = parse_metal_log(LOG);
        assert_eq!(f.gpu_name.as_deref(), Some("MTL0 (Apple M5 Pro)"));
        assert_eq!(f.gpu_families, ["MTLGPUFamilyApple10", "MTLGPUFamilyMetal4"]);
        assert_eq!(f.simdgroup_reduction, Some(true));
        assert_eq!(f.simdgroup_mm, Some(true));
        assert_eq!(f.has_unified_memory, Some(true));
        assert_eq!(f.has_bfloat, Some(true));
        assert_eq!(f.has_tensor, Some(true));
        assert_eq!(f.use_residency_sets, Some(true));
        assert_eq!(f.use_shared_buffers, Some(true));
        assert_eq!(f.recommended_max_working_set_mb, Some(40200.90));
    }

    #[test]
    fn false_values_and_unknown_lines() {
        let f = parse_metal_log(
            "ggml_metal_device_init: has tensor            = false\n\
             ggml_metal_device_init: has bfloat            = maybe\n\
             ggml_metal_device_init: something new         = 12\n\
             some_other_func: has unified memory = true\n\
             garbage",
        );
        assert_eq!(f.has_tensor, Some(false));
        assert_eq!(f.has_bfloat, None); // unparsable value is "absent", never a guess
        assert_eq!(f.has_unified_memory, None); // other function prefixes are ignored
        assert_eq!(f.recommended_max_working_set_mb, None);
        assert_eq!(parse_metal_log(""), MetalLogFacts::default());
    }

    #[test]
    fn budget_is_ninety_percent_of_the_working_set() {
        // Phase 0 device facts: 40,200,896,512 recommended bytes.
        assert_eq!(memory_budget(40_200_896_512), 36_180_806_860);
        assert_eq!(memory_budget(0), 0);
        assert_eq!(memory_budget(u64::MAX), (u64::MAX as u128 * 9 / 10) as u64);
    }

    fn metal_dev(total: u64) -> DeviceCaps {
        DeviceCaps {
            index: 0,
            name: "MTL0".into(),
            description: "Apple M5 Pro".into(),
            dev_type: 1,
            memory_free_bytes: total - 1000,
            memory_total_bytes: total,
            buffer_from_host_ptr: true,
            mmap_support: true,
            async_compute: true,
        }
    }

    #[test]
    fn capabilities_from_fixture() {
        let dev = metal_dev(40_200_896_512);
        let c = Capabilities::from_parts(&parse_metal_log(LOG), &dev, vec![dev.clone()]);
        assert!(c.has_tensor && c.has_bfloat && c.use_residency_sets && c.has_unified_memory);
        assert_eq!(c.recommended_working_set_bytes, 40_200_896_512);
        assert_eq!(c.memory_budget_bytes, 36_180_806_860);
        assert!(c.log_matches_dev_memory);
        assert!(c.missing_log_facts.is_empty());
        // The record round-trips through JSON.
        let back: Capabilities = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn incomplete_log_is_reported_not_guessed() {
        let dev = metal_dev(40_200_896_512);
        let c = Capabilities::from_parts(&parse_metal_log("ggml_metal_device_init: has bfloat = true"), &dev, vec![]);
        assert!(c.has_bfloat && !c.has_tensor && !c.use_residency_sets);
        assert!(c.missing_log_facts.contains(&"has tensor".to_string()));
        assert!(c.missing_log_facts.contains(&"recommendedMaxWorkingSetSize".to_string()));
        assert!(!c.log_matches_dev_memory);
    }
}
