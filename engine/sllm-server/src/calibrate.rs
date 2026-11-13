//! Startup calibration: a few seconds of timing on the real GPU, run once before the engine is built, for ANY GGUF.
//!
//! It times a decode step at batch 1, 8 and 32 (synthetic tokens, one sequence each, a shared 256-token context) and a
//! prefill chunk at 128, 256, 512, 1024 and 2048 tokens, on a throw-away libllama context that is freed before the real
//! engine loads. From the two curves [`choose`] sets the knobs the user did not pass:
//!
//! * the **prefill chunk** (and `n_ubatch`, which the engine keeps equal to it): the smallest size (at least 256, the
//!   scheduler's smallest fair share) whose prefill throughput is within 10% of the best measured, since larger chunks only lengthen the step without going faster;
//! * the **step token budget**: how many prefill tokens fit in [`STEP_TARGET_S`] at that throughput, a whole number of
//!   chunks, between two chunks and [`MAX_STEP_TOKENS`];
//! * the **batch knee** (reported, not applied): the largest measured decode batch whose step time stays within
//!   [`FLAT_TOL`] of the batch-1 step, where the step stops being bandwidth bound.
//!
//! `--calibrate on` runs it; it is OFF by default for now (it showed no gain and made the step budget vary from run to
//! run, which hurts benchmark reproducibility); off keeps the previous defaults. `/caps` `calibration` carries the curves and the chosen knobs.

use crate::profile::{self, Profile};
use llama_ref::sys;
use serde_json::{json, Value};
use std::ffi::CString;
use std::path::Path;
use std::time::Instant;

/// Decode batches timed.
pub const DECODE_BATCHES: [usize; 3] = [1, 8, 32];
/// Prefill chunk sizes timed.
pub const PREFILL_CHUNKS: [usize; 5] = [128, 256, 512, 1024, 2048];
/// Tokens of shared context every decode row attends to.
const DEPTH: usize = 256;
/// Stop the prefill sweep when it has taken this long (a 32B dense model would otherwise take half a minute).
const PREFILL_BUDGET_S: f64 = 6.0;
/// A prefill size within this share of the best throughput counts as saturated.
const SATURATION: f64 = 0.90;
/// The prefill time one scheduler step should aim for.
pub const STEP_TARGET_S: f64 = 0.5;
pub const MAX_STEP_TOKENS: usize = 4096;
/// The smallest prefill chunk calibration will pick (the scheduler's `MIN_FAIR_CHUNK`).
pub const MIN_CHUNK: usize = crate::sched::MIN_FAIR_CHUNK;
/// A decode step this much slower than the batch-1 step is no longer "flat".
pub const FLAT_TOL: f64 = 1.25;

unsafe extern "C" {
    fn llama_memory_seq_rm(mem: sys::llama_memory_t, seq: sys::llama_seq_id, p0: sys::llama_pos, p1: sys::llama_pos) -> bool;
    fn llama_memory_seq_cp(mem: sys::llama_memory_t, src: sys::llama_seq_id, dst: sys::llama_seq_id, p0: sys::llama_pos, p1: sys::llama_pos);
    fn llama_synchronize(ctx: *mut sys::llama_context);
}

/// `--calibrate on|off`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    On,
    Off,
}

impl Mode {
    pub fn parse(s: &str) -> Result<Mode, String> {
        match s {
            "on" => Ok(Mode::On),
            "off" => Ok(Mode::Off),
            o => Err(format!("--calibrate: expected on or off, got {o:?}")),
        }
    }
}

/// Median step time in ms per decode batch, and ms per prefill chunk size.
#[derive(Debug, Clone, Default)]
pub struct Curves {
    pub decode: Vec<(usize, f64)>,
    pub prefill: Vec<(usize, f64)>,
}

/// What the curves say.
#[derive(Debug, Clone, PartialEq)]
pub struct Knobs {
    pub chunk: usize,
    pub step_tokens: usize,
    /// Largest measured decode batch whose step time is within [`FLAT_TOL`] of batch 1.
    pub batch_knee: usize,
}

/// The knobs from the curves (pure: unit-tested on synthetic curves).
pub fn choose(c: &Curves) -> Option<Knobs> {
    if c.prefill.is_empty() || c.decode.is_empty() {
        return None;
    }
    let rate = |&(t, ms): &(usize, f64)| t as f64 / (ms / 1e3);
    let best = c.prefill.iter().map(rate).fold(0.0, f64::max);
    // never under the scheduler's smallest fair prefill share: many small prefill groups ran several times slower (sched.rs)
    let sat = c.prefill.iter().filter(|p| p.0 >= MIN_CHUNK).find(|p| rate(p) >= SATURATION * best).or_else(|| c.prefill.last())?;
    let chunk = sat.0;
    let step_tokens = ((rate(sat) * STEP_TARGET_S) as usize / chunk * chunk).clamp(2 * chunk, MAX_STEP_TOKENS.max(2 * chunk));
    let t1 = c.decode.iter().find(|d| d.0 == 1).map_or(c.decode[0].1, |d| d.1);
    let batch_knee = c.decode.iter().filter(|d| d.1 <= FLAT_TOL * t1).map(|d| d.0).max().unwrap_or(1);
    Some(Knobs { chunk, step_tokens, batch_knee })
}

#[derive(Debug, Clone)]
pub struct Calibration {
    pub curves: Curves,
    pub knobs: Option<Knobs>,
    pub measured_s: f64,
    /// `bytes of a batch-1 step / its measured time`: what the machine actually delivered (GB/s).
    pub effective_gbs: Option<f64>,
    pub roofline_ms_batch1: Option<f64>,
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Time `reps` runs of `f` (after `warm` untimed ones), ms, median.
fn time_ms(warm: usize, reps: usize, mut f: impl FnMut() -> Result<(), String>) -> Result<f64, String> {
    for _ in 0..warm {
        f()?;
    }
    let mut v = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        f()?;
        v.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    Ok(median(&mut v))
}

struct Ctx {
    model: *mut sys::llama_model,
    ctx: *mut sys::llama_context,
    batch: sys::llama_batch,
    n_vocab: usize,
}

impl Drop for Ctx {
    fn drop(&mut self) {
        unsafe {
            sys::llama_batch_free(self.batch);
            sys::llama_free(self.ctx);
            sys::llama_model_free(self.model);
        }
    }
}

impl Ctx {
    /// A synthetic token: spread over the vocabulary (so an MoE routes rows to different experts), never a low special id.
    fn tok(&self, i: usize) -> i32 {
        let span = self.n_vocab.saturating_sub(2000).max(1);
        (1000 + (i.wrapping_mul(2654435761)) % span) as i32
    }

    /// Decode `rows` = (sequence, position, token, logits).
    fn decode(&mut self, rows: &[(i32, i32, i32, bool)]) -> Result<(), String> {
        unsafe {
            let b = &mut self.batch;
            for (k, &(sid, pos, tok, lg)) in rows.iter().enumerate() {
                *b.token.add(k) = tok;
                *b.pos.add(k) = pos;
                *b.n_seq_id.add(k) = 1;
                **b.seq_id.add(k) = sid;
                *b.logits.add(k) = lg as i8;
            }
            b.n_tokens = rows.len() as i32;
            let rc = sys::llama_decode(self.ctx, *b);
            llama_synchronize(self.ctx);
            if rc != 0 {
                return Err(format!("llama_decode returned {rc}"));
            }
        }
        Ok(())
    }

    fn mem(&self) -> sys::llama_memory_t {
        unsafe { sys::llama_get_memory(self.ctx) }
    }
}

/// Run the calibration on `model`. Takes the libllama log guard (nothing else logs meanwhile).
pub fn run(model: &Path, prof: &Profile, bandwidth_gbs: Option<f64>) -> Result<Calibration, String> {
    let t_start = Instant::now();
    let mut cx = Ctx::open(model)?;
    let mut curves = Curves::default();
    cx.time_decode(&mut curves)?;
    cx.time_prefill(prof, &mut curves)?;
    drop(cx);

    let knobs = choose(&curves);
    let b1_ms = curves.decode.iter().find(|d| d.0 == 1).map(|d| d.1);
    let bytes1 = prof.decode_step_bytes(profile::KvType::F16, 1, DEPTH as u64) as f64;
    Ok(Calibration {
        effective_gbs: b1_ms.map(|ms| bytes1 / (ms / 1e3) / 1e9),
        roofline_ms_batch1: bandwidth_gbs.map(|bw| bytes1 / (bw * 1e9) * 1e3),
        curves,
        knobs,
        measured_s: t_start.elapsed().as_secs_f64(),
    })
}

impl Ctx {
    /// The throw-away model and context, sized for the largest decode batch and prefill chunk timed.
    fn open(model: &Path) -> Result<Ctx, String> {
        let max_chunk = *PREFILL_CHUNKS.last().unwrap();
        let max_batch = *DECODE_BATCHES.last().unwrap();
        let c = CString::new(model.to_string_lossy().as_bytes()).map_err(|_| "model path contains NUL".to_string())?;
        let (made, _log) = profile::capture_log(|| unsafe { Self::load(&c, max_batch, max_chunk) });
        made
    }

    /// The libllama calls of [`Ctx::open`] (inside the log capture).
    unsafe fn load(c: &CString, max_batch: usize, max_chunk: usize) -> Result<Ctx, String> {
        let mut mp = sys::llama_model_default_params();
        mp.n_gpu_layers = 999;
        let m = sys::llama_model_load_from_file(c.as_ptr(), mp);
        if m.is_null() {
            return Err("calibration: llama_model_load_from_file failed".to_string());
        }
        let n_vocab = sys::llama_vocab_n_tokens(sys::llama_model_get_vocab(m)) as usize;
        let mut cp = sys::llama_context_default_params();
        cp.n_ctx = 4096;
        cp.n_seq_max = max_batch as u32;
        cp.n_batch = max_chunk as u32;
        cp.n_ubatch = max_chunk as u32;
        cp.flash_attn_type = sys::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.kv_unified = true;
        cp.no_perf = true;
        let ctx = sys::llama_init_from_model(m, cp);
        if ctx.is_null() {
            sys::llama_model_free(m);
            return Err("calibration: llama_init_from_model failed".to_string());
        }
        Ok(Ctx { model: m, ctx, batch: sys::llama_batch_init(max_chunk as i32, 0, 1), n_vocab })
    }

    /// The decode curve: one row per sequence, all attending to the same DEPTH tokens (seq 0 prefilled, the others share
    /// its cells).
    fn time_decode(&mut self, curves: &mut Curves) -> Result<(), String> {
        let max_batch = *DECODE_BATCHES.last().unwrap();
        let fill: Vec<(i32, i32, i32, bool)> = (0..DEPTH).map(|i| (0, i as i32, self.tok(i + 7), i + 1 == DEPTH)).collect();
        self.decode(&fill)?;
        for s in 1..max_batch as i32 {
            unsafe { llama_memory_seq_cp(self.mem(), 0, s, 0, DEPTH as i32) };
        }
        for &b in &DECODE_BATCHES {
            let rows: Vec<(i32, i32, i32, bool)> = (0..b).map(|i| (i as i32, DEPTH as i32, self.tok(1000 + i), true)).collect();
            let ms = time_ms(2, 5, || {
                self.decode(&rows)?;
                for s in 0..b as i32 {
                    unsafe { llama_memory_seq_rm(self.mem(), s, DEPTH as i32, -1) };
                }
                Ok(())
            })?;
            curves.decode.push((b, ms));
        }
        Ok(())
    }

    /// The prefill curve: one sequence, T tokens from position 0, logits for the last (until [`PREFILL_BUDGET_S`]).
    fn time_prefill(&mut self, prof: &Profile, curves: &mut Curves) -> Result<(), String> {
        let sweep_start = Instant::now();
        for &t in PREFILL_CHUNKS.iter().filter(|&&t| (t as u64) <= prof.n_ctx_train.max(128)) {
            if sweep_start.elapsed().as_secs_f64() > PREFILL_BUDGET_S && !curves.prefill.is_empty() {
                break;
            }
            curves.prefill.push((t, self.time_prefill_of(t)?));
        }
        Ok(())
    }

    /// One prefill of `t` tokens, timed: a warm-up, then 1 to 3 runs by how long the warm-up took.
    fn time_prefill_of(&mut self, t: usize) -> Result<f64, String> {
        let rows: Vec<(i32, i32, i32, bool)> = (0..t).map(|i| (0, i as i32, self.tok(i + 31), i + 1 == t)).collect();
        let mut once = || -> Result<(), String> {
            unsafe { llama_memory_seq_rm(self.mem(), -1, -1, -1) };
            self.decode(&rows)
        };
        let first = Instant::now();
        once()?; // warm-up (pipelines, buffers)
        let est = first.elapsed().as_secs_f64();
        let reps = if est < 0.15 { 3 } else if est < 0.6 { 2 } else { 1 };
        time_ms(0, reps, &mut once)
    }
}

impl Calibration {
    /// `/caps` `calibration`. `applied` says which knobs replaced a default (the others were passed on the command line).
    pub fn to_json(&self, applied: &Applied) -> Value {
        let decode: Vec<Value> = self
            .curves
            .decode
            .iter()
            .map(|&(b, ms)| json!({"batch": b, "step_ms": ms, "tokens_per_s": b as f64 / (ms / 1e3), "ms_per_token": ms / b as f64}))
            .collect();
        let prefill: Vec<Value> = self.curves.prefill.iter().map(|&(t, ms)| json!({"tokens": t, "ms": ms, "tokens_per_s": t as f64 / (ms / 1e3)})).collect();
        json!({
            "mode": "on", "measured_s": self.measured_s, "depth": DEPTH,
            "decode": decode, "prefill": prefill,
            "batch_knee": self.knobs.as_ref().map(|k| k.batch_knee),
            "flat_tolerance": FLAT_TOL, "saturation": SATURATION, "step_target_s": STEP_TARGET_S,
            "effective_bandwidth_gbs": self.effective_gbs, "roofline_ms_batch1": self.roofline_ms_batch1,
            "chosen": self.knobs.as_ref().map(|k| json!({"chunk": k.chunk, "n_ubatch": k.chunk, "step_tokens": k.step_tokens})),
            "applied": {"chunk": applied.chunk, "n_ubatch": applied.n_ubatch, "step_tokens": applied.step_tokens},
            "note": "knobs the user passed on the command line are not overridden (applied=false)",
        })
    }

    /// The one startup line.
    pub fn summary_line(&self, applied: &Applied) -> String {
        let d: Vec<String> = self.curves.decode.iter().map(|&(b, ms)| format!("b{b} {ms:.1}ms")).collect();
        let p: Vec<String> = self.curves.prefill.iter().map(|&(t, ms)| format!("{t} {:.0}t/s", t as f64 / (ms / 1e3))).collect();
        let k = match &self.knobs {
            Some(k) => format!(
                "chunk {}{} step {}{} (batch knee {})",
                k.chunk,
                if applied.chunk { "" } else { " (kept: set by flag)" },
                k.step_tokens,
                if applied.step_tokens { "" } else { " (kept: set by flag)" },
                k.batch_knee
            ),
            None => "no knobs".into(),
        };
        format!("calibration {:.1}s: decode [{}] prefill [{}] -> {k}", self.measured_s, d.join(", "), p.join(", "))
    }
}

/// Which knobs the calibration replaced.
#[derive(Debug, Clone, Copy, Default)]
pub struct Applied {
    pub chunk: bool,
    pub n_ubatch: bool,
    pub step_tokens: bool,
}

/// Which knobs the user passed explicitly (calibration never overrides those).
#[derive(Debug, Clone, Copy, Default)]
pub struct Explicit {
    pub chunk: bool,
    pub ubatch: bool,
    pub step_tokens: bool,
}

/// Values to use for the knobs that were not passed.
#[derive(Debug, Clone, Copy, Default)]
pub struct Tuned {
    pub chunk: Option<usize>,
    pub n_ubatch: Option<usize>,
    pub step_tokens: Option<usize>,
}

/// HOOK(profile) entry point of `main`: read the profile, print its line, calibrate (unless off), publish both for the
/// engine's `/caps`, and return the knobs to set. A calibration failure is a warning, never fatal.
pub fn startup(model: &Path, mode: Mode, explicit: Explicit) -> Result<Tuned, String> {
    let prof = profile::load_profile(model).map_err(|e| format!("model profile: {e}"))?;
    eprintln!("sllm-server: {}", prof.summary_line());
    let machine = profile::Machine::detect();
    let mut tuned = Tuned::default();
    let mut json = json!({"mode": "off"});
    if mode == Mode::On {
        (tuned, json) = calibrate_now(model, &prof, &machine, explicit);
    }
    profile::set_startup(profile::Startup { profile: prof, machine, calibration: json });
    Ok(tuned)
}

/// `--calibrate on` at startup: the knobs to set and the `/caps` `calibration` object (a failure: the defaults stay).
fn calibrate_now(model: &Path, prof: &Profile, machine: &profile::Machine, explicit: Explicit) -> (Tuned, Value) {
    // the ggml device registry must exist before libllama creates it (sllm-core captures the Metal device log there)
    let _ = sllm_core::backend::registry();
    match run(model, prof, machine.bandwidth_gbs) {
        Ok(c) => {
            let applied = Applied { chunk: !explicit.chunk, n_ubatch: !explicit.ubatch && !explicit.chunk, step_tokens: !explicit.step_tokens };
            let tuned = c.knobs.as_ref().map_or_else(Tuned::default, |k| k.tuned(&applied));
            eprintln!("sllm-server: {}", c.summary_line(&applied));
            (tuned, c.to_json(&applied))
        }
        Err(e) => {
            eprintln!("sllm-server: warning: calibration failed ({e}); keeping the default knobs");
            (Tuned::default(), json!({"mode": "on", "error": e}))
        }
    }
}

impl Knobs {
    /// The knobs the calibration may replace (`applied`), as values to set.
    fn tuned(&self, applied: &Applied) -> Tuned {
        Tuned {
            chunk: applied.chunk.then_some(self.chunk),
            n_ubatch: applied.n_ubatch.then_some(self.chunk),
            step_tokens: applied.step_tokens.then_some(self.step_tokens),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curves(pre: &[(usize, f64)], dec: &[(usize, f64)]) -> Curves {
        Curves { prefill: pre.to_vec(), decode: dec.to_vec() }
    }

    #[test]
    fn saturated_prefill_picks_the_smallest_good_chunk() {
        // 4,000 tok/s at 128, 8,000 at 256, 10,000 at 512 and beyond
        let c = curves(&[(128, 32.0), (256, 32.0), (512, 51.2), (1024, 102.4), (2048, 204.8)], &[(1, 3.0), (8, 3.3), (32, 6.0)]);
        let k = choose(&c).unwrap();
        assert_eq!(k.chunk, 512); // 8,000 < 0.9 x 10,000
        assert_eq!(k.step_tokens, 4096.min(5000 / 512 * 512)); // 0.5 s at 10,000 tok/s = 5,000 tokens -> 4,096 cap
        assert_eq!(k.batch_knee, 8);
    }

    #[test]
    fn slow_model_gets_small_steps_and_flat_decode_a_wide_knee() {
        let c = curves(&[(128, 400.0), (256, 800.0), (512, 1600.0)], &[(1, 10.0), (8, 10.5), (32, 11.0)]);
        let k = choose(&c).unwrap();
        assert_eq!(k.chunk, 256); // 320 tok/s at every size: the smallest is saturated, but not under the fair share (256)
        assert_eq!(k.step_tokens, 512); // 160 tokens in 0.5 s, but never under two chunks
        assert_eq!(k.batch_knee, 32);
    }

    #[test]
    fn no_curves_no_knobs() {
        assert!(choose(&Curves::default()).is_none());
    }

    #[test]
    fn mode_parses() {
        assert_eq!(Mode::parse("on"), Ok(Mode::On));
        assert_eq!(Mode::parse("off"), Ok(Mode::Off));
        assert!(Mode::parse("file.json").is_err());
    }
}
