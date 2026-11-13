//! Batch-invariance probe on libllama: does a sequence's decode depend on which other sequences share its step?
//!
//! batchinv_probe MODEL.gguf [OTHERS=1,3,7,15] [GEN=24]
//! Per-sequence KV streams (as the server runs). Sequence 0 decodes GEN greedy tokens alone (the reference), then again
//! next to K other decoding sequences for each K. Its logits are compared bitwise step by step, and on the first decode
//! step every graph node's first row (sequence 0's row) is hashed through the eval callback, so the first op whose output
//! changes with the batch is named. Run under engine/scripts/with_gpu_lock.py.

use llama_ref::sys;
use std::ffi::{c_void, CStr, CString};

unsafe extern "C" {
    fn llama_memory_seq_rm(mem: sys::llama_memory_t, seq: i32, p0: i32, p1: i32) -> bool;
    fn llama_state_seq_get_size(ctx: *mut sys::llama_context, seq: i32) -> usize;
    fn llama_state_seq_get_data(ctx: *mut sys::llama_context, dst: *mut u8, size: usize, seq: i32) -> usize;
    fn llama_state_seq_set_data(ctx: *mut sys::llama_context, src: *const u8, size: usize, seq: i32) -> usize;
}

#[link(name = "ggml-base")]
unsafe extern "C" {
    fn ggml_backend_tensor_get(tensor: *const sys::ggml_tensor, data: *mut c_void, offset: usize, size: usize);
    fn ggml_nbytes(tensor: *const sys::ggml_tensor) -> usize;
}

const GGML_TYPE_F32: u32 = 0;
const PLEN: usize = 256;

fn fnv(b: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn argmax(row: &[f32]) -> i32 {
    let (mut b, mut bv) = (0usize, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            b = i;
        }
    }
    b as i32
}

/// Eval-callback state: when `on`, every F32 node's first row is hashed into `nodes` as (name, hash).
struct Trace {
    on: bool,
    nodes: Vec<(String, u64)>,
    info: Vec<String>,
}

unsafe extern "C" fn cb(t: *mut sys::ggml_tensor, ask: bool, ud: *mut c_void) -> bool {
    let tr = unsafe { &mut *(ud as *mut Trace) };
    if !tr.on {
        return !ask; // ask: no data wanted; otherwise keep going
    }
    if ask {
        return true;
    }
    let t = unsafe { &*t };
    if t.type_ as u32 != GGML_TYPE_F32 || t.nb[0] != 4 {
        return true;
    }
    // the tensor's bytes, as ggml_nbytes counts them; read the first row (sequence 0's), never past the end
    let nbytes = unsafe { ggml_nbytes(t) };
    if nbytes == 0 || t.data.is_null() && t.view_src.is_null() && t.buffer.is_null() {
        return true;
    }
    let row = (t.ne[0].max(1) as usize * 4).min(nbytes);
    let mut buf = vec![0u8; row];
    unsafe { ggml_backend_tensor_get(t, buf.as_mut_ptr() as *mut c_void, 0, row) };
    let name = unsafe { CStr::from_ptr(t.name.as_ptr()) }.to_string_lossy().into_owned();
    let src = |i: usize| if t.src[i].is_null() { String::from("-") } else { let x = unsafe { &*t.src[i] }; format!("{}:type{}:ne{:?}", unsafe { CStr::from_ptr(x.name.as_ptr()) }.to_string_lossy(), x.type_ as u32, x.ne) };
    tr.info.push(format!("op{} ne{:?} src0[{}] src1[{}]", t.op as u32, t.ne, src(0), src(1)));
    tr.nodes.push((name, fnv(&buf)));
    true
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model = CString::new(a[1].as_bytes()).unwrap();
    let others: Vec<usize> = a.get(2).map_or("1,3,7,15".into(), |s| s.clone()).split(',').map(|x| x.parse().unwrap()).collect();
    let gen: usize = a.get(3).map_or(24, |s| s.parse().unwrap());
    let n_seq = others.iter().max().unwrap() + 1;
    let mut trace = Box::new(Trace { on: false, nodes: Vec::new(), info: Vec::new() });
    unsafe {
        let mut mp = sys::llama_model_default_params();
        mp.n_gpu_layers = 999;
        let m = sys::llama_model_load_from_file(model.as_ptr(), mp);
        assert!(!m.is_null());
        let n_vocab = sys::llama_vocab_n_tokens(sys::llama_model_get_vocab(m)) as usize;
        let mut cp = sys::llama_context_default_params();
        cp.n_ctx = (n_seq * 1024) as u32;
        cp.n_seq_max = n_seq as u32;
        cp.n_ubatch = 512;
        cp.n_batch = 2048;
        cp.flash_attn_type = sys::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.kv_unified = false;
        cp.cb_eval = Some(cb);
        cp.cb_eval_user_data = &mut *trace as *mut Trace as *mut c_void;
        let ctx = sys::llama_init_from_model(m, cp);
        assert!(!ctx.is_null());
        let mem = sys::llama_get_memory(ctx);
        let mut batch = sys::llama_batch_init(2048, 0, 1);
        // decode rows (seq, pos, token); logits for every row; returns the logits per row
        let mut decode = |rows: &[(i32, usize, i32)]| -> Vec<Vec<f32>> {
            let b = &mut batch;
            for (k, &(s, p, t)) in rows.iter().enumerate() {
                *b.token.add(k) = t;
                *b.pos.add(k) = p as i32;
                *b.n_seq_id.add(k) = 1;
                **b.seq_id.add(k) = s;
                *b.logits.add(k) = 1;
            }
            b.n_tokens = rows.len() as i32;
            assert_eq!(sys::llama_decode(ctx, *b), 0, "llama_decode");
            sys::llama_synchronize(ctx);
            (0..rows.len()).map(|k| std::slice::from_raw_parts(sys::llama_get_logits_ith(ctx, k as i32), n_vocab).to_vec()).collect()
        };
        // prompts: fixed pseudo-random ids, one prefill decode per sequence
        let mut x = 12345u64;
        let mut first = vec![0i32; n_seq];
        for s in 0..n_seq {
            let rows: Vec<(i32, usize, i32)> = (0..PLEN)
                .map(|p| {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (s as i32, p, 1000 + ((x >> 33) % 20000) as i32)
                })
                .collect();
            let out = decode(&rows);
            first[s] = argmax(out.last().unwrap());
        }
        // the state of every sequence right after its prompt (recurrent and hybrid caches cannot drop a tail, so a run
        // starts from a restored snapshot instead of a cut)
        let snaps: Vec<Vec<u8>> = (0..n_seq)
            .map(|s| {
                let n = llama_state_seq_get_size(ctx, s as i32);
                let mut b = vec![0u8; n];
                let got = llama_state_seq_get_data(ctx, b.as_mut_ptr(), n, s as i32);
                b.truncate(got);
                b
            })
            .collect();
        let reset = |n: usize| {
            for s in 0..n {
                llama_memory_seq_rm(mem, s as i32, -1, -1);
                assert!(llama_state_seq_set_data(ctx, snaps[s].as_ptr(), snaps[s].len(), s as i32) > 0, "state restore");
            }
        };
        // run: sequence 0 plus `k` others decode together for `gen` steps; returns seq 0's logit hashes and step-0 trace
        let mut step_ms: Vec<f64> = Vec::new();
        let mut run = |k: usize, decode: &mut dyn FnMut(&[(i32, usize, i32)]) -> Vec<Vec<f32>>, trace: &mut Trace| -> (Vec<u64>, Vec<(String, u64)>, f64) {
            step_ms.clear();
            reset(n_seq);
            let mut tok: Vec<i32> = first.clone();
            let mut hs = Vec::new();
            let mut nodes = Vec::new();
            for j in 0..gen {
                let rows: Vec<(i32, usize, i32)> = (0..=k).map(|s| (s as i32, PLEN + j, tok[s])).collect();
                trace.on = j == 0;
                trace.nodes.clear();
                trace.info.clear();
                let t0 = std::time::Instant::now();
                let out = decode(&rows);
                if j > 0 {
                    step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
                }
                // the eval callback wrote through a raw pointer: make the compiler re-read the trace
                let trace: &mut Trace = std::hint::black_box(&mut *trace);
                if j == 0 {
                    nodes = std::mem::take(&mut trace.nodes);
                    if k == 1 {
                        for (i, inf) in trace.info.iter().enumerate().skip(48).take(8) {
                            println!("  node {i} {} {}", nodes[i].0, inf);
                        }
                    }
                }
                trace.on = false;
                hs.push(fnv(std::slice::from_raw_parts(out[0].as_ptr() as *const u8, n_vocab * 4)));
                for s in 0..=k {
                    tok[s] = argmax(&out[s]);
                }
            }
            step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = step_ms.get(step_ms.len() / 2).copied().unwrap_or(0.0);
            (hs, nodes, med)
        };
        let (ref_h, ref_nodes, ms0) = run(0, &mut decode, &mut trace);
        let (again, _, _) = run(0, &mut decode, &mut trace);
        println!("model {} | {} graph nodes traced | alone twice identical: {} | step alone {:.2} ms", a[1], ref_nodes.len(), again == ref_h, ms0);
        for &k in &others {
            let (h, nodes, ms) = run(k, &mut decode, &mut trace);
            let same = h.iter().zip(&ref_h).filter(|(x, y)| x == y).count();
            let first_diff_step = h.iter().zip(&ref_h).position(|(x, y)| x != y);
            let first_node = ref_nodes.iter().zip(&nodes).position(|(r, n)| r.0 == n.0 && r.1 != n.1).map(|i| (i, ref_nodes[i].0.clone()));
            let mismatch_names = ref_nodes.iter().zip(&nodes).filter(|(r, n)| r.0 != n.0).count();
            println!(
                "next to {k:2} others: step {ms:.2} ms | seq-0 logits bitwise equal {same}/{gen} steps, first different step {:?}; first differing node {:?} ({} name mismatches)",
                first_diff_step, first_node, mismatch_names
            );
        }
    }
}
