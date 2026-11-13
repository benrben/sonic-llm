//! Probe for prompt-lookup speculative decoding on libllama: the cost of a verify step (k+1 rows per sequence) against a
//! plain step (1 row per sequence), and whether the verify rows are bitwise what sequential 1-row decode gives.
//!
//! spec_probe MODEL.gguf N_SEQ PROMPT_LEN [GEN=40] [TEXT_FILE]
//! Run under engine/scripts/with_gpu_lock.py. Env: SLLM_QMM32_N1=1 etc. reach the patched libggml-metal as usual.

use llama_ref::{sys, Vocab};
use std::ffi::CString;
use std::time::Instant;

unsafe extern "C" {
    fn llama_memory_seq_rm(mem: sys::llama_memory_t, seq: i32, p0: i32, p1: i32) -> bool;
}

fn fnv(row: &[f32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for v in row {
        h ^= v.to_bits() as u64;
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

struct Ctx {
    ctx: *mut sys::llama_context,
    batch: sys::llama_batch,
    n_vocab: usize,
}

type Rows = Vec<(i32, usize, Vec<i32>, bool)>;

impl Ctx {
    /// Decode rows `(seq, pos0, toks, logits_all)`; returns, per row, the logits of each of its tokens that has them.
    fn decode(&mut self, rows: &Rows) -> Vec<Vec<Vec<f32>>> {
        let mut out_idx: Vec<Vec<i32>> = Vec::new();
        let mut k = 0usize;
        unsafe {
            let b = &mut self.batch;
            for (sid, pos0, toks, all) in rows {
                let mut idx = Vec::new();
                for (i, &t) in toks.iter().enumerate() {
                    *b.token.add(k) = t;
                    *b.pos.add(k) = (*pos0 + i) as i32;
                    *b.n_seq_id.add(k) = 1;
                    **b.seq_id.add(k) = *sid;
                    let want = *all || i + 1 == toks.len();
                    *b.logits.add(k) = want as i8;
                    if want {
                        idx.push(k as i32);
                    }
                    k += 1;
                }
                out_idx.push(idx);
            }
            b.n_tokens = k as i32;
            let rc = sys::llama_decode(self.ctx, *b);
            assert_eq!(rc, 0, "llama_decode");
            sys::llama_synchronize(self.ctx);
        }
        out_idx
            .iter()
            .map(|idx| {
                idx.iter()
                    .map(|&i| unsafe { std::slice::from_raw_parts(sys::llama_get_logits_ith(self.ctx, i), self.n_vocab).to_vec() })
                    .collect()
            })
            .collect()
    }

    /// Time one step of `r` rows per sequence (logits on for every row); `do_argmax` adds the host argmax of every row.
    fn time_step(&mut self, n_seq: usize, pos0: usize, r: usize, tok: i32, do_argmax: bool, mem: sys::llama_memory_t) -> f64 {
        let mut k = 0usize;
        unsafe {
            let b = &mut self.batch;
            for s in 0..n_seq {
                for i in 0..r {
                    *b.token.add(k) = tok;
                    *b.pos.add(k) = (pos0 + i) as i32;
                    *b.n_seq_id.add(k) = 1;
                    **b.seq_id.add(k) = s as i32;
                    *b.logits.add(k) = 1;
                    k += 1;
                }
            }
            b.n_tokens = k as i32;
            let t0 = Instant::now();
            let rc = sys::llama_decode(self.ctx, *b);
            assert_eq!(rc, 0);
            sys::llama_synchronize(self.ctx);
            let mut sink = 0i64;
            if do_argmax {
                for i in 0..k {
                    let row = std::slice::from_raw_parts(sys::llama_get_logits_ith(self.ctx, i as i32), self.n_vocab);
                    sink += argmax(row) as i64;
                }
            }
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(sink);
            for s in 0..n_seq {
                llama_memory_seq_rm(mem, s as i32, pos0 as i32, -1);
            }
            dt
        }
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model = std::path::PathBuf::from(&a[1]);
    let n_seq: usize = a[2].parse().unwrap();
    let plen: usize = a[3].parse().unwrap();
    let gen: usize = a.get(4).map_or(40, |s| s.parse().unwrap());
    let text_file = a.get(5).cloned().unwrap_or_else(|| "engine/vendor/sllm-metal-b11669/README.md".into());
    let text = std::fs::read_to_string(&text_file).unwrap();
    let vocab = Vocab::load(&model).unwrap();
    let n_vocab = vocab.n_vocab();
    let words: Vec<&str> = text.split_whitespace().collect();
    // a distinct natural-text prompt per sequence
    let prompts: Vec<Vec<i32>> = (0..n_seq)
        .map(|s| {
            let mut w = (s * 997) % words.len().saturating_sub(100).max(1);
            let mut ids = vocab.tokenize(&words[w..(w + 40).min(words.len())].join(" "), true, false).unwrap();
            while ids.len() < plen {
                w = (w + 40) % words.len().saturating_sub(41).max(1);
                let t = vocab.tokenize(&format!(" {}", words[w..(w + 40).min(words.len())].join(" ")), false, false).unwrap();
                ids.extend(t);
            }
            ids.truncate(plen);
            ids
        })
        .collect();
    unsafe {
        let path = CString::new(model.to_string_lossy().as_bytes()).unwrap();
        let mut mp = sys::llama_model_default_params();
        mp.n_gpu_layers = 999;
        let m = sys::llama_model_load_from_file(path.as_ptr(), mp);
        assert!(!m.is_null());
        let mut cp = sys::llama_context_default_params();
        cp.n_ctx = ((plen + gen + 64) * n_seq).div_ceil(256) as u32 * 256;
        cp.n_seq_max = n_seq as u32;
        cp.n_ubatch = 512;
        cp.n_batch = 2048;
        cp.flash_attn_type = sys::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.kv_unified = std::env::var_os("PROBE_NONUNIFIED").is_none();
        let ctx = sys::llama_init_from_model(m, cp);
        assert!(!ctx.is_null());
        let mem = sys::llama_get_memory(ctx);
        let mut c = Ctx { ctx, batch: sys::llama_batch_init(2048, 0, 1), n_vocab };
        // prefill every sequence (chunks of 512), one sequence per decode
        let mut t0s = vec![0i32; n_seq];
        for s in 0..n_seq {
            let mut pos = 0;
            while pos < plen {
                let end = (pos + 512).min(plen);
                let r = c.decode(&vec![(s as i32, pos, prompts[s][pos..end].to_vec(), false)]);
                if end == plen {
                    t0s[s] = argmax(&r[0][0]);
                }
                pos = end;
            }
        }
        // reference: sequential 1-row decode, all sequences together: step j feeds toks[s][j]
        let mut toks: Vec<Vec<i32>> = t0s.iter().map(|&t| vec![t]).collect();
        let mut refl: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_seq];
        for j in 0..gen {
            let rows: Rows = (0..n_seq).map(|s| (s as i32, plen + j, vec![toks[s][j]], false)).collect();
            let out = c.decode(&rows);
            for s in 0..n_seq {
                let l = out[s][0].clone();
                toks[s].push(argmax(&l));
                refl[s].push(l);
            }
        }
        // determinism of the reference itself
        for s in 0..n_seq {
            llama_memory_seq_rm(mem, s as i32, plen as i32, -1);
        }
        let mut det_diff = 0usize;
        for j in 0..gen {
            let rows: Rows = (0..n_seq).map(|s| (s as i32, plen + j, vec![toks[s][j]], false)).collect();
            let out = c.decode(&rows);
            for s in 0..n_seq {
                if fnv(&out[s][0]) != fnv(&refl[s][j]) {
                    det_diff += 1;
                }
            }
        }
        println!("N={n_seq} P={plen} gen={gen}: reference rerun: {det_diff}/{} rows differ bitwise", n_seq * gen);
        // verify runs: `pattern` rows per sequence per step
        let pats: Vec<(&str, Vec<usize>)> = vec![
            ("all5", vec![5usize; n_seq]),
            ("all3", vec![3; n_seq]),
            ("mixed(odd=5,even=1)", (0..n_seq).map(|s| if s % 2 == 1 { 5 } else { 1 }).collect()),
        ];
        let timing_only = std::env::var_os("PROBE_TIMING_ONLY").is_some();
        for (name, pat) in pats {
            if timing_only {
                break;
            }
            if name.starts_with("mixed") && n_seq < 2 {
                continue;
            }
            for s in 0..n_seq {
                llama_memory_seq_rm(mem, s as i32, plen as i32, -1);
            }
            let mut ptr = vec![0usize; n_seq];
            let (mut rows_cmp, mut bit_eq, mut am_eq, mut maxd) = (0usize, 0usize, 0usize, 0f32);
            let (mut m5, mut b5, mut a5) = (0usize, 0usize, 0usize);
            let mut steps = 0;
            while ptr.iter().any(|&p| p < gen) {
                let mut rows: Rows = Vec::new();
                let mut who = Vec::new();
                for s in 0..n_seq {
                    let n = pat[s].min(gen - ptr[s]);
                    if n == 0 {
                        continue;
                    }
                    rows.push((s as i32, plen + ptr[s], toks[s][ptr[s]..ptr[s] + n].to_vec(), true));
                    who.push((s, n));
                }
                let out = c.decode(&rows);
                for (ri, &(s, n)) in who.iter().enumerate() {
                    for i in 0..n {
                        let l = &out[ri][i];
                        let r = &refl[s][ptr[s] + i];
                        let eq = fnv(l) == fnv(r);
                        let aeq = argmax(l) == argmax(r);
                        let d = l.iter().zip(r).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                        rows_cmp += 1;
                        bit_eq += eq as usize;
                        am_eq += aeq as usize;
                        maxd = maxd.max(d);
                        if n > 1 {
                            m5 += 1;
                            b5 += eq as usize;
                            a5 += aeq as usize;
                        }
                    }
                    ptr[s] += n;
                }
                steps += 1;
            }
            println!(
                "  {name:22} steps={steps} rows={rows_cmp} bitwise-equal={bit_eq} ({:.1}%) argmax-equal={am_eq} ({:.1}%) max|dlogit|={maxd:.4}   [rows in multi-row groups: {m5}, bitwise {b5}, argmax {a5}]",
                100.0 * bit_eq as f64 / rows_cmp as f64,
                100.0 * am_eq as f64 / rows_cmp as f64
            );
        }
        if std::env::var_os("PROBE_NO_TIMING").is_some() {
            return;
        }
        // timing
        for s in 0..n_seq {
            llama_memory_seq_rm(mem, s as i32, plen as i32, -1);
        }
        let p0 = plen;
        let tok = toks[0][0];
        for do_am in [false, true] {
            let mut res = Vec::new();
            for r in [1usize, 2, 3, 5, 9] {
                for _ in 0..3 {
                    c.time_step(n_seq, p0, r, tok, do_am, mem);
                }
                let mut v: Vec<f64> = (0..25).map(|_| c.time_step(n_seq, p0, r, tok, do_am, mem)).collect();
                res.push((r, median(&mut v)));
            }
            let base = res[0].1;
            let s: Vec<String> = res.iter().map(|(r, t)| format!("{r}row {t:.2}ms (c={:.3})", if *r > 1 { (t / base - 1.0) / (*r as f64 - 1.0) } else { 0.0 })).collect();
            println!("  timing N={n_seq} P={plen} host-argmax={do_am}: {}", s.join("  "));
        }
        sys::llama_batch_free(c.batch);
        sys::llama_free(ctx);
        sys::llama_model_free(m);
    }
}
