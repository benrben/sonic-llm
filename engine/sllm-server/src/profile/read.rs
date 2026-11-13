//! The profile read from the GGUF header, and what libllama settles or checks on top of it.

use super::*;

fn arr_or_scalar(g: &Gguf, key: &str, n_layer: usize) -> Option<Vec<u32>> {
    if let Some(a) = g.nums(key) {
        let mut v: Vec<u32> = a.iter().map(|&x| x as u32).collect();
        v.resize(n_layer, *v.last().unwrap_or(&0));
        return Some(v);
    }
    g.num(key).map(|x| vec![x as u32; n_layer])
}

/// llama.cpp's `set_swa_pattern(n, dense_first = false)`: layers `il % n < n - 1` are SWA, every n-th is full attention.
pub(super) fn period_pattern(n_layer: usize, period: u32) -> Vec<bool> {
    (0..n_layer).map(|il| period == 0 || (il as u32 % period) < period - 1).collect()
}

/// A hash of the file's size and sampled blocks (head, middle, tail), hex.
pub fn fingerprint(path: &Path) -> Result<(u64, String), String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let len = f.metadata().map_err(|e| e.to_string())?.len();
    let mut h: u64 = 0xcbf29ce484222325 ^ len;
    let block = 4usize << 20;
    let mut buf = vec![0u8; block];
    let mut at = |off: u64, f: &mut std::fs::File| -> Result<(), String> {
        f.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        for &b in &buf[..n] {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        Ok(())
    };
    at(0, &mut f)?;
    if len > 3 * block as u64 {
        at(len / 2, &mut f)?;
        at(len - block as u64, &mut f)?;
    }
    Ok((len, format!("{len:x}-{h:016x}")))
}

/// The per-head key and value dimensions of a model's layers (the SWA layers' own when the model has them).
struct HeadDims {
    key_len: u32,
    val_len: u32,
    key_len_swa: u32,
    val_len_swa: u32,
    /// MLA: the cache holds one latent per token and no V.
    mla: bool,
}

impl HeadDims {
    fn read(g: &Gguf, k: &dyn Fn(&str) -> String, n_embd: u64, heads: &[u32]) -> HeadDims {
        // llama.cpp: n_embd_head_k = n_embd / n_head(0) unless key_length says otherwise
        let n_head0 = heads.first().copied().unwrap_or(0).max(1) as u64;
        let dflt = (n_embd / n_head0) as u32;
        let key_len = g.num(&k("attention.key_length")).map(|x| x as u32).unwrap_or(dflt);
        let val_len = g.num(&k("attention.value_length")).map(|x| x as u32).unwrap_or(dflt);
        let key_len_swa = g.num(&k("attention.key_length_swa")).map(|x| x as u32).unwrap_or(key_len);
        let val_len_swa = g.num(&k("attention.value_length_swa")).map(|x| x as u32).unwrap_or(val_len);
        let mla = g.has(&k("attention.key_length_mla")) && g.has(&k("attention.value_length_mla"));
        HeadDims { key_len, val_len, key_len_swa, val_len_swa, mla }
    }

    /// The value dimension of the full-attention layers (none with MLA).
    fn head_dim_v(&self) -> u32 {
        if self.mla { 0 } else { self.val_len }
    }

    /// What a layer with `n_head_kv` KV heads stores per cell.
    fn layer(&self, n_head_kv: u32, swa: bool) -> LayerKv {
        LayerKv {
            n_head_kv,
            k_dim: if swa { self.key_len_swa } else { self.key_len },
            v_dim: if self.mla { 0 } else if swa { self.val_len_swa } else { self.val_len },
            swa,
        }
    }
}

/// The sliding-window layer map of a model with a window of `n_swa` (0: none), and where it came from.
fn swa_layer_map(g: &Gguf, k: &dyn Fn(&str) -> String, n_layer: usize, n_swa: u32) -> (Vec<bool>, SwaSource) {
    if n_swa == 0 {
        (vec![false; n_layer], SwaSource::None)
    } else if let Some(a) = g.nums(&k("attention.sliding_window_pattern")) {
        let mut v: Vec<bool> = a.iter().map(|&x| x != 0.0).collect();
        v.resize(n_layer, false);
        (v, SwaSource::MetadataArray)
    } else if let Some(p) = g.num(&k("attention.sliding_window_pattern")) {
        (period_pattern(n_layer, p as u32), SwaSource::MetadataPeriod)
    } else {
        (vec![true; n_layer], SwaSource::Assumed)
    }
}

/// The tensor table's bytes by how a decode step reads them (see the `weights_*` fields of [`Profile`]).
#[derive(Default)]
struct WeightBytes {
    total: u64,
    dense: u64,
    experts: u64,
    gather: u64,
    embd_row_bytes: u64,
}

impl WeightBytes {
    fn of(tensors: &[TensorInfo], tied_head: bool) -> WeightBytes {
        let mut w = WeightBytes::default();
        for t in tensors {
            w.total += t.bytes;
            if t.name.contains("_exps") {
                w.experts += t.bytes;
            } else if t.name.ends_with("token_embd.weight") {
                w.add_embedding(t, tied_head);
            } else {
                w.dense += t.bytes;
            }
        }
        w
    }

    /// An embedding table: gathered, unless it is the tied output head.
    fn add_embedding(&mut self, t: &TensorInfo, tied_head: bool) {
        if t.name == "token_embd.weight" {
            self.embd_row_bytes = t.bytes / t.ne[1].max(1) as u64;
            if tied_head {
                self.dense += t.bytes; // read in full as the output head
                return;
            }
        }
        self.gather += t.bytes;
    }
}

/// Whether the model is recurrent (a state and no attention layer) and whether it is hybrid (a state beside attention).
fn state_kinds(tensors: &[TensorInfo], layers: &[LayerKv]) -> (bool, bool) {
    let has_state = tensors.iter().any(|t| t.name.contains(".ssm_") || t.name.contains(".time_mix_") || t.name.contains("shortconv") || t.name.contains(".conv1d"));
    let n_attn = layers.iter().filter(|l| l.n_head_kv > 0).count();
    let recurrent = has_state && n_attn == 0;
    let hybrid = has_state && n_attn > 0;
    (recurrent, hybrid)
}

impl Profile {
    /// Read the profile from the GGUF header. Pure computation: no model is loaded and no GPU is touched.
    pub fn from_gguf(path: &Path) -> Result<Profile, String> {
        let g = Gguf::open(path)?;
        let (file_bytes, fingerprint) = fingerprint(path)?;
        let arch = g.str("general.architecture").ok_or("no general.architecture")?;
        let k = |s: &str| format!("{arch}.{s}");
        let mut notes = Vec::new();
        let n_layer = g.num(&k("block_count")).ok_or_else(|| format!("no {arch}.block_count"))? as usize;
        let n_embd = g.num(&k("embedding_length")).ok_or_else(|| format!("no {arch}.embedding_length"))? as u64;
        let heads = arr_or_scalar(&g, &k("attention.head_count"), n_layer).unwrap_or_else(|| vec![0; n_layer]);
        let heads_kv = arr_or_scalar(&g, &k("attention.head_count_kv"), n_layer).unwrap_or_else(|| heads.clone());
        let dims = HeadDims::read(&g, &k, n_embd, &heads);
        let n_swa = g.num(&k("attention.sliding_window")).map(|x| x as u32).unwrap_or(0);

        // sliding-window layer map
        let (swa_map, mut swa_source) = swa_layer_map(&g, &k, n_layer, n_swa);
        if swa_source == SwaSource::Assumed {
            notes.push(format!("sliding_window {n_swa} without a layer map in the metadata: resolved by the libllama probe"));
        }
        let layers: Vec<LayerKv> = (0..n_layer).map(|il| dims.layer(heads_kv[il], swa_map[il])).collect();
        if n_swa > 0 && !layers.iter().any(|l| l.swa) {
            swa_source = SwaSource::None;
        }

        let n_expert = g.num(&k("expert_count")).unwrap_or(0.0) as u32;
        let n_expert_used = g.num(&k("expert_used_count")).unwrap_or(0.0) as u32;
        let n_expert_shared = g.num(&k("expert_shared_count")).unwrap_or(0.0) as u32;
        let n_vocab = g.arr_len("tokenizer.ggml.tokens").map(|n| n as u64).or_else(|| g.num(&k("vocab_size")).map(|x| x as u64)).unwrap_or(0);
        let n_ctx_train = g.num(&k("context_length")).unwrap_or(0.0) as u64;

        // weights: the tensor table. llama.cpp names expert stacks `*_exps*`; embedding tables are gathered.
        let tensors = g.tensors();
        let has_output = tensors.iter().any(|t| t.name == "output.weight");
        let tied_head = !has_output;
        let w = WeightBytes::of(&tensors, tied_head);
        let (recurrent, hybrid) = state_kinds(&tensors, &layers);
        let n_head = heads.iter().copied().max().unwrap_or(0);
        let n_head_kv = heads_kv.iter().copied().max().unwrap_or(0);
        Ok(Profile {
            path: path.to_path_buf(),
            name: g.str("general.name"),
            arch,
            file_bytes,
            fingerprint,
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            head_dim_k: dims.key_len,
            head_dim_v: dims.head_dim_v(),
            n_vocab,
            n_ctx_train,
            n_swa,
            layers,
            swa_source,
            n_expert,
            n_expert_used,
            n_expert_shared,
            recurrent,
            hybrid,
            mla: dims.mla,
            weights_total: w.total,
            weights_dense: w.dense,
            weights_experts: w.experts,
            weights_gather: w.gather,
            embd_row_bytes: w.embd_row_bytes,
            tied_head,
            notes,
        })
    }
}

impl Profile {
    /// Settle what the metadata could not say, by asking libllama (a `no_alloc` context, nothing is allocated): the SWA
    /// layer count when the file has a sliding window but no layer map.
    pub fn resolve_with_libllama(&mut self) {
        if self.swa_source != SwaSource::Assumed {
            return;
        }
        match probe_memory(&self.path, &ProbeParams { n_gpu_layers: 0, cells: 1024, n_seq: 1, n_ubatch: 32, n_batch: 32, ..Default::default() }) {
            Ok(p) if p.kv.len() == 2 => self.take_layer_counts(p.kv[0].layers, p.kv[1].layers),
            Ok(p) if p.kv.len() == 1 => {
                // libllama kept a single cache: no layer is SWA after all
                for l in self.layers.iter_mut() {
                    l.swa = false;
                }
                self.swa_source = SwaSource::None;
                self.notes.clear();
            }
            Ok(_) => self.notes.push("libllama probe: no KV cache line".into()),
            Err(e) => self.notes.push(format!("libllama probe failed: {e}")),
        }
    }

    /// The probe's two caches held `n_full` full-attention and `n_swa` SWA layers: laid out with llama.cpp's period rule
    /// (the first `n_swa` layers when the rule gives another count); counts that do not add up are only noted.
    fn take_layer_counts(&mut self, n_full: usize, n_swa: usize) {
        if n_swa + n_full == self.n_layer && n_full > 0 {
            let period = (self.n_layer / n_full) as u32;
            let map = period_pattern(self.n_layer, period);
            let map = if map.iter().filter(|&&s| s).count() == n_swa { map } else { (0..self.n_layer).map(|il| il < n_swa).collect() };
            for (l, s) in self.layers.iter_mut().zip(map) {
                l.swa = s;
            }
            self.swa_source = SwaSource::Libllama;
            self.notes.clear();
            return;
        }
        self.notes.push(format!("libllama probe: {n_full} full + {n_swa} SWA layers of {}", self.n_layer));
    }

    /// Compare the header-derived profile with a loaded `llama_model`; returns the differences (also kept in `notes`).
    ///
    /// # Safety
    /// `model` is a valid loaded model.
    pub unsafe fn verify_model(&mut self, model: *const sys::llama_model) -> Vec<String> {
        let mut diff = Vec::new();
        unsafe {
            let chk = |what: &str, mine: i64, theirs: i64, diff: &mut Vec<String>| {
                if mine != theirs {
                    diff.push(format!("{what}: profile {mine}, libllama {theirs}"));
                }
            };
            chk("n_layer", self.n_layer as i64, ffi::llama_model_n_layer(model) as i64, &mut diff);
            chk("n_embd", self.n_embd as i64, sys::llama_model_n_embd(model) as i64, &mut diff);
            chk("n_head(0)", self.layers.first().map_or(0, |_| self.n_head as i64), ffi::llama_model_n_head(model) as i64, &mut diff);
            chk("n_swa", self.n_swa as i64, ffi::llama_model_n_swa(model) as i64, &mut diff);
            chk("n_ctx_train", self.n_ctx_train as i64, ffi::llama_model_n_ctx_train(model) as i64, &mut diff);
            let size = ffi::llama_model_size(model);
            if size != self.weights_total {
                diff.push(format!("weights: profile {} bytes, libllama {size}", self.weights_total));
            }
            let (rec, hyb) = (ffi::llama_model_is_recurrent(model), ffi::llama_model_is_hybrid(model));
            // libllama is authoritative for the state kinds
            if rec != self.recurrent || hyb != self.hybrid {
                diff.push(format!("recurrent/hybrid: profile {}/{}, libllama {rec}/{hyb} (libllama wins)", self.recurrent, self.hybrid));
                self.recurrent = rec;
                self.hybrid = hyb;
            }
        }
        self.notes.extend(diff.iter().cloned());
        diff
    }
}
