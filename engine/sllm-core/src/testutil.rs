//! Test helpers: a tiny GGUF v3 writer for synthetic model headers, and the real-model paths.

use std::io::Write;
use std::path::{Path, PathBuf};

pub const QWEN3_PATH: &str = "/Users/benreich/Projects/crazyidea/research/inference_runtime/converted/qwen3-0.6b-Q4_0.gguf";
pub const GEMMA3_PATH: &str = "/Users/benreich/Projects/crazyidea/models/gguf/gemma-3-4b-it-qat-Q4_0.gguf";

/// Path of a real model; the test fails loudly when it is missing unless SLLM_TEST_ALLOW_MISSING_MODELS=1.
pub fn real_model(path: &str) -> Option<PathBuf> {
    let p = PathBuf::from(path);
    if p.exists() {
        return Some(p);
    }
    if std::env::var("SLLM_TEST_ALLOW_MISSING_MODELS").as_deref() == Ok("1") {
        eprintln!("skipping: {path} not found");
        return None;
    }
    panic!("model file {path} not found (set SLLM_TEST_ALLOW_MISSING_MODELS=1 to skip)");
}

pub enum Kv {
    U32(u32),
    F32(f32),
    Str(String),
    StrArr(Vec<String>),
}

#[derive(Default)]
pub struct Builder {
    pub kvs: Vec<(String, Kv)>,
    /// (name, ne) -- all tensors are F32 and carry zeros.
    pub tensors: Vec<(String, Vec<i64>)>,
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

impl Builder {
    pub fn kv(mut self, k: &str, v: Kv) -> Self {
        self.kvs.push((k.to_string(), v));
        self
    }
    pub fn tensor(mut self, name: &str, ne: &[i64]) -> Self {
        self.tensors.push((name.to_string(), ne.to_vec()));
        self
    }

    /// Serialise. `truncate_data_by` removes that many bytes from the end (simulates a cut-off download).
    pub fn build(&self, truncate_data_by: usize) -> Vec<u8> {
        const ALIGN: usize = 32;
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.kvs.len() as u64).to_le_bytes());
        for (k, v) in &self.kvs {
            put_str(&mut out, k);
            match v {
                Kv::U32(x) => {
                    out.extend_from_slice(&4u32.to_le_bytes());
                    out.extend_from_slice(&x.to_le_bytes());
                }
                Kv::F32(x) => {
                    out.extend_from_slice(&6u32.to_le_bytes());
                    out.extend_from_slice(&x.to_le_bytes());
                }
                Kv::Str(s) => {
                    out.extend_from_slice(&8u32.to_le_bytes());
                    put_str(&mut out, s);
                }
                Kv::StrArr(a) => {
                    out.extend_from_slice(&9u32.to_le_bytes());
                    out.extend_from_slice(&8u32.to_le_bytes());
                    out.extend_from_slice(&(a.len() as u64).to_le_bytes());
                    for s in a {
                        put_str(&mut out, s);
                    }
                }
            }
        }
        let mut offset = 0usize;
        let mut sizes = Vec::new();
        for (name, ne) in &self.tensors {
            put_str(&mut out, name);
            out.extend_from_slice(&(ne.len() as u32).to_le_bytes());
            for d in ne {
                out.extend_from_slice(&(*d as u64).to_le_bytes());
            }
            out.extend_from_slice(&0u32.to_le_bytes()); // GGML_TYPE_F32
            out.extend_from_slice(&(offset as u64).to_le_bytes());
            let bytes = 4 * ne.iter().product::<i64>() as usize;
            let padded = bytes.div_ceil(ALIGN) * ALIGN;
            sizes.push(padded);
            offset += padded;
        }
        while out.len() % ALIGN != 0 {
            out.push(0);
        }
        out.resize(out.len() + offset - truncate_data_by, 0);
        out
    }

    pub fn write_to(&self, dir: &Path, name: &str, truncate_data_by: usize) -> PathBuf {
        let p = dir.join(name);
        std::fs::File::create(&p).unwrap().write_all(&self.build(truncate_data_by)).unwrap();
        p
    }
}

/// A scratch directory under the system temp dir, removed on drop.
pub struct TempDir(pub PathBuf);
impl TempDir {
    pub fn new(tag: &str) -> TempDir {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("sllm-core-test-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Header + tensors of a tiny but consistent model of the given architecture.
pub fn tiny_model(arch: &str) -> Builder {
    let (e, ff, vocab, hd, nh, nkv, layers) = (8i64, 16i64, 10i64, 4i64, 4i64, 2i64, 3u32);
    let p = |s: &str| format!("{arch}.{s}");
    let mut b = Builder::default()
        .kv("general.architecture", Kv::Str(arch.into()))
        .kv("general.name", Kv::Str(format!("tiny-{arch}")))
        .kv(&p("block_count"), Kv::U32(layers))
        .kv(&p("context_length"), Kv::U32(512))
        .kv(&p("embedding_length"), Kv::U32(e as u32))
        .kv(&p("feed_forward_length"), Kv::U32(ff as u32))
        .kv(&p("attention.head_count"), Kv::U32(nh as u32))
        .kv(&p("attention.head_count_kv"), Kv::U32(nkv as u32))
        .kv(&p("attention.key_length"), Kv::U32(hd as u32))
        .kv(&p("attention.layer_norm_rms_epsilon"), Kv::F32(1e-6))
        .kv(&p("rope.freq_base"), Kv::F32(1_000_000.0))
        .kv("tokenizer.ggml.tokens", Kv::StrArr((0..vocab).map(|i| format!("t{i}")).collect()));
    if arch == "gemma3" {
        b = b
            .kv(&p("attention.sliding_window"), Kv::U32(64))
            .kv(&p("rope.scaling.type"), Kv::Str("linear".into()))
            .kv(&p("rope.scaling.factor"), Kv::F32(8.0));
    }
    b = b.tensor("token_embd.weight", &[e, vocab]).tensor("output_norm.weight", &[e]);
    for il in 0..layers {
        b = b
            .tensor(&format!("blk.{il}.attn_q.weight"), &[e, hd * nh])
            .tensor(&format!("blk.{il}.attn_k.weight"), &[e, hd * nkv])
            .tensor(&format!("blk.{il}.ffn_up.weight"), &[e, ff])
            .tensor(&format!("blk.{il}.ffn_down.weight"), &[ff, e]);
    }
    b
}
