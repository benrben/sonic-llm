use super::*;
use crate::testutil::*;

fn golden(name: &str) -> serde_json::Value {
    let p = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// Compare a parsed file with the dump gguf-py produced for the same file.
fn check_against_gguf_py(path: &str, golden_name: &str, expect_tensors: usize) {
    let Some(p) = real_model(path) else { return };
    let g = GgufFile::open(&p).unwrap();
    let want = golden(golden_name);
    assert_eq!(g.tensors().len(), expect_tensors);
    assert_eq!(want["n_tensors"].as_u64().unwrap() as usize, expect_tensors);
    assert_eq!(g.version() as u64, want["version"].as_u64().unwrap());
    assert_eq!(g.alignment() as u64, want["alignment"].as_u64().unwrap());
    assert_eq!(g.data_offset() as u64, want["data_offset"].as_u64().unwrap());
    let wt = want["tensors"].as_array().unwrap();
    for (i, (t, w)) in g.tensors().iter().zip(wt).enumerate() {
        assert_eq!(t.name, w["name"].as_str().unwrap(), "tensor #{i} name");
        assert_eq!(t.type_name.to_uppercase(), w["type"].as_str().unwrap(), "{} type", t.name);
        let ne: Vec<i64> = w["ne"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
        assert_eq!(t.ne.to_vec(), ne, "{} ne", t.name);
        assert_eq!(t.offset, w["offset"].as_u64().unwrap(), "{} offset", t.name);
        assert_eq!(t.nbytes, w["nbytes"].as_u64().unwrap(), "{} nbytes", t.name);
    }
    // scalar metadata and array lengths
    let got = g.dump_json();
    for (k, v) in want["kv"].as_object().unwrap() {
        let gv = &got["kv"][k];
        match (v.as_f64(), gv.as_f64()) {
            (Some(a), Some(b)) if v.is_f64() => assert!((a - b).abs() <= a.abs() * 1e-9, "{k}: {a} vs {b}"),
            _ => assert_eq!(gv, v, "kv {k}"),
        }
    }
    assert_eq!(got["kv"].as_object().unwrap().len(), want["kv"].as_object().unwrap().len());
    assert_eq!(got["arrays"], want["arrays"]);
}

#[test]
fn qwen3_header_matches_gguf_py() {
    check_against_gguf_py(QWEN3_PATH, "gguf_py_qwen3-0.6b-Q4_0.json", 310);
}

#[test]
fn gemma3_header_matches_gguf_py() {
    check_against_gguf_py(GEMMA3_PATH, "gguf_py_gemma-3-4b-it-qat-Q4_0.json", 444);
}

#[test]
fn synthetic_header_roundtrip() {
    let dir = TempDir::new("gguf-rt");
    let p = tiny_model("qwen3").write_to(&dir.0, "tiny.gguf", 0);
    let g = GgufFile::open(&p).unwrap();
    assert_eq!(g.version(), 3);
    assert_eq!(g.alignment(), 32);
    assert_eq!(g.data_offset() % 32, 0);
    assert_eq!(g.get_str("general.architecture").as_deref(), Some("qwen3"));
    assert_eq!(g.get_u32("qwen3.block_count"), Some(3));
    assert_eq!(g.get_f32("qwen3.rope.freq_base"), Some(1_000_000.0));
    assert_eq!(g.arr_len("tokenizer.ggml.tokens"), Some(10));
    assert_eq!(g.arr_str("tokenizer.ggml.tokens", 3).as_deref(), Some("t3"));
    assert_eq!(g.arr_str("tokenizer.ggml.tokens", 10), None);
    assert_eq!(g.get("nope"), None);
    let t = g.tensor("blk.1.ffn_down.weight").unwrap();
    assert_eq!(t.ne, [16, 8, 1, 1]);
    assert_eq!(t.type_name, "f32");
    assert_eq!(t.nbytes, 16 * 8 * 4);
    assert_eq!(t.offset % 32, 0);
}

#[test]
fn truncated_data_is_rejected() {
    let dir = TempDir::new("gguf-trunc");
    let p = tiny_model("qwen3").write_to(&dir.0, "cut.gguf", 100);
    let e = GgufFile::open(&p).err().expect("truncated file must not open");
    assert!(e.to_string().contains("extends past the end"), "{e}");
}

#[test]
fn garbage_and_missing_files_are_rejected() {
    let dir = TempDir::new("gguf-bad");
    let p = dir.0.join("junk.gguf");
    std::fs::write(&p, b"not a gguf file at all, just text").unwrap();
    assert!(matches!(GgufFile::open(&p), Err(Error::Gguf(_))));
    assert!(matches!(GgufFile::open(dir.0.join("absent.gguf")), Err(Error::Io(_))));
}
