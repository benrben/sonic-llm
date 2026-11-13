//! Integration tests against the real Qwen3-0.6B GGUF. They skip (and say so) when the GGUF is not on this machine.
//! The GPU test loads the model on Metal: run `cargo test` under `engine/scripts/with_gpu_lock.py`.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn qwen() -> Option<PathBuf> {
    let candidates = [
        "/Users/benreich/Projects/crazyidea/research/inference_runtime/converted/qwen3-0.6b-Q4_0.gguf",
        "../../research/inference_runtime/converted/qwen3-0.6b-Q4_0.gguf",
    ];
    let p = candidates.iter().map(PathBuf::from).find(|p| p.is_file());
    if p.is_none() {
        eprintln!("SKIP: qwen3-0.6b-Q4_0.gguf not found");
    }
    p
}

fn fixture_ids(name: &str, take: usize) -> Vec<i32> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures").join(name);
    let lists = llama_ref::inputs::read_id_lists(&p).unwrap();
    lists[0][..take].to_vec()
}

#[test]
fn tokenizer_matches_fixture_and_handles_special_tokens() {
    let _g = ONE_AT_A_TIME.lock().unwrap();
    let Some(gguf) = qwen() else { return };
    // Qwen3 adds no BOS; the chat markers are one token each only when special parsing is on.
    let plain = llama_ref::tokenize(&gguf, "hello world", true, false).unwrap();
    assert_eq!(plain, llama_ref::tokenize(&gguf, "hello world", false, false).unwrap());
    let parsed = llama_ref::tokenize(&gguf, "<|im_start|>user", false, true).unwrap();
    assert_eq!(parsed[0], 151644);
    let unparsed = llama_ref::tokenize(&gguf, "<|im_start|>user", false, false).unwrap();
    assert!(unparsed.len() > parsed.len());
    // The first WikiText line, as the fixture was built.
    let fx = fixture_ids("wikitext2k-qwen3-0.6b-q4.json", 4);
    let t = llama_ref::tokenize(&gguf, " = Robert Boulter = \n", true, false).unwrap();
    assert_eq!(&t[..4.min(t.len())], &fx[..4.min(t.len())]);
    let (n_vocab, _bos, adds_bos) = llama_ref::vocab_info(&gguf).unwrap();
    assert_eq!(n_vocab, 151936);
    assert!(!adds_bos);
}

#[test]
fn logits_are_deterministic_and_agree_with_greedy() {
    if std::env::var_os("LLAMA_REF_SKIP_GPU").is_some() {
        return;
    }
    let _g = ONE_AT_A_TIME.lock().unwrap();
    let Some(gguf) = qwen() else { return };
    let ids = fixture_ids("wikitext2k-qwen3-0.6b-q4.json", 600); // two ubatch chunks (512 + 88)
    let a = llama_ref::reference_logits(&gguf, &ids, 768, 512).unwrap();
    let b = llama_ref::reference_logits(&gguf, &ids, 768, 512).unwrap();
    assert_eq!((a.n_tokens, a.n_vocab), (600, 151936));
    assert_eq!(a.data.len(), 600 * 151936);
    assert!(a.data.iter().all(|v| v.is_finite()));
    assert!(a.data == b.data, "two runs differ");
    assert_eq!(a.settings.n_ubatch, 512);
    assert_eq!(a.settings.flash_attn, "enabled");

    // Greedy from the same prompt: first token is the argmax of the last logits row, and the run repeats.
    let g1 = llama_ref::reference_greedy(&gguf, &ids, 6).unwrap();
    let g2 = llama_ref::reference_greedy(&gguf, &ids, 6).unwrap();
    assert_eq!(g1, g2);
    assert_eq!(g1.ids.len(), 6);
    let last = &a.data[599 * a.n_vocab..];
    let (mut best, mut bi) = (f32::MIN, 0usize);
    for (i, &v) in last.iter().enumerate() {
        if v > best {
            best = v;
            bi = i;
        }
    }
    assert_eq!(g1.ids[0], bi as i32);
    assert_eq!(g1.top2[0].id1, bi as i32);
    assert!(g1.top2[0].logit1 >= g1.top2[0].logit2);
}
