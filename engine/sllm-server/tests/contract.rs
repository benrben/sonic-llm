//! Contract tests: `sllm-server --fake` against a REAL llama-server b11669 `/completion` capture
//! (`fixtures/capture_llama_fixture.py`: qwen3-0.6b-Q4_0, `prompt` = token ids, `return_tokens`, `n_predict` 8, ignore_eos).
//!
//! What must match: the JSON keys of every chunk (recursively through objects), the JSON type of every value, the SSE
//! framing (`data: <json>\n\n`, no `[DONE]`, no other fields), the content type, and the invariants of the numbers
//! (counts, `tokens_cached`, timings consistency). Token ids and texts are the fake's own.
//! Known, documented divergences: the error STATUS for a malformed body (llama 400, here 422 when the JSON parses), and
//! `id_slot` of the final chunk.

mod common;
use common::*;
use serde_json::{json, Value};
use std::collections::BTreeSet;

fn fixture_meta() -> Value {
    serde_json::from_slice(&std::fs::read(fixtures_dir().join("llama_capture_meta.json")).expect("meta fixture")).unwrap()
}

fn fixture_stream() -> String {
    String::from_utf8(std::fs::read(fixtures_dir().join("llama_completion_stream.sse")).expect("sse fixture")).unwrap()
}

fn fixture_json() -> Value {
    serde_json::from_slice(&std::fs::read(fixtures_dir().join("llama_completion_nonstream.json")).unwrap()).unwrap()
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Every key path (objects only; arrays are leaves) with the JSON type found there.
fn shape(v: &Value, prefix: &str, out: &mut BTreeSet<(String, &'static str)>) {
    if let Value::Object(m) = v {
        for (k, x) in m {
            let p = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
            // null in the capture means "not applicable"; compare presence only for those
            out.insert((p.clone(), kind(x)));
            shape(x, &p, out);
        }
    }
}

fn shape_of(v: &Value) -> BTreeSet<(String, &'static str)> {
    let mut s = BTreeSet::new();
    shape(v, "", &mut s);
    s
}

fn assert_same_shape(what: &str, ours: &Value, theirs: &Value) {
    let (a, b) = (shape_of(ours), shape_of(theirs));
    // Numbers are numbers whether llama printed 0.0 or 0; compare keys exactly and types except null/number
    let keys = |s: &BTreeSet<(String, &str)>| s.iter().map(|(k, _)| k.clone()).collect::<BTreeSet<_>>();
    let (ka, kb) = (keys(&a), keys(&b));
    let missing: Vec<_> = kb.difference(&ka).collect();
    let extra: Vec<_> = ka.difference(&kb).collect();
    assert!(missing.is_empty() && extra.is_empty(), "{what}: keys differ from llama-server. missing here: {missing:?}; extra here: {extra:?}");
    let ta: Vec<_> = a.difference(&b).filter(|(k, _)| !b.iter().any(|(k2, t2)| k2 == k && *t2 == "null")).collect();
    assert!(ta.is_empty(), "{what}: value types differ from llama-server: ours-only {ta:?}");
}

fn request_ids(meta: &Value) -> Vec<i32> {
    meta["request"]["prompt"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect()
}

#[test]
fn fixture_is_what_the_contract_assumes() {
    // A guard on the capture itself: if someone re-captures with other settings the tests below must be revisited.
    let meta = fixture_meta();
    assert_eq!(meta["request"]["n_predict"], 8);
    assert_eq!(meta["request"]["stream"], true);
    assert_eq!(meta["request"]["return_tokens"], true);
    assert!(meta["model"].as_str().unwrap().ends_with("qwen3-0.6b-Q4_0.gguf"));
    let raw = fixture_stream();
    let events = sse_events(&raw);
    assert_eq!(events.len(), 9, "8 token chunks and the final chunk");
    assert!(!raw.contains("[DONE]"));
}

#[tokio::test(flavor = "multi_thread")]
async fn sse_framing_and_chunk_keys_match_llama_server() {
    let meta = fixture_meta();
    let ids = request_ids(&meta);
    let llama_raw = fixture_stream();
    let llama = sse_json(&llama_raw);
    let srv = spawn(fast());
    let body = json!({"prompt": ids, "n_predict": 8, "temperature": 0.0, "stream": true, "cache_prompt": false,
                      "ignore_eos": true, "return_tokens": true, "return_progress": false});
    let r = srv.post("/completion", &body).await;
    assert_eq!(r.status(), 200);
    let ct = r.headers()["content-type"].to_str().unwrap().to_string();
    assert!(ct.starts_with("text/event-stream"), "{ct}");
    let raw = r.text().await.unwrap();

    // framing: byte-for-byte the same grammar as the capture
    assert!(raw.starts_with("data: {") && raw.ends_with("}\n\n") && !raw.contains("[DONE]"));
    assert_eq!(raw.matches("data: ").count(), llama_raw.matches("data: ").count(), "number of events");
    for (ours, theirs) in sse_events(&raw).iter().zip(sse_events(&llama_raw).iter()) {
        assert_eq!(ours.starts_with("{\"") && ours.ends_with('}'), theirs.starts_with("{\"") && theirs.ends_with('}'));
    }

    let chunks = sse_json(&raw);
    assert_eq!(chunks.len(), llama.len());
    for i in 0..8 {
        assert_same_shape(&format!("token chunk {i}"), &chunks[i], &llama[i]);
    }
    assert_same_shape("final chunk", &chunks[8], &llama[8]);

    // the numbers follow the same rules as llama-server's
    for (i, c) in chunks[..8].iter().enumerate() {
        assert_eq!(c["index"], 0);
        assert_eq!(c["stop"], false);
        assert_eq!(c["id_slot"], llama[i]["id_slot"], "id_slot of a token chunk");
        assert_eq!(c["tokens"].as_array().unwrap().len(), 1);
        assert_eq!(c["tokens_predicted"], i + 1);
        assert_eq!(c["tokens_evaluated"], llama[i]["tokens_evaluated"]);
        assert_eq!(c["tokens_evaluated"], ids.len());
    }
    let (f, lf) = (&chunks[8], &llama[8]);
    for k in ["index", "content", "stop", "stop_type", "stopping_word", "truncated", "has_new_line", "tokens_predicted",
              "tokens_evaluated", "tokens_cached", "prompt"] {
        // `prompt` differs by design (llama echoes the detokenized prompt; we send "")
        if k != "prompt" {
            assert_eq!(f[k], lf[k], "final chunk field {k}");
        }
    }
    assert_eq!(f["tokens"], json!([]));
    assert_eq!(f["id_slot"], lf["id_slot"]);
    assert_eq!(f["stop_type"], "limit");
    assert_eq!(f["tokens_cached"], 19, "12 prompt + 8 predicted - 1");
    for k in ["cache_n", "prompt_n", "predicted_n"] {
        assert_eq!(f["timings"][k], lf["timings"][k], "timings.{k}");
    }
    let t = &f["timings"];
    let (pn, pms, gn, gms) = (t["prompt_n"].as_f64().unwrap(), t["prompt_ms"].as_f64().unwrap(), t["predicted_n"].as_f64().unwrap(), t["predicted_ms"].as_f64().unwrap());
    assert!(pms >= 0.0 && gms >= 0.0);
    assert!((t["prompt_per_token_ms"].as_f64().unwrap() - pms / pn).abs() < 1e-9);
    assert!((t["predicted_per_token_ms"].as_f64().unwrap() - gms / gn).abs() < 1e-9);
    if gms > 0.0 {
        assert!((t["predicted_per_second"].as_f64().unwrap() - gn * 1000.0 / gms).abs() < 1e-6 * gn * 1000.0 / gms);
    }
    // the generation settings echo what was asked
    assert_eq!(f["generation_settings"]["n_predict"], 8);
    assert_eq!(f["generation_settings"]["ignore_eos"], true);
    assert_eq!(f["generation_settings"]["stream"], true);
}

#[tokio::test(flavor = "multi_thread")]
async fn non_stream_response_keys_match_llama_server() {
    let meta = fixture_meta();
    let ids = request_ids(&meta);
    let llama = fixture_json();
    let srv = spawn(fast());
    let body = json!({"prompt": ids, "n_predict": 8, "temperature": 0.0, "stream": false, "cache_prompt": false,
                      "ignore_eos": true, "return_tokens": true, "return_progress": false});
    let r = srv.post("/completion", &body).await;
    assert_eq!(r.status(), 200);
    assert!(r.headers()["content-type"].to_str().unwrap().starts_with("application/json"));
    let v: Value = r.json().await.unwrap();
    assert_same_shape("non-stream response", &v, &llama);
    assert_eq!(v["tokens"].as_array().unwrap().len(), 8);
    assert_eq!(v["tokens_predicted"], llama["tokens_predicted"]);
    assert_eq!(v["tokens_evaluated"], llama["tokens_evaluated"]);
    assert_eq!(v["tokens_cached"], llama["tokens_cached"]);
    assert_eq!(v["stop_type"], llama["stop_type"]);
    assert_eq!(v["id_slot"], llama["id_slot"]);
    assert_eq!(v["generation_settings"]["stream"], false);
    // content is the concatenation of the token texts
    assert!(v["content"].as_str().unwrap().len() > 8);
}

#[tokio::test(flavor = "multi_thread")]
async fn streamed_and_non_streamed_agree_token_for_token() {
    let ids: Vec<i32> = (100..112).collect();
    let srv = spawn(fast());
    let (_, chunks) = srv.stream("/completion", &json!({"prompt": ids, "n_predict": 8, "stream": true, "return_tokens": true, "ignore_eos": true})).await;
    let v = srv.complete(&ids, 8, json!({})).await;
    assert_eq!(tokens_of(&chunks), v["tokens"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect::<Vec<_>>());
    let streamed: String = chunks.iter().map(|c| c["content"].as_str().unwrap()).collect();
    assert_eq!(streamed, v["content"].as_str().unwrap());
}

#[tokio::test(flavor = "multi_thread")]
async fn health_and_error_bodies_match_llama_server() {
    let meta = fixture_meta();
    let srv = spawn(fast());
    let h = srv.get("/health").await;
    assert_eq!(h.status(), 200);
    let hv: Value = h.json().await.unwrap();
    assert_eq!(hv, serde_json::from_str::<Value>(meta["health_body"].as_str().unwrap()).unwrap());

    // unknown path: same status and body keys
    let ours = srv.get("/nope").await;
    assert_eq!(ours.status(), 404);
    let ov: Value = ours.json().await.unwrap();
    let theirs: Value = serde_json::from_str(meta["errors"]["unknown_path"]["body"].as_str().unwrap()).unwrap();
    assert_eq!(meta["errors"]["unknown_path"]["status"], 404);
    assert_same_shape("404 body", &ov, &theirs);
    assert_eq!(ov["error"]["code"], 404);
    assert_eq!(ov["error"]["type"], theirs["error"]["type"]);

    // missing prompt / wrong type: llama answers 400, and so does this server (a body that does not fit the request), body keys equal
    for (name, body) in [("missing_prompt", json!({"n_predict": 4})), ("bad_n_predict_type", json!({"prompt": [1, 2], "n_predict": "x"}))] {
        let r = srv.post("/completion", &body).await;
        let status = r.status().as_u16();
        let v: Value = r.json().await.unwrap();
        let llama: Value = serde_json::from_str(meta["errors"][name]["body"].as_str().unwrap()).unwrap();
        assert_eq!(meta["errors"][name]["status"], 400);
        assert_eq!(status, 400, "{name}");
        assert_same_shape(&format!("{name} body"), &v, &llama);
        assert_eq!(v["error"]["code"], 400);
        assert_eq!(v["error"]["type"], llama["error"]["type"], "{name}: error type");
    }
}
