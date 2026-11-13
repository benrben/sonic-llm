//! llama-server interface compatibility on the fake engine (no GPU): path aliases, the 501s, `/props`, `/slots`,
//! `/detokenize`, `/apply-template`, `/tokenize` pieces, `system_fingerprint` / `timings` / `usage` details, `--api-key`,
//! `--alias`, `reasoning_content`, `json_schema` / `response_format` and `logprobs`. The gap report is
//! `sllm/results/interface-2026-11-13/report-before.md`.

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::api::llama_compat::{CompatConfig, ReasoningFormat};
use sllm_server::fake::FakeConfig;
use sllm_server::tokenizer::Tokenizer;
use sllm_server::ServerConfig;

fn scripted(pieces: &[&str]) -> Srv {
    spawn(FakeConfig { script: Some(pieces.iter().map(|s| s.to_string()).collect()), ..fast() })
}

fn with_compat(fake: FakeConfig, compat: CompatConfig) -> Srv {
    spawn_with(fake, Tokenizer::Bytes, ServerConfig { port: 0, compat, ..Default::default() })
}

fn chat_body(extra: Value) -> Value {
    let mut b = json!({"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4});
    for (k, v) in extra.as_object().into_iter().flatten() {
        b[k] = v.clone();
    }
    b
}

const TIMING_KEYS: [&str; 9] = ["cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second"];

fn assert_timings(v: &Value, what: &str) {
    for k in TIMING_KEYS {
        assert!(v["timings"][k].is_number(), "{what}: timings.{k} in {v}");
    }
}

fn err_of(v: &Value) -> (i64, String, String) {
    let e = &v["error"];
    (e["code"].as_i64().unwrap(), e["type"].as_str().unwrap().to_string(), e["message"].as_str().unwrap().to_string())
}

// -- paths ---------------------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn path_aliases_have_the_same_handlers() {
    let srv = spawn(fast());
    assert_eq!(srv.get_json("/v1/health").await, json!({"status": "ok"}));
    // /completions is /completion
    let a = srv.complete(&[1, 2, 3], 4, json!({})).await;
    let r = srv.post("/completions", &json!({"prompt": [1, 2, 3], "n_predict": 4, "temperature": 0.0, "ignore_eos": true, "return_tokens": true})).await;
    assert_eq!(r.status(), 200);
    let b: Value = r.json().await.unwrap();
    assert_eq!(a["tokens"], b["tokens"]);
    // /chat/completions is /v1/chat/completions
    let r = srv.post("/chat/completions", &chat_body(json!({}))).await;
    assert_eq!(r.status(), 200);
    assert_eq!(r.json::<Value>().await.unwrap()["object"], "chat.completion");
    // /models is /v1/models
    let (m1, m2) = (srv.get_json("/v1/models").await, srv.get_json("/models").await);
    assert_eq!(m1["data"][0]["id"], m2["data"][0]["id"]);
    assert_eq!(m1.as_object().unwrap().keys().collect::<Vec<_>>(), m2.as_object().unwrap().keys().collect::<Vec<_>>());
    assert_eq!(srv.get_json("/lora-adapters").await, json!([]));
    // wrong method on an alias is 405, not a silent success
    assert_eq!(srv.get("/completions").await.status(), 405);
}

#[tokio::test(flavor = "multi_thread")]
async fn what_the_server_does_not_do_is_501_in_llama_servers_shape() {
    let srv = spawn(fast());
    for (path, what) in [
        ("/embedding", "embeddings"), ("/embeddings", "embeddings"), ("/v1/embeddings", "embeddings"),
        ("/rerank", "reranking"), ("/reranking", "reranking"), ("/v1/rerank", "reranking"), ("/v1/reranking", "reranking"),
    ] {
        let r = srv.post(path, &json!({"content": "x", "input": "x", "query": "q", "documents": ["d"]})).await;
        assert_eq!(r.status(), 501, "{path}");
        let (code, kind, msg) = err_of(&r.json().await.unwrap());
        assert_eq!((code, kind.as_str()), (501, "not_supported_error"), "{path}");
        assert!(msg.contains(what), "{path}: {msg}");
    }
    let r = srv.post("/props", &json!({})).await;
    assert_eq!(r.status(), 501);
    // the endpoints that are not implemented yet are still 404, not a fake answer
    for p in ["/infill", "/v1/responses", "/slots/0"] {
        assert_eq!(srv.post(p, &json!({})).await.status(), 404, "{p}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn props_and_slots_describe_the_server() {
    let srv = spawn(FakeConfig { n_ctx: 2048, model: "/models/m.gguf".into(), ..fast() });
    let p = srv.get_json("/props").await;
    for k in ["default_generation_settings", "total_slots", "model_alias", "model_path", "modalities", "chat_template", "bos_token", "eos_token", "build_info", "is_sleeping", "endpoint_slots"] {
        assert!(p.get(k).is_some(), "/props lacks {k}: {p}");
    }
    assert_eq!(p["model_path"], "/models/m.gguf");
    assert_eq!(p["model_alias"], "/models/m.gguf");
    assert_eq!(p["default_generation_settings"]["n_ctx"], 2048);
    let params = &p["default_generation_settings"]["params"];
    assert_eq!(params["temperature"].as_f64().unwrap() as f32, 0.8);
    for k in ["top_k", "top_p", "min_p", "seed", "max_tokens", "samplers", "repeat_penalty", "n_predict"] {
        assert!(params.get(k).is_some(), "params lacks {k}");
    }
    assert!(params.get("grammar").is_none() && params.get("stop").is_none(), "per-request fields are not default settings");
    let n = p["total_slots"].as_u64().unwrap();
    let slots = srv.get_json("/slots").await;
    let slots = slots.as_array().unwrap();
    assert_eq!(slots.len() as u64, n);
    assert_eq!(slots[0], json!({"id": 0, "n_ctx": 2048, "speculative": false, "is_processing": false}));
}

#[tokio::test(flavor = "multi_thread")]
async fn tokenize_pieces_detokenize_and_apply_template() {
    let srv = spawn(fast());
    let t: Value = srv.post("/tokenize", &json!({"content": "Hi", "with_pieces": true, "add_special": true})).await.json().await.unwrap();
    assert_eq!(t["tokens"], json!([{"id": 72, "piece": "H"}, {"id": 105, "piece": "i"}]));
    // a piece that is not valid UTF-8 on its own is the list of its bytes
    let d: Value = srv.post("/detokenize", &json!({"tokens": [72, 105]})).await.json().await.unwrap();
    assert_eq!(d, json!({"content": "Hi"}));
    assert_eq!(srv.post("/detokenize", &json!({})).await.json::<Value>().await.unwrap(), json!({"content": ""}));
    assert_eq!(srv.post("/detokenize", &json!({"tokens": "x"})).await.status(), 400);
    // plain tokenize is unchanged
    let t: Value = srv.post("/tokenize", &json!({"content": "Hi"})).await.json().await.unwrap();
    assert_eq!(t["tokens"], json!([72, 105]));

    let r: Value = srv.post("/apply-template", &json!({"messages": [{"role": "user", "content": "Say hi."}]})).await.json().await.unwrap();
    assert_eq!(r["prompt"], "<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n");
    let r = srv.post("/apply-template", &json!({})).await;
    assert_eq!(r.status(), 400);
    assert_eq!(err_of(&r.json().await.unwrap()), (400, "invalid_request_error".into(), "'messages' is required".into()));
}

#[tokio::test(flavor = "multi_thread")]
async fn models_list_has_llama_servers_fields_and_keeps_the_anthropic_ones() {
    let srv = spawn(FakeConfig { n_ctx: 512, model: "/models/m.gguf".into(), ..fast() });
    let m = srv.get_json("/v1/models").await;
    assert_eq!(m["object"], "list");
    let e = &m["data"][0];
    assert_eq!((e["object"].clone(), e["owned_by"].clone(), e["id"].clone()), (json!("model"), json!("sllm"), json!("m")));
    assert!(e["created"].as_u64().unwrap() > 1_700_000_000);
    assert_eq!(e["aliases"], json!(["/models/m.gguf"]));
    assert_eq!(e["meta"]["n_ctx"], 512);
    for k in ["vocab_type", "n_vocab", "n_ctx_train", "n_embd", "n_params", "size", "ftype"] {
        assert!(e["meta"].get(k).is_some(), "meta lacks {k}");
    }
    assert_eq!(m["models"][0]["name"], "/models/m.gguf");
    assert_eq!(m["models"][0]["details"]["format"], "gguf");
    // what Claude Code reads is still there
    assert_eq!((e["type"].clone(), e["display_name"].clone()), (json!("model"), json!("m")));
    assert_eq!((m["has_more"].clone(), m["first_id"].clone()), (json!(false), json!("m")));
}

// -- reply fields ----------------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn replies_carry_system_fingerprint_timings_and_cached_tokens() {
    let srv = spawn(fast());
    let v: Value = srv.post("/v1/chat/completions", &chat_body(json!({}))).await.json().await.unwrap();
    assert!(v["system_fingerprint"].as_str().unwrap().starts_with("sllm-"));
    assert_timings(&v, "chat");
    assert_eq!(v["usage"]["prompt_tokens_details"], json!({"cached_tokens": 0}));
    assert_eq!(v["timings"]["predicted_n"], v["usage"]["completion_tokens"]);

    let v: Value = srv.post("/v1/completions", &json!({"prompt": "hello", "max_tokens": 3})).await.json().await.unwrap();
    assert!(v["system_fingerprint"].is_string());
    assert_timings(&v, "v1 completions");
    assert_eq!(v["usage"]["prompt_tokens_details"], json!({"cached_tokens": 0}));
}

#[tokio::test(flavor = "multi_thread")]
async fn streams_stamp_every_chunk_and_put_timings_on_the_last() {
    let srv = spawn(fast());
    // chat, no usage chunk: the finish chunk is the last and carries the timings
    let (_, chunks) = srv.stream("/v1/chat/completions", &chat_body(json!({"stream": true}))).await;
    assert!(chunks.iter().all(|c| c["system_fingerprint"].is_string()), "{chunks:?}");
    let timed: Vec<usize> = chunks.iter().enumerate().filter(|(_, c)| c.get("timings").is_some()).map(|(i, _)| i).collect();
    assert_eq!(timed, vec![chunks.len() - 1]);
    assert_timings(chunks.last().unwrap(), "chat stream");
    assert_eq!(chunks.last().unwrap()["choices"][0]["finish_reason"], "length");
    // with include_usage the trailing usage chunk is the last
    let (_, chunks) = srv.stream("/v1/chat/completions", &chat_body(json!({"stream": true, "stream_options": {"include_usage": true}}))).await;
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"], json!([]));
    assert_timings(last, "usage chunk");
    assert_eq!(chunks.iter().filter(|c| c.get("timings").is_some()).count(), 1);
    assert_eq!(last["usage"]["prompt_tokens_details"], json!({"cached_tokens": 0}));
    // /v1/completions the same
    let (_, chunks) = srv.stream("/v1/completions", &json!({"prompt": "hi", "max_tokens": 3, "stream": true})).await;
    assert!(chunks.iter().all(|c| c["system_fingerprint"].is_string()));
    assert_eq!(chunks.iter().filter(|c| c.get("timings").is_some()).count(), 1);
    assert!(chunks.last().unwrap().get("timings").is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn completion_stream_chunks_carry_the_token_ids() {
    let srv = spawn(fast());
    let (_, chunks) = srv.stream("/completion", &json!({"prompt": [1, 2, 3], "n_predict": 4, "temperature": 0.0, "ignore_eos": true, "stream": true})).await;
    let ids = tokens_of(&chunks);
    assert_eq!(ids.len(), 4, "one id per token chunk, without return_tokens: {chunks:?}");
    assert!(chunks[..4].iter().all(|c| c["tokens"].as_array().unwrap().len() == 1));
    // and they are the ids the whole reply has
    let whole = srv.complete(&[1, 2, 3], 4, json!({})).await;
    assert_eq!(whole["tokens"].as_array().unwrap().iter().map(|t| t.as_i64().unwrap()).collect::<Vec<_>>(), ids);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_body_that_does_not_fit_the_request_is_400() {
    let srv = spawn(fast());
    for p in ["/v1/chat/completions", "/chat/completions"] {
        let r = srv.post(p, &json!({"max_tokens": 4})).await;
        assert_eq!(r.status(), 400, "{p}");
        assert_eq!(err_of(&r.json().await.unwrap()), (400, "invalid_request_error".into(), "'messages' is required".into()));
    }
    for (p, body) in [("/completion", json!({"prompt": [1], "n_predict": "x"})), ("/v1/completions", json!({"max_tokens": 3})), ("/tokenize", json!({}))] {
        let r = srv.post(p, &body).await;
        assert_eq!(r.status(), 400, "{p}");
        assert_eq!(err_of(&r.json().await.unwrap()).1, "invalid_request_error");
    }
}

// -- --api-key and --alias ---------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn api_keys_guard_every_route_but_the_health_checks() {
    let srv = with_compat(fast(), CompatConfig { api_keys: vec!["k1".into(), "k2".into()], ..Default::default() });
    for p in ["/health", "/v1/health"] {
        assert_eq!(srv.get(p).await.status(), 200, "{p} is public");
    }
    for p in ["/v1/models", "/props", "/metrics", "/caps", "/nope"] {
        let r = srv.get(p).await;
        assert_eq!(r.status(), 401, "{p}");
        assert_eq!(r.json::<Value>().await.unwrap(), json!({"error": {"message": "Invalid API Key", "type": "authentication_error", "code": 401}}));
    }
    let r = srv.post("/v1/chat/completions", &chat_body(json!({}))).await;
    assert_eq!(r.status(), 401);
    let get = |path: &str, header: &str, value: &str| srv.http.get(srv.url(path)).header(header, value).send();
    assert_eq!(get("/v1/models", "authorization", "Bearer k1").await.unwrap().status(), 200);
    assert_eq!(get("/v1/models", "authorization", "k2").await.unwrap().status(), 200, "the Bearer prefix is optional");
    assert_eq!(get("/v1/models", "x-api-key", "k2").await.unwrap().status(), 200, "the Anthropic header");
    assert_eq!(get("/v1/models", "authorization", "Bearer nope").await.unwrap().status(), 401);
    let ok = srv.http.post(srv.url("/v1/chat/completions")).header("authorization", "Bearer k1").json(&chat_body(json!({}))).send().await.unwrap();
    assert_eq!(ok.status(), 200);
    // no keys configured: nothing is checked
    let open = spawn(fast());
    assert_eq!(open.get("/v1/models").await.status(), 200);
}

#[tokio::test(flavor = "multi_thread")]
async fn alias_is_the_model_name_everywhere() {
    let srv = with_compat(fast(), CompatConfig { alias: Some("sonic".into()), aliases: vec!["sllm".into()], ..Default::default() });
    let m = srv.get_json("/v1/models").await;
    assert_eq!(m["data"][0]["id"], "sonic");
    assert_eq!(m["data"][0]["aliases"].as_array().unwrap().last().unwrap(), "sllm");
    assert_eq!(srv.get_json("/props").await["model_alias"], "sonic");
    // a reply to a request that names no model says it
    let mut body = chat_body(json!({}));
    body.as_object_mut().unwrap().remove("model");
    let v: Value = srv.post("/v1/chat/completions", &body).await.json().await.unwrap();
    assert_eq!(v["model"], "sonic");
    let v: Value = srv.post("/v1/completions", &json!({"prompt": "hi", "max_tokens": 2})).await.json().await.unwrap();
    assert_eq!(v["model"], "sonic");
    // a request that names one gets it back
    let v: Value = srv.post("/v1/chat/completions", &chat_body(json!({"model": "other"}))).await.json().await.unwrap();
    assert_eq!(v["model"], "other");
}

// -- reasoning_content -------------------------------------------------------------------------------------------------

async fn chat_reply(srv: &Srv, extra: Value) -> Value {
    let mut body = chat_body(extra);
    body["max_tokens"] = json!(50);
    srv.post("/v1/chat/completions", &body).await.json().await.unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn think_text_moves_to_reasoning_content_as_in_llama_server() {
    let srv = scripted(&["<think>", "a plan", "</think>", "\n\nHello"]);
    let v = chat_reply(&srv, json!({})).await;
    let msg = &v["choices"][0]["message"];
    assert_eq!((msg["content"].as_str().unwrap(), msg["reasoning_content"].as_str().unwrap()), ("Hello", "a plan"));
    // the stream says it in deltas
    let (_, chunks) = srv.stream("/v1/chat/completions", &chat_body(json!({"stream": true, "max_tokens": 50}))).await;
    let join = |key: &str| chunks.iter().filter_map(|c| c["choices"].get(0).and_then(|ch| ch["delta"][key].as_str())).collect::<String>();
    assert_eq!((join("reasoning_content").as_str(), join("content").as_str()), ("a plan", "Hello"));
    // reasoning_format none leaves the text as the model wrote it
    let v = chat_reply(&srv, json!({"reasoning_format": "none"})).await;
    let msg = &v["choices"][0]["message"];
    assert_eq!(msg["content"], "<think>a plan</think>\n\nHello");
    assert!(msg.get("reasoning_content").is_none());
    assert_eq!(chat_reply(&srv, json!({"reasoning_format": "deepseek"})).await["choices"][0]["message"]["reasoning_content"], "a plan");
    // deepseek-legacy is not implemented: refused, not ignored
    let r = srv.post("/v1/chat/completions", &chat_body(json!({"reasoning_format": "deepseek-legacy"}))).await;
    assert_eq!(r.status(), 422);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_reply_without_think_tags_is_the_same_bytes_with_or_without_the_split() {
    let on = scripted(&["Hel", "lo ", "<b>wor", "ld</b>", " < 3"]);
    let off = with_compat(FakeConfig { script: Some(["Hel", "lo ", "<b>wor", "ld</b>", " < 3"].iter().map(|s| s.to_string()).collect()), ..fast() }, CompatConfig { reasoning: ReasoningFormat::None, ..Default::default() });
    let (a, b) = (chat_reply(&on, json!({})).await, chat_reply(&off, json!({})).await);
    assert_eq!(a["choices"][0]["message"], b["choices"][0]["message"]);
    assert_eq!(a["choices"][0]["message"]["content"], "Hello <b>world</b> < 3");
    assert!(a["choices"][0]["message"].get("reasoning_content").is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn the_server_default_reasoning_format_can_be_none() {
    let srv = with_compat(FakeConfig { script: Some(vec!["<think>x</think>y".into()]), ..fast() }, CompatConfig { reasoning: ReasoningFormat::None, ..Default::default() });
    let v = chat_reply(&srv, json!({})).await;
    assert_eq!(v["choices"][0]["message"]["content"], "<think>x</think>y");
    // a request can still ask for the split
    let v = chat_reply(&srv, json!({"reasoning_format": "deepseek"})).await;
    assert_eq!(v["choices"][0]["message"]["reasoning_content"], "x");
}

// -- json_schema, response_format, grammar -------------------------------------------------------------------------------

const SCHEMA: fn() -> Value = || json!({"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]});

#[tokio::test(flavor = "multi_thread")]
async fn json_schema_on_completion_becomes_the_grammar() {
    let srv = spawn(fast());
    let r = srv.post("/completion", &json!({"prompt": [1, 2], "n_predict": 2, "json_schema": SCHEMA()})).await;
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await.unwrap();
    let g = v["generation_settings"]["grammar"].as_str().unwrap();
    assert!(g.contains("root ::= \"{\" space name-kv \",\" space age-kv space \"}\""), "{g}");
    // /v1/completions takes it too
    let r = srv.post("/v1/completions", &json!({"prompt": "hi", "max_tokens": 2, "json_schema": SCHEMA()})).await;
    assert_eq!(r.status(), 200);
    // both a grammar and a schema: 400, as llama-server
    let r = srv.post("/completion", &json!({"prompt": [1, 2], "n_predict": 2, "json_schema": SCHEMA(), "grammar": "root ::= \"a\""})).await;
    assert_eq!(r.status(), 400);
    assert!(err_of(&r.json().await.unwrap()).2.contains("both"));
    // a schema the grammar could not enforce is refused, not half applied
    for bad in [json!({"type": "string", "pattern": "^a+$"}), json!({"type": "integer", "minimum": 3}), json!({"allOf": [{"type": "object"}]})] {
        let r = srv.post("/completion", &json!({"prompt": [1, 2], "n_predict": 2, "json_schema": bad})).await;
        assert_eq!(r.status(), 400, "{bad}");
        assert!(err_of(&r.json().await.unwrap()).2.contains("JSON schema"));
    }
    // empty values are "not asked for", as before
    for empty in [json!(null), json!({}), json!(""), json!(false)] {
        let r = srv.post("/completion", &json!({"prompt": [1, 2], "n_predict": 2, "json_schema": empty})).await;
        assert_eq!(r.status(), 200, "{empty}");
    }
    // agent turns still refuse it
    let a: Value = srv.post("/v1/agents", &json!({})).await.json().await.unwrap();
    let r = srv.post(&format!("/v1/agents/{}/turn", a["id"].as_str().unwrap()), &json!({"prompt": [1, 2], "n_predict": 2, "json_schema": SCHEMA()})).await;
    assert_eq!(r.status(), 400);
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_response_format_and_json_schema() {
    let srv = spawn(fast());
    for rf in [
        json!({"type": "json_object"}),
        json!({"type": "json_schema", "json_schema": {"name": "p", "schema": SCHEMA()}}),
        json!({"type": "text"}),
    ] {
        let r = srv.post("/v1/chat/completions", &chat_body(json!({"response_format": rf}))).await;
        assert_eq!(r.status(), 200, "{rf}: {}", r.text().await.unwrap());
    }
    let r = srv.post("/v1/chat/completions", &chat_body(json!({"json_schema": SCHEMA()}))).await;
    assert_eq!(r.status(), 200);
    let r = srv.post("/v1/chat/completions", &chat_body(json!({"grammar": "root ::= \"a\""}))).await;
    assert_eq!(r.status(), 200, "a grammar on chat has its effect now, not silently ignored");
    // the refusals: an unknown type, a missing schema, a schema and a grammar, tools, a level
    for (extra, why) in [
        (json!({"response_format": {"type": "xml"}}), "type"),
        (json!({"response_format": {"type": "json_schema"}}), "schema"),
        (json!({"response_format": {"type": "json_object"}, "grammar": "root ::= \"a\""}), "both"),
        (json!({"response_format": {"type": "json_object"}, "tools": [{"type": "function", "function": {"name": "f"}}]}), "tools"),
        (json!({"response_format": {"type": "json_object"}, "level": "fast"}), "level"),
        (json!({"json_schema": {"type": "string", "pattern": "^a$"}}), "pattern"),
    ] {
        let r = srv.post("/v1/chat/completions", &chat_body(extra.clone())).await;
        assert_eq!(r.status(), 400, "{extra}");
        assert!(err_of(&r.json().await.unwrap()).2.contains(why), "{extra}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_template_kwargs_reads_only_enable_thinking() {
    let srv = spawn(fast());
    for ok in [json!({"enable_thinking": true}), json!({"enable_thinking": false}), json!({}), json!(null)] {
        let r = srv.post("/v1/chat/completions", &chat_body(json!({"chat_template_kwargs": ok}))).await;
        assert_eq!(r.status(), 200, "{ok}");
    }
    for (bad, why) in [(json!({"x": 1}), "`x`"), (json!({"enable_thinking": "yes"}), "boolean"), (json!(5), "object")] {
        let r = srv.post("/v1/chat/completions", &chat_body(json!({"chat_template_kwargs": bad}))).await;
        assert_eq!(r.status(), 400, "{bad}");
        assert!(err_of(&r.json().await.unwrap()).2.contains(why), "{bad}");
    }
    assert_eq!(srv.post("/v1/chat/completions", &chat_body(json!({"chat_template_kwargs": {"enable_thinking": true}, "level": "fast"}))).await.status(), 400);
    // the switch reaches the model's template: Qwen3 closes an empty think block itself unless thinking is on
    let Some(gguf) = qwen_gguf() else { return };
    let q = spawn_with(fast(), Tokenizer::gguf(gguf), ServerConfig { port: 0, ..Default::default() });
    let msgs = json!([{"role": "user", "content": "hi"}]);
    let off: Value = q.post("/apply-template", &json!({"messages": msgs})).await.json().await.unwrap();
    let on: Value = q.post("/apply-template", &json!({"messages": msgs, "chat_template_kwargs": {"enable_thinking": true}})).await.json().await.unwrap();
    assert!(off["prompt"].as_str().unwrap().ends_with("<think>\n\n</think>\n\n"), "{off}");
    assert!(on["prompt"].as_str().unwrap().ends_with("<|im_start|>assistant\n"), "{on}");
}

// -- logprobs --------------------------------------------------------------------------------------------------------

fn with_probs() -> Srv {
    spawn(FakeConfig { natural_len: Some(5), token_probs: Some((0.5, 0.25)), ..fast() })
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_logprobs_come_from_the_probability_path() {
    let srv = with_probs();
    let v: Value = srv.post("/v1/chat/completions", &chat_body(json!({"max_tokens": 3, "logprobs": true, "top_logprobs": 2}))).await.json().await.unwrap();
    let content = v["choices"][0]["logprobs"]["content"].as_array().unwrap();
    assert_eq!(content.len(), 3, "one entry per generated token");
    for e in content {
        assert!(e["token"].is_string() && e["id"].is_i64() && e["bytes"].is_array());
        assert!((e["logprob"].as_f64().unwrap() - 0.5f64.ln()).abs() < 1e-6);
        let top = e["top_logprobs"].as_array().unwrap();
        assert_eq!(top.len(), 2);
        assert!(top[0]["logprob"].as_f64().unwrap() >= top[1]["logprob"].as_f64().unwrap());
    }
    // top_logprobs 0 or absent: the chosen token's alone
    let v: Value = srv.post("/v1/chat/completions", &chat_body(json!({"max_tokens": 2, "logprobs": true}))).await.json().await.unwrap();
    assert_eq!(v["choices"][0]["logprobs"]["content"][0]["top_logprobs"], json!([]));
    // not asked for: no logprobs object
    let v: Value = srv.post("/v1/chat/completions", &chat_body(json!({"max_tokens": 2, "logprobs": false}))).await.json().await.unwrap();
    assert!(v["choices"][0].get("logprobs").is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn logprobs_that_cannot_be_served_are_refused() {
    let srv = with_probs();
    for (extra, why) in [
        (json!({"logprobs": true, "top_logprobs": 3}), "top_logprobs"),
        (json!({"logprobs": true, "stream": true}), "stream"),
        (json!({"top_logprobs": 1}), "top_logprobs"),
        (json!({"logprobs": 5}), "logprobs"),
    ] {
        let r = srv.post("/v1/chat/completions", &chat_body(extra.clone())).await;
        assert_eq!(r.status(), 400, "{extra}");
        assert!(err_of(&r.json().await.unwrap()).2.contains(why), "{extra}");
    }
    // /v1/completions: logprobs 3 is more than the engine serves; with stream it is refused too
    assert_eq!(srv.post("/v1/completions", &json!({"prompt": "hi", "max_tokens": 2, "logprobs": 3})).await.status(), 400);
    assert_eq!(srv.post("/v1/completions", &json!({"prompt": "hi", "max_tokens": 2, "logprobs": 1, "stream": true})).await.status(), 400);
    // logprobs under a level is a conflict
    assert_eq!(srv.post("/v1/chat/completions", &chat_body(json!({"logprobs": true, "level": "fast"}))).await.status(), 400);
}

#[tokio::test(flavor = "multi_thread")]
async fn text_completion_logprobs() {
    let srv = with_probs();
    let v: Value = srv.post("/v1/completions", &json!({"prompt": "hi", "max_tokens": 3, "logprobs": 2})).await.json().await.unwrap();
    let content = v["choices"][0]["logprobs"]["content"].as_array().unwrap();
    assert_eq!(content.len(), 3);
    assert_eq!(content[0]["top_logprobs"].as_array().unwrap().len(), 2);
    // without it the field stays null, as before
    let v: Value = srv.post("/v1/completions", &json!({"prompt": "hi", "max_tokens": 3})).await.json().await.unwrap();
    assert!(v["choices"][0]["logprobs"].is_null());
}

// -- the command line ------------------------------------------------------------------------------------------------

mod cli {
    use serde_json::{json, Value};
    use std::io::{BufRead, BufReader, Read};
    use std::process::{Command, Stdio};

    const BIN: &str = env!("CARGO_BIN_EXE_sllm-server");

    fn run(args: &[&str]) -> (Option<i32>, String, String) {
        let out = Command::new(BIN).args(args).output().unwrap();
        (out.status.code(), String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned())
    }

    #[test]
    fn help_and_version_say_sonic_llm() {
        let (code, out, _) = run(&["--version"]);
        assert_eq!(code, Some(0));
        assert!(out.starts_with(&format!("Sonic LLM server (sllm-server) {} (commit ", env!("CARGO_PKG_VERSION"))), "{out}");
        let (code, out, _) = run(&["--help"]);
        assert_eq!(code, Some(0));
        assert!(out.starts_with("Sonic LLM server (sllm-server)\n"), "{out}");
        assert!(out.contains("-np/--parallel") && out.contains("--api-key"), "the help lists the llama-server spellings");
    }

    #[test]
    fn llama_server_spellings_parse() {
        // everything before --version is read, so a bad flag or a missing value would exit 2 first
        let (code, _, err) = run(&["-m", "/x.gguf", "-c", "512", "--ctx-size", "256", "-t", "4", "-np", "2", "--parallel", "3", "-ub", "128", "--ubatch-size", "64", "--host", "127.0.0.1", "--port", "9", "-a", "n,m", "--api-key", "a,b", "--reasoning-format", "none", "--version"]);
        assert_eq!(code, Some(0), "{err}");
        for bad in [vec!["--reasoning-format", "xml", "--version"], vec!["-np", "x", "--version"], vec!["--api-key"]] {
            assert_eq!(run(&bad).0, Some(2), "{bad:?}");
        }
    }

    #[test]
    fn flags_with_no_meaning_here_are_accepted_with_a_note() {
        let (code, _, err) = run(&["-ngl", "99", "--n-gpu-layers", "10", "--jinja", "-fa", "on", "-fa", "--mlock", "--no-mmap", "-b", "2048", "-kvu", "--slots", "--metrics", "-cb", "--version"]);
        assert_eq!(code, Some(0), "{err}");
        for flag in ["-ngl", "--jinja", "-fa", "--mlock", "--no-mmap", "-b", "-kvu", "--slots", "--metrics", "-cb"] {
            assert!(err.contains(&format!("{flag} is accepted and ignored")), "{flag}: {err}");
        }
        // `-fa on` took its value: `on` was not read as a flag (that would be an unknown argument); `-ngl` needs one
        assert_eq!(run(&["-ngl"]).0, Some(2));
        // any other flag is still an error
        let (code, _, err) = run(&["--not-a-flag", "--version"]);
        assert_eq!((code, err.contains("unknown argument")), (Some(2), true));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn alias_and_api_key_work_end_to_end() {
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port().to_string();
        let mut child = Command::new(BIN)
            .args(["--fake", "--host", "127.0.0.1", "--port", &port, "-a", "sonic", "--api-key", "sekret", "-c", "512", "-np", "2", "--jinja", "-fa", "on", "-ngl", "99"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut out = BufReader::new(child.stdout.take().unwrap());
        let mut line = String::new();
        out.read_line(&mut line).unwrap(); // the version line
        line.clear();
        out.read_line(&mut line).unwrap(); // the banner: the engine is up
        assert!(line.starts_with("Sonic LLM server: engine "), "{line}");
        let c = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        assert_eq!(c.get(format!("{base}/health")).send().await.unwrap().status(), 200);
        assert_eq!(c.get(format!("{base}/v1/models")).send().await.unwrap().status(), 401);
        let m: Value = c.get(format!("{base}/v1/models")).bearer_auth("sekret").send().await.unwrap().json().await.unwrap();
        assert_eq!(m["data"][0]["id"], "sonic");
        let v: Value = c.post(format!("{base}/v1/chat/completions")).bearer_auth("sekret").json(&json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2})).send().await.unwrap().json().await.unwrap();
        assert_eq!(v["model"], "sonic");
        child.kill().unwrap();
        child.wait().unwrap();
        let mut err = String::new();
        child.stderr.take().unwrap().read_to_string(&mut err).unwrap();
        assert!(err.contains("-np/--parallel 2 is --agents 2, which sizes the cache only with --auto-mode"), "{err}");
        for flag in ["--jinja", "-fa", "-ngl"] {
            assert!(err.contains(&format!("{flag} is accepted and ignored")), "{err}");
        }
    }
}
