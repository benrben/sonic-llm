//! `POST /v1/chat/completions` and stop strings, on the fake engine with a scripted output (no GPU): the OpenAI wire,
//! tool calls parsed out of the text (whole, one character per token, two in a reply, text before, tag-like text that is
//! not a call, malformed JSON), stop strings across token boundaries on all three text routes, the refusals, and
//! `render_chat` with exact expected prompts.

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::fake::FakeConfig;

fn scripted(pieces: &[&str]) -> Srv {
    spawn(FakeConfig { script: Some(pieces.iter().map(|s| s.to_string()).collect()), ..fast() })
}

fn chars(s: &str) -> Vec<String> {
    s.chars().map(|c| c.to_string()).collect()
}

fn scripted_chars(s: &str) -> Srv {
    spawn(FakeConfig { script: Some(chars(s)), ..fast() })
}

fn tools() -> Value {
    json!([{"type": "function", "function": {"name": "get_weather", "description": "Weather of a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}])
}

fn chat_body(extra: Value) -> Value {
    let mut b = json!({"model": "fake", "messages": [{"role": "user", "content": "hi"}]});
    for (k, v) in extra.as_object().into_iter().flatten() {
        b[k] = v.clone();
    }
    b
}

/// What a chat reply says, read the same way from the whole reply and from the stream.
#[derive(Debug, PartialEq)]
struct Reply {
    content: String,
    calls: Vec<(String, Value)>,
    finish: String,
}

struct Full {
    reply: Reply,
    usage: Value,
    /// Every non-empty content delta of a stream (empty for a non-stream reply).
    deltas: Vec<String>,
}

fn call_of(c: &Value) -> (String, Value) {
    assert_eq!(c["type"], "function");
    assert!(c["id"].as_str().is_some_and(|i| i.starts_with("call_")), "{c}");
    let args = c["function"]["arguments"].as_str().expect("arguments is a JSON string");
    (c["function"]["name"].as_str().unwrap().to_string(), serde_json::from_str(args).expect("arguments parse"))
}

async fn whole(srv: &Srv, body: &Value) -> Full {
    let mut b = body.clone();
    b["stream"] = json!(false);
    let r = srv.post("/v1/chat/completions", &b).await;
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["object"], "chat.completion");
    let ch = &v["choices"][0];
    assert_eq!((ch["index"].clone(), ch["message"]["role"].clone()), (json!(0), json!("assistant")));
    let calls: Vec<(String, Value)> = ch["message"]["tool_calls"].as_array().map(|a| a.iter().map(call_of).collect()).unwrap_or_default();
    let content = match &ch["message"]["content"] {
        Value::Null => String::new(),
        c => c.as_str().unwrap().to_string(),
    };
    if !calls.is_empty() && content.is_empty() {
        assert!(ch["message"]["content"].is_null(), "a reply that is only tool calls has content null");
    }
    Full { reply: Reply { content, calls, finish: ch["finish_reason"].as_str().unwrap().to_string() }, usage: v["usage"].clone(), deltas: vec![] }
}

async fn streamed(srv: &Srv, body: &Value) -> Full {
    let mut b = body.clone();
    b["stream"] = json!(true);
    let r = srv.post("/v1/chat/completions", &b).await;
    assert_eq!(r.status(), 200);
    assert!(r.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));
    let raw = r.text().await.unwrap();
    let events = sse_events(&raw);
    assert_eq!(events.last().unwrap(), "[DONE]");
    let chunks = sse_json(&raw);
    assert!(chunks.iter().all(|c| c["object"] == "chat.completion.chunk" && c["id"] == chunks[0]["id"]));
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant", "the first chunk carries the role");
    let (mut content, mut deltas, mut calls, mut finish, mut usage) = (String::new(), vec![], vec![], None, Value::Null);
    for c in &chunks {
        if c["choices"].as_array().unwrap().is_empty() {
            continue; // the trailing usage chunk of stream_options.include_usage
        }
        let ch = &c["choices"][0];
        assert!(finish.is_none(), "nothing follows the chunk with finish_reason: {c}");
        if let Some(t) = ch["delta"]["content"].as_str().filter(|t| !t.is_empty()) {
            content.push_str(t);
            deltas.push(t.to_string());
        }
        for tc in ch["delta"]["tool_calls"].as_array().into_iter().flatten() {
            assert_eq!(tc["index"], calls.len(), "tool call deltas come in order");
            calls.push(call_of(tc));
        }
        if let Some(f) = ch["finish_reason"].as_str() {
            finish = Some(f.to_string());
            usage = c["usage"].clone();
        }
    }
    Full { reply: Reply { content, calls, finish: finish.expect("a chunk with finish_reason") }, usage, deltas }
}

/// Non-stream and stream must say the same thing; returns it.
async fn both(srv: &Srv, body: &Value) -> Full {
    let (w, s) = (whole(srv, body).await, streamed(srv, body).await);
    assert_eq!(w.reply, s.reply, "stream and non-stream differ");
    assert_eq!(w.usage, s.usage, "usage differs");
    s
}

const CALL: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

// -- the wire -------------------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn stream_and_non_stream_give_the_same_text_and_finish_reason() {
    let srv = scripted(&["Hel", "lo ", "wor", "ld"]);
    let f = both(&srv, &chat_body(json!({}))).await;
    assert_eq!(f.reply, Reply { content: "Hello world".into(), calls: vec![], finish: "stop".into() });
    assert_eq!(f.deltas, ["Hel", "lo ", "wor", "ld"], "the stream deltas concatenate to the non-stream content");
    // the limit ends both with "length", at the same text
    let f = both(&srv, &chat_body(json!({"max_tokens": 2}))).await;
    assert_eq!(f.reply, Reply { content: "Hello ".into(), calls: vec![], finish: "length".into() });
    // max_completion_tokens is the newer name of max_tokens
    let f = both(&srv, &chat_body(json!({"max_completion_tokens": 1}))).await;
    assert_eq!((f.reply.content.as_str(), f.reply.finish.as_str()), ("Hel", "length"));
}

#[tokio::test(flavor = "multi_thread")]
async fn usage_counts_prompt_completion_and_cached_tokens() {
    let srv = scripted(&["a", "b", "c"]);
    let f = both(&srv, &chat_body(json!({}))).await;
    let prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
    // the test server has the byte tokenizer: one token per byte of the rendered prompt
    assert_eq!(f.usage["prompt_tokens"], prompt.len());
    assert_eq!(f.usage["completion_tokens"], 3);
    assert_eq!(f.usage["total_tokens"], prompt.len() + 3);
    assert!(f.usage["prompt_tokens_details"]["cached_tokens"].is_u64(), "{}", f.usage);
    // stream_options.include_usage adds OpenAI's trailing chunk with empty choices
    let r = srv.post("/v1/chat/completions", &chat_body(json!({"stream": true, "stream_options": {"include_usage": true}}))).await;
    let chunks = sse_json(&r.text().await.unwrap());
    let last = chunks.last().unwrap();
    assert_eq!((last["choices"].clone(), last["usage"]["completion_tokens"].clone()), (json!([]), json!(3)));
}

#[tokio::test(flavor = "multi_thread")]
async fn tools_are_rendered_into_the_prompt() {
    let srv = scripted(&["ok"]);
    let without = whole(&srv, &chat_body(json!({}))).await.usage["prompt_tokens"].as_u64().unwrap();
    let with = whole(&srv, &chat_body(json!({"tools": tools()}))).await.usage["prompt_tokens"].as_u64().unwrap();
    let none = whole(&srv, &chat_body(json!({"tools": tools(), "tool_choice": "none"}))).await.usage["prompt_tokens"].as_u64().unwrap();
    let auto = whole(&srv, &chat_body(json!({"tools": tools(), "tool_choice": "auto"}))).await.usage["prompt_tokens"].as_u64().unwrap();
    assert!(with > without + 100, "the tool definition is in the prompt: {with} vs {without}");
    assert_eq!(auto, with);
    assert_eq!(none, without, "tool_choice none: the tools are not shown to the model");
}

// -- tool calls -------------------------------------------------------------------------------------------------------

fn paris() -> (String, Value) {
    ("get_weather".into(), json!({"city": "Paris"}))
}

#[tokio::test(flavor = "multi_thread")]
async fn a_tool_call_whole() {
    let srv = scripted(&[CALL]);
    let f = both(&srv, &chat_body(json!({"tools": tools()}))).await;
    assert_eq!(f.reply, Reply { content: String::new(), calls: vec![paris()], finish: "tool_calls".into() });
    assert!(f.deltas.is_empty(), "no content: {:?}", f.deltas);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_tool_call_split_over_many_tokens_shows_no_tag_text() {
    let srv = scripted_chars(CALL);
    let f = both(&srv, &chat_body(json!({"tools": tools()}))).await;
    assert_eq!(f.reply, Reply { content: String::new(), calls: vec![paris()], finish: "tool_calls".into() });
    assert!(f.deltas.is_empty(), "text that may start a tool tag is held back, never shown: {:?}", f.deltas);
    // the call arrives once, whole: one chunk with the name and the complete arguments
    let r = srv.post("/v1/chat/completions", &chat_body(json!({"tools": tools(), "stream": true}))).await;
    let with_calls = sse_json(&r.text().await.unwrap()).into_iter().filter(|c| c["choices"][0]["delta"]["tool_calls"].is_array()).count();
    assert_eq!(with_calls, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn two_tool_calls_in_one_reply() {
    let second = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}\n</tool_call>";
    let text = format!("{CALL}\n{second}");
    for srv in [scripted(&[&text]), scripted_chars(&text)] {
        let f = both(&srv, &chat_body(json!({"tools": tools()}))).await;
        assert_eq!(f.reply.calls, vec![paris(), ("get_weather".into(), json!({"city": "Rome"}))]);
        assert_eq!((f.reply.content.as_str(), f.reply.finish.as_str()), ("", "tool_calls"), "the newline between two calls is not content");
        let ids: Vec<String> = whole(&srv, &chat_body(json!({"tools": tools()}))).await.reply.calls.iter().map(|c| c.0.clone()).collect();
        assert_eq!(ids.len(), 2);
    }
    let r = srv_ids(&scripted(&[&text])).await;
    assert_ne!(r[0], r[1], "each call has its own id");
}

async fn srv_ids(srv: &Srv) -> Vec<String> {
    let v: Value = srv.post("/v1/chat/completions", &chat_body(json!({"tools": tools()}))).await.json().await.unwrap();
    v["choices"][0]["message"]["tool_calls"].as_array().unwrap().iter().map(|c| c["id"].as_str().unwrap().to_string()).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn text_before_a_tool_call_is_content() {
    let text = format!("Let me check.\n{CALL}");
    for srv in [scripted(&[&text]), scripted_chars(&text)] {
        let f = both(&srv, &chat_body(json!({"tools": tools()}))).await;
        assert_eq!(f.reply, Reply { content: "Let me check.\n".into(), calls: vec![paris()], finish: "tool_calls".into() });
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn tag_like_text_that_is_not_a_tool_call_is_content() {
    let srv = scripted(&["<tool", "kit"]);
    let f = both(&srv, &chat_body(json!({"tools": tools()}))).await;
    assert_eq!(f.reply, Reply { content: "<toolkit".into(), calls: vec![], finish: "stop".into() });
    assert_eq!(f.deltas, ["<toolkit"], "\"<tool\" is held back until it is clear what it is");
    // ... and the same one character per token, with a lone "<" and a text that ends on a tag start
    for text in ["a < b and <tool_x>", "ends with <tool_ca"] {
        let f = both(&scripted_chars(text), &chat_body(json!({"tools": tools()}))).await;
        assert_eq!(f.reply, Reply { content: text.into(), calls: vec![], finish: "stop".into() }, "{text}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn malformed_json_in_a_tag_is_returned_as_the_text_it_was() {
    // the rule: a tag whose body is not a call (bad JSON, no name) stays content, verbatim; no call, finish_reason "stop"
    for text in [
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {oops}\n</tool_call>",
        "<tool_call>\n{\"arguments\": {}}\n</tool_call>",
        "<tool_call>\nnot json\n</tool_call>",
        "<tool_call>\n{\"name\": \"get_weather\", \"arg", // cut off: never closed
    ] {
        for srv in [scripted(&[text]), scripted_chars(text)] {
            let f = both(&srv, &chat_body(json!({"tools": tools()}))).await;
            assert_eq!(f.reply, Reply { content: text.into(), calls: vec![], finish: "stop".into() }, "{text}");
        }
    }
    // a good call after a bad one is still a call
    let text = format!("<tool_call>\nbroken\n</tool_call>\n{CALL}");
    let f = both(&scripted_chars(&text), &chat_body(json!({"tools": tools()}))).await;
    assert_eq!(f.reply.calls, vec![paris()]);
    assert_eq!(f.reply.content, "<tool_call>\nbroken\n</tool_call>\n");
}

#[tokio::test(flavor = "multi_thread")]
async fn without_tools_or_with_tool_choice_none_the_tag_is_plain_content() {
    let srv = scripted(&[CALL]);
    for body in [chat_body(json!({})), chat_body(json!({"tools": tools(), "tool_choice": "none"})), chat_body(json!({"tools": []}))] {
        let f = both(&srv, &body).await;
        assert_eq!(f.reply, Reply { content: CALL.into(), calls: vec![], finish: "stop".into() });
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_conversation_with_tool_calls_and_results_is_accepted() {
    let srv = scripted(&["It is sunny."]);
    let body = json!({"model": "fake", "tools": tools(), "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]},
        {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 21C"},
    ]});
    let f = both(&srv, &body).await;
    assert_eq!(f.reply.content, "It is sunny.");
    // an image part is not supported (no vision): 422, not silently dropped
    let mut b = body.clone();
    b["messages"][1]["content"] = json!([{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]);
    assert_eq!(srv.post("/v1/chat/completions", &b).await.status(), 422);
}

// -- stop strings -----------------------------------------------------------------------------------------------------

/// The text of one reply on each of the three text routes, non-stream and stream, and the reason it ended.
async fn on_every_route(srv: &Srv, stop: Value) -> Vec<(String, String, Value)> {
    let mut out = vec![];
    // chat
    let body = chat_body(json!({"stop": stop}));
    let (w, s) = (whole(srv, &body).await, streamed(srv, &body).await);
    assert_eq!(w.reply, s.reply);
    out.push(("chat".to_string(), w.reply.content, json!(w.reply.finish)));
    // /v1/completions
    let body = json!({"prompt": "hi", "max_tokens": 50, "stop": stop});
    let v: Value = srv.post("/v1/completions", &body).await.json().await.unwrap();
    let (text, finish) = (v["choices"][0]["text"].as_str().unwrap().to_string(), v["choices"][0]["finish_reason"].clone());
    let (_, chunks) = srv.stream("/v1/completions", &json!({"prompt": "hi", "max_tokens": 50, "stop": stop, "stream": true})).await;
    let streamed_text: String = chunks.iter().map(|c| c["choices"].get(0).and_then(|c| c["text"].as_str()).unwrap_or("").to_string()).collect();
    assert_eq!(streamed_text, text, "/v1/completions: stream and non-stream");
    assert_eq!(chunks.last().unwrap()["choices"][0]["finish_reason"], finish);
    out.push(("/v1/completions".to_string(), text, finish));
    // /completion
    let body = json!({"prompt": "hi", "n_predict": 50, "stop": stop, "stream": false});
    let v: Value = srv.post("/completion", &body).await.json().await.unwrap();
    let (_, chunks) = srv.stream("/completion", &json!({"prompt": "hi", "n_predict": 50, "stop": stop, "stream": true})).await;
    let streamed_text: String = chunks.iter().map(|c| c["content"].as_str().unwrap().to_string()).collect();
    assert_eq!(streamed_text, v["content"].as_str().unwrap(), "/completion: stream and non-stream");
    assert_eq!(chunks.last().unwrap()["stop_type"], v["stop_type"]);
    assert_eq!(chunks.last().unwrap()["stopping_word"], v["stopping_word"]);
    out.push(("/completion".to_string(), v["content"].as_str().unwrap().to_string(), json!(v["stop_type"])));
    out
}

async fn check_stop(pieces: &[&str], stop: Value, expect: &str, word: &str) {
    let srv = scripted(pieces);
    let all = pieces.concat();
    // negative control: without `stop` the output contains the stop text, so the cut below is the stop's doing
    let ctl = on_every_route(&srv, json!(null)).await;
    for (route, text, _) in &ctl {
        assert_eq!(text, &all, "{route} without stop");
    }
    assert!(all.contains(word) && all != expect);
    for (route, text, reason) in on_every_route(&srv, stop.clone()).await {
        assert_eq!(text, expect, "{route} with stop {stop}");
        assert!(!text.contains(word), "{route}: the stop string is not output");
        assert!(reason == json!("stop") || reason == json!("word"), "{route}: {reason}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_at_a_token_boundary() {
    check_stop(&["foo ", "STOP", " bar"], json!("STOP"), "foo ", "STOP").await;
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_split_across_tokens_is_held_back_and_never_sent() {
    check_stop(&["foo S", "TO", "P bar"], json!("STOP"), "foo ", "STOP").await;
    // one character per token
    let pieces = chars("foo STOP bar");
    let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
    check_stop(&refs, json!("STOP"), "foo ", "STOP").await;
    // the stream: "S", "ST", "STO" never go out as content
    let srv = scripted(&["foo S", "TO", "P bar"]);
    let f = streamed(&srv, &chat_body(json!({"stop": "STOP"}))).await;
    assert_eq!(f.deltas, ["foo "]);
    assert_eq!(f.reply.finish, "stop");
    // completion_tokens is what ran to the match: three tokens
    assert_eq!(f.usage["completion_tokens"], 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_in_a_list() {
    check_stop(&["one ", "two ", "three"], json!(["XYZ", "two"]), "one ", "two").await;
    // the stop string that completes first wins
    check_stop(&["a-b-c"], json!(["c", "-b"]), "a", "-b").await;
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_with_an_overlapping_prefix() {
    // "ab" against "aab": the first "a" is held back, then released when the second "a" shows it was not a start
    check_stop(&["a", "a", "b", "c"], json!("ab"), "a", "ab").await;
    check_stop(&["a", "ab", "cd"], json!("abc"), "a", "abc").await;
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_at_the_very_end_of_the_output() {
    check_stop(&["foo ", "STOP"], json!("STOP"), "foo ", "STOP").await;
    // the output ends in the middle of a stop string: it is not a stop, and the held text is released
    let srv = scripted(&["foo S", "TO"]);
    for (route, text, reason) in on_every_route(&srv, json!("STOP")).await {
        assert_eq!(text, "foo STO", "{route}");
        assert!(reason == json!("stop") || reason == json!("eos"), "{route}: {reason}");
    }
    // and where the limit ends it
    let f = both(&srv, &chat_body(json!({"stop": "STOP", "max_tokens": 1}))).await;
    assert_eq!((f.reply.content.as_str(), f.reply.finish.as_str()), ("foo S", "length"));
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_is_reported_on_the_wire_and_the_server_goes_on() {
    let srv = scripted(&["foo ", "STOP", " bar"]);
    let v: Value = srv.post("/completion", &json!({"prompt": "hi", "n_predict": 50, "stop": ["STOP"], "return_tokens": true})).await.json().await.unwrap();
    assert_eq!((v["stop_type"].clone(), v["stopping_word"].clone(), v["stop"].clone()), (json!("word"), json!("STOP"), json!(true)));
    assert_eq!(v["generation_settings"]["stop"], json!(["STOP"]));
    assert_eq!(v["tokens_predicted"], 2);
    // the cancelled request is gone and the next one runs
    let v: Value = srv.post("/completion", &json!({"prompt": "hi", "n_predict": 50})).await.json().await.unwrap();
    assert_eq!((v["content"].clone(), v["stop_type"].clone(), v["stopping_word"].clone()), (json!("foo STOP bar"), json!("eos"), json!("")));
    srv.wait_metrics("idle", |m| m["active_requests"] == 0).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stop_string_cancels_the_generation_instead_of_running_it_out() {
    let mut pieces = vec!["foo ".to_string(), "STOP".to_string()];
    pieces.extend((0..400).map(|_| "x".to_string()));
    let srv = spawn(FakeConfig { script: Some(pieces), token_ms: 2.0, ..fast() });
    for path in ["/v1/chat/completions", "/completion", "/v1/completions"] {
        let body = match path {
            "/v1/chat/completions" => chat_body(json!({"stop": "STOP"})),
            "/completion" => json!({"prompt": "hi", "n_predict": 500, "stop": "STOP"}),
            _ => json!({"prompt": "hi", "max_tokens": 500, "stop": "STOP"}),
        };
        let before = srv.metrics().await["tokens_generated"].as_u64().unwrap();
        let r = srv.post(path, &body).await;
        assert_eq!(r.status(), 200);
        r.bytes().await.unwrap();
        let m = srv.wait_metrics("idle", |m| m["active_requests"] == 0).await;
        let made = m["tokens_generated"].as_u64().unwrap() - before;
        assert!(made < 100, "{path}: the engine made {made} of 402 tokens after the stop string");
    }
    assert_eq!(srv.metrics().await["requests"]["cancelled"], 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_with_an_empty_value_is_no_stop_and_a_wrong_type_is_400() {
    let srv = scripted(&["foo STOP"]);
    for stop in [json!(null), json!(""), json!([]), json!([""])] {
        let f = whole(&srv, &chat_body(json!({"stop": stop}))).await;
        assert_eq!(f.reply.content, "foo STOP");
    }
    for stop in [json!(5), json!({"a": 1}), json!(["ok", 3])] {
        for (path, body) in [("/v1/chat/completions", chat_body(json!({"stop": stop}))), ("/completion", json!({"prompt": "hi", "stop": stop}))] {
            assert_eq!(srv.post(path, &body).await.status(), 400, "{path} {stop}"); // the body's type does not fit: 400
        }
    }
}

// -- refusals and errors --------------------------------------------------------------------------------------------------

fn err_of(v: &Value) -> (i64, String, String) {
    let e = &v["error"];
    (e["code"].as_i64().unwrap(), e["type"].as_str().unwrap().to_string(), e["message"].as_str().unwrap().to_string())
}

#[tokio::test(flavor = "multi_thread")]
async fn forced_tool_choice_and_other_constraints_are_refused() {
    let srv = scripted(&["x"]);
    let named = json!({"type": "function", "function": {"name": "get_weather"}});
    for tc in [json!("required"), named] {
        let r = srv.post("/v1/chat/completions", &chat_body(json!({"tools": tools(), "tool_choice": tc}))).await;
        assert_eq!(r.status(), 400, "{tc}");
        let (code, kind, msg) = err_of(&r.json().await.unwrap());
        assert_eq!((code, kind.as_str()), (400, "invalid_request_error"));
        assert!(msg.contains("tool_choice"), "{msg}");
    }
    // negative control: auto and none are accepted
    for tc in ["auto", "none"] {
        assert_eq!(srv.post("/v1/chat/completions", &chat_body(json!({"tools": tools(), "tool_choice": tc}))).await.status(), 200, "{tc}");
    }
    assert_eq!(srv.post("/v1/chat/completions", &chat_body(json!({"tools": tools(), "tool_choice": "banana"}))).await.status(), 422);
    for (f, v) in [("functions", json!([{"name": "f"}])), ("logit_bias", json!({"1": 5})), ("chat_template_kwargs", json!({"x": 1}))] {
        let r = srv.post("/v1/chat/completions", &chat_body(json!({f: v}))).await;
        assert_eq!(r.status(), 400, "{f}");
        assert!(err_of(&r.json().await.unwrap()).2.contains(&format!("`{f}`")));
    }
    // a malformed tool is 422
    assert_eq!(srv.post("/v1/chat/completions", &chat_body(json!({"tools": [{"type": "function"}]}))).await.status(), 422);
    assert_eq!(srv.post("/v1/chat/completions", &chat_body(json!({"n": 2}))).await.status(), 422);
    assert_eq!(srv.post("/v1/chat/completions", &chat_body(json!({"temperature": -1}))).await.status(), 422);
}

#[tokio::test(flavor = "multi_thread")]
async fn tools_stay_refused_on_the_other_routes() {
    let srv = scripted(&["x"]);
    let a: Value = srv.post("/v1/agents", &json!({})).await.json().await.unwrap();
    let id = a["id"].as_str().unwrap();
    for (path, base) in [
        ("/completion".to_string(), json!({"prompt": "hi", "n_predict": 1})),
        ("/v1/completions".to_string(), json!({"prompt": "hi", "max_tokens": 1})),
        (format!("/v1/agents/{id}/turn"), json!({"prompt": [1, 2], "n_predict": 1, "stream": false})),
    ] {
        assert_eq!(srv.post(&path, &base).await.status(), 200, "{path} without the field");
        for (f, v) in [("tools", tools()), ("tool_choice", json!("auto"))] {
            let mut b = base.clone();
            b[f] = v;
            let r = srv.post(&path, &b).await;
            assert_eq!(r.status(), 400, "{path} {f}");
            assert!(err_of(&r.json().await.unwrap()).2.contains(&format!("`{f}`")));
        }
    }
    // stop is honoured on /completion and /v1/completions, and still refused on an agent's turn and event
    for (path, base) in [
        (format!("/v1/agents/{id}/turn"), json!({"prompt": [1, 2], "n_predict": 1, "stream": false})),
        (format!("/v1/agents/{id}/event"), json!({"name": "check", "content": "x", "dry_run": true})),
    ] {
        let mut b = base.clone();
        b["stop"] = json!(["\n"]);
        let r = srv.post(&path, &b).await;
        assert_eq!(r.status(), 400, "{path}");
        assert!(err_of(&r.json().await.unwrap()).2.contains("`stop`"));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn bad_requests_get_the_right_status() {
    let srv = scripted(&["x"]);
    // no usable `messages` is 400 (as llama-server); messages that are not valid messages are 422 (as an empty prompt is on /v1/completions)
    for (body, status) in [(json!({"model": "fake", "messages": []}), 400), (json!({"model": "fake"}), 400), (json!({"messages": "hi"}), 400), (json!({"messages": [{"content": "no role"}]}), 422), (json!({"messages": ["hi"]}), 422)] {
        let r = srv.post("/v1/chat/completions", &body).await;
        assert_eq!(r.status().as_u16() as i64, status as i64, "{body}");
        let (code, kind, _) = err_of(&r.json().await.unwrap());
        assert_eq!((code, kind.as_str()), (status as i64, "invalid_request_error"), "{body}");
    }
    let (_, _, msg) = err_of(&srv.post("/v1/chat/completions", &json!({"model": "fake"})).await.json().await.unwrap());
    assert_eq!(msg, "'messages' is required", "llama-server's words");
    let (_, _, msg) = err_of(&srv.post("/v1/chat/completions", &json!({"messages": []})).await.json().await.unwrap());
    assert!(msg.contains("messages"), "{msg}");
    // not JSON, or nothing: 400
    assert_eq!(srv.post_raw("/v1/chat/completions", "{nope").await.status(), 400);
    assert_eq!(srv.post_raw("/v1/chat/completions", "").await.status(), 400);
    // wrong method
    assert_eq!(srv.get("/v1/chat/completions").await.status(), 405);
}

#[tokio::test(flavor = "multi_thread")]
async fn caps_lists_the_route_and_what_it_honours() {
    let srv = scripted(&["x"]);
    let caps = srv.get_json("/caps").await;
    assert!(caps["endpoints"].as_array().unwrap().iter().any(|e| e == "/v1/chat/completions"));
    let chat = &caps["chat"];
    assert!(chat["honoured"].as_array().unwrap().iter().any(|f| f == "level"), "chat.honoured lacks level");
    assert_eq!(chat["levels"], serde_json::json!(["off", "sonic", "fast", "normal", "deep"]));
    let list = |k: &str| -> Vec<String> { chat[k].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect() };
    for f in ["messages", "tools", "tool_choice", "stop", "stream", "max_tokens", "temperature"] {
        assert!(list("honoured").contains(&f.to_string()), "chat.honoured lacks {f}");
    }
    assert_eq!(chat["tool_choice"], json!(["auto", "none"]));
    assert_eq!(chat["route"], "/v1/chat/completions");
    assert!(chat["tool_call_format"].is_string());
    assert!(!list("refused").contains(&"stop".to_string()));
    assert!(!list("refused").contains(&"response_format".to_string()), "response_format is honoured on chat now");
    assert!(list("honoured").contains(&"response_format".to_string()) && list("refused").contains(&"logit_bias".to_string()));
    assert!(!list("refused").contains(&"tools".to_string()));
    // stop is honoured on the completion routes, and no longer refused there
    assert!(caps["completion"]["honoured"].as_array().unwrap().iter().any(|f| f == "stop"));
    assert!(!caps["completion"]["refused"].as_array().unwrap().iter().any(|f| f == "stop"));
    assert!(caps["completion"]["refused"].as_array().unwrap().iter().any(|f| f == "tools"));
}

// -- render_chat: exact prompts -------------------------------------------------------------------------------------------

use sllm_server::chat::{ChatTemplate, RenderError};
use sllm_server::toolcall::{Part, ToolCall, ToolFormat, ToolParser};

// the keys in the order tools() writes them: the template prints them as sent
const TOOL_JSON: &str = r#"{"type": "function", "function": {"name": "get_weather", "description": "Weather of a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}"#;

fn conversation() -> Vec<Value> {
    vec![
        json!({"role": "system", "content": "You are helpful."}),
        json!({"role": "user", "content": "Weather in Paris?"}),
        // OpenAI's wire: content null, arguments a JSON string
        json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}),
        json!({"role": "tool", "tool_call_id": "call_1", "content": "sunny, 21C"}),
    ]
}

#[test]
fn render_chat_chatml_with_tools_and_tool_results() {
    let t = ChatTemplate::chatml();
    let got = t.render_chat(&conversation(), tools().as_array().unwrap(), true, false).unwrap();
    let want = format!(
        "<|im_start|>system\nYou are helpful.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{TOOL_JSON}\n</tools>\n\n\
         For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n\
         <tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call><|im_end|>\n\
         <|im_start|>user\nWeather in Paris?<|im_end|>\n\
         <|im_start|>assistant\n\n<tool_call>\n{{\"name\": \"get_weather\", \"arguments\": {{\"city\": \"Paris\"}}}}\n</tool_call><|im_end|>\n\
         <|im_start|>user\n<tool_response>\nsunny, 21C\n</tool_response><|im_end|>\n\
         <|im_start|>assistant\n"
    );
    assert_eq!(got, want);
    // no tools: no tools block, the system message stays a system message
    let got = t.render_chat(&conversation(), &[], true, false).unwrap();
    assert!(got.starts_with("<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nWeather in Paris?<|im_end|>\n"), "{got}");
    assert!(!got.contains("<tools>"));
    // without the generation prompt the text ends at the last message
    assert!(t.render_chat(&conversation(), &[], false, false).unwrap().ends_with("</tool_response><|im_end|>\n"));
    // a tool call written by the template is read back by the parser: the round trip
    let call_text = &got[got.find("<tool_call>").unwrap()..got.find("</tool_call>").unwrap() + "</tool_call>".len()];
    let mut p = ToolParser::new(ToolFormat::QWEN_JSON);
    let mut parts = p.push(call_text);
    parts.extend(p.finish());
    assert_eq!(parts, vec![Part::Call(ToolCall { name: "get_weather".into(), arguments: "{\"city\":\"Paris\"}".into() })]);
}

#[test]
fn render_chat_reads_openai_messages_the_way_templates_expect() {
    let t = ChatTemplate::chatml();
    let msgs = [
        json!({"role": "developer", "content": [{"type": "text", "text": "Be "}, {"type": "text", "text": "brief."}]}),
        json!({"role": "user", "content": "hi"}),
        json!({"role": "assistant", "content": "hello", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": {"a": 1}}}]}),
    ];
    // developer is system; text parts are joined; arguments already an object, or a string holding JSON, print alike
    let want = "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n\
                <|im_start|>assistant\nhello\n<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call><|im_end|>\n";
    assert_eq!(t.render_chat(&msgs, &[], false, false).unwrap(), want);
    let mut as_string = msgs.clone();
    as_string[2]["tool_calls"][0]["function"]["arguments"] = json!("{\"a\": 1}");
    assert_eq!(t.render_chat(&as_string, &[], false, false).unwrap(), want);
    // errors: a message the request got wrong, and the template refusing
    let img = [json!({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]})];
    assert!(matches!(t.render_chat(&img, &[], true, false), Err(RenderError::Request(m)) if m.contains("image_url")));
    assert!(matches!(t.render_chat(&[json!({"content": "x"})], &[], true, false), Err(RenderError::Request(_))));
    let strict = ChatTemplate::new("{% for m in messages %}{% if m.role != 'user' %}{{ raise_exception('only user') }}{% endif %}{{ m.content }}{% endfor %}", "", "").unwrap();
    assert!(matches!(strict.render_chat(&conversation(), &[], true, false), Err(RenderError::Template(m)) if m.contains("only user")));
}

/// A Gemma-style template: BOS, `model` for the assistant, the system text and the tools in the first user turn, tool
/// calls as `tool_code` blocks and results as `tool_output` blocks (a test fixture, not a shipped model's template).
const GEMMA_STYLE: &str = r#"{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}{%- set first = messages[0]['content'] + '\n\n' -%}{%- set rest = messages[1:] -%}{%- else -%}{%- set first = '' -%}{%- set rest = messages -%}{%- endif -%}
{%- if tools -%}{%- set first = first + 'You can call these functions by replying with a tool_code block:\n' + (tools | map(attribute='function') | map('tojson') | join('\n')) + '\n\n' -%}{%- endif -%}
{%- for m in rest -%}
{%- if m['role'] == 'assistant' -%}
{{ '<start_of_turn>model\n' + m['content'] | trim }}
{%- for tc in m['tool_calls'] or [] -%}{{ '```tool_code\n' + tc.function.name + '(' + (tc.function.arguments | tojson) + ')\n```' }}{%- endfor -%}
{{ '<end_of_turn>\n' }}
{%- elif m['role'] == 'tool' -%}
{{ '<start_of_turn>user\n```tool_output\n' + m['content'] | trim + '\n```<end_of_turn>\n' }}
{%- else -%}
{{ '<start_of_turn>user\n' + (first if loop.first else '') + m['content'] | trim + '<end_of_turn>\n' }}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{ '<start_of_turn>model\n' }}{%- endif -%}"#;

#[test]
fn render_chat_gemma_style_with_tools_and_tool_results() {
    let t = ChatTemplate::new(GEMMA_STYLE, "<bos>", "<eos>").unwrap();
    let fn_json = r#"{"name": "get_weather", "description": "Weather of a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}"#;
    let got = t.render_chat(&conversation(), tools().as_array().unwrap(), true, false).unwrap();
    let want = format!(
        "<bos><start_of_turn>user\nYou are helpful.\n\nYou can call these functions by replying with a tool_code block:\n{fn_json}\n\nWeather in Paris?<end_of_turn>\n\
         <start_of_turn>model\n```tool_code\nget_weather({{\"city\": \"Paris\"}})\n```<end_of_turn>\n\
         <start_of_turn>user\n```tool_output\nsunny, 21C\n```<end_of_turn>\n\
         <start_of_turn>model\n"
    );
    assert_eq!(got, want);
    // without tools or a system message
    let got = t.render_chat(&[json!({"role": "user", "content": "  hi  "})], &[], true, false).unwrap();
    assert_eq!(got, "<bos><start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n");
    // this template has no tool-call syntax of any known format: the chat route has no parser for it
    assert_eq!(t.tool_format(), None);
    assert_eq!(ChatTemplate::chatml().tool_format(), Some(ToolFormat::QWEN_JSON));
}

// -- answer levels (`"level"`) ----------------------------------------------------------------------------------------------
//
// The fake engine's text is a hash of the prompt (or the script, the same for every prompt), so a prompt is checked by the
// reply it gives: the level's reply must equal a plain chat reply whose system message is the stage prompt.

use sllm_server::levels::{answers_agree, CHAT, LONG, ONE, TIGHT};

async fn chat_json(srv: &Srv, body: &Value) -> Value {
    let r = srv.post("/v1/chat/completions", body).await;
    assert_eq!(r.status(), 200, "{body}");
    r.json().await.unwrap()
}

fn content(v: &Value) -> String {
    v["choices"][0]["message"]["content"].as_str().unwrap().to_string()
}

/// A plain chat reply with `system` as the system message (before the caller's own, when it sends one).
async fn plain(srv: &Srv, system: &str, extra: Value) -> Value {
    let mut b = chat_body(json!({"messages": [{"role": "system", "content": system}, {"role": "user", "content": "hi"}]}));
    for (k, v) in extra.as_object().into_iter().flatten() {
        b[k] = v.clone();
    }
    chat_json(srv, &b).await
}

fn stages(v: &Value) -> Vec<(String, i64)> {
    v["level"]["stages"].as_array().unwrap().iter().map(|s| (s["prompt"].as_str().unwrap().to_string(), s["tokens"].as_i64().unwrap())).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn sonic_and_fast_answer_with_their_prompt_and_cap() {
    let srv = spawn(fast());
    // fast's cap (4096) is also the test context (n_ctx 4096): what the prompt leaves
    let chat_prompt = plain(&srv, CHAT, json!({"max_tokens": 1})).await["usage"]["prompt_tokens"].as_i64().unwrap();
    for (level, prompt, name, cap) in [("sonic", ONE, "ONE", 1024), ("fast", CHAT, "CHAT", 4096 - chat_prompt)] {
        // ignore_eos: the answer runs to the cap
        let r = chat_json(&srv, &chat_body(json!({"level": level, "ignore_eos": true}))).await;
        let want = plain(&srv, prompt, json!({"ignore_eos": true, "max_tokens": cap})).await;
        assert_eq!(content(&r), content(&want), "{level}: the stage prompt is the system message");
        assert_eq!(r["usage"], want["usage"], "{level}");
        assert_eq!(r["usage"]["completion_tokens"], cap, "{level}: the stage's cap");
        assert_eq!(r["choices"][0]["finish_reason"], "length");
        assert_eq!(r["level"], json!({"name": level, "stages": [{"prompt": name, "tokens": cap}], "check": null}));
        // the caller's max_tokens is lower than the cap
        let r = chat_json(&srv, &chat_body(json!({"level": level, "ignore_eos": true, "max_tokens": 7}))).await;
        assert_eq!(r["usage"]["completion_tokens"], 7, "{level}");
        // and a caller's system message follows the stage prompt
        let mine = "You manage agents.";
        let r = chat_json(&srv, &chat_body(json!({"level": level, "messages": [{"role": "system", "content": mine}, {"role": "user", "content": "hi"}], "max_tokens": 20, "ignore_eos": true}))).await;
        let want = plain(&srv, &format!("{prompt}\n\n{mine}"), json!({"max_tokens": 20, "ignore_eos": true})).await;
        assert_eq!(content(&r), content(&want), "{level}: stage prompt, blank line, the caller's system");
        // an empty system message is replaced, not kept
        let r = chat_json(&srv, &chat_body(json!({"level": level, "messages": [{"role": "system", "content": " "}, {"role": "user", "content": "hi"}], "max_tokens": 20, "ignore_eos": true}))).await;
        let want = plain(&srv, prompt, json!({"max_tokens": 20, "ignore_eos": true})).await;
        assert_eq!(content(&r), content(&want), "{level}: no caller system");
    }
    // the caller's limit above the cap: min(stage cap, caller limit, what the context leaves)
    let r = chat_json(&srv, &chat_body(json!({"level": "fast", "max_tokens": 5000, "ignore_eos": true}))).await;
    assert_eq!(r["usage"]["completion_tokens"], 4096 - chat_prompt, "min(stage cap, caller limit, context)");
    let r = chat_json(&srv, &chat_body(json!({"level": "fast", "max_tokens": 3000, "ignore_eos": true}))).await;
    assert_eq!(r["usage"]["completion_tokens"], 3000);
}

#[tokio::test(flavor = "multi_thread")]
async fn sonic_and_fast_stream_the_answer_and_end_with_the_level() {
    let srv = scripted(&["Hel", "lo ", "wor", "ld"]);
    for level in ["sonic", "fast"] {
        let body = chat_body(json!({"level": level}));
        let w = whole(&srv, &body).await;
        let s = streamed(&srv, &body).await;
        assert_eq!(w.reply, s.reply);
        assert_eq!(s.deltas, ["Hel", "lo ", "wor", "ld"], "{level}: streams as it is written");
        assert_eq!(w.usage, s.usage);
        let (raw, chunks) = srv.stream("/v1/chat/completions", &chat_body(json!({"level": level, "stream": true, "stream_options": {"include_usage": true}}))).await;
        assert!(raw.ends_with("data: [DONE]\n\n"));
        let last = chunks.iter().find(|c| c["choices"][0]["finish_reason"].is_string()).unwrap();
        assert_eq!(last["level"]["name"], level);
        assert_eq!(last["level"]["check"], Value::Null);
        assert_eq!(last["level"]["stages"].as_array().unwrap().len(), 1);
        assert_eq!(last["usage"]["completion_tokens"], 4);
        assert!(chunks.last().unwrap()["choices"].as_array().unwrap().is_empty(), "the trailing usage chunk");
    }
    // the non-stream response carries the level object and the usual usage
    let r = chat_json(&srv, &chat_body(json!({"level": "sonic"}))).await;
    assert_eq!(r["level"], json!({"name": "sonic", "stages": [{"prompt": "ONE", "tokens": 4}], "check": null}));
    assert_eq!(r["usage"]["completion_tokens"], 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn no_level_off_and_null_are_the_plain_path() {
    let srv = spawn(fast());
    let a = chat_json(&srv, &chat_body(json!({}))).await;
    for lv in [json!(null), json!("off")] {
        let b = chat_json(&srv, &chat_body(json!({"level": lv}))).await;
        assert_eq!((content(&a), &a["usage"]), (content(&b), &b["usage"]));
        assert!(b.get("level").is_none(), "no level object on the plain path");
    }
    // the plain path has no stage prompt
    let want = plain(&srv, ONE, json!({})).await;
    assert_ne!(content(&a), content(&want));
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unknown_level_and_a_level_with_constraints_are_refused() {
    let srv = spawn(fast());
    for bad in [json!("turbo"), json!("Sonic"), json!(""), json!(3), json!(true), json!(["deep"])] {
        let r = srv.post("/v1/chat/completions", &chat_body(json!({"level": bad}))).await;
        assert_eq!(r.status(), 422, "{bad}");
        let (code, kind, msg) = err_of(&r.json().await.unwrap());
        assert_eq!((code, kind.as_str()), (422, "invalid_request_error"));
        assert!(msg.contains("level"), "{msg}");
    }
    let named = json!({"type": "function", "function": {"name": "get_weather"}});
    for (f, v) in [
        ("tools", tools()),
        ("tool_choice", json!("none")),
        ("tool_choice", json!("auto")),
        ("tool_choice", named),
        ("response_format", json!({"type": "json_object"})),
        ("grammar", json!("root ::= \"a\"")),
    ] {
        for level in ["sonic", "fast", "normal", "deep"] {
            let r = srv.post("/v1/chat/completions", &chat_body(json!({"level": level, f: v}))).await;
            assert_eq!(r.status(), 400, "{level} + {f}");
            let (_, kind, msg) = err_of(&r.json().await.unwrap());
            assert_eq!(kind, "invalid_request_error");
            assert!(msg.contains("`level`") && msg.contains(&format!("`{f}`")), "{msg}");
        }
        // negative control: the same field without a level (or with "off") is not refused for the level's sake
        let r = srv.post("/v1/chat/completions", &chat_body(json!({"level": "off", f: v}))).await;
        if r.status() == 400 {
            assert!(!err_of(&r.json().await.unwrap()).2.contains("`level`"), "{f}");
        }
    }
    // an empty tools list is not asked for
    assert_eq!(srv.post("/v1/chat/completions", &chat_body(json!({"level": "sonic", "tools": [], "max_tokens": 2}))).await.status(), 200);
    // a refused first stage is an HTTP status even when streaming (a prompt that fills the context)
    let big = "x".repeat(5000);
    for stream in [false, true] {
        let r = srv.post("/v1/chat/completions", &chat_body(json!({"level": "fast", "stream": stream, "messages": [{"role": "user", "content": big}]}))).await;
        assert_eq!(r.status(), 400, "stream {stream}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn normal_and_deep_return_the_tight_answer_when_one_and_tight_agree() {
    // a scripted fake writes the same text for every prompt: ONE and TIGHT agree
    let srv = scripted(&["Hel", "lo ", "wor", "ld"]);
    for level in ["normal", "deep"] {
    let body = chat_body(json!({"level": level}));
    let w = whole(&srv, &body).await;
    assert_eq!(w.reply, Reply { content: "Hello world".into(), calls: vec![], finish: "stop".into() });
    let r = chat_json(&srv, &body).await;
    assert_eq!(r["level"], json!({"name": level, "stages": [{"prompt": "ONE", "tokens": 4}, {"prompt": "TIGHT", "tokens": 4}],
                                   "check": "the two short answers agree (Hello world)"}));
    // usage counts both stages: the prompts of ONE and TIGHT differ in length by the length of their prompts
    let one = plain(&srv, ONE, json!({})).await["usage"]["prompt_tokens"].as_i64().unwrap();
    let tight = plain(&srv, TIGHT, json!({})).await["usage"]["prompt_tokens"].as_i64().unwrap();
    assert_eq!(r["usage"]["prompt_tokens"], one + tight);
    assert_eq!(r["usage"]["completion_tokens"], 8);
    assert_eq!(r["usage"]["total_tokens"], one + tight + 8);
    // streaming: nothing while the short answers are written, then the kept TIGHT answer whole in one delta
    let s = streamed(&srv, &body).await;
    assert_eq!(s.deltas, ["Hello world"]);
    assert_eq!(s.usage, r["usage"]);
    assert_eq!(s.reply.finish, "stop");
    let (_, chunks) = srv.stream("/v1/chat/completions", &chat_body(json!({"level": level, "stream": true}))).await;
    let last = chunks.iter().find(|c| c["choices"][0]["finish_reason"].is_string()).unwrap();
    assert_eq!(last["level"], r["level"]);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn normal_and_deep_answer_again_with_long_when_one_and_tight_do_not_agree() {
    // an unscripted fake writes a different text for a different prompt
    let srv = spawn(FakeConfig { natural_len: Some(12), ..fast() });
    let one = content(&plain(&srv, ONE, json!({})).await);
    let tight = content(&plain(&srv, TIGHT, json!({})).await);
    let long = content(&plain(&srv, LONG, json!({})).await);
    let (same, _) = answers_agree(&one, &tight);
    assert!(!same, "this fake pair must disagree for the test to mean anything: {one:?} / {tight:?}");
    for level in ["normal", "deep"] {
    let body = chat_body(json!({"level": level}));
    let r = chat_json(&srv, &body).await;
    assert_eq!(content(&r), long, "the LONG answer is the answer");
    assert_eq!(stages(&r), [("ONE".to_string(), 12), ("TIGHT".to_string(), 12), ("LONG".to_string(), 12)]);
    assert_eq!(r["level"]["check"], "the two short answers do not agree: answering again step by step");
    assert_eq!(r["usage"]["completion_tokens"], 36);
    // streaming: the LONG answer streams as it is written (one delta per token), the short ones not at all
    let s = streamed(&srv, &body).await;
    assert_eq!(s.reply.content, long);
    assert_eq!(s.deltas.len(), 12);
    assert_eq!(s.usage, r["usage"]);
    // LONG's cap: 6144, cut to what the context leaves (n_ctx 4096)
    let r = chat_json(&srv, &chat_body(json!({"level": level, "ignore_eos": true}))).await;
    let ts = stages(&r);
    assert_eq!((ts[0].1, ts[1].1), (1024, 2048));
    let long_prompt = plain(&srv, LONG, json!({"max_tokens": 1})).await["usage"]["prompt_tokens"].as_i64().unwrap();
    assert_eq!(ts[2], ("LONG".to_string(), 4096 - long_prompt), "min(6144, what the context leaves)");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn normal_ignores_token_probabilities() {
    // an engine that serves token probabilities (every token a near tie) changes nothing: normal is the ONE/TIGHT/LONG recipe
    let unsure = spawn(FakeConfig { natural_len: Some(10), token_probs: Some((0.5, 0.45)), ..fast() });
    let one = content(&plain(&unsure, ONE, json!({})).await);
    let tight = content(&plain(&unsure, TIGHT, json!({})).await);
    let (same, why) = answers_agree(&one, &tight);
    assert!(!same, "this fake pair must disagree for the test to mean anything: {one:?} / {tight:?}");
    let r = chat_json(&unsure, &chat_body(json!({"level": "normal"}))).await;
    assert_eq!(content(&r), content(&plain(&unsure, LONG, json!({})).await), "the LONG answer");
    assert_eq!(stages(&r), [("ONE".to_string(), 10), ("TIGHT".to_string(), 10), ("LONG".to_string(), 10)]);
    assert_eq!(r["level"]["check"], why);
    // the same engine without probabilities gives the same stages
    let none = spawn(FakeConfig { natural_len: Some(10), ..fast() });
    let r = chat_json(&none, &chat_body(json!({"level": "normal"}))).await;
    assert_eq!(stages(&r).len(), 3);
    // agreeing answers: no LONG stage at all
    let agree = scripted(&["Hel", "lo ", "wor", "ld"]);
    let r = chat_json(&agree, &chat_body(json!({"level": "normal"}))).await;
    assert_eq!(stages(&r).len(), 2);
    assert_eq!(content(&r), "Hello world");
}
