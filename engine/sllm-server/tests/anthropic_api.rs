//! `POST /v1/messages`, `/v1/messages/count_tokens` and `/v1/models` on the fake engine with a scripted output (no GPU):
//! Anthropic's SSE event sequence (exact event types and block indices), stream and non-stream agreeing, `count_tokens`
//! equal to the tokens a message prefills, system as a string, blocks or an in-messages entry, the tool_use / tool_result
//! round trip, thinking blocks (split tags, an unclosed tag, tag-like text), stop sequences, usage, and Anthropic-shaped errors.

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

const CALL: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
const CALL_ROME: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}\n</tool_call>";

fn tools() -> Value {
    json!([{"name": "get_weather", "description": "Weather of a city", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}])
}

fn openai_tools() -> Value {
    json!([{"type": "function", "function": {"name": "get_weather", "description": "Weather of a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])
}

fn body(extra: Value) -> Value {
    let mut b = json!({"model": "fake", "max_tokens": 256, "messages": [{"role": "user", "content": "hi"}]});
    for (k, v) in extra.as_object().into_iter().flatten() {
        b[k] = v.clone();
    }
    b
}

const PLAIN_PROMPT: &str = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";

// -- reading the wire -------------------------------------------------------------------------------------------------

/// Anthropic's framing: `event: <type>\ndata: <json>\n\n`, and the event name is the data's `type`.
fn sse(raw: &str) -> Vec<(String, Value)> {
    assert!(raw.is_empty() || raw.ends_with("\n\n"), "SSE body must end with a blank line: {raw:?}");
    raw.split("\n\n")
        .filter(|e| !e.is_empty())
        .map(|e| {
            let mut lines = e.split('\n');
            let name = lines.next().and_then(|l| l.strip_prefix("event: ")).unwrap_or_else(|| panic!("event without `event: `: {e:?}")).to_string();
            let data = lines.next().and_then(|l| l.strip_prefix("data: ")).unwrap_or_else(|| panic!("event without `data: `: {e:?}"));
            assert!(lines.next().is_none(), "extra lines in an event: {e:?}");
            let v: Value = serde_json::from_str(data).expect("data is JSON");
            assert_eq!(v["type"], name.as_str(), "the event name is the data's type");
            (name, v)
        })
        .collect()
}

async fn stream_raw(srv: &Srv, b: &Value) -> Vec<(String, Value)> {
    let mut b = b.clone();
    b["stream"] = json!(true);
    let r = srv.post("/v1/messages", &b).await;
    assert_eq!(r.status(), 200, "{b}");
    assert!(r.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));
    sse(&r.text().await.unwrap())
}

/// The event types with the block index and kind where they have one: the golden sequences below.
fn shape(events: &[(String, Value)]) -> Vec<String> {
    events
        .iter()
        .map(|(n, v)| match n.as_str() {
            "content_block_start" => format!("start:{}:{}", v["index"], v["content_block"]["type"].as_str().unwrap()),
            "content_block_delta" => format!("delta:{}:{}", v["index"], v["delta"]["type"].as_str().unwrap()),
            "content_block_stop" => format!("stop:{}", v["index"]),
            other => other.to_string(),
        })
        .collect()
}

async fn shape_of(srv: &Srv, b: &Value) -> Vec<String> {
    shape(&stream_raw(srv, b).await)
}

/// A reply as an SDK reads it from a stream: the content blocks, the stop reason, the usage.
#[derive(Debug, PartialEq, Clone)]
struct Reply {
    content: Vec<Value>,
    stop_reason: String,
    stop_sequence: Value,
    usage: Value,
}

fn fold(events: &[(String, Value)]) -> Reply {
    let (mut content, mut partial) = (Vec::<Value>::new(), String::new());
    let (mut stop_reason, mut stop_sequence, mut usage) = (String::new(), Value::Null, json!({}));
    let mut start_usage = Value::Null;
    for (n, v) in events {
        match n.as_str() {
            "message_start" => {
                assert_eq!(v["message"]["role"], "assistant");
                assert_eq!(v["message"]["content"], json!([]));
                assert_eq!(v["message"]["stop_reason"], Value::Null);
                start_usage = v["message"]["usage"].clone();
            }
            "content_block_start" => {
                assert_eq!(v["index"], content.len());
                content.push(v["content_block"].clone());
                partial.clear();
            }
            "content_block_delta" => {
                assert_eq!(v["index"], content.len() - 1);
                let b = content.last_mut().unwrap();
                let d = &v["delta"];
                match d["type"].as_str().unwrap() {
                    "text_delta" => b["text"] = json!(format!("{}{}", b["text"].as_str().unwrap(), d["text"].as_str().unwrap())),
                    "thinking_delta" => b["thinking"] = json!(format!("{}{}", b["thinking"].as_str().unwrap(), d["thinking"].as_str().unwrap())),
                    "signature_delta" => b["signature"] = d["signature"].clone(),
                    "input_json_delta" => partial.push_str(d["partial_json"].as_str().unwrap()),
                    t => panic!("delta type {t}"),
                }
            }
            "content_block_stop" => {
                assert_eq!(v["index"], content.len() - 1);
                let b = content.last_mut().unwrap();
                if b["type"] == "tool_use" {
                    b["input"] = serde_json::from_str(&partial).expect("the partial_json chunks join to JSON");
                }
            }
            "message_delta" => {
                stop_reason = v["delta"]["stop_reason"].as_str().unwrap().to_string();
                stop_sequence = v["delta"]["stop_sequence"].clone();
                usage = v["usage"].clone();
            }
            _ => {}
        }
    }
    assert!(start_usage["input_tokens"].is_u64(), "message_start carries input_tokens: {start_usage}");
    Reply { content, stop_reason, stop_sequence, usage }
}

async fn streamed(srv: &Srv, b: &Value) -> Reply {
    fold(&stream_raw(srv, b).await)
}

async fn whole(srv: &Srv, b: &Value) -> Reply {
    let mut b = b.clone();
    b["stream"] = json!(false);
    let r = srv.post("/v1/messages", &b).await;
    assert_eq!(r.status(), 200, "{b}");
    assert!(r.headers().contains_key("x-request-id"));
    let v: Value = r.json().await.unwrap();
    assert_eq!((v["type"].clone(), v["role"].clone()), (json!("message"), json!("assistant")));
    assert!(v["id"].as_str().unwrap().starts_with("msg_"));
    Reply { content: v["content"].as_array().unwrap().clone(), stop_reason: v["stop_reason"].as_str().unwrap().to_string(), stop_sequence: v["stop_sequence"].clone(), usage: v["usage"].clone() }
}

/// Stream and non-stream must say the same thing; returns it. Block ids are made per request, so they are compared apart.
async fn both(srv: &Srv, b: &Value) -> Reply {
    let (w, s) = (whole(srv, b).await, streamed(srv, b).await);
    let strip = |r: &Reply| -> Vec<Value> {
        r.content
            .iter()
            .map(|c| {
                let mut c = c.clone();
                if c["type"] == "tool_use" {
                    assert!(c["id"].as_str().unwrap().starts_with("toolu_"), "{c}");
                    c["id"] = json!("ID");
                }
                c
            })
            .collect()
    };
    assert_eq!(strip(&w), strip(&s), "content differs between non-stream and stream");
    assert_eq!((&w.stop_reason, &w.stop_sequence), (&s.stop_reason, &s.stop_sequence));
    assert_eq!(w.usage, s.usage, "usage differs");
    s
}

fn text_of(r: &Reply) -> String {
    r.content.iter().filter(|c| c["type"] == "text").map(|c| c["text"].as_str().unwrap()).collect()
}

async fn count(srv: &Srv, b: &Value) -> u64 {
    let r = srv.post("/v1/messages/count_tokens", b).await;
    assert_eq!(r.status(), 200, "{b}");
    let v: Value = r.json().await.unwrap();
    assert_eq!(v.as_object().unwrap().len(), 1, "count_tokens answers {{input_tokens}} only: {v}");
    v["input_tokens"].as_u64().unwrap()
}

async fn input_tokens(srv: &Srv, b: &Value) -> u64 {
    let r = whole(srv, b).await;
    assert_eq!(streamed(srv, b).await.usage["input_tokens"], r.usage["input_tokens"]);
    r.usage["input_tokens"].as_u64().unwrap()
}

// -- the event sequence ----------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn golden_text_only() {
    let srv = scripted(&["Hel", "lo"]);
    let ev = stream_raw(&srv, &body(json!({}))).await;
    assert_eq!(
        shape(&ev),
        ["message_start", "start:0:text", "ping", "delta:0:text_delta", "delta:0:text_delta", "stop:0", "message_delta", "message_stop"]
    );
    // the exact payloads of the first and the last events
    let ms = &ev[0].1["message"];
    assert!(ms["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!((ms["type"].clone(), ms["model"].clone()), (json!("message"), json!("fake")));
    assert_eq!(ms["usage"]["input_tokens"], PLAIN_PROMPT.len());
    assert_eq!(ev[1].1["content_block"], json!({"type": "text", "text": ""}));
    assert_eq!(ev[3].1["delta"], json!({"type": "text_delta", "text": "Hel"}));
    let md = &ev[ev.len() - 2].1;
    assert_eq!(md["delta"], json!({"stop_reason": "end_turn", "stop_sequence": null}));
    assert_eq!(md["usage"]["output_tokens"], 2);
    assert_eq!(ev.last().unwrap().1, json!({"type": "message_stop"}));
}

#[tokio::test(flavor = "multi_thread")]
async fn golden_text_then_tool_use() {
    let srv = scripted(&["Let me check.\n", CALL]);
    let b = body(json!({"tools": tools()}));
    let ev = stream_raw(&srv, &b).await;
    assert_eq!(
        shape(&ev),
        ["message_start", "start:0:text", "ping", "delta:0:text_delta", "stop:0", "start:1:tool_use", "delta:1:input_json_delta", "stop:1", "message_delta", "message_stop"]
    );
    let start = &ev[5].1["content_block"];
    assert_eq!((start["type"].clone(), start["name"].clone(), start["input"].clone()), (json!("tool_use"), json!("get_weather"), json!({})));
    assert!(start["id"].as_str().unwrap().starts_with("toolu_"));
    assert_eq!(ev[8].1["delta"]["stop_reason"], "tool_use");
    let r = both(&srv, &b).await;
    assert_eq!(r.content[0], json!({"type": "text", "text": "Let me check.\n"}));
    assert_eq!((r.content[1]["name"].clone(), r.content[1]["input"].clone()), (json!("get_weather"), json!({"city": "Paris"})));
    assert_eq!(r.stop_reason, "tool_use");
}

#[tokio::test(flavor = "multi_thread")]
async fn golden_two_tool_uses() {
    let srv = scripted(&[&format!("{CALL}\n{CALL_ROME}")]);
    let b = body(json!({"tools": tools()}));
    assert_eq!(
        shape_of(&srv, &b).await,
        ["message_start", "start:0:tool_use", "ping", "delta:0:input_json_delta", "stop:0", "start:1:tool_use", "delta:1:input_json_delta", "stop:1", "message_delta", "message_stop"]
    );
    let r = both(&srv, &b).await;
    let inputs: Vec<Value> = r.content.iter().map(|c| c["input"].clone()).collect();
    assert_eq!(inputs, [json!({"city": "Paris"}), json!({"city": "Rome"})]);
    assert_ne!(r.content[0]["id"], r.content[1]["id"], "each tool_use has its own id");
    assert_eq!(r.stop_reason, "tool_use");
}

#[tokio::test(flavor = "multi_thread")]
async fn golden_thinking_text_tool_use() {
    let srv = scripted(&["<think>\nhmm\n</think>\n\nAnswer.\n", CALL]);
    let b = body(json!({"tools": tools(), "thinking": {"type": "enabled", "budget_tokens": 1024}}));
    let ev = stream_raw(&srv, &b).await;
    assert_eq!(
        shape(&ev),
        [
            "message_start", "start:0:thinking", "ping", "delta:0:thinking_delta", "delta:0:signature_delta", "stop:0", "start:1:text", "delta:1:text_delta", "stop:1",
            "start:2:tool_use", "delta:2:input_json_delta", "stop:2", "message_delta", "message_stop"
        ]
    );
    assert_eq!(ev[1].1["content_block"]["type"], "thinking");
    assert_eq!(ev[3].1["delta"], json!({"type": "thinking_delta", "thinking": "hmm"}));
    let r = both(&srv, &b).await;
    assert_eq!(r.content[0]["thinking"], "hmm");
    assert_eq!(r.content[1], json!({"type": "text", "text": "Answer.\n"}));
    assert_eq!(r.content[2]["input"], json!({"city": "Paris"}));
}

#[tokio::test(flavor = "multi_thread")]
async fn golden_max_tokens() {
    let srv = scripted(&["a", "b", "c", "d"]);
    let b = body(json!({"max_tokens": 2}));
    assert_eq!(
        shape_of(&srv, &b).await,
        ["message_start", "start:0:text", "ping", "delta:0:text_delta", "delta:0:text_delta", "stop:0", "message_delta", "message_stop"]
    );
    let r = both(&srv, &b).await;
    assert_eq!((text_of(&r).as_str(), r.stop_reason.as_str(), r.usage["output_tokens"].clone()), ("ab", "max_tokens", json!(2)));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_tool_call_split_one_character_per_token_streams_json_that_joins_exactly() {
    let srv = scripted_chars(CALL);
    let b = body(json!({"tools": tools()}));
    let ev = stream_raw(&srv, &b).await;
    let partial: Vec<&str> = ev.iter().filter(|(n, _)| n == "content_block_delta").map(|(_, v)| v["delta"]["partial_json"].as_str().unwrap()).collect();
    assert!(!partial.is_empty());
    assert_eq!(partial.concat(), "{\"city\":\"Paris\"}", "the chunks join to the exact JSON");
    // no tag text goes out as a text block while the call is being read
    assert!(shape(&ev).iter().all(|s| !s.contains("text")), "{:?}", shape(&ev));
    let r = both(&srv, &b).await;
    assert_eq!((r.content.len(), r.content[0]["input"].clone()), (1, json!({"city": "Paris"})));
}

#[tokio::test(flavor = "multi_thread")]
async fn stream_and_non_stream_agree() {
    let cases: Vec<(Srv, Value)> = vec![
        (scripted(&["Hel", "lo ", "wor", "ld"]), body(json!({}))),
        (scripted(&["Hel", "lo ", "wor", "ld"]), body(json!({"max_tokens": 3}))),
        (scripted_chars(&format!("Sure.\n{CALL}")), body(json!({"tools": tools()}))),
        (scripted(&["<think>why</think>", "ok"]), body(json!({"thinking": {"type": "enabled", "budget_tokens": 2048}}))),
        (scripted(&["foo ", "STOP", " bar"]), body(json!({"stop_sequences": ["STOP"]}))),
        (scripted(&[]), body(json!({}))),
    ];
    for (srv, b) in &cases {
        let r = both(srv, b).await;
        assert!(r.usage["input_tokens"].as_u64().unwrap() > 0 && r.usage.get("cache_read_input_tokens").is_some() && r.usage.get("cache_creation_input_tokens").is_some(), "{}", r.usage);
    }
    let r = both(&cases[0].0, &cases[0].1).await;
    assert_eq!((text_of(&r).as_str(), r.stop_reason.as_str()), ("Hello world", "end_turn"));
    let r = both(&cases[1].0, &cases[1].1).await;
    assert_eq!((text_of(&r).as_str(), r.stop_reason.as_str()), ("Hello wor", "max_tokens"));
    let r = both(&cases[4].0, &cases[4].1).await;
    assert_eq!((text_of(&r).as_str(), r.stop_reason.as_str(), r.stop_sequence.clone()), ("foo ", "stop_sequence", json!("STOP")));
    // an empty reply is one empty text block, in both
    let r = both(&cases[5].0, &cases[5].1).await;
    assert_eq!(r.content, [json!({"type": "text", "text": ""})]);
}

#[tokio::test(flavor = "multi_thread")]
async fn usage_names_all_four_counts_and_cache_read_follows_the_cache() {
    let srv = scripted(&["a", "b", "c"]);
    let r = both(&srv, &body(json!({}))).await;
    assert_eq!(r.usage, json!({"input_tokens": PLAIN_PROMPT.len(), "output_tokens": 3, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}));
}

// -- count_tokens -----------------------------------------------------------------------------------------------------

fn round_trip() -> Value {
    body(json!({"tools": tools(), "system": "You are helpful.", "messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": [{"type": "text", "text": "Checking."}, {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny, 21C"}]},
    ]}))
}

#[tokio::test(flavor = "multi_thread")]
async fn count_tokens_equals_the_tokens_a_message_prefills() {
    let srv = scripted(&["ok"]);
    let bodies = vec![
        body(json!({})),
        body(json!({"system": "Be brief."})),
        body(json!({"system": [{"type": "text", "text": "Be brief."}, {"type": "text", "text": "Be kind.", "cache_control": {"type": "ephemeral"}}]})),
        body(json!({"tools": tools(), "messages": [{"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]}]})),
        round_trip(),
        body(json!({"tools": tools(), "tool_choice": {"type": "none"}})),
    ];
    let mut seen = vec![];
    for b in &bodies {
        let (c, m) = (count(&srv, b).await, input_tokens(&srv, b).await);
        assert_eq!(c, m, "count_tokens vs usage.input_tokens for {b}");
        seen.push(c);
    }
    // the plain body is the bytes of the template's prompt (the test server tokenizes one token per byte)
    assert_eq!(seen[0], PLAIN_PROMPT.len() as u64);
    // negative control: a different body counts differently (not a constant)
    let mut distinct = seen.clone();
    distinct.sort();
    distinct.dedup();
    assert!(distinct.len() >= 5, "{seen:?}");
    for (i, extra) in [json!({"system": "Be brief!!!"}), json!({"messages": [{"role": "user", "content": "hi there"}]})].into_iter().enumerate() {
        let mut changed = bodies[1].clone();
        for (k, v) in extra.as_object().unwrap() {
            changed[k] = v.clone();
        }
        assert_ne!(count(&srv, &changed).await, seen[1], "variant {i} must change the count");
    }
    let mut no_tools = bodies[3].clone();
    no_tools.as_object_mut().unwrap().remove("tools");
    assert_ne!(count(&srv, &no_tools).await, seen[3], "dropping the tools changes the count");
    // tool_choice none hides the tools from the model: the count is the count without tools
    assert_eq!(seen[5], seen[0]);
    // count_tokens does not need max_tokens, and runs nothing
    let mut no_max = bodies[0].clone();
    no_max.as_object_mut().unwrap().remove("max_tokens");
    assert_eq!(count(&srv, &no_max).await, seen[0]);
    // count_tokens runs nothing: the generation counter only moved for the messages above
    let before = srv.metrics().await["tokens_generated"].as_u64().unwrap();
    count(&srv, &bodies[4]).await;
    assert_eq!(srv.metrics().await["tokens_generated"].as_u64().unwrap(), before);
}

// -- system, messages, tools -----------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn system_as_a_string_as_blocks_and_inside_messages_render_the_same() {
    let srv = scripted(&["ok"]);
    let want = "<|im_start|>system\nBe brief.<|im_end|>\n".len() + PLAIN_PROMPT.len();
    let as_string = body(json!({"system": "Be brief."}));
    let as_blocks = body(json!({"system": [{"type": "text", "text": "Be brief."}]}));
    let in_messages = body(json!({"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]}));
    let in_messages_blocks = body(json!({"messages": [{"role": "system", "content": [{"type": "text", "text": "Be brief."}]}, {"role": "user", "content": "hi"}]}));
    for b in [&as_string, &as_blocks, &in_messages, &in_messages_blocks] {
        assert_eq!(count(&srv, b).await, want as u64, "{b}");
        assert_eq!(both(&srv, b).await.usage["input_tokens"], want);
    }
    // several system blocks, and a top-level plus an in-messages one, are joined with a blank line
    let joined = body(json!({"system": "A\n\nB"}));
    let blocks = body(json!({"system": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]}));
    let both_places = body(json!({"system": "A", "messages": [{"role": "system", "content": "B"}, {"role": "user", "content": "hi"}]}));
    let n = count(&srv, &joined).await;
    assert_eq!((count(&srv, &blocks).await, count(&srv, &both_places).await), (n, n));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_tool_use_and_tool_result_round_trip_renders_like_the_equivalent_openai_conversation() {
    let srv = scripted(&["It is sunny."]);
    let b = round_trip();
    let oai = json!({"model": "fake", "tools": openai_tools(), "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "Checking.", "tool_calls": [{"id": "toolu_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny, 21C"},
    ]});
    let v: Value = srv.post("/v1/chat/completions", &oai).await.json().await.unwrap();
    let want = v["usage"]["prompt_tokens"].as_u64().unwrap();
    assert_eq!(count(&srv, &b).await, want, "the Anthropic conversation renders the bytes the OpenAI one does");
    let r = both(&srv, &b).await;
    assert_eq!((text_of(&r).as_str(), r.stop_reason.as_str()), ("It is sunny.", "end_turn"));
    // a tool_result may hold blocks, and an error result is still a result
    let mut with_blocks = b.clone();
    with_blocks["messages"][2]["content"][0]["content"] = json!([{"type": "text", "text": "sunny, 21C"}]);
    assert_eq!(count(&srv, &with_blocks).await, want);
    with_blocks["messages"][2]["content"][0]["is_error"] = json!(true);
    assert!(count(&srv, &with_blocks).await > 0);
    // thinking blocks of an earlier turn (with their signature) are accepted and not rendered
    let mut with_thinking = b.clone();
    with_thinking["messages"][1]["content"].as_array_mut().unwrap().insert(0, json!({"type": "thinking", "thinking": "hmm", "signature": "abc"}));
    assert_eq!(count(&srv, &with_thinking).await, want);
}

#[tokio::test(flavor = "multi_thread")]
async fn sampling_fields_and_metadata_are_accepted() {
    let srv = scripted(&["ok"]);
    let b = body(json!({"temperature": 0.5, "top_p": 0.9, "top_k": 40, "metadata": {"user_id": "u1"}, "tool_choice": {"type": "auto", "disable_parallel_tool_use": true}, "tools": tools(), "stream": false}));
    assert_eq!(text_of(&whole(&srv, &b).await), "ok");
    assert_eq!(srv.post("/v1/messages", &body(json!({"tool_choice": {"type": "none"}}))).await.status(), 200);
}

// -- stop sequences ---------------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn stop_sequences_cut_the_text_across_tokens_and_are_reported() {
    for srv in [scripted(&["foo S", "TO", "P bar"]), scripted_chars("foo STOP bar")] {
        let b = body(json!({"stop_sequences": ["STOP"]}));
        let r = both(&srv, &b).await;
        assert_eq!((text_of(&r).as_str(), r.stop_reason.as_str(), r.stop_sequence.clone()), ("foo ", "stop_sequence", json!("STOP")));
        // negative control: without it the text runs on
        let r = both(&srv, &body(json!({}))).await;
        assert_eq!((text_of(&r).as_str(), r.stop_reason.as_str(), r.stop_sequence.clone()), ("foo STOP bar", "end_turn", Value::Null));
    }
}

// -- thinking ---------------------------------------------------------------------------------------------------------

fn thinking_on() -> Value {
    json!({"thinking": {"type": "enabled", "budget_tokens": 1024}})
}

fn kinds(r: &Reply) -> Vec<(String, String)> {
    r.content
        .iter()
        .map(|c| match c["type"].as_str().unwrap() {
            "thinking" => ("thinking".to_string(), c["thinking"].as_str().unwrap().to_string()),
            "text" => ("text".to_string(), c["text"].as_str().unwrap().to_string()),
            t => (t.to_string(), String::new()),
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn thinking_tags_split_across_tokens_make_a_thinking_block() {
    let text = "<think>\nlet me see\n</think>\n\nThe answer.";
    let want = vec![("thinking".to_string(), "let me see".to_string()), ("text".to_string(), "The answer.".to_string())];
    for srv in [scripted(&[text]), scripted_chars(text), scripted(&["<thi", "nk>\nlet me see\n</th", "ink>\n\nThe", " answer."])] {
        assert_eq!(kinds(&both(&srv, &body(thinking_on())).await), want);
    }
    // thinking is off unless asked for: the same text stays text
    let srv = scripted(&[text]);
    let r = both(&srv, &body(json!({}))).await;
    assert_eq!(kinds(&r), [("text".to_string(), text.to_string())]);
    let r = both(&srv, &body(json!({"thinking": {"type": "disabled"}}))).await;
    assert_eq!(kinds(&r), [("text".to_string(), text.to_string())]);
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unclosed_think_tag_at_the_end_is_a_thinking_block() {
    for srv in [scripted(&["<think>\nstill thin", "king"]), scripted_chars("<think>\nstill thinking")] {
        let r = both(&srv, &body(thinking_on())).await;
        assert_eq!(kinds(&r), [("thinking".to_string(), "still thinking".to_string())]);
        assert_eq!(r.stop_reason, "end_turn");
    }
    // cut off by max_tokens inside the tag's own text: nothing is lost
    let srv = scripted(&["<think>\nabc</th", "ink>"]);
    let r = both(&srv, &body(json!({"max_tokens": 1, "thinking": {"type": "enabled", "budget_tokens": 1024}}))).await;
    assert_eq!((kinds(&r), r.stop_reason.as_str()), (vec![("thinking".to_string(), "abc</th".to_string())], "max_tokens"));
}

#[tokio::test(flavor = "multi_thread")]
async fn tag_like_text_that_is_not_a_think_tag_is_text() {
    for pieces in [vec!["<thin", "gs are fun"], vec!["<thin"], vec!["a <think b", "> c"], vec!["<", "thinking>"]] {
        let all = pieces.concat();
        for srv in [scripted(&pieces), scripted_chars(&all)] {
            let r = both(&srv, &body(thinking_on())).await;
            assert_eq!(kinds(&r), [("text".to_string(), all.clone())], "{all}");
        }
    }
    // a "</think>" with no open tag, and a "<think>" after the answer began, are plain text
    let srv = scripted(&["ok </think> and <think>no</think>"]);
    let r = both(&srv, &body(thinking_on())).await;
    assert_eq!(kinds(&r), [("text".to_string(), "ok </think> and <think>no</think>".to_string())]);
}

// -- errors -----------------------------------------------------------------------------------------------------------

async fn error_of(srv: &Srv, path: &str, b: &Value) -> (u16, Value) {
    let r = srv.post(path, b).await;
    let status = r.status().as_u16();
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["type"], "error", "{v}");
    assert!(v["error"]["message"].as_str().is_some_and(|m| !m.is_empty()), "{v}");
    let mut keys = v.as_object().unwrap().keys().collect::<Vec<_>>();
    keys.sort(); // (the body keeps the order it was written in)
    assert_eq!(keys, ["error", "type"], "{v}");
    (status, v["error"].clone())
}

#[tokio::test(flavor = "multi_thread")]
async fn errors_have_anthropics_shape_and_status() {
    let srv = scripted(&["ok"]);
    // max_tokens is required on /v1/messages
    let mut b = body(json!({}));
    b.as_object_mut().unwrap().remove("max_tokens");
    let (status, e) = error_of(&srv, "/v1/messages", &b).await;
    assert_eq!((status, e["type"].clone()), (400, json!("invalid_request_error")));
    assert!(e["message"].as_str().unwrap().contains("max_tokens"), "{e}");
    // ... and zero is not a count
    assert_eq!(error_of(&srv, "/v1/messages", &body(json!({"max_tokens": 0}))).await.0, 400);
    // tool_choice any and tool need grammar-constrained decoding
    for tc in [json!({"type": "any"}), json!({"type": "tool", "name": "get_weather"})] {
        let (status, e) = error_of(&srv, "/v1/messages", &body(json!({"tools": tools(), "tool_choice": tc}))).await;
        assert_eq!((status, e["type"].clone()), (400, json!("invalid_request_error")), "{tc}");
        assert!(e["message"].as_str().unwrap().contains("tool_choice"), "{e}");
    }
    // an image block: no vision
    let img = json!([{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]);
    let (status, e) = error_of(&srv, "/v1/messages", &body(json!({"messages": [{"role": "user", "content": img}]}))).await;
    assert_eq!((status, e["type"].clone()), (400, json!("invalid_request_error")));
    assert!(e["message"].as_str().unwrap().contains("image"), "{e}");
    // an image inside a tool_result too
    let tr = json!([{"type": "tool_result", "tool_use_id": "t", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]);
    assert_eq!(error_of(&srv, "/v1/messages", &body(json!({"messages": [{"role": "user", "content": tr}]}))).await.0, 400);
    // the same body is refused by count_tokens too
    assert_eq!(error_of(&srv, "/v1/messages/count_tokens", &body(json!({"tools": tools(), "tool_choice": {"type": "any"}}))).await.0, 400);
    // messages missing or empty, a bad role, not JSON, an empty body
    let mut b = body(json!({}));
    b.as_object_mut().unwrap().remove("messages");
    assert_eq!(error_of(&srv, "/v1/messages", &b).await.0, 400);
    assert_eq!(error_of(&srv, "/v1/messages", &body(json!({"messages": []}))).await.0, 400);
    assert_eq!(error_of(&srv, "/v1/messages", &body(json!({"messages": [{"role": "robot", "content": "hi"}]}))).await.0, 400);
    for raw in ["{not json", ""] {
        let r = srv.post_raw("/v1/messages", raw).await;
        assert_eq!(r.status(), 400);
        assert_eq!(r.json::<Value>().await.unwrap()["type"], "error");
    }
    // the prompt does not fit the context: Anthropic's own words, which Claude Code reads as "compact now"
    let srv = spawn(FakeConfig { n_ctx: 512, script: Some(vec!["ok".into()]), ..fast() });
    let big = body(json!({"messages": [{"role": "user", "content": "x".repeat(2000)}]}));
    let (status, e) = error_of(&srv, "/v1/messages", &big).await;
    assert_eq!((status, e["type"].clone()), (400, json!("invalid_request_error")));
    assert!(e["message"].as_str().unwrap().starts_with("prompt is too long: "), "{e}");
    // the server goes on
    assert_eq!(srv.post("/v1/messages", &body(json!({}))).await.status(), 200);
}

// -- models and caps --------------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn models_list_and_get() {
    let srv = spawn(FakeConfig { model: "/models/qwen3-0.6b-Q4_0.gguf".into(), ..fast() });
    let v = srv.get_json("/v1/models").await;
    assert_eq!((v["has_more"].clone(), v["data"].as_array().unwrap().len()), (json!(false), 1));
    let m = &v["data"][0];
    assert_eq!((m["type"].clone(), m["id"].clone(), m["display_name"].clone()), (json!("model"), json!("qwen3-0.6b-Q4_0"), json!("qwen3-0.6b-Q4_0")));
    assert!(m["created_at"].as_str().unwrap().ends_with('Z'), "{m}");
    assert_eq!((v["first_id"].clone(), v["last_id"].clone()), (m["id"].clone(), m["id"].clone()));
    let got = srv.get_json("/v1/models/qwen3-0.6b-Q4_0").await;
    assert_eq!(&got, m);
    let r = srv.get("/v1/models/claude-opus-9").await;
    assert_eq!(r.status(), 404);
    let e: Value = r.json().await.unwrap();
    assert_eq!((e["type"].clone(), e["error"]["type"].clone()), (json!("error"), json!("not_found_error")));
}

#[tokio::test(flavor = "multi_thread")]
async fn caps_lists_the_routes() {
    let srv = scripted(&["ok"]);
    let c = srv.get_json("/caps").await;
    let ends: Vec<&str> = c["endpoints"].as_array().unwrap().iter().map(|e| e.as_str().unwrap()).collect();
    for e in ["/v1/messages", "/v1/messages/count_tokens", "/v1/models", "/v1/models/{id}"] {
        assert!(ends.contains(&e), "{e} missing from {ends:?}");
    }
    assert_eq!(c["anthropic"]["tool_choice"], json!(["auto", "none"]));
    assert!(c["anthropic"]["tool_result_limit"].is_string(), "{}", c["anthropic"]);
}

// -- canonicalization: the same prompt bytes for what carries no meaning -----------------------------------------------

use sllm_server::anthropic::{self, AnthropicConfig, ToolResultLimit, CAPS};
use sllm_server::chat::ChatTemplate;
use sllm_server::tokenizer::Tokenizer;
use sllm_server::ServerConfig;

/// The prompt the library renders for a request body (the test server's template is the built-in ChatML one).
fn rendered(b: &Value) -> String {
    let c = anthropic::convert(b.get("system"), b["messages"].as_array().unwrap(), b.get("tools")).unwrap();
    ChatTemplate::chatml().render_chat(&c.messages, &c.tools, true, false).unwrap()
}

fn bash_tool(desc: &str) -> Value {
    json!({"name": "bash", "description": desc, "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}, "timeout": {"type": "integer"}}}})
}

fn read_tool() -> Value {
    json!({"name": "read", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}})
}

#[tokio::test(flavor = "multi_thread")]
async fn requests_that_differ_only_in_volatile_parts_render_the_same_prompt() {
    let srv = scripted(&["ok"]);
    let sys = |billing: &str| json!([{"type": "text", "text": billing}, {"type": "text", "text": "You are Claude Code."}]);
    let a = body(json!({"system": sys("x-anthropic-billing-header: cc_version=2.1.1.abc; cc_entrypoint=cli; cch=1;"), "tools": [bash_tool("Run"), read_tool()]}));
    // another hash (and a different length), the tools in the other order, the keys of the tool object itself in another order
    // (the order inside its schema is the same: that order is kept as sent)
    let b = body(json!({"system": sys("x-anthropic-billing-header: cc_version=2.1.2.zyxwv; cc_entrypoint=cli; cch=777777777;"),
                        "tools": [read_tool(), {"input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}, "timeout": {"type": "integer"}}}, "description": "Run", "name": "bash"}]}));
    assert_eq!(rendered(&a), rendered(&b), "the library renders the same bytes");
    // ... and the route prefills exactly those bytes (one token per byte), for both
    let (ca, cb) = (count(&srv, &a).await, count(&srv, &b).await);
    assert_eq!((ca, cb), (rendered(&a).len() as u64, rendered(&b).len() as u64));
    assert_eq!(input_tokens(&srv, &a).await, ca);
    assert!(!rendered(&a).contains("billing"));

    // negative controls: a real change is a different prompt, on the wire too
    let mut changed = vec![];
    changed.push(body(json!({"system": sys("x-anthropic-billing-header: cch=1;").as_array().map(|a| json!([a[0], {"type": "text", "text": "You are Claude Code, v2."}])).unwrap(), "tools": [bash_tool("Run"), read_tool()]})));
    changed.push(body(json!({"system": sys("x-anthropic-billing-header: cch=1;"), "tools": [bash_tool("Run it"), read_tool()]})));
    changed.push(body(json!({"system": sys("x-anthropic-billing-header: cch=1;"), "tools": [bash_tool("Run")]})));
    changed.push(body(json!({"system": sys("x-anthropic-billing-header: cch=1;"), "tools": [bash_tool("Run"), read_tool()], "messages": [{"role": "user", "content": "hi!"}]})));
    changed.push(body(json!({"system": [{"type": "text", "text": "x-anthropic-billing-header: cch=1;"}, {"type": "text", "text": "You are Claude Code.\nToday's date is 2026-10-30."}], "tools": [bash_tool("Run"), read_tool()]})));
    for c in &changed {
        assert_ne!(rendered(c), rendered(&a), "{c}");
        assert_ne!(count(&srv, c).await, ca, "{c}");
    }
    assert_ne!(rendered(&changed[0]), rendered(&changed[1]));
    // key order inside a schema is kept as sent, so reordering it is a different prompt (and the route prefills those bytes)
    let reordered = body(json!({"system": sys("x-anthropic-billing-header: cch=1;"),
                                "tools": [read_tool(), {"name": "bash", "description": "Run", "input_schema": {"properties": {"timeout": {"type": "integer"}, "cmd": {"type": "string"}}, "type": "object"}}]}));
    assert_ne!(rendered(&reordered), rendered(&a));
    assert!(rendered(&reordered).contains("\"properties\": {\"timeout\"") && rendered(&a).contains("\"properties\": {\"cmd\""));
    assert_eq!(count(&srv, &reordered).await, rendered(&reordered).len() as u64);
}

/// Claude Code (2.1.x) sends `system` entries inside `messages`, one per turn (`<total_tokens>N tokens left</total_tokens>`),
/// after the tool result: hoisting each to the front system prompt made every turn's prompt differ at the front.
#[tokio::test(flavor = "multi_thread")]
async fn a_system_message_after_an_assistant_turn_stays_in_place_so_the_next_turns_prompt_extends_this_one() {
    let user = |t: &str| json!({"role": "user", "content": [{"type": "text", "text": t}]});
    let env = json!({"role": "system", "content": [{"type": "text", "text": "# Environment\nPlatform: darwin"}]});
    let counter = |n: u32| json!({"role": "system", "content": [{"type": "text", "text": format!("<total_tokens>{n} tokens left</total_tokens>"), "cache_control": {"type": "ephemeral"}}]});
    let call = json!({"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "a.py"}}]});
    let result = json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "print(1)"}]});
    let answer = json!({"role": "assistant", "content": [{"type": "text", "text": "It prints 1."}]});
    let req = |messages: Vec<Value>| body(json!({"system": "You are Claude Code.", "tools": [read_tool()], "messages": messages}));
    let a = req(vec![user("read a.py"), env.clone()]);
    let b = req(vec![user("read a.py"), env.clone(), call.clone(), result.clone(), counter(100)]);
    let c = req(vec![user("read a.py"), env.clone(), call.clone(), result.clone(), counter(100), answer.clone(), user("thanks"), counter(90)]);
    let gen = "<|im_start|>assistant\n";
    let head = |b: &Value| rendered(b).strip_suffix(gen).expect("ends with the generation prompt").to_string();
    // each request's prompt (up to the reply it asks for) is the start of the next one's: a cache can serve it
    assert!(rendered(&b).starts_with(&head(&a)), "turn 2 extends turn 1:\n{}\n---\n{}", rendered(&a), rendered(&b));
    assert!(rendered(&c).starts_with(&head(&b)), "turn 3 extends turn 2:\n{}\n---\n{}", rendered(&b), rendered(&c));
    // the counters are still told to the model, where they were sent
    assert!(rendered(&b).contains("<total_tokens>100 tokens left</total_tokens>") && rendered(&c).contains("<total_tokens>90 tokens left</total_tokens>"));
    assert!(rendered(&c).find("<total_tokens>100").unwrap() < rendered(&c).find("<total_tokens>90").unwrap());
    // ... and on the wire the route prefills exactly those bytes
    let srv = scripted(&["ok"]);
    assert_eq!(count(&srv, &c).await, rendered(&c).len() as u64);
    // negative controls: a system message before the first assistant turn is still the front system prompt (it is stable),
    // and a changed early part is still a different prompt start
    assert!(rendered(&a).starts_with("<|im_start|>system\nYou are Claude Code.\n\n# Environment\nPlatform: darwin"));
    let other_env = json!({"role": "system", "content": "# Environment\nPlatform: linux"});
    let b2 = req(vec![user("read a.py"), other_env, call.clone(), result.clone(), counter(100)]);
    assert!(!rendered(&b2).starts_with(&head(&a)), "a changed front is not a prefix");
}

// -- keep-alive -------------------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn a_slow_prefill_is_kept_alive_with_ping_events() {
    let fake = FakeConfig { script: Some(vec!["ok".into()]), prefill_ms: 400.0, ..fast() };
    let cfg = ServerConfig { port: 0, anthropic: AnthropicConfig { ping_ms: 40, ..Default::default() }, ..Default::default() };
    let srv = spawn_with(fake, Tokenizer::Bytes, cfg);
    let ev = stream_raw(&srv, &body(json!({}))).await;
    let s = shape(&ev);
    let first_block = s.iter().position(|e| e.starts_with("start:")).unwrap();
    let before = s[..first_block].iter().filter(|e| *e == "ping").count();
    assert_eq!(s[0], "message_start", "the stream opens before the prefill is done");
    assert!(before >= 3, "pings during the prefill: {s:?}");
    assert_eq!(s[first_block + 1], "ping");
    // the non-stream reply has none, and the answer is the same
    assert_eq!(text_of(&whole(&srv, &body(json!({}))).await), "ok");
}

// -- tool-result truncation -------------------------------------------------------------------------------------------

const N_CTX: usize = 4096;
const RESERVE: usize = 256;

fn limited(limit: ToolResultLimit) -> Srv {
    let cfg = ServerConfig { port: 0, anthropic: AnthropicConfig { tool_result_limit: limit, ..Default::default() }, ..Default::default() };
    spawn_with(FakeConfig { script: Some(vec!["ok".into()]), n_ctx: N_CTX, ..fast() }, Tokenizer::Bytes, cfg)
}

/// A tool result of `n` characters: a numbered line each, so the head and the tail can be told apart.
fn log(n: usize) -> String {
    let mut s = String::new();
    let mut i = 0;
    while s.len() < n {
        s.push_str(&format!("line {i:05}\n"));
        i += 1;
    }
    s.truncate(n);
    s
}

fn with_results(results: &[String]) -> Value {
    let uses: Vec<Value> = (0..results.len()).map(|i| json!({"type": "tool_use", "id": format!("toolu_{i}"), "name": "bash", "input": {"cmd": format!("cat {i}")}})).collect();
    let rs: Vec<Value> = results.iter().enumerate().map(|(i, r)| json!({"type": "tool_result", "tool_use_id": format!("toolu_{i}"), "content": r})).collect();
    body(json!({"tools": [bash_tool("Run")], "messages": [
        {"role": "user", "content": "run them"},
        {"role": "assistant", "content": uses},
        {"role": "user", "content": rs},
    ]}))
}

/// The prompt length the ladder should settle on, worked out from the library alone.
fn expected_len(b: &Value) -> (usize, usize) {
    let c = anthropic::convert(b.get("system"), b["messages"].as_array().unwrap(), b.get("tools")).unwrap();
    let t = ChatTemplate::chatml();
    for cap in CAPS {
        let (m, n) = anthropic::cut_tool_results(&c.messages, cap);
        let len = t.render_chat(&m, &c.tools, true, false).unwrap().len();
        if n > 0 && len + RESERVE <= N_CTX {
            return (len, n);
        }
    }
    panic!("nothing in the ladder fits");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_oversized_tool_result_is_cut_to_fit_instead_of_the_prompt_being_refused() {
    let b = with_results(&[log(20_000)]);
    let before = rendered(&b).len();
    assert!(before > N_CTX * 4);
    // without the rule (off) the prompt is refused: this is the failure the rule fixes
    let off = limited(ToolResultLimit::Off);
    let r = off.post("/v1/messages", &b).await;
    assert_eq!(r.status(), 400);
    let e: Value = r.json().await.unwrap();
    assert!(e["error"]["message"].as_str().unwrap().starts_with(&format!("prompt is too long: {before} tokens > {N_CTX} maximum")), "{e}");
    assert_eq!(off.post("/v1/messages/count_tokens", &b).await.json::<Value>().await.unwrap()["input_tokens"], before, "count_tokens does not cut when off");

    // with it (the default) the request succeeds, at the length the ladder gives
    let srv = limited(ToolResultLimit::Auto);
    let (want, cut) = expected_len(&b);
    let r = srv.post("/v1/messages", &b).await;
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["x-sllm-tool-results-cut"], cut.to_string().as_str());
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["usage"]["input_tokens"], want);
    assert!(want + RESERVE <= N_CTX && want > N_CTX / 4, "cut to fit, not to nothing: {want}");
    assert_eq!(v["content"][0]["text"], "ok");
    // count_tokens agrees, and so does the stream
    assert_eq!(count(&srv, &b).await, want as u64);
    assert_eq!(streamed(&srv, &b).await.usage["input_tokens"], want);
    // a second turn with the same history cuts the same way (a stable prefix)
    assert_eq!(count(&srv, &b).await, count(&srv, &b).await);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_cut_keeps_head_tail_marker_and_the_tool_use_pairing() {
    let b = with_results(&[log(20_000), "short result".to_string(), log(9_000)]);
    let c = anthropic::convert(b.get("system"), b["messages"].as_array().unwrap(), b.get("tools")).unwrap();
    let (cut, n) = anthropic::cut_tool_results(&c.messages, 2048);
    assert_eq!(n, 2, "only the two big results change");
    assert_eq!(cut.len(), c.messages.len(), "no message is dropped");
    let tools: Vec<(&Value, &Value)> = c.messages.iter().zip(&cut).filter(|(m, _)| m["role"] == "tool").collect();
    assert_eq!(tools.len(), 3);
    for (old, new) in &tools {
        assert_eq!(old["tool_call_id"], new["tool_call_id"], "every tool_result keeps the id of its tool_use");
    }
    assert_eq!(tools[1].0, tools[1].1, "a result under the cap is byte-identical");
    let big = tools[0].1["content"].as_str().unwrap();
    assert!(big.starts_with("line 00000\n") && big.contains("characters cut to fit the context") && big.ends_with(&log(20_000)[19_990..]), "{big}");
    assert!(big.chars().count() <= 2048);
    // the calls that asked for them are untouched
    let calls = |ms: &[Value]| -> Vec<Value> { ms.iter().filter_map(|m| m.get("tool_calls").cloned()).collect() };
    assert_eq!(calls(&cut), calls(&c.messages));
    // and the cut prompt still has one tool_response per result, in order
    let p = ChatTemplate::chatml().render_chat(&cut, &c.tools, true, false).unwrap();
    assert_eq!(p.matches("<tool_response>").count(), 3);
    assert_eq!(p.matches("\"arguments\": {\"cmd\"").count(), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_prompt_that_fits_is_byte_identical_whatever_the_limit() {
    let b = with_results(&[log(1_500), "short".to_string()]);
    let want = rendered(&b).len() as u64;
    assert!(want < N_CTX as u64 - RESERVE as u64);
    for limit in [ToolResultLimit::Auto, ToolResultLimit::Off, ToolResultLimit::Chars(4096)] {
        let srv = limited(limit);
        assert_eq!(count(&srv, &b).await, want, "{limit:?}");
        let r = srv.post("/v1/messages", &b).await;
        assert!(!r.headers().contains_key("x-sllm-tool-results-cut"), "{limit:?}");
        assert_eq!(r.json::<Value>().await.unwrap()["usage"]["input_tokens"], want);
    }
    // a prompt just under the context is not touched by auto either, however big its results
    let near = with_results(&[log(2_800)]);
    assert!(rendered(&near).len() < N_CTX);
    assert_eq!(count(&limited(ToolResultLimit::Auto), &near).await, rendered(&near).len() as u64);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_hard_limit_cuts_every_result_always_and_the_flag_values_parse() {
    let b = with_results(&[log(3_000)]);
    let off = limited(ToolResultLimit::Off);
    let hard = limited(ToolResultLimit::Chars(1_000));
    let (a, h) = (count(&off, &b).await, count(&hard, &b).await);
    assert!(h + 1_500 < a, "{h} vs {a}");
    assert_eq!(streamed(&hard, &b).await.usage["input_tokens"], h);
}

#[tokio::test(flavor = "multi_thread")]
async fn when_even_the_smallest_cut_does_not_fit_the_prompt_is_refused_as_too_long() {
    let mut b = with_results(&[log(20_000)]);
    b["messages"][0]["content"] = json!("x".repeat(6_000));
    let srv = limited(ToolResultLimit::Auto);
    let r = srv.post("/v1/messages", &b).await;
    assert_eq!(r.status(), 400);
    let e: Value = r.json().await.unwrap();
    assert!(e["error"]["message"].as_str().unwrap().starts_with("prompt is too long: "), "{e}");
}
