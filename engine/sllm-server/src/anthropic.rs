//! Anthropic Messages requests as chat-template input (`POST /v1/messages`, `/v1/messages/count_tokens`), pure and testable
//! without a server: [`convert`] turns `system`, `messages` and `tools` into the OpenAI-shaped messages and tools that
//! [`crate::chat::ChatTemplate::render_chat`] reads, applying the canonicalization rules below, and [`cut_tool_results`]
//! cuts oversized tool results. The route is `api/anthropic_api.rs`.
//!
//! Conversion: `system` (a string or text blocks) and the `system` entries inside `messages` become one system message at the
//! front, the pieces joined with a blank line; the text blocks of one message are joined with a blank line; a `tool_use`
//! block of an assistant message is a `tool_calls` entry (its `input` object is `arguments`); a `tool_result` block of a
//! user message is a `tool` message (its `content` a string or text blocks), in the order the blocks came, with the text
//! around it as user messages; `thinking` and `redacted_thinking` blocks of earlier turns are dropped (templates do not
//! render them; their signature is not checked); `cache_control` is ignored. An image, a document or any other block is an
//! error, so is a tool without `input_schema` (a server tool).
//!
//! Canonicalization, so that the rendered prompt is byte-stable across the turns of one conversation (a prompt cache hits only
//! on an identical prefix). Each rule changes only what carries no meaning:
//! 1. `x-anthropic-billing-header:` lines of the system text are dropped: Claude Code puts a per-request hash there (vllm-mlx
//!    strips it for the same reason). A block that is nothing else disappears.
//! 2. Tools are ordered by name (stable for equal names): the same set in another order is the same prompt.
//! 3. Arguments are re-serialised compactly from the parsed `input`, so white space in the client's JSON never reaches the
//!    prompt. Object KEY ORDER is kept as sent (serde_json and minijinja `preserve_order`, as llama.cpp does): the order of
//!    a tool's schema properties and of a call's arguments is part of what the model reads. Claude Code re-sends the same
//!    order every turn, so the prompt stays byte-stable; a client that reorders keys between requests gets a different
//!    prompt (no cache hit past that point), by rule. Tools are the exception to "as sent": rule 2 sorts them by name.
//! 4. Text blocks with no text are dropped (a client may send `{"type": "text", "text": ""}`; Anthropic itself refuses them).
//!
//! What is NOT normalized: message text (including dates and `<system-reminder>` blocks, which are content), the order of
//! messages, the set of tools (a tool added mid-session changes the prompt).

use serde_json::{json, Map, Value};

/// The messages and tools [`crate::chat::ChatTemplate::render_chat`] takes, and what the request asked of the cache.
#[derive(Debug, Clone, PartialEq)]
pub struct Converted {
    pub messages: Vec<Value>,
    pub tools: Vec<Value>,
    /// Some block, tool or system block carries `cache_control`: the client counts cache reads apart from `input_tokens`.
    pub cache_control: bool,
}

/// `--tool-result-limit`: what to do with oversized tool results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolResultLimit {
    /// Change nothing unless the prompt would be refused for not fitting the context; then cut the tool results to the
    /// largest cap of [`CAPS`] that makes it fit (default).
    Auto,
    /// Never cut: a prompt that does not fit is refused.
    Off,
    /// Every tool result is cut to this many characters, always.
    Chars(usize),
}

impl ToolResultLimit {
    pub fn parse(s: &str) -> Result<ToolResultLimit, String> {
        match s {
            "auto" => Ok(ToolResultLimit::Auto),
            "off" => Ok(ToolResultLimit::Off),
            n => match n.parse::<usize>() {
                Ok(n) if n >= MIN_CAP => Ok(ToolResultLimit::Chars(n)),
                _ => Err(format!("expected auto, off or a number of characters >= {MIN_CAP}, got {s:?}")),
            },
        }
    }

    pub fn name(self) -> String {
        match self {
            ToolResultLimit::Auto => "auto".into(),
            ToolResultLimit::Off => "off".into(),
            ToolResultLimit::Chars(n) => n.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    pub tool_result_limit: ToolResultLimit,
    /// An SSE `ping` goes out when no event has for this long (a long prefill); 0: never. Not a flag; tests set it.
    pub ping_ms: u64,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        AnthropicConfig { tool_result_limit: ToolResultLimit::Auto, ping_ms: 10_000 }
    }
}

/// The caps `auto` tries, largest first, in characters per tool result. A fixed ladder, not a fit to the exact overflow: the
/// cut of a result then stays the same over many turns, so the prompt stays a stable prefix until the next step down.
pub const CAPS: [usize; 9] = [65_536, 32_768, 16_384, 8_192, 4_096, 2_048, 1_024, 512, 256];
pub const MIN_CAP: usize = 256;
/// The marker's own length is at most this (it names the number of characters cut).
const MARKER_ROOM: usize = 64;

/// `text` cut to about `cap` characters, keeping its head (two thirds) and its tail (one third) around a marker that says
/// how much was cut. Text of at most `cap` characters is returned unchanged. Deterministic: the same text and cap give the
/// same bytes, so a cut result re-renders identically every turn.
pub fn cut_middle(text: &str, cap: usize) -> String {
    let n = text.chars().count();
    if n <= cap {
        return text.to_string();
    }
    let keep = cap.saturating_sub(MARKER_ROOM).max(1);
    let (head, tail) = (keep - keep / 3, keep / 3);
    let at = |k: usize| text.char_indices().nth(k).map_or(text.len(), |(i, _)| i);
    format!("{}\n[... {} characters cut to fit the context ...]\n{}", &text[..at(head)], n - head - tail, &text[at(n - tail)..])
}

/// `messages` with every tool result cut to `cap` characters ([`cut_middle`]); the message, its role and its
/// `tool_call_id` are kept, so each `tool_use` still has its `tool_result`. Also the number of results that changed.
pub fn cut_tool_results(messages: &[Value], cap: usize) -> (Vec<Value>, usize) {
    let mut cut = 0;
    let out = messages
        .iter()
        .map(|m| {
            let mut m = m.clone();
            if m["role"] == "tool" {
                if let Some(c) = m["content"].as_str().map(|c| cut_middle(c, cap)).filter(|c| Some(c.as_str()) != m["content"].as_str()) {
                    m["content"] = Value::String(c);
                    cut += 1;
                }
            }
            m
        })
        .collect();
    (out, cut)
}

const BILLING: &str = "x-anthropic-billing-header:";

/// Rule 1: the system text without its `x-anthropic-billing-header:` lines; the text itself when there are none.
pub fn canonical_system_text(text: &str) -> String {
    if !text.lines().any(|l| l.trim_start().starts_with(BILLING)) {
        return text.to_string();
    }
    text.split('\n').filter(|l| !l.trim_start().starts_with(BILLING)).collect::<Vec<_>>().join("\n").trim_matches('\n').to_string()
}

/// The text of a content that is a string or a list of text blocks (`what` names it in an error); blocks joined with a
/// blank line, empty ones dropped (rule 4).
fn text_of(content: &Value, what: &str) -> Result<String, String> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(blocks) => {
            let mut parts = Vec::new();
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let t = b.get("text").and_then(Value::as_str).ok_or_else(|| format!("{what}: a text block needs a string `text`"))?;
                        if !t.is_empty() {
                            parts.push(t);
                        }
                    }
                    Some(t) => return Err(unsupported_block(t, what)),
                    None => return Err(format!("{what}: each content block needs a `type`")),
                }
            }
            Ok(parts.join("\n\n"))
        }
        _ => Err(format!("{what}: content must be a string or a list of content blocks")),
    }
}

fn unsupported_block(kind: &str, what: &str) -> String {
    match kind {
        "image" | "document" | "audio" | "video" => format!("{what}: `{kind}` content blocks are not supported (this model has no vision); send text"),
        k => format!("{what}: content block type `{k}` is not supported"),
    }
}

fn has_cache_control(v: &Value) -> bool {
    v.get("cache_control").is_some_and(|c| !c.is_null())
}

/// The OpenAI-shaped messages and tools for an Anthropic request. `Err`: the request is not valid or asks for what the
/// server does not have (the message says what).
pub fn convert(system: Option<&Value>, messages: &[Value], tools: Option<&Value>) -> Result<Converted, String> {
    let mut cache_control = false;
    let mut sys: Vec<String> = Vec::new();
    if let Some(s) = system.filter(|s| !s.is_null()) {
        convert_system(s, &mut sys, &mut cache_control)?;
    }

    let mut out: Vec<Value> = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        convert_message(i, m, &mut sys, &mut out, &mut cache_control)?;
    }
    if !sys.is_empty() {
        out.insert(0, json!({"role": "system", "content": sys.join("\n\n")}));
    }

    let mut tool_list = Vec::new();
    if let Some(t) = tools.filter(|t| !t.is_null()) {
        for (i, t) in t.as_array().ok_or("tools: must be a list")?.iter().enumerate() {
            cache_control |= has_cache_control(t);
            tool_list.push(convert_tool(i, t)?);
        }
        // rule 2
        tool_list.sort_by(|a, b| a.pointer("/function/name").and_then(Value::as_str).cmp(&b.pointer("/function/name").and_then(Value::as_str)));
    }
    Ok(Converted { messages: out, tools: tool_list, cache_control })
}

/// A part of the front system prompt, unless it is empty once canonical.
fn add_system(text: String, sys: &mut Vec<String>) {
    let text = canonical_system_text(&text);
    if !text.is_empty() {
        sys.push(text);
    }
}

/// The request's `system`: a string, or text blocks (one block at a time).
fn convert_system(s: &Value, sys: &mut Vec<String>, cache_control: &mut bool) -> Result<(), String> {
    match s {
        Value::String(t) => add_system(t.clone(), sys),
        Value::Array(blocks) => {
            for (i, b) in blocks.iter().enumerate() {
                *cache_control |= has_cache_control(b);
                // one block at a time, so that a block that is only the billing line disappears
                add_system(text_of(&json!([b]), &format!("system.{i}"))?, sys);
            }
        }
        _ => return Err("system: must be a string or a list of text blocks".into()),
    }
    Ok(())
}

/// Message `i` of the request, onto `out` (a system entry before the first assistant turn onto `sys`).
fn convert_message(i: usize, m: &Value, sys: &mut Vec<String>, out: &mut Vec<Value>, cache_control: &mut bool) -> Result<(), String> {
    let what = format!("messages.{i}");
    let o = m.as_object().ok_or_else(|| format!("{what}: must be an object"))?;
    let role = o.get("role").and_then(Value::as_str).ok_or_else(|| format!("{what}.role: Field required"))?;
    let content = o.get("content").ok_or_else(|| format!("{what}.content: Field required"))?;
    if let Value::Array(blocks) = content {
        *cache_control |= blocks.iter().any(has_cache_control);
    }
    place_message(role, content, &what, sys, out)
}

/// A system entry after the first assistant turn: a user turn where it was sent (unless it is empty once canonical).
fn late_system(content: &Value, what: &str, out: &mut Vec<Value>) -> Result<(), String> {
    let text = canonical_system_text(&text_of(content, &format!("{what}.content"))?);
    if !text.is_empty() {
        out.push(json!({"role": "user", "content": text}));
    }
    Ok(())
}

/// A message of `role` onto `out`, or onto `sys`.
fn place_message(role: &str, content: &Value, what: &str, sys: &mut Vec<String>, out: &mut Vec<Value>) -> Result<(), String> {
    match role {
        // Rule 5: a system entry before the first assistant turn is part of the front system prompt; one after it (Claude Code
        // sends a `<total_tokens>` counter after every tool result) stays where it was sent, as a user turn. Hoisted, each
        // turn's new entry would change the front of the prompt, and no cache could serve the turns before it.
        "system" if out.iter().any(|m| m["role"] == "assistant") => late_system(content, what, out)?,
        "system" => add_system(text_of(content, &format!("{what}.content"))?, sys),
        "user" => user_message(content, what, out)?,
        "assistant" => out.push(assistant_message(content, what)?),
        r => return Err(format!("{what}.role: Input should be 'user' or 'assistant', got {r:?}")),
    }
    Ok(())
}

/// Tool `i` of the request as an OpenAI function (custom tools with an `input_schema` only).
fn convert_tool(i: usize, t: &Value) -> Result<Value, String> {
    let what = format!("tools.{i}");
    let kind = t.get("type").and_then(Value::as_str).unwrap_or("custom");
    let name = t.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()).ok_or_else(|| format!("{what}.name: Field required"))?;
    if kind != "custom" || t.get("input_schema").is_none() {
        return Err(format!("{what}: tool type {kind:?} is not supported (only custom tools with an `input_schema`; {name} looks like a server tool)"));
    }
    let mut f = Map::new();
    f.insert("name".into(), json!(name));
    if let Some(d) = t.get("description").filter(|d| d.is_string()) {
        f.insert("description".into(), d.clone());
    }
    f.insert("parameters".into(), t["input_schema"].clone());
    Ok(json!({"type": "function", "function": f}))
}

/// The text of a `text` content block onto `parts`, unless it is empty.
fn push_block_text<'a>(b: &'a Value, what: &str, parts: &mut Vec<&'a str>) -> Result<(), String> {
    let t = b.get("text").and_then(Value::as_str).ok_or_else(|| format!("{what}: a text block needs a string `text`"))?;
    if !t.is_empty() {
        parts.push(t);
    }
    Ok(())
}

/// The error for a content block that is not one of a message's kinds.
fn bad_block(kind: Option<&str>, what: &str) -> String {
    match kind {
        Some(t) => unsupported_block(t, what),
        None => format!("{what}: each content block needs a `type`"),
    }
}

fn user_message(content: &Value, what: &str, out: &mut Vec<Value>) -> Result<(), String> {
    let Value::Array(blocks) = content else {
        out.push(json!({"role": "user", "content": text_of(content, &format!("{what}.content"))?}));
        return Ok(());
    };
    let mut pending: Vec<&str> = Vec::new();
    let flush = |pending: &mut Vec<&str>, out: &mut Vec<Value>| {
        if !pending.is_empty() {
            out.push(json!({"role": "user", "content": pending.join("\n\n")}));
            pending.clear();
        }
    };
    for (j, b) in blocks.iter().enumerate() {
        let what = format!("{what}.content.{j}");
        match b.get("type").and_then(Value::as_str) {
            Some("text") => push_block_text(b, &what, &mut pending)?,
            Some("tool_result") => {
                flush(&mut pending, out);
                let id = b.get("tool_use_id").and_then(Value::as_str).ok_or_else(|| format!("{what}.tool_use_id: Field required"))?;
                let text = text_of(b.get("content").unwrap_or(&Value::Null), &format!("{what}.content"))?;
                out.push(json!({"role": "tool", "tool_call_id": id, "content": text}));
            }
            kind => return Err(bad_block(kind, &what)),
        }
    }
    flush(&mut pending, out);
    Ok(())
}

/// A `tool_use` block as an OpenAI tool call.
fn tool_use_call(b: &Value, what: &str) -> Result<Value, String> {
    let id = b.get("id").and_then(Value::as_str).ok_or_else(|| format!("{what}.id: Field required"))?;
    let name = b.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()).ok_or_else(|| format!("{what}.name: Field required"))?;
    let input = match b.get("input") {
        None | Some(Value::Null) => json!({}),
        Some(i @ Value::Object(_)) => i.clone(),
        Some(_) => return Err(format!("{what}.input: must be an object")),
    };
    Ok(json!({"id": id, "type": "function", "function": {"name": name, "arguments": input}}))
}

fn assistant_message(content: &Value, what: &str) -> Result<Value, String> {
    let Value::Array(blocks) = content else {
        return Ok(json!({"role": "assistant", "content": text_of(content, &format!("{what}.content"))?}));
    };
    let (mut text, mut calls) = (Vec::<&str>::new(), Vec::<Value>::new());
    for (j, b) in blocks.iter().enumerate() {
        let what = format!("{what}.content.{j}");
        match b.get("type").and_then(Value::as_str) {
            Some("text") => push_block_text(b, &what, &mut text)?,
            Some("tool_use") => calls.push(tool_use_call(b, &what)?),
            Some("thinking") | Some("redacted_thinking") => {}
            kind => return Err(bad_block(kind, &what)),
        }
    }
    let mut m = json!({"role": "assistant", "content": text.join("\n\n")});
    if !calls.is_empty() {
        m["tool_calls"] = Value::Array(calls);
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::ChatTemplate;

    fn render(c: &Converted) -> String {
        ChatTemplate::chatml().render_chat(&c.messages, &c.tools, true, false).unwrap()
    }

    fn conv(system: Value, messages: Value, tools: Value) -> Converted {
        convert(Some(&system), messages.as_array().unwrap(), Some(&tools)).unwrap()
    }

    #[test]
    fn cut_middle_keeps_head_and_tail_and_says_how_much() {
        let text: String = (0..1000).map(|i| char::from(b'a' + (i % 26) as u8)).collect();
        assert_eq!(cut_middle(&text, 1000), text);
        assert_eq!(cut_middle(&text, 5000), text);
        let cut = cut_middle(&text, 300);
        assert!(cut.chars().count() <= 300, "{}", cut.chars().count());
        let (head, rest) = cut.split_once("\n[... ").unwrap();
        let (marker, tail) = rest.split_once(" characters cut to fit the context ...]\n").unwrap();
        assert!(text.starts_with(head) && text.ends_with(tail), "head and tail are the text's own");
        assert_eq!(head.len() + tail.len() + marker.parse::<usize>().unwrap(), 1000, "the marker counts what was cut");
        assert!(head.len() > tail.len() && tail.len() > 0);
        assert_eq!(cut_middle(&text, 300), cut, "deterministic");
        // multi-byte text is cut on characters
        let wide: String = "日本語".repeat(300);
        let cut = cut_middle(&wide, 300);
        assert!(cut.chars().count() <= 300 && cut.starts_with("日本語") && cut.contains("characters cut"), "{cut}");
    }

    #[test]
    fn billing_lines_are_dropped_and_other_system_text_is_not_touched() {
        let line = "x-anthropic-billing-header: cc_version=2.1.9.abc; cc_entrypoint=cli; cch=12345;";
        assert_eq!(canonical_system_text(line), "");
        assert_eq!(canonical_system_text(&format!("{line}\nYou are Claude Code.")), "You are Claude Code.");
        assert_eq!(canonical_system_text(&format!("A\n{line}\nB")), "A\nB");
        for same in ["You are X.", "  indented\n\nkeeps  spaces \n", "Today's date is 2026-10-30.", "mentions x-anthropic-billing-header: mid-line"] {
            assert_eq!(canonical_system_text(same), same);
        }
    }

    #[test]
    fn requests_that_differ_only_in_what_carries_no_meaning_render_the_same_bytes() {
        let tool = |n: &str| json!({"name": n, "description": "d", "input_schema": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}});
        let base = conv(
            json!([{"type": "text", "text": "x-anthropic-billing-header: cc_version=1.2.3.aaa; cch=11111;"}, {"type": "text", "text": "You are Claude Code."}]),
            json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "/a", "n": 1}}]},
                   {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}]),
            json!([tool("read"), tool("bash"), tool("edit")]),
        );
        let same = conv(
            // another hash in the billing line, the blocks with cache_control and an empty text block
            json!([{"type": "text", "text": "x-anthropic-billing-header: cc_version=1.2.4.zzz; cch=99999999999;"}, {"type": "text", "text": ""}, {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}}]),
            // arguments written with other white space (the same keys in the same order)
            json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read", "input": serde_json::from_str::<Value>("{ \"path\" : \"/a\",\n \"n\":1 }").unwrap()}]},
                   {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "ok"}]}]}]),
            // the tools in another order (and one tool's own keys in another order: only the order of the tool objects' keys is free,
            // not the order inside its schema)
            json!([{"description": "d", "name": "edit", "input_schema": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}}, tool("read"), tool("bash")]),
        );
        assert_eq!(render(&base), render(&same));
        assert!(render(&base).contains("You are Claude Code.") && !render(&base).contains("billing"));
        // the tools are in name order in the prompt
        let p = render(&base);
        let at = |n: &str| p.find(&format!("\"name\": \"{n}\"")).unwrap();
        assert!(at("bash") < at("edit") && at("edit") < at("read"));

        // negative controls: each real change changes the bytes
        let changed = |system: Value, messages: Value, tools: Value| render(&conv(system, messages, tools));
        let (s, m, t) = (
            json!([{"type": "text", "text": "x-anthropic-billing-header: cc_version=1.2.3.aaa; cch=11111;"}, {"type": "text", "text": "You are Claude Code."}]),
            json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "/a", "n": 1}}]},
                   {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}]),
            json!([tool("read"), tool("bash"), tool("edit")]),
        );
        assert_eq!(changed(s.clone(), m.clone(), t.clone()), render(&base), "the control itself is stable");
        let mut s2 = s.clone();
        s2[1]["text"] = json!("You are Claude Code!");
        assert_ne!(changed(s2, m.clone(), t.clone()), render(&base), "system text");
        let mut s3 = s.clone();
        s3[1]["text"] = json!("You are Claude Code.\nToday's date is 2026-10-31.");
        assert_ne!(changed(s3, m.clone(), t.clone()), render(&base), "a date line is content, not noise");
        let mut m2 = m.clone();
        m2[1]["content"][0]["input"]["n"] = json!(2);
        assert_ne!(changed(s.clone(), m2, t.clone()), render(&base), "a tool argument value");
        let mut m3 = m.clone();
        m3[2]["content"][0]["content"] = json!("ok!");
        assert_ne!(changed(s.clone(), m3, t.clone()), render(&base), "a tool result");
        let mut t2 = t.clone();
        t2[0]["description"] = json!("other");
        assert_ne!(changed(s.clone(), m.clone(), t2), render(&base), "a tool description");
        let t3 = json!([tool("read"), tool("bash")]);
        assert_ne!(changed(s.clone(), m.clone(), t3), render(&base), "a missing tool");
        // key order is meaning the model reads and is kept as sent (rule 3): reordered arguments, or a reordered schema, are a
        // different prompt, and each is rendered in the order it was sent
        let mut m4 = m.clone();
        m4[1]["content"][0]["input"] = serde_json::from_str("{\"n\": 1, \"path\": \"/a\"}").unwrap();
        let reordered_args = changed(s.clone(), m4, t.clone());
        assert_ne!(reordered_args, render(&base), "arguments with their keys in another order");
        assert!(render(&base).contains("{\"path\": \"/a\", \"n\": 1}") && reordered_args.contains("{\"n\": 1, \"path\": \"/a\"}"));
        let mut t4 = t.clone();
        t4[0]["input_schema"] = serde_json::from_str("{\"properties\": {\"b\": {\"type\": \"integer\"}, \"a\": {\"type\": \"string\"}}, \"type\": \"object\"}").unwrap();
        assert_ne!(changed(s.clone(), m.clone(), t4), render(&base), "a schema with its keys in another order");
    }

    #[test]
    fn the_round_trip_renders_the_tool_result_where_the_template_puts_it() {
        let c = conv(
            json!("Be brief."),
            json!([{"role": "user", "content": "Weather?"},
                   {"role": "assistant", "content": [{"type": "text", "text": "Checking."}, {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}}]},
                   {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "sunny"}, {"type": "text", "text": "And Rome?"}]}]),
            json!([{"name": "get_weather", "description": "W", "input_schema": {"type": "object"}}]),
        );
        let p = render(&c);
        assert!(p.starts_with("<|im_start|>system\nBe brief.\n\n# Tools\n"), "{p}");
        assert!(p.ends_with(
            "<|im_start|>user\nWeather?<|im_end|>\n\
             <|im_start|>assistant\nChecking.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call><|im_end|>\n\
             <|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n\
             <|im_start|>user\nAnd Rome?<|im_end|>\n\
             <|im_start|>assistant\n"
        ), "{p}");
    }

    #[test]
    fn system_pieces_are_one_system_message_at_the_front() {
        let c = convert(
            Some(&json!("A")),
            &[json!({"role": "user", "content": "q"}), json!({"role": "system", "content": [{"type": "text", "text": "B"}]}), json!({"role": "assistant", "content": "a"})],
            None,
        )
        .unwrap();
        assert_eq!(c.messages, vec![json!({"role": "system", "content": "A\n\nB"}), json!({"role": "user", "content": "q"}), json!({"role": "assistant", "content": "a"})]);
    }

    #[test]
    fn what_is_not_supported_is_an_error_that_says_so() {
        let err = |messages: Value, tools: Value| convert(None, messages.as_array().unwrap(), Some(&tools)).unwrap_err();
        assert!(err(json!([{"role": "user", "content": [{"type": "image", "source": {}}]}]), json!(null)).contains("image"));
        assert!(err(json!([{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": [{"type": "image"}]}]}]), json!(null)).contains("image"));
        assert!(err(json!([{"role": "user", "content": [{"type": "document"}]}]), json!(null)).contains("document"));
        assert!(err(json!([{"role": "tool", "content": "x"}]), json!(null)).contains("role"));
        assert!(err(json!([{"role": "user"}]), json!(null)).contains("content"));
        assert!(err(json!([{"role": "user", "content": "x"}]), json!([{"type": "web_search_20250305", "name": "web_search"}])).contains("server tool"));
    }

    #[test]
    fn tool_results_are_cut_not_dropped_and_what_fits_is_untouched() {
        let big = "x".repeat(5000);
        let msgs = vec![json!({"role": "user", "content": "q"}), json!({"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": {}}}]}),
                        json!({"role": "tool", "tool_call_id": "t1", "content": big}), json!({"role": "tool", "tool_call_id": "t2", "content": "short"})];
        let (same, n) = cut_tool_results(&msgs, 5000);
        assert_eq!((same, n), (msgs.clone(), 0));
        let (cut, n) = cut_tool_results(&msgs, 1000);
        assert_eq!(n, 1);
        assert_eq!(cut.len(), msgs.len());
        assert_eq!(cut[2]["tool_call_id"], "t1", "the pairing key is kept");
        assert_eq!(cut[3], msgs[3]);
        assert!(cut[2]["content"].as_str().unwrap().contains("characters cut") && cut[2]["content"].as_str().unwrap().len() <= 1000);
        assert_eq!(cut[..2], msgs[..2]);
    }

    #[test]
    fn the_limit_flag_parses() {
        let accepted = [("auto", ToolResultLimit::Auto), ("off", ToolResultLimit::Off), ("4096", ToolResultLimit::Chars(4096))];
        for (text, limit) in accepted {
            assert_eq!(ToolResultLimit::parse(text), Ok(limit));
        }
        assert!(ToolResultLimit::parse("100").is_err() && ToolResultLimit::parse("x").is_err());
    }
}
