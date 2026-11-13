//! Chat prompts ([`ChatTemplate::render_chat`], `POST /v1/chat/completions`) and events for agents: an outside event becomes text in the model's own chat format (`POST /v1/agents/{id}/event`).
//!
//! An agent's context ends with its last reply (the end-of-turn token is never fed). An event is appended as what the
//! model's chat template renders after that reply:
//!
//! * **synthetic** (default): the reply gets a tool call `name(arguments)` and the event is that call's result. The model
//!   reads the event the way it reads any tool result, so no prompt wording is needed (the "synthetic tool call").
//! * **tool_result**: the reply already ended with a tool call (the agent asked for it); the event is its result.
//!
//! The text comes from rendering the GGUF's Jinja chat template twice over marker strings, the way the playground's
//! agent loop derives its framing ([`ChatTemplate::event_text`]). A template without a tool role (Gemma 3 raises, some
//! skip the message) gets the event as a user message instead, `Framing::User`. Without a GGUF (`--fake`) a ChatML
//! template with Qwen's tool format stands in.
//!
//! Rendering follows Hugging Face's: trim_blocks and lstrip_blocks on, `tojson` as Python's `json.dumps` (", " and ": "
//! separators, no HTML escaping, non-ASCII kept), Python string methods (minijinja-contrib pycompat), `raise_exception`,
//! `strftime_now`, `add_generation_prompt` and `enable_thinking` false (Qwen 3 then opens the reply with an empty think
//! block), `strftime_now` from the real clock (UTC; llama.cpp uses local time), and, as llama.cpp does, a system message merged into the
//! first turn ("system\nuser") for a template that never reads it (Mistral). Object keys keep their insertion order (serde_json and minijinja `preserve_order`), as Python does.

use minijinja::value::Kwargs;
use minijinja::{Environment, Error, ErrorKind, Value as J};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

const SYS: &str = "\u{1}SYS\u{1}";
const USR: &str = "\u{1}USR\u{1}";
const ANS: &str = "\u{1}ANS\u{1}";
const EVT: &str = "\u{1}EVT\u{1}";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    Tool,
    User,
}

impl Framing {
    pub fn name(self) -> &'static str {
        match self {
            Framing::Tool => "tool",
            Framing::User => "user",
        }
    }
}

pub struct Event<'a> {
    pub name: &'a str,
    pub arguments: &'a Value,
    pub content: &'a str,
    /// The reply gets a tool call made up for the event; false: the event answers the reply's own tool call.
    pub synthetic: bool,
    pub call_id: &'a str,
}

pub struct ChatTemplate {
    env: Environment<'static>,
    bos: String,
    eos: String,
    pub source: String,
    /// Seconds since 1970 that `strftime_now` formats (as UTC); `NO_CLOCK`: the real clock.
    clock: Arc<AtomicI64>,
    /// The template reads a system message (probed once); false: render_chat merges it into the next message.
    system_role: bool,
    /// The vocabulary's marker-shaped tokens (`<|x|>`, `[X]`), for [`crate::toolcall::ToolFormat::detect`].
    specials: std::collections::HashSet<String>,
}

const NO_CLOCK: i64 = i64::MIN;

/// ChatML with Qwen's tool format: the stand-in without a GGUF.
const CHATML_TOOLS: &str = r##"{%- if tools %}
{{- '<|im_start|>system\n' }}
{%- if messages[0].role == "system" %}{{- messages[0].content + '\n\n' }}{% endif %}
{{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
{%- for tool in tools %}{{- "\n" + (tool | tojson) }}{% endfor %}
{{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- endif %}
{%- for message in messages %}
{%- if message.role == "system" and tools and loop.first %}
{%- elif message.role == "tool" %}
{{- '<|im_start|>user\n<tool_response>\n' + message.content + '\n</tool_response><|im_end|>\n' }}
{%- else %}
{{- '<|im_start|>' + message.role + '\n' + message.content }}
{%- if message.tool_calls %}{% for tc in message.tool_calls %}
{{- '\n<tool_call>\n{"name": "' + tc.function.name + '", "arguments": ' + (tc.function.arguments | tojson) + '}\n</tool_call>' }}
{%- endfor %}{% endif %}
{{- '<|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{% endif %}"##;

impl ChatTemplate {
    pub fn new(source: &str, bos: &str, eos: &str) -> Result<ChatTemplate, String> {
        let mut env = Environment::new();
        env.set_syntax(minijinja::syntax::SyntaxConfig::builder().trim_blocks(true).lstrip_blocks(true).build().map_err(|e| e.to_string())?);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_function("raise_exception", |msg: String| -> Result<J, Error> { Err(Error::new(ErrorKind::InvalidOperation, msg)) });
        let clock = Arc::new(AtomicI64::new(NO_CLOCK));
        let c = clock.clone();
        env.add_function("strftime_now", move |fmt: String| strftime(&fmt, now(&c)));
        env.add_template_owned("chat", source.to_string()).map_err(|e| format!("chat template: {e}"))?;
        let mut t = ChatTemplate { env, bos: bos.into(), eos: eos.into(), source: source.into(), clock, system_role: true, specials: Default::default() };
        // llama.cpp's probe: a template that does not print a system message's content cannot take one
        t.system_role = t.render(&json!([{"role": "system", "content": SYS}, {"role": "user", "content": USR}])).is_ok_and(|r| r.contains(SYS));
        Ok(t)
    }

    /// `date_string` and `datetime` are variables llama.cpp gives every template (Llama 3.1 prints `date_string`).
    fn strftime_now(&self, fmt: &str) -> String {
        strftime(fmt, now(&self.clock))
    }

    /// Pin what `strftime_now` formats (seconds since 1970, read as UTC), for reproducible prompts.
    pub fn set_clock(&self, unix_seconds: i64) {
        self.clock.store(unix_seconds, Ordering::Relaxed);
    }

    pub fn chatml() -> ChatTemplate {
        ChatTemplate::new(CHATML_TOOLS, "", "<|im_end|>").expect("the built-in template parses")
    }

    /// The template stored in the GGUF, with its BOS and EOS token text.
    pub fn from_gguf(path: &Path) -> Result<ChatTemplate, String> {
        let g = sllm_core::gguf::GgufFile::open(path).map_err(|e| e.to_string())?;
        let src = g.get_str("tokenizer.chat_template").ok_or_else(|| format!("{} has no chat template", path.display()))?;
        let tok = |key: &str| g.get_u32(key).and_then(|i| g.arr_str("tokenizer.ggml.tokens", i as usize)).unwrap_or_default();
        let mut t = ChatTemplate::new(&src, &tok("tokenizer.ggml.bos_token_id"), &tok("tokenizer.ggml.eos_token_id"))?;
        let shaped = |s: &str| s.len() <= 48 && ((s.starts_with('<') && s.ends_with('>')) || (s.starts_with('[') && s.ends_with(']')));
        t.specials = (0..g.arr_len("tokenizer.ggml.tokens").unwrap_or(0)).filter_map(|i| g.arr_str("tokenizer.ggml.tokens", i)).filter(|s| shaped(s)).collect();
        Ok(t)
    }

    /// Declare the vocabulary's marker-shaped tokens (what [`ChatTemplate::from_gguf`] reads itself).
    pub fn with_specials(mut self, tokens: &[&str]) -> ChatTemplate {
        self.specials = tokens.iter().map(|t| t.to_string()).collect();
        self
    }

    pub fn bos(&self) -> &str {
        &self.bos
    }

    pub fn eos(&self) -> &str {
        &self.eos
    }

    /// The tool-call format this template writes ([`crate::toolcall`]); `None`: it has no tool syntax.
    pub fn tool_format(&self) -> Option<crate::toolcall::ToolFormat> {
        crate::toolcall::ToolFormat::detect(&self.source, &self.specials)
    }

    fn render(&self, messages: &Value) -> Result<String, String> {
        self.render_with(messages, &Value::Null, true, false)
    }

    fn render_with(&self, messages: &Value, tools: &Value, add_generation_prompt: bool, enable_thinking: bool) -> Result<String, String> {
        let t = self.env.get_template("chat").map_err(|e| e.to_string())?;
        t.render(minijinja::context! {
            messages => J::from(minijinja::value::Serde(messages)), add_generation_prompt => add_generation_prompt, enable_thinking => enable_thinking,
            date_string => self.strftime_now("%d %b %Y"), datetime => self.strftime_now("%b %d %Y"),
            tools => if tools.is_null() { J::from(()) } else { J::from(minijinja::value::Serde(tools)) }, bos_token => &self.bos, eos_token => &self.eos,
        })
        .map_err(|e| e.to_string())
    }

    /// The turn frames of this template, read off a render of three turns: the text between the turns, and the frame that
    /// opens a user turn. The first turn can have a default system message in front of it, so the user frame is what that
    /// text and the text between turns end with, without the close they share with the reply frame.
    pub fn turn_frames(&self) -> Result<crate::compaction::TurnFrames, String> {
        const MARK: [&str; 3] = ["SLLMFRAMEA", "SLLMFRAMEB", "SLLMFRAMEC"];
        let msgs = json!([{"role": "user", "content": MARK[0]}, {"role": "assistant", "content": MARK[1]}, {"role": "user", "content": MARK[2]}]);
        let s = self.render_with(&msgs, &Value::Null, true, false)?;
        let at = |m: &str| s.find(m).ok_or_else(|| format!("the template dropped {m}"));
        let (a, b, c) = (at(MARK[0])?, at(MARK[1])? + MARK[1].len(), at(MARK[2])?);
        let (before, to_user, to_reply) = (&s[..a], &s[b..c], &s[c + MARK[2].len()..]);
        let tail = before.bytes().rev().zip(to_user.bytes().rev()).take_while(|(x, y)| x == y).count();
        let mut user_open = &to_user[to_user.len() - tail..];
        if tail == to_user.len() {
            let shared = to_user.bytes().zip(to_reply.bytes()).take_while(|(x, y)| x == y).count();
            if let Some(i) = to_user[..shared].rfind('\n') {
                user_open = &to_user[i + 1..];
            }
        }
        Ok(crate::compaction::TurnFrames { user_open: user_open.to_string(), to_user: to_user.to_string(), to_reply: to_reply.to_string() })
    }

    /// The prompt for an OpenAI chat request: `messages` (roles system/developer, user, assistant with `tool_calls`, tool;
    /// `content` a string, null or a list of text parts) and `tools` (`[{"type": "function", "function": {...}}]`) rendered
    /// by the GGUF's template, the way llama-server `--jinja` renders them. The messages are first put in the shape Hugging
    /// Face templates read: `content` is always a string, `developer` is `system`, and a call's `arguments` (a JSON string
    /// in OpenAI's wire) is the object it holds, so `arguments | tojson` writes JSON once, not a quoted string.
    ///
    /// `RenderError::Request`: the messages cannot be rendered (an image part, no role). `RenderError::Template`: the
    /// template itself refused them (`raise_exception`, e.g. roles that must alternate).
    pub fn render_chat(&self, messages: &[Value], tools: &[Value], add_generation_prompt: bool, enable_thinking: bool) -> Result<String, RenderError> {
        let mut msgs = messages.iter().map(normalize_message).collect::<Result<Vec<_>, _>>().map_err(RenderError::Request)?;
        if !self.system_role && msgs.first().is_some_and(|m| m["role"] == "system") {
            let sys = msgs.remove(0);
            if let Some(next) = msgs.first_mut() {
                next["content"] = json!(format!("{}\n{}", sys["content"].as_str().unwrap_or(""), next["content"].as_str().unwrap_or("")));
            }
        }
        let tools = if tools.is_empty() { Value::Null } else { Value::Array(tools.to_vec()) };
        self.render_with(&Value::Array(msgs), &tools, add_generation_prompt, enable_thinking).map_err(RenderError::Template)
    }

    /// The text that follows the agent's last reply when `ev` arrives, and how the template could frame it.
    pub fn event_text(&self, ev: &Event) -> Result<(String, Framing), String> {
        let call = json!({"id": ev.call_id, "type": "function", "function": {"name": ev.name, "arguments": ev.arguments}});
        let reply = if ev.synthetic { json!({"role": "assistant", "content": ANS, "tool_calls": [call]}) } else { json!({"role": "assistant", "content": ANS}) };
        let tool = json!({"role": "tool", "content": EVT, "name": ev.name, "tool_call_id": ev.call_id});
        let label = if ev.synthetic { "event" } else { "tool result" };
        let as_user = json!({"role": "user", "content": EVT});
        let mut err = String::new();
        for (framing, reply, msg) in [(Framing::Tool, &reply, &tool), (Framing::User, &json!({"role": "assistant", "content": ANS}), &as_user)] {
            // with a system message, then (templates without a system role) the system text merged into the first user turn
            for head in [json!([{"role": "system", "content": SYS}, {"role": "user", "content": USR}]), json!([{"role": "user", "content": format!("{SYS}\n\n{USR}")}])] {
                let mut msgs = head.as_array().unwrap().clone();
                msgs.push(reply.clone());
                msgs.push(msg.clone());
                match self.render(&Value::Array(msgs)) {
                    Ok(r) => {
                        if let Some(tail) = after_reply(&r) {
                            let body = match framing {
                                Framing::Tool => ev.content.to_string(),
                                Framing::User => format!("[{label}: {}]\n{}", ev.name, ev.content),
                            };
                            return Ok((tail.replacen(EVT, &body, 1), framing));
                        }
                        err = "the template does not render the message after the reply".into();
                    }
                    Err(e) => err = e,
                }
            }
        }
        Err(format!("the chat template cannot frame an event: {err}"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RenderError {
    /// The request's messages are not renderable.
    Request(String),
    /// The chat template raised.
    Template(String),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderError::Request(m) => write!(f, "{m}"),
            RenderError::Template(m) => write!(f, "the chat template cannot render these messages: {m}"),
        }
    }
}

/// One OpenAI message in the shape chat templates read (see [`ChatTemplate::render_chat`]).
fn normalize_message(m: &Value) -> Result<Value, String> {
    let mut o = m.as_object().ok_or("each message must be an object")?.clone();
    let role = o.get("role").and_then(Value::as_str).ok_or("each message needs a string `role`")?;
    if role == "developer" {
        o.insert("role".into(), json!("system"));
    }
    let content = match o.get("content") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for p in parts {
                match p {
                    Value::String(s) => text.push_str(s),
                    Value::Object(po) if po.get("type").and_then(Value::as_str) == Some("text") => {
                        text.push_str(po.get("text").and_then(Value::as_str).ok_or("a text content part needs a string `text`")?)
                    }
                    Value::Object(po) => return Err(format!("content part type {:?} is not supported (text only)", po.get("type").and_then(Value::as_str).unwrap_or("?"))),
                    _ => return Err("a content part must be an object".into()),
                }
            }
            text
        }
        Some(_) => return Err("`content` must be a string, null or a list of text parts".into()),
    };
    o.insert("content".into(), Value::String(content));
    if let Some(calls) = o.get_mut("tool_calls").and_then(Value::as_array_mut) {
        for c in calls {
            if let Some(a) = c.pointer_mut("/function/arguments") {
                if let Some(parsed) = a.as_str().and_then(|s| serde_json::from_str::<Value>(s).ok()) {
                    *a = parsed;
                }
            }
        }
    }
    Ok(Value::Object(o))
}

/// What a render holds after the reply marker, when the event marker is in it once.
fn after_reply(r: &str) -> Option<&str> {
    if r.matches(ANS).count() != 1 {
        return None;
    }
    let tail = &r[r.find(ANS)? + ANS.len()..];
    (tail.matches(EVT).count() == 1).then_some(tail)
}

/// Python's `json.dumps(x, ensure_ascii=False, indent=indent)`, as Hugging Face's `tojson`; `indent` positional or keyword.
fn tojson(v: J, pos_indent: Option<usize>, kw: Kwargs) -> Result<J, Error> {
    let indent = match kw.get::<Option<usize>>("indent")? {
        Some(n) => Some(n),
        None => pos_indent,
    };
    // the ways llama.cpp's and Hugging Face's tojson differ from this one are refused, not ignored
    for unsupported in ["ensure_ascii", "sort_keys", "separators"] {
        if kw.get::<Option<J>>(unsupported)?.is_some_and(|x| x.is_true() || unsupported == "separators") {
            return Err(Error::new(ErrorKind::InvalidOperation, format!("tojson({unsupported}=...) is not supported")));
        }
    }
    kw.assert_all_used()?;
    let v: Value = serde_json::to_value(&v).map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
    let mut out = String::new();
    py_json(&v, indent, 0, &mut out);
    Ok(J::from_safe_string(out))
}

fn py_json(v: &Value, indent: Option<usize>, depth: usize, out: &mut String) {
    let (open_sep, item_sep, close_sep) = match indent {
        Some(n) => (format!("\n{}", " ".repeat(n * (depth + 1))), format!(",\n{}", " ".repeat(n * (depth + 1))), format!("\n{}", " ".repeat(n * depth))),
        None => (String::new(), ", ".to_string(), String::new()),
    };
    match v {
        Value::Array(a) if !a.is_empty() => {
            out.push('[');
            out.push_str(&open_sep);
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(&item_sep);
                }
                py_json(x, indent, depth + 1, out);
            }
            out.push_str(&close_sep);
            out.push(']');
        }
        Value::Object(m) if !m.is_empty() => {
            out.push('{');
            out.push_str(&open_sep);
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(&item_sep);
                }
                out.push_str(&serde_json::to_string(k).unwrap());
                out.push_str(": ");
                py_json(x, indent, depth + 1, out);
            }
            out.push_str(&close_sep);
            out.push('}');
        }
        Value::Number(n) if n.is_f64() => {
            let f = n.as_f64().unwrap();
            out.push_str(&if f.fract() == 0.0 && f.abs() < 1e16 { format!("{f:.1}") } else { n.to_string() });
        }
        other => out.push_str(&serde_json::to_string(other).unwrap()),
    }
}

fn now(clock: &AtomicI64) -> i64 {
    match clock.load(Ordering::Relaxed) {
        NO_CLOCK => std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64),
        t => t,
    }
}

/// The civil date (year, month, day) of a day count since 1970-01-01 (Howard Hinnant's algorithm).
pub(crate) fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + (m <= 2) as i64, m, d)
}

/// C `strftime` for the directives chat templates use, on a UTC time.
fn strftime(fmt: &str, unix: i64) -> String {
    const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    const DAYS: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    let (days, secs) = (unix.div_euclid(86400), unix.rem_euclid(86400));
    let (y, m, d) = civil_from_days(days);
    let wd = DAYS[(days + 4).rem_euclid(7) as usize];
    let mut out = String::new();
    let mut it = fmt.chars();
    while let Some(c) = it.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('Y') => out.push_str(&y.to_string()),
            Some('y') => out.push_str(&format!("{:02}", y.rem_euclid(100))),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('e') => out.push_str(&format!("{d:2}")),
            Some('B') => out.push_str(MONTHS[m as usize - 1]),
            Some('b') | Some('h') => out.push_str(&MONTHS[m as usize - 1][..3]),
            Some('A') => out.push_str(wd),
            Some('a') => out.push_str(&wd[..3]),
            Some('H') => out.push_str(&format!("{:02}", secs / 3600)),
            Some('M') => out.push_str(&format!("{:02}", secs / 60 % 60)),
            Some('S') => out.push_str(&format!("{:02}", secs % 60)),
            Some('F') => out.push_str(&format!("{y}-{m:02}-{d:02}")),
            Some('%') => out.push('%'),
            Some(o) => {
                out.push('%');
                out.push(o);
            }
            None => out.push('%'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strftime_formats_a_utc_date() {
        // 2026-10-30 14:05:09 UTC, a Friday
        let t = 1793369109;
        assert_eq!(strftime("%d %b %Y|%d %B %Y|%Y-%m-%d|%A %H:%M:%S %%", t), "30 Oct 2026|30 October 2026|2026-10-30|Friday 14:05:09 %");
        assert_eq!(strftime("%Y-%m-%d", 0), "1970-01-01");
        assert_eq!(strftime("%d %b %Y", 1722000000), "26 Jul 2024");
    }

    #[test]
    fn tojson_takes_indent_as_a_keyword() {
        let src = "{% set x = {'a': [1]} %}{{ x | tojson(indent=2) }}";
        let out = ChatTemplate::new(src, "", "").unwrap().render_with(&json!([]), &Value::Null, true, false).unwrap();
        assert_eq!(out, "{\n  \"a\": [\n    1\n  ]\n}");
    }

    #[test]
    fn a_template_without_a_system_role_gets_the_system_text_in_the_first_user_turn() {
        let src = "{% for m in messages %}{% if m.role == 'system' %}{{ raise_exception('no system') }}{% endif %}[{{ m.role }}:{{ m.content }}]{% endfor %}";
        let t = ChatTemplate::new(src, "", "").unwrap();
        let out = t.render_chat(&[json!({"role": "system", "content": "S"}), json!({"role": "user", "content": "U"})], &[], true, false).unwrap();
        assert_eq!(out, "[user:S\nU]");
    }

    fn ev<'a>(args: &'a Value, synthetic: bool) -> Event<'a> {
        Event { name: "inbox", arguments: args, content: "mail from ops: disk 91% full", synthetic, call_id: "call_1" }
    }

    #[test]
    fn chatml_puts_the_event_in_a_tool_response_after_a_made_up_call() {
        let args = json!({"since": 3, "box": "ops"});
        let (t, f) = ChatTemplate::chatml().event_text(&ev(&args, true)).unwrap();
        assert_eq!(f, Framing::Tool);
        assert_eq!(
            t,
            "\n<tool_call>\n{\"name\": \"inbox\", \"arguments\": {\"since\": 3, \"box\": \"ops\"}}\n</tool_call><|im_end|>\n\
             <|im_start|>user\n<tool_response>\nmail from ops: disk 91% full\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
        );
        let (t, _) = ChatTemplate::chatml().event_text(&ev(&args, false)).unwrap();
        assert!(t.starts_with("<|im_end|>\n<|im_start|>user\n<tool_response>\nmail from ops"), "{t:?}");
    }

    #[test]
    fn a_template_without_a_tool_role_gets_a_user_message() {
        // Gemma 3's shape: no system role, alternation enforced, the assistant is "model"
        let src = r#"{{ bos_token }}{%- if messages[0]['role'] == 'system' -%}{%- set first = messages[0]['content'] + '\n\n' -%}{%- set loop_messages = messages[1:] -%}{%- else -%}{%- set first = "" -%}{%- set loop_messages = messages -%}{%- endif -%}
{%- for message in loop_messages -%}
{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}{{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}{%- endif -%}
{%- if message['role'] == 'assistant' -%}{%- set role = "model" -%}{%- else -%}{%- set role = message['role'] -%}{%- endif -%}
{{ '<start_of_turn>' + role + '\n' + (first if loop.first else "") }}{{ message['content'] | trim }}{{ '<end_of_turn>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}{{'<start_of_turn>model\n'}}{%- endif -%}"#;
        let t = ChatTemplate::new(src, "<bos>", "<eos>").unwrap();
        let args = json!({});
        let (text, f) = t.event_text(&ev(&args, true)).unwrap();
        assert_eq!(f, Framing::User);
        assert_eq!(text, "<end_of_turn>\n<start_of_turn>user\n[event: inbox]\nmail from ops: disk 91% full<end_of_turn>\n<start_of_turn>model\n");
    }

    #[test]
    fn tojson_matches_python_json_dumps() {
        let mut s = String::new();
        py_json(&json!({"a": [1, 2.0, "x\"y\n", "é"], "b": {}, "c": null, "d": true}), None, 0, &mut s);
        assert_eq!(s, r#"{"a": [1, 2.0, "x\"y\n", "é"], "b": {}, "c": null, "d": true}"#);
        let mut s = String::new();
        py_json(&json!({"a": [1]}), Some(2), 0, &mut s);
        assert_eq!(s, "{\n  \"a\": [\n    1\n  ]\n}");
    }
}
