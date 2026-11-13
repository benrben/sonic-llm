//! Tool calls out of generated text, for every model family, in the API layer (the engine never sees them).
//!
//! ONE parser, [`ToolParser`], driven by a [`FormatSpec`]: plain data that says where a call starts and ends and how its
//! body is written. A model family is a row in [`FORMATS`], not a parser: a new model whose template uses a known style needs
//! no code, and [`ToolFormat::detect`] finds the row from the chat template's source and the GGUF's special tokens.
//!
//! A spec holds: how a call starts ([`Start`]: a marker, or the whole reply), how it ends ([`Close`]), how the body between
//! is written ([`Body`]: JSON object, JSON list, `<function=...>` XML, Gemma's `call:name{key:value}`, name-then-JSON),
//! spans that are not content (`hide`: reasoning channels; `strip`: channel/turn markers), and the `preserved` special tokens
//! the reply text must show (control tokens render as empty text unless asked for, see [`ToolFormat::preserves`]).
//!
//! The parser reads the reply as it streams. Text that may be the start of a marker (a suffix that is a proper prefix of one)
//! is held back and never shown as content; a call's body is held until its close, then returned as one whole [`ToolCall`]
//! (so a call never goes out as partial JSON). The result does not depend on how the text was cut into pieces: one piece, or
//! one character per piece, give the same [`Part`]s up to the splitting of content runs.
//!
//! Rules (tested):
//! * a call whose body is not a call (not JSON, not an object, no `name`, `arguments` neither an object nor a string, a
//!   broken tag) is NOT dropped and not an error: the marker, body and close come back as content, verbatim, and the reply's
//!   finish_reason stays what it was (`stop`, not `tool_calls`).
//! * a call that is never closed (the reply was cut off, or the model stopped) comes back as content at
//!   [`ToolParser::finish`], verbatim. Exception: a format whose call runs to the end of the reply ([`Close::End`], and
//!   [`Close::MarkerOrEnd`]) is closed by the end; there a body cut short only by missing `}` / `]` / `"` is completed first
//!   (llama-server does the same), anything else broken is content.
//! * white space directly after a call, up to the next non-white-space character, is not content (so the newline between
//!   two calls, or after the last, is dropped). White space before a call stays in the content that precedes it.
//! * a call's name is not checked against the request's `tools`.
//! * argument values: JSON formats and Gemma carry typed values already. In the XML format every value is text; with the
//!   request's tools ([`ToolParser::with_tools`]) a parameter whose schema type is only `string` stays a string, any other
//!   value that is valid JSON becomes that JSON (`3`, `true`, `{"a": 1}`), and the rest stays a string (llama-server's rule).
//!   An unknown tool or parameter: JSON when it parses, else a string.
//! * `hide` spans (reasoning channels) are dropped from content, also when never closed.

use crate::stop::take_through;
use Need::{Src, Tok};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};

mod body;

use body::{gemma_call, json_call, name_then_json, parse_json, xml_call};

/// What a template or its tokenizer must show for a format to be the one.
#[derive(Debug, Clone, Copy)]
pub enum Need {
    /// The chat template's source contains this text.
    Src(&'static str),
    /// The vocabulary has a special token with exactly this text.
    Tok(&'static str),
}

#[derive(Debug, Clone, Copy)]
pub enum Start {
    /// A call starts at any of these markers (the text before is content).
    Markers(&'static [&'static str]),
    /// The reply is a call when its first non-white-space character is `{`; otherwise it is all content.
    Reply,
}

#[derive(Debug, Clone, Copy)]
pub enum Close {
    /// The body ends at this text; without it the call is unclosed (content).
    Marker(&'static str),
    /// The body runs to the end of the reply.
    End,
    /// The body ends at this text, or at the end of the reply.
    MarkerOrEnd(&'static str),
}

#[derive(Debug, Clone, Copy)]
pub enum Body {
    /// JSON with the name under `name` and the arguments under the first of `args` that is present; `list`: a JSON array of such objects.
    Json { name: &'static str, args: &'static [&'static str], list: bool },
    /// `<function=NAME><parameter=KEY>VALUE</parameter>...</function>`, values as text (typed by the tools' schemas).
    Xml,
    /// Gemma 4: `call:NAME{key:value,...}`, strings in `<|"|>...<|"|>`, bare numbers, booleans, null, nested `{}` and `[]`.
    Gemma,
    /// `NAME` then anything up to this separator, then the arguments as JSON (gpt-oss: `NAME <|constrain|>json<|message|>{...}`).
    NameThenJson(&'static str),
}

#[derive(Debug)]
pub struct FormatSpec {
    pub name: &'static str,
    /// All must hold for [`ToolFormat::detect`] to pick this format.
    pub detect: &'static [Need],
    pub start: Start,
    pub close: Close,
    pub body: Body,
    /// (open, close) spans that are not content: reasoning channels. Dropped.
    pub hide: &'static [(&'static str, &'static str)],
    /// Markers that are not content (channel and turn headers). Dropped.
    pub strip: &'static [&'static str],
    /// Special tokens the reply text must show: the markers above and the structure tokens between them.
    pub preserved: &'static [&'static str],
}

const JSON_ARGS: &[&str] = &["arguments", "parameters"];

/// What the formats below start from: nothing hidden, stripped or preserved (each sets what it needs).
const BARE: FormatSpec = FormatSpec { name: "", detect: &[], start: Start::Reply, close: Close::End, body: Body::Xml, hide: &[], strip: &[], preserved: &[] };

const QWEN_JSON: FormatSpec = FormatSpec {
    name: "qwen-json",
    detect: &[Src("<tool_call>")],
    start: Start::Markers(&["<tool_call>"]),
    close: Close::Marker("</tool_call>"),
    body: Body::Json { name: "name", args: JSON_ARGS, list: false },
    preserved: &["<tool_call>", "</tool_call>"],
    ..BARE
};

// qwen-json's tags around a body in XML
const QWEN_XML: FormatSpec = FormatSpec {
    name: "qwen-xml",
    detect: &[Src("<tool_call>"), Src("<function="), Src("<parameter=")],
    body: Body::Xml,
    ..QWEN_JSON
};

const GEMMA4: FormatSpec = FormatSpec {
    name: "gemma4",
    detect: &[Src("<|tool_call>call:")],
    start: Start::Markers(&["<|tool_call>"]),
    close: Close::Marker("<tool_call|>"),
    body: Body::Gemma,
    hide: &[("<|channel>thought", "<channel|>")],
    preserved: &["<|tool_call>", "<tool_call|>", "<|channel>", "<channel|>", "<|\"|>"],
    ..BARE
};

const GPT_OSS: FormatSpec = FormatSpec {
    name: "gpt-oss",
    detect: &[Src("<|channel|>"), Src("functions")],
    start: Start::Markers(&["<|channel|>commentary to=functions.", "<|channel|>analysis to=functions.", "<|start|>assistant to=functions."]),
    close: Close::MarkerOrEnd("<|call|>"),
    body: Body::NameThenJson("<|message|>"),
    hide: &[("<|channel|>analysis<|message|>", "<|end|>")],
    strip: &["<|start|>assistant", "<|channel|>final<|message|>", "<|channel|>commentary<|message|>", "<|end|>"],
    preserved: &["<|channel|>", "<|constrain|>", "<|message|>", "<|start|>", "<|end|>", "<|call|>"],
};

const MISTRAL: FormatSpec = FormatSpec {
    name: "mistral",
    detect: &[Src("[TOOL_CALLS]"), Tok("[TOOL_CALLS]")],
    start: Start::Markers(&["[TOOL_CALLS]"]),
    close: Close::End,
    body: Body::Json { name: "name", args: JSON_ARGS, list: true },
    preserved: &["[TOOL_CALLS]"],
    ..BARE
};

const LLAMA31: FormatSpec = FormatSpec {
    name: "llama31-json",
    detect: &[Src("<|python_tag|>"), Src("\"parameters\"")],
    start: Start::Reply,
    close: Close::End,
    body: Body::Json { name: "name", args: &["parameters", "arguments"], list: false },
    ..BARE
};

/// A model's tool-call format: a row of [`FORMATS`].
#[derive(Clone, Copy)]
pub struct ToolFormat(&'static FormatSpec);

impl PartialEq for ToolFormat {
    fn eq(&self, o: &ToolFormat) -> bool {
        self.0.name == o.0.name
    }
}

impl std::fmt::Debug for ToolFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ToolFormat({})", self.0.name)
    }
}

/// In detection order, most specific first: Gemma 4's `<|tool_call>` contains `<tool_call>`, and Qwen 3.5's XML template
/// contains Qwen's JSON markers, so both come before the plain Qwen JSON format.
pub const FORMATS: &[ToolFormat] =
    &[ToolFormat(&GEMMA4), ToolFormat(&QWEN_XML), ToolFormat(&GPT_OSS), ToolFormat(&QWEN_JSON), ToolFormat(&MISTRAL), ToolFormat(&LLAMA31)];

impl ToolFormat {
    pub const QWEN_JSON: ToolFormat = ToolFormat(&QWEN_JSON);
    pub const QWEN_XML: ToolFormat = ToolFormat(&QWEN_XML);
    pub const GEMMA4: ToolFormat = ToolFormat(&GEMMA4);
    pub const GPT_OSS: ToolFormat = ToolFormat(&GPT_OSS);
    pub const MISTRAL: ToolFormat = ToolFormat(&MISTRAL);
    pub const LLAMA31: ToolFormat = ToolFormat(&LLAMA31);

    /// The format a chat template writes tool calls in, from its source and the vocabulary's special tokens; `None` when no
    /// format fits (the template has no tool syntax: llama-server shows such a model no tools and parses no calls).
    pub fn detect(template_source: &str, specials: &HashSet<String>) -> Option<ToolFormat> {
        FORMATS.iter().copied().find(|f| {
            f.0.detect.iter().all(|n| match n {
                Need::Src(s) => template_source.contains(s),
                Need::Tok(t) => specials.contains(*t),
            })
        })
    }

    pub fn name(self) -> &'static str {
        self.0.name
    }

    pub fn spec(self) -> &'static FormatSpec {
        self.0
    }

    /// Is this special token's text part of the format (so a reply must show it, though control tokens print as nothing)?
    pub fn preserves(self, piece: &str) -> bool {
        self.0.preserved.contains(&piece)
    }
}

/// One call. `arguments` is JSON text, as OpenAI's `function.arguments`.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Part {
    Content(String),
    Call(ToolCall),
    /// Reasoning text ([`crate::thinking`]); only a pipe that has a thinking parser makes it (the Anthropic route).
    Thinking(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Act {
    Call,
    Hide(&'static str),
    Strip,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    /// A [`Start::Reply`] format before the first non-white-space character.
    Undecided,
    Text,
    Hidden(&'static str),
    /// In a call that started with this marker (empty for a whole-reply call).
    Call(&'static str),
}

enum Scan {
    /// A marker at this byte offset.
    Hit(usize, &'static str, Act),
    /// From this offset the text may still become a marker: wait.
    Hold(usize),
    Nothing,
}

#[derive(Debug, Clone)]
pub struct ToolParser {
    format: ToolFormat,
    /// Tool name to its `parameters.properties`, for typing XML values.
    props: HashMap<String, Map<String, Value>>,
    buf: String,
    state: State,
    after_call: bool,
}

impl ToolParser {
    pub fn new(format: ToolFormat) -> ToolParser {
        let state = if matches!(format.0.start, Start::Reply) { State::Undecided } else { State::Text };
        ToolParser { format, props: HashMap::new(), buf: String::new(), state, after_call: false }
    }

    /// Know the request's tools (`[{"type": "function", "function": {"name", "parameters"}}]`), to type text values.
    pub fn with_tools(mut self, tools: &[Value]) -> ToolParser {
        for t in tools {
            let f = &t["function"];
            if let (Some(n), Some(p)) = (f["name"].as_str(), f["parameters"]["properties"].as_object()) {
                self.props.insert(n.to_string(), p.clone());
            }
        }
        self
    }

    pub fn format(&self) -> ToolFormat {
        self.format
    }

    pub fn push(&mut self, text: &str) -> Vec<Part> {
        self.buf.push_str(text);
        self.run(false)
    }

    /// The reply ended: what is still held was not a call, except a call that runs to the end of the reply.
    pub fn finish(&mut self) -> Vec<Part> {
        let mut out = self.run(true);
        let rest = std::mem::take(&mut self.buf);
        match std::mem::replace(&mut self.state, State::Text) {
            State::Call(open) => match self.format.0.close {
                Close::End | Close::MarkerOrEnd(_) => self.call_done(open, &rest, "", true, &mut out),
                Close::Marker(_) => self.content(format!("{open}{rest}"), &mut out),
            },
            State::Hidden(_) => {}
            State::Undecided | State::Text => self.content(rest, &mut out),
        }
        out
    }

    fn markers(&self) -> Vec<(&'static str, Act)> {
        let spec = self.format.0;
        let mut m = Vec::new();
        if let Start::Markers(s) = spec.start {
            m.extend(s.iter().map(|&s| (s, Act::Call)));
        }
        m.extend(spec.hide.iter().map(|&(o, c)| (o, Act::Hide(c))));
        m.extend(spec.strip.iter().map(|&s| (s, Act::Strip)));
        m
    }

    /// The first marker in `buf`: the earliest position where one matches in full (the longest there) and no longer one can
    /// still come, or from where the text may still become one (not at the end of the reply).
    fn scan(&self, eof: bool) -> Scan {
        let markers = self.markers();
        for (i, _) in self.buf.char_indices() {
            let rest = &self.buf[i..];
            if !eof && markers.iter().any(|(m, _)| m.len() > rest.len() && m.starts_with(rest)) {
                return Scan::Hold(i);
            }
            if let Some(&(m, act)) = markers.iter().filter(|(m, _)| rest.starts_with(m)).max_by_key(|(m, _)| m.len()) {
                return Scan::Hit(i, m, act);
            }
        }
        Scan::Nothing
    }

    fn run(&mut self, eof: bool) -> Vec<Part> {
        let mut out = Vec::new();
        loop {
            let more = match self.state {
                State::Undecided => self.decide(),
                State::Text => self.text(eof, &mut out),
                State::Hidden(until) => self.hidden(until),
                State::Call(open) => self.call(open, &mut out),
            };
            if !more {
                return out;
            }
        }
    }

    /// A whole-reply format: the reply is a call when it starts with `{`. False: wait for more text.
    fn decide(&mut self) -> bool {
        let t = self.buf.trim_start();
        if t.is_empty() {
            return false;
        }
        self.state = if t.starts_with('{') { State::Call("") } else { State::Text };
        true
    }

    /// Text up to the next marker is content. False: the rest is shown or held, wait for more text.
    fn text(&mut self, eof: bool, out: &mut Vec<Part>) -> bool {
        match self.scan(eof) {
            Scan::Hit(i, m, act) => {
                let before = take_through(&mut self.buf, i, m.len());
                self.content(before, out);
                match act {
                    Act::Call => self.state = State::Call(m),
                    Act::Hide(close) => self.state = State::Hidden(close),
                    Act::Strip => {}
                }
                true
            }
            Scan::Hold(i) => {
                let rest = self.buf.split_off(i);
                let shown = std::mem::replace(&mut self.buf, rest);
                self.content(shown, out);
                false
            }
            Scan::Nothing => {
                let shown = std::mem::take(&mut self.buf);
                self.content(shown, out);
                false
            }
        }
    }

    /// A hidden span is dropped up to its close. False: no close yet, wait for more text.
    fn hidden(&mut self, until: &'static str) -> bool {
        match self.buf.find(until) {
            Some(q) => {
                self.buf = self.buf.split_off(q + until.len());
                self.state = State::Text;
                true
            }
            None => {
                // dropped; only a tail that may still become the close is kept
                let keep = (1..until.len()).rev().find(|&n| until.is_char_boundary(n) && self.buf.ends_with(&until[..n])).unwrap_or(0);
                self.buf = self.buf.split_off(self.buf.len() - keep);
                false
            }
        }
    }

    /// A call's body up to its close marker. False: no close yet (or a call that runs to the end), wait for more text.
    fn call(&mut self, open: &'static str, out: &mut Vec<Part>) -> bool {
        let (Close::Marker(close) | Close::MarkerOrEnd(close)) = self.format.0.close else { return false };
        let Some(q) = self.buf.find(close) else { return false };
        let body = take_through(&mut self.buf, q, close.len());
        self.call_done(open, &body, close, false, out);
        self.state = State::Text;
        true
    }

    /// A call's body is complete: the calls it holds, or the whole text back as content.
    fn call_done(&mut self, open: &str, body: &str, close: &str, at_end: bool, out: &mut Vec<Part>) {
        match self.parse_body(body, at_end) {
            Some(calls) => {
                out.extend(calls.into_iter().map(Part::Call));
                self.after_call = true;
            }
            None => self.content(format!("{open}{body}{close}"), out),
        }
    }

    fn content(&mut self, text: String, out: &mut Vec<Part>) {
        let text = if self.after_call { text.trim_start().to_string() } else { text };
        if text.is_empty() {
            return;
        }
        self.after_call = false;
        out.push(Part::Content(text));
    }

    /// The calls a body holds (`heal`: the reply ended here, so a body cut short may be completed).
    fn parse_body(&self, body: &str, heal: bool) -> Option<Vec<ToolCall>> {
        match self.format.0.body {
            Body::Json { name, args, list } => {
                let v = parse_json(body.trim(), heal)?;
                let calls: Vec<ToolCall> = if list { v.as_array()?.iter().map(|c| json_call(c, name, args)).collect::<Option<_>>()? } else { vec![json_call(&v, name, args)?] };
                (!calls.is_empty()).then_some(calls)
            }
            Body::Xml => xml_call(body, &self.props).map(|c| vec![c]),
            Body::Gemma => gemma_call(body).map(|c| vec![c]),
            Body::NameThenJson(sep) => name_then_json(body, sep, heal).map(|c| vec![c]),
        }
    }
}

#[cfg(test)]
mod tests;
