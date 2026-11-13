//! Call bodies: the JSON, XML, Gemma and name-then-JSON forms a call's text is read from.

use serde_json::{Map, Value};
use std::collections::HashMap;

use super::ToolCall;

pub(super) fn parse_json(s: &str, heal: bool) -> Option<Value> {
    serde_json::from_str(s).ok().or_else(|| if heal { serde_json::from_str(&heal_json(s)).ok() } else { None })
}

/// What a cut-off JSON text left open: the closers it still owes, innermost last, and whether it ended inside a string
/// (right after a backslash).
#[derive(Default)]
struct JsonTail {
    stack: Vec<char>,
    in_str: bool,
    esc: bool,
}

impl JsonTail {
    fn feed(&mut self, c: char) {
        if self.in_str {
            self.feed_in_string(c);
        } else {
            self.feed_outside(c);
        }
    }

    fn feed_in_string(&mut self, c: char) {
        match (self.esc, c) {
            (true, _) => self.esc = false,
            (false, '\\') => self.esc = true,
            (false, '"') => self.in_str = false,
            _ => {}
        }
    }

    fn feed_outside(&mut self, c: char) {
        match c {
            '"' => self.in_str = true,
            '{' => self.stack.push('}'),
            '[' => self.stack.push(']'),
            '}' | ']' => {
                self.stack.pop();
            }
            _ => {}
        }
    }
}

/// `s` with the `"` / `]` / `}` it is missing at its end (a JSON text that was cut off).
fn heal_json(s: &str) -> String {
    let mut tail = JsonTail::default();
    for c in s.chars() {
        tail.feed(c);
    }
    let mut out = s.to_string();
    if tail.esc {
        out.pop();
    }
    if tail.in_str {
        out.push('"');
    }
    while let Some(c) = tail.stack.pop() {
        out.push(c);
    }
    out
}

pub(super) fn json_call(v: &Value, name_key: &str, arg_keys: &[&str]) -> Option<ToolCall> {
    let o = v.as_object()?;
    let name = o.get(name_key)?.as_str().filter(|n| !n.is_empty())?.to_string();
    let arguments = match arg_keys.iter().find_map(|k| o.get(*k)) {
        None | Some(Value::Null) => "{}".to_string(),
        Some(a @ Value::Object(_)) => a.to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(_) => return None,
    };
    Some(ToolCall { name, arguments })
}

fn valid_name(n: &str) -> bool {
    !n.is_empty() && !n.contains(|c: char| c.is_whitespace() || matches!(c, '<' | '>' | '{' | '}'))
}

/// `NAME`, anything, `sep`, then the arguments as a JSON object (empty: no arguments).
pub(super) fn name_then_json(body: &str, sep: &str, heal: bool) -> Option<ToolCall> {
    let b = body.trim_start();
    let end = b.find(|c: char| c.is_whitespace() || c == '<').unwrap_or(b.len());
    let name = &b[..end];
    let (_, args) = b[end..].split_once(sep)?;
    let args = args.trim();
    let arguments = if args.is_empty() {
        "{}".to_string()
    } else {
        match parse_json(args, heal)? {
            a @ Value::Object(_) => a.to_string(),
            _ => return None,
        }
    };
    valid_name(name).then(|| ToolCall { name: name.to_string(), arguments })
}

/// `<function=NAME>` `<parameter=KEY>` `VALUE` `</parameter>` ... `</function>`, one newline inside each tag's edge.
pub(super) fn xml_call(body: &str, props: &HashMap<String, Map<String, Value>>) -> Option<ToolCall> {
    let (name, mut rest) = body.trim().strip_prefix("<function=")?.split_once('>')?;
    if !valid_name(name) {
        return None;
    }
    let mut args = Map::new();
    while let Some(r) = rest.trim_start().strip_prefix("<parameter=") {
        let (key, r) = r.split_once('>')?;
        let (val, r) = r.split_once("</parameter>")?;
        let val = val.strip_prefix('\n').unwrap_or(val);
        let val = val.strip_suffix('\n').unwrap_or(val);
        args.insert(key.to_string(), typed(val, props.get(name).and_then(|p| p.get(key))));
        rest = r;
    }
    (rest.trim() == "</function>").then(|| ToolCall { name: name.to_string(), arguments: Value::Object(args).to_string() })
}

/// A parameter's text as a value: a string when the schema says only `string`, else its JSON when it is JSON, else a string.
fn typed(text: &str, schema: Option<&Value>) -> Value {
    let only_string = schema.is_some_and(|s| match &s["type"] {
        Value::String(t) => t == "string",
        Value::Array(a) => a.len() == 1 && a[0] == "string",
        _ => false,
    });
    if only_string {
        return Value::String(text.to_string());
    }
    serde_json::from_str(text.trim()).unwrap_or_else(|_| Value::String(text.to_string()))
}

/// `call:NAME{key:value,...}`.
pub(super) fn gemma_call(body: &str) -> Option<ToolCall> {
    let b = body.trim().strip_prefix("call:")?;
    let at = b.find('{')?;
    let name = b[..at].trim();
    let (args, used) = gemma_value(b, at)?;
    (valid_name(name) && b[used..].trim().is_empty() && args.is_object()).then(|| ToolCall { name: name.to_string(), arguments: args.to_string() })
}

const GEMMA_QUOTE: &str = "<|\"|>";

fn skip_ws(s: &str, i: usize) -> usize {
    i + (s.len() - i - s[i..].trim_start().len())
}

/// The value at byte `i` of `s` and the offset after it.
fn gemma_value(s: &str, i: usize) -> Option<(Value, usize)> {
    let i = skip_ws(s, i);
    let rest = &s[i..];
    if let Some(r) = rest.strip_prefix(GEMMA_QUOTE) {
        let q = r.find(GEMMA_QUOTE)?;
        return Some((Value::String(r[..q].to_string()), i + GEMMA_QUOTE.len() + q + GEMMA_QUOTE.len()));
    }
    if rest.starts_with('{') {
        return gemma_object(s, i);
    }
    if rest.starts_with('[') {
        return gemma_array(s, i);
    }
    let end = rest.find(|c: char| c.is_whitespace() || matches!(c, ',' | '}' | ']')).unwrap_or(rest.len());
    let v: Value = serde_json::from_str(&rest[..end]).ok()?;
    (v.is_number() || v.is_boolean() || v.is_null()).then_some((v, i + end))
}

/// A bracketed list in Gemma's syntax whose opening bracket is at byte `i` of `s`, ending at `close`: `item` reads the item at
/// the offset it is given and returns it with the offset after it. The items and the offset after the closing bracket.
fn gemma_list<T>(s: &str, i: usize, close: char, mut item: impl FnMut(usize) -> Option<(T, usize)>) -> Option<(Vec<T>, usize)> {
    let (mut items, mut i) = (Vec::new(), skip_ws(s, i + 1));
    if s[i..].starts_with(close) {
        return Some((items, i + 1));
    }
    loop {
        let (v, next) = item(i)?;
        items.push(v);
        i = skip_ws(s, next);
        match s[i..].chars().next()? {
            ',' => i = skip_ws(s, i + 1),
            c if c == close => return Some((items, i + 1)),
            _ => return None,
        }
    }
}

/// The `{key:value,...}` at byte `i` of `s` and the offset after it.
fn gemma_object(s: &str, i: usize) -> Option<(Value, usize)> {
    let (entries, end) = gemma_list(s, i, '}', |i| {
        let colon = i + s[i..].find(':')?;
        let key = s[i..colon].trim();
        if key.is_empty() || key.contains('}') {
            return None;
        }
        let (v, next) = gemma_value(s, colon + 1)?;
        Some(((key.to_string(), v), next))
    })?;
    Some((Value::Object(entries.into_iter().collect()), end))
}

/// The `[value,...]` at byte `i` of `s` and the offset after it.
fn gemma_array(s: &str, i: usize) -> Option<(Value, usize)> {
    let (items, end) = gemma_list(s, i, ']', |i| gemma_value(s, i))?;
    Some((Value::Array(items), end))
}
