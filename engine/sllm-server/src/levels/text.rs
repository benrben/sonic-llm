//! The python `re` helpers of the answer levels, written out by hand: box and bold parts, numbers, thinking cut-off.

/// The text inside the `{...}` that opens at `text[start] == '{'` (nested braces counted); `None` when it never closes.
fn brace_arg(text: &str, start: usize) -> Option<&str> {
    let mut depth = 0i32;
    for (i, c) in text[start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start + 1..start + i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// The matches of `\\boxed\s*\{` (non-overlapping, left to right) as the byte index just after each `{`.
fn box_opens(text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(p) = text[from..].find("\\boxed") {
        let mut i = from + p + "\\boxed".len();
        let rest = &text[i..];
        let trimmed = rest.trim_start();
        i += rest.len() - trimmed.len();
        if trimmed.starts_with('{') {
            out.push(i + 1);
            from = i + 1;
        } else {
            from += p + 1;
        }
    }
    out
}

/// (start, end) byte span of the content inside the last closed `\boxed{...}` of `text`.
pub(super) fn box_span(text: &str) -> Option<(usize, usize)> {
    box_opens(text).into_iter().rev().find_map(|after| brace_arg(text, after - 1).map(|arg| (after, after + arg.len())))
}

/// The content of the last `\boxed{...}` in `text` (`None` without one, or when its brace never closes).
pub fn last_boxed(text: &str) -> Option<&str> {
    box_span(text).map(|(a, b)| &text[a..b])
}

/// The first `\text{..}` or `\mathrm{..}` (with a `}`-free body) at or after `from`: (start, end, body).
fn find_text_cmd(s: &str, from: usize) -> Option<(usize, usize, &str)> {
    let mut i = from;
    while let Some(p) = s[i..].find('\\') {
        let at = i + p;
        for cmd in ["\\text{", "\\mathrm{"] {
            if s[at..].starts_with(cmd) {
                let body_start = at + cmd.len();
                if let Some(close) = s[body_start..].find('}') {
                    return Some((at, body_start + close + 1, &s[body_start..body_start + close]));
                }
            }
        }
        i = at + 1;
    }
    None
}

/// `\d{1,3}(,\d{3})+(\.\d+)?` with an optional sign, the whole string.
fn is_comma_number(s: &str) -> bool {
    let t = s.strip_prefix(['-', '+']).unwrap_or(s);
    let (int, frac) = match t.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (t, None),
    };
    if frac.is_some_and(|f| f.is_empty() || !f.bytes().all(|b| b.is_ascii_digit())) {
        return false;
    }
    let mut groups = int.split(',');
    let first = groups.next().unwrap_or("");
    if first.is_empty() || first.len() > 3 || !first.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    let rest: Vec<&str> = groups.collect();
    !rest.is_empty() && rest.iter().all(|g| g.len() == 3 && g.bytes().all(|b| b.is_ascii_digit()))
}

/// `\d+(\.\d+)?` with an optional sign, the whole string.
fn is_plain_number(s: &str) -> bool {
    let t = s.strip_prefix(['-', '+']).unwrap_or(s);
    let digits = |x: &str| !x.is_empty() && x.bytes().all(|b| b.is_ascii_digit());
    match t.split_once('.') {
        Some((i, f)) => digits(i) && digits(f),
        None => digits(t),
    }
}

/// `format(Decimal(s).normalize(), "f")` for a plain number: no sign `+`, no leading zeros, no trailing zeros after the point.
fn normalize_decimal(s: &str) -> String {
    let (neg, t) = match s.strip_prefix('-') {
        Some(t) => (true, t),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let (int, frac) = t.split_once('.').unwrap_or((t, ""));
    let int = int.trim_start_matches('0');
    let frac = frac.trim_end_matches('0');
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    out.push_str(if int.is_empty() { "0" } else { int });
    if !frac.is_empty() {
        out.push('.');
        out.push_str(frac);
    }
    out
}

/// The comparable form of a box content (`\dfrac` is `\frac`, no `$`, `%`, spaces or thousands commas, 18.0 is 18 ...).
pub fn canon_box(s: &str) -> String {
    // \text{x} and \mathrm{x} are their body
    let mut t = String::new();
    let mut from = 0;
    while let Some((a, b, body)) = find_text_cmd(s, from) {
        t.push_str(&s[from..a]);
        t.push_str(body);
        from = b;
    }
    t.push_str(&s[from..]);
    let t = t.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "");
    // a backslash before one of , ; : ! space is dropped with it
    let mut u = String::new();
    let mut it = t.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\\' && it.peek().is_some_and(|n| matches!(n, ',' | ';' | ':' | '!' | ' ')) {
            it.next();
        } else {
            u.push(c);
        }
    }
    let u = u.replace("\\$", "").replace('$', "").replace("\\%", "").replace('%', "");
    let u: String = u.chars().filter(|c| !c.is_whitespace()).collect::<String>().to_lowercase();
    let mut s = u.trim_end_matches('.').to_string();
    if is_comma_number(&s) {
        s = s.replace(',', "");
    }
    if is_plain_number(&s) {
        s = normalize_decimal(&s);
    }
    s
}

/// Lower-case, punctuation stripped, whitespace collapsed.
pub fn normalize_text(text: &str) -> String {
    let kept: String = text.to_lowercase().chars().filter(|c| c.is_alphanumeric() || *c == '_' || c.is_whitespace()).collect();
    kept.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Python `str.splitlines()`.
pub(super) fn splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        let is_break = matches!(c, '\n' | '\r' | '\u{b}' | '\u{c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}');
        if is_break {
            out.push(&s[start..i]);
            let mut next = i + c.len_utf8();
            if c == '\r' && it.peek().is_some_and(|(_, n)| *n == '\n') {
                it.next();
                next += 1;
            }
            start = next;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// `\*\*(.+?)\*\*` findall: the contents, left to right (`.` does not match a newline).
pub(super) fn bold_parts(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < s.len() {
        if !s[i..].starts_with("**") {
            i += s[i..].chars().next().map_or(1, char::len_utf8);
            continue;
        }
        let body = i + 2;
        let mut found = None;
        // `.+?`: at least one char, as few as possible, none a newline
        for (k, c) in s[body..].char_indices() {
            if c == '\n' {
                break;
            }
            let j = body + k + c.len_utf8();
            if s[j..].starts_with("**") {
                found = Some(j);
                break;
            }
        }
        match found {
            Some(end) => {
                out.push(&s[body..end]);
                i = end + 2;
            }
            None => i += 1,
        }
    }
    out
}

/// The index after the run of ASCII digits (and, with `commas`, commas) that starts at `j`.
fn skip_digits(b: &[u8], mut j: usize, commas: bool) -> usize {
    while j < b.len() && (b[j].is_ascii_digit() || (commas && b[j] == b',')) {
        j += 1;
    }
    j
}

/// The end of the number that starts at byte `start` (`-?\d[\d,]*(?:\.\d+)?`); `None` when none starts there.
fn number_end(b: &[u8], start: usize) -> Option<usize> {
    let j = if b[start] == b'-' { start + 1 } else { start };
    if j >= b.len() || !b[j].is_ascii_digit() {
        return None;
    }
    let j = skip_digits(b, j, true);
    if j + 1 < b.len() && b[j] == b'.' && b[j + 1].is_ascii_digit() {
        return Some(skip_digits(b, j + 1, false));
    }
    Some(j)
}

/// `-?\d[\d,]*(?:\.\d+)?` findall.
pub(super) fn numbers(s: &str) -> Vec<&str> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        match number_end(b, i) {
            Some(j) => {
                out.push(&s[i..j]);
                i = j;
            }
            None => i += 1,
        }
    }
    out
}

/// `_THOUGHT_END` at byte `i`: the length of the match (`</think>` or harmony's final channel opening).
fn thought_end_at(s: &str, i: usize) -> Option<usize> {
    let r = &s[i..];
    if r.starts_with("</think>") {
        return Some("</think>".len());
    }
    let mut p = 0;
    for opt in ["<|end|>", "<|start|>"] {
        if r[p..].starts_with(opt) {
            p += opt.len();
        }
    }
    if !r[p..].starts_with("assistant") {
        return None;
    }
    p += "assistant".len();
    if r[p..].starts_with("<|channel|>") {
        p += "<|channel|>".len();
    }
    if !r[p..].starts_with("final") {
        return None;
    }
    p += "final".len();
    if r[p..].starts_with("<|message|>") {
        p += "<|message|>".len();
    }
    Some(p)
}

/// `_THOUGHT_START.match(s)`: `^\s*(<think>|<|channel|>analysis<|message|>|analysis(?=\S))`.
fn thought_starts(s: &str) -> bool {
    let r = s.trim_start();
    r.starts_with("<think>")
        || r.starts_with("<|channel|>analysis<|message|>")
        || r.strip_prefix("analysis").is_some_and(|after| after.chars().next().is_some_and(|c| !c.is_whitespace()))
}

/// The answer part of a reply: what follows the thinking (up to `</think>`, or harmony's final channel); the whole text
/// without thinking.
pub fn split_thinking(text: &str) -> &str {
    let found = text.char_indices().find_map(|(i, _)| thought_end_at(text, i).map(|len| (i, len)));
    let Some((at, len)) = found else { return text };
    if !(thought_starts(text) || text[at..].starts_with("</think>")) {
        return text;
    }
    text[at + len..].trim_start()
}

/// The answer of a reply: end-of-generation tokens and the thinking cut off.
pub fn final_part(text: &str, eog: &[&str]) -> String {
    let mut t = text;
    for tok in eog {
        t = t.split(tok).next().unwrap_or("");
    }
    split_thinking(t).trim().to_string()
}
