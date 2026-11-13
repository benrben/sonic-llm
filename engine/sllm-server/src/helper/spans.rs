//! Extract by constrained span copying (phase 3 helper lane).
//!
//! While the helper decodes, its output is held to **substrings of the source text**, at the byte level: a span is the
//! set of source byte positions where the bytes emitted so far match, and the next token is allowed only if its bytes
//! continue the match at one of them (at the start of a span: if they occur anywhere in the source). The model may
//! tokenize a span its own way (`"A"` + `"-778"` where the source was tokenized `" A"` `"-"` `"778"`), but every byte it
//! emits is a source byte in the source order, so a returned span always occurs in the source, with its offsets, and an
//! ID cannot be invented. Besides span tokens the model may end a span (a newline token), end the output (an
//! end-of-generation token) or say there is nothing (`NONE`, only as its first token). Spans never hold a newline.
//!
//! The choice is the best-scoring allowed token: the candidates are taken in descending logit order and the first one
//! the constraint allows wins (exactly the argmax over the allowed set, without listing that set).

/// One extracted span: `text == source[start..end]` (byte offsets; `char_start`/`char_end` in Unicode scalar values).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Span {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub char_start: usize,
    pub char_end: usize,
}

/// The span being decoded: its length in bytes and every source position where it still matches.
#[derive(Debug, Clone, Default)]
pub struct SpanState {
    starts: Vec<usize>,
    len: usize,
}

fn find_all(hay: &[u8], needle: &[u8]) -> Vec<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return Vec::new();
    }
    let first = needle[0];
    (0..=hay.len() - needle.len()).filter(|&i| hay[i] == first && &hay[i..i + needle.len()] == needle).collect()
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || needle.len() > hay.len() {
        return false;
    }
    let first = needle[0];
    (0..=hay.len() - needle.len()).any(|i| hay[i] == first && &hay[i..i + needle.len()] == needle)
}

impl SpanState {
    pub fn new() -> SpanState {
        SpanState::default()
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// May a token of bytes `piece` come next?
    pub fn allows(&self, src: &[u8], piece: &[u8]) -> bool {
        if piece.is_empty() || piece.contains(&b'\n') {
            return false;
        }
        if self.len == 0 {
            return contains(src, piece);
        }
        self.starts.iter().any(|&s| src[s + self.len..].starts_with(piece))
    }

    /// Append `piece`. False (and the span unchanged) when it is not allowed.
    pub fn push(&mut self, src: &[u8], piece: &[u8]) -> bool {
        if !self.allows(src, piece) {
            return false;
        }
        if self.len == 0 {
            self.starts = find_all(src, piece);
        } else {
            let len = self.len;
            self.starts.retain(|&s| src[s + len..].starts_with(piece));
        }
        self.len += piece.len();
        true
    }

    /// The bytes `[a, b)` of the earliest occurrence.
    pub fn run(&self) -> Option<(usize, usize)> {
        (self.len > 0).then(|| (self.starts[0], self.starts[0] + self.len))
    }
}

/// The source span of the bytes `[a, b)`, trimmed of surrounding whitespace and moved inward to character boundaries;
/// `None` when nothing is left.
pub fn locate(text: &str, a: usize, b: usize) -> Option<Span> {
    if a >= b || b > text.len() {
        return None;
    }
    let (s, e) = trim_span(text, a, b);
    if s >= e {
        return None;
    }
    let t = text[s..e].to_string();
    let char_start = text[..s].chars().count();
    let char_end = char_start + t.chars().count();
    Some(Span { text: t, start: s, end: e, char_start, char_end })
}

/// `[a, b)` moved inward to character boundaries, then trimmed of surrounding whitespace.
fn trim_span(text: &str, a: usize, b: usize) -> (usize, usize) {
    let (mut s, mut e) = (a, b);
    let src = text.as_bytes();
    while s < e && !text.is_char_boundary(s) {
        s += 1;
    }
    while e > s && !text.is_char_boundary(e) {
        e -= 1;
    }
    while s < e && src[s].is_ascii_whitespace() {
        s += 1;
    }
    while e > s && src[e - 1].is_ascii_whitespace() {
        e -= 1;
    }
    (s, e)
}

/// What the constrained decoder emits at one step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    /// A span token.
    Token(i32),
    /// Close the current span.
    Sep(i32),
    /// End the output.
    End(i32),
    /// "Nothing to extract" (only as the first token of the output).
    None(i32),
}

/// How many of the best-scoring tokens are tried before every token of the vocabulary is.
const TOP_K: usize = 512;

/// The best span token the state allows: the top tokens in descending order, then (rarely) all of them.
fn best_span_token(logits: &[f32], pieces: &[Vec<u8>], src: &[u8], st: &SpanState) -> Option<usize> {
    let n = logits.len().min(pieces.len());
    let ok = |t: usize| logits[t].is_finite() && st.allows(src, &pieces[t]);
    let mut idx: Vec<usize> = (0..n).collect();
    let k = TOP_K.min(n);
    let mut tok = None;
    if k > 0 {
        idx.select_nth_unstable_by(k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
        let mut top = idx[..k].to_vec();
        top.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        tok = top.into_iter().find(|&t| ok(t));
        if tok.is_none() {
            tok = (0..n).filter(|&t| ok(t)).max_by(|&a, &b| logits[a].total_cmp(&logits[b]));
        }
    }
    tok
}

/// The best allowed choice for one row of logits: span tokens the state allows, separators (only inside a span),
/// end-of-generation tokens, and `none` (only when given).
pub fn choose(logits: &[f32], pieces: &[Vec<u8>], src: &[u8], st: &SpanState, seps: &[i32], eog: &[i32], none: Option<i32>) -> Choice {
    let mut best = (f32::NEG_INFINITY, Choice::End(eog.first().copied().unwrap_or(0)));
    let mut consider = |t: i32, c: Choice| {
        if let Some(&l) = logits.get(t as usize) {
            if l > best.0 {
                best = (l, c);
            }
        }
    };
    if !st.is_empty() {
        for &t in seps {
            consider(t, Choice::Sep(t));
        }
    }
    for &t in eog {
        consider(t, Choice::End(t));
    }
    if let Some(t) = none {
        consider(t, Choice::None(t));
    }
    if let Some(t) = best_span_token(logits, pieces, src, st) {
        if logits[t] > best.0 {
            best = (logits[t], Choice::Token(t as i32));
        }
    }
    best.1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(words: &[&str]) -> Vec<Vec<u8>> {
        words.iter().map(|w| w.as_bytes().to_vec()).collect()
    }

    #[test]
    fn a_span_may_be_tokenized_differently_from_the_source() {
        let text = "Order A-77812 shipped to bob@example.com";
        let src = text.as_bytes();
        let mut st = SpanState::new();
        // the model starts without the space the source token had, and splits the digits its own way
        for p in ["A", "-778", "12"] {
            assert!(st.push(src, p.as_bytes()), "{p}");
        }
        let (a, b) = st.run().unwrap();
        let s = locate(text, a, b).unwrap();
        assert_eq!(s.text, "A-77812");
        assert_eq!(&text[s.start..s.end], "A-77812");
        assert_eq!((s.char_start, s.char_end), (6, 13));
    }

    #[test]
    fn only_continuations_of_the_match_are_allowed() {
        let src = b"a b c a b d";
        let mut st = SpanState::new();
        assert!(st.push(src, b"a b"));
        assert!(st.allows(src, b" c") && st.allows(src, b" d"), "the match at 0 and at 6 continue differently");
        assert!(!st.allows(src, b" e"));
        assert!(!st.push(src, b" a"), "a token that does not continue the span is refused");
        assert!(st.push(src, b" d"));
        assert_eq!(st.run(), Some((6, 11)));
        assert!(!SpanState::new().allows(src, b"zz"), "a start that occurs nowhere");
    }

    #[test]
    fn spans_never_hold_a_newline() {
        let src = b"id: X1\nid: X2";
        let mut st = SpanState::new();
        assert!(st.push(src, b"X1"));
        assert!(!st.allows(src, b"\n"));
        assert!(!SpanState::new().allows(src, b"\nid"));
        assert!(SpanState::new().allows(src, b"X2"));
    }

    #[test]
    fn locate_trims_whitespace_and_respects_char_boundaries() {
        let text = "café  ID-9 ";
        let s = locate(text, 4, text.len()).unwrap(); // starts inside the é: moved to the next boundary, spaces trimmed
        assert_eq!(s.text, "ID-9");
        assert_eq!(&text[s.start..s.end], "ID-9");
        assert!(locate(text, 5, 7).is_none(), "whitespace only is no span");
        let t = "x ÉT-9921 y";
        let s = locate(t, 1, 10).unwrap();
        assert_eq!(s.text, "ÉT-9921");
        assert_eq!((s.char_start, s.char_end), (2, 9));
    }

    #[test]
    fn choose_takes_the_best_allowed_token() {
        let pieces = vocab(&["zzz", "A-1", " A", "-1", "\n", "<eos>", "NONE", "B"]);
        let src = b"see A-1 and B";
        let mut logits = vec![0f32; pieces.len()];
        logits[0] = 9.0; // the favourite occurs nowhere in the source
        logits[1] = 3.0;
        logits[2] = 2.0;
        let st = SpanState::new();
        assert_eq!(choose(&logits, &pieces, src, &st, &[4], &[5], None), Choice::Token(1));
        logits[6] = 8.0;
        assert_eq!(choose(&logits, &pieces, src, &st, &[4], &[5], Some(6)), Choice::None(6));
        logits[4] = 7.0;
        assert_eq!(choose(&logits, &pieces, src, &st, &[4], &[5], None), Choice::Token(1), "no separator outside a span");
        let mut st = SpanState::new();
        assert!(st.push(src, b"A-1"));
        assert_eq!(choose(&logits, &pieces, src, &st, &[4], &[5], None), Choice::Sep(4));
        logits[5] = 7.5;
        assert_eq!(choose(&logits, &pieces, src, &st, &[4], &[5], None), Choice::End(5));
    }

    #[test]
    fn choose_finds_an_allowed_token_outside_the_top_k() {
        let n = 3000;
        let mut pieces: Vec<Vec<u8>> = (0..n).map(|i| format!("q{i}q").into_bytes()).collect();
        pieces[2999] = b"ok".to_vec();
        let mut logits: Vec<f32> = (0..n).map(|i| 10.0 - i as f32 * 1e-3).collect();
        logits[2999] = -5.0;
        let c = choose(&logits, &pieces, b"it is ok", &SpanState::new(), &[], &[], None);
        assert_eq!(c, Choice::Token(2999));
    }

    /// Random sources and random allowed tokens: every span is a substring at its offsets.
    #[test]
    fn random_spans_are_always_substrings() {
        let words = ["A-1", " B-22", " c", " d@x.io", " 42", "\n", " é", " A-1", " z", "1", "-", "B", "@"];
        let mut rnd = crate::helper::seeded(12345);
        for _ in 0..500 {
            let text: String = (0..1 + rnd(30)).map(|_| words[rnd(words.len())]).collect();
            let src = text.as_bytes();
            let mut st = SpanState::new();
            for _ in 0..1 + rnd(8) {
                let cands: Vec<&str> = words.iter().copied().filter(|w| st.allows(src, w.as_bytes())).collect();
                if cands.is_empty() {
                    break;
                }
                assert!(st.push(src, cands[rnd(cands.len())].as_bytes()));
            }
            if let Some((a, b)) = st.run() {
                if let Some(s) = locate(&text, a, b) {
                    assert_eq!(&text[s.start..s.end], s.text);
                    assert!(!s.text.contains('\n'));
                }
            }
        }
    }
}
