//! `<think>...</think>` out of generated text, for the Anthropic route (thinking blocks), in the API layer.
//!
//! A [`ThinkParser`] reads the reply as it streams and returns [`ThinkPart`]s: reasoning text and answer text. Like
//! [`crate::toolcall::ToolParser`], text that may be the start of a tag (a suffix that is a proper prefix of it) is held
//! back until it is clear what it is, so a tag split over tokens is found and a lookalike (`<thin`, `<think b>`) comes back
//! as text, verbatim. The parts do not depend on how the text was cut into pieces, up to the splitting of runs.
//!
//! Rules (tested):
//! * `<think>` opens reasoning only at the start of the reply (white space before it is allowed). Later in the reply it is
//!   text: a model that writes the tag in an explanation does not lose the rest of its answer to a thinking block. A
//!   template whose prompt already ends inside the tag ([`prompt_opens_thinking`], DeepSeek-R1 and QwQ style) starts the
//!   parser in reasoning.
//! * `</think>` closes reasoning. Without an open tag it is text.
//! * White space directly after `<think>` and directly after `</think>`, and the white space that ends the reasoning
//!   before `</think>`, is not output (`<think>\nhmm\n</think>\n\nAnswer` is reasoning `hmm`, then text `Answer`).
//! * A reply that ends inside reasoning (cut off, or the model stopped) is reasoning, up to the last character; a held
//!   partial `</thi` is part of it.

use crate::stop::take_through;

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

#[derive(Debug, Clone, PartialEq)]
pub enum ThinkPart {
    Thinking(String),
    Text(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    /// Nothing but white space seen yet: `<think>` may still come.
    Start,
    Thinking,
    Answer,
}

#[derive(Debug, Clone)]
pub struct ThinkParser {
    state: State,
    buf: String,
    /// White space is dropped from the next text of the current run (after an open or a close tag).
    skip_ws: bool,
}

/// Does the rendered prompt end inside an open `<think>` (the model continues the reasoning)?
pub fn prompt_opens_thinking(prompt: &str) -> bool {
    match prompt.rfind(OPEN) {
        Some(p) => {
            let after = &prompt[p + OPEN.len()..];
            !after.contains(CLOSE) && after.trim().is_empty()
        }
        None => false,
    }
}

impl ThinkParser {
    pub fn new(start_in_thinking: bool) -> ThinkParser {
        ThinkParser { state: if start_in_thinking { State::Thinking } else { State::Start }, buf: String::new(), skip_ws: start_in_thinking }
    }

    pub fn push(&mut self, text: &str) -> Vec<ThinkPart> {
        let mut out = Vec::new();
        self.buf.push_str(text);
        loop {
            match self.state {
                State::Start => {
                    let t = self.buf.trim_start();
                    if let Some(rest) = t.strip_prefix(OPEN) {
                        self.buf = rest.to_string();
                        self.state = State::Thinking;
                        self.skip_ws = true;
                    } else if t.is_empty() || OPEN.starts_with(t) {
                        return out; // `<think>` may still come
                    } else {
                        self.state = State::Answer;
                    }
                }
                State::Answer => {
                    let text = std::mem::take(&mut self.buf);
                    self.text(text, &mut out);
                    return out;
                }
                State::Thinking => {
                    if let Some(p) = self.buf.find(CLOSE) {
                        let body = take_through(&mut self.buf, p, CLOSE.len());
                        self.thinking(body.trim_end().to_string(), &mut out);
                        self.state = State::Answer;
                        self.skip_ws = true;
                        continue;
                    }
                    // hold back a tail that could still become `</think>`, and the white space before it
                    let keep = (1..CLOSE.len()).rev().find(|&n| self.buf.ends_with(&CLOSE[..n])).unwrap_or(0);
                    let tail = self.buf.split_off(self.buf.len() - keep);
                    let body = std::mem::take(&mut self.buf);
                    let shown = body.trim_end();
                    self.buf = format!("{}{tail}", &body[shown.len()..]);
                    self.thinking(shown.to_string(), &mut out);
                    return out;
                }
            }
        }
    }

    /// The reply ended: what is still held was not a tag.
    pub fn finish(&mut self) -> Vec<ThinkPart> {
        let mut out = Vec::new();
        let rest = std::mem::take(&mut self.buf);
        match self.state {
            State::Thinking => self.thinking(rest.trim_end().to_string(), &mut out),
            State::Start | State::Answer => self.text(rest, &mut out),
        }
        out
    }

    fn thinking(&mut self, text: String, out: &mut Vec<ThinkPart>) {
        self.emit(text, ThinkPart::Thinking, out);
    }

    fn text(&mut self, text: String, out: &mut Vec<ThinkPart>) {
        self.emit(text, ThinkPart::Text, out);
    }

    /// `text` as a part (made by `part`), unless nothing is left of it; white space at the start of a piece is dropped while
    /// `skip_ws` is set, and the first thing that is kept ends that.
    fn emit(&mut self, text: String, part: fn(String) -> ThinkPart, out: &mut Vec<ThinkPart>) {
        let text = if self.skip_ws { text.trim_start().to_string() } else { text };
        if !text.is_empty() {
            self.skip_ws = false;
            out.push(part(text));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All parts for `pieces`, runs of one kind merged (how the text was cut may split a run).
    fn parse(start_in_thinking: bool, pieces: &[&str]) -> Vec<ThinkPart> {
        let mut p = ThinkParser::new(start_in_thinking);
        let mut all: Vec<ThinkPart> = pieces.iter().flat_map(|s| p.push(s)).collect();
        all.extend(p.finish());
        let mut out: Vec<ThinkPart> = Vec::new();
        for part in all {
            match (out.last_mut(), part) {
                (Some(ThinkPart::Thinking(a)), ThinkPart::Thinking(b)) | (Some(ThinkPart::Text(a)), ThinkPart::Text(b)) => a.push_str(&b),
                (_, part) => out.push(part),
            }
        }
        out
    }

    fn same_however_cut(text: &str) -> Vec<ThinkPart> {
        let whole = parse(false, &[text]);
        let chars: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        assert_eq!(parse(false, &chars.iter().map(String::as_str).collect::<Vec<_>>()), whole, "one char per piece: {text:?}");
        for i in (0..=text.len()).filter(|&i| text.is_char_boundary(i)) {
            assert_eq!(parse(false, &[&text[..i], &text[i..]]), whole, "cut at {i}: {text:?}");
        }
        whole
    }

    fn th(s: &str) -> ThinkPart {
        ThinkPart::Thinking(s.into())
    }

    fn tx(s: &str) -> ThinkPart {
        ThinkPart::Text(s.into())
    }

    #[test]
    fn reasoning_then_answer_however_the_text_is_cut() {
        assert_eq!(same_however_cut("<think>\nhmm\n</think>\n\nAnswer."), vec![th("hmm"), tx("Answer.")]);
        assert_eq!(same_however_cut("  <think>a b</think>c"), vec![th("a b"), tx("c")]);
        assert_eq!(same_however_cut("<think></think>x"), vec![tx("x")]);
        assert_eq!(same_however_cut("no tags here"), vec![tx("no tags here")]);
        assert_eq!(same_however_cut("<think>\nline 1\n\nline 2\n</think>"), vec![th("line 1\n\nline 2")]);
    }

    #[test]
    fn lookalikes_are_text_verbatim() {
        for text in ["<thin", "<thin gs", "<think b>x", "<", " <th", "a <think>b</think>", "</think> x", "<thinking>"] {
            assert_eq!(same_however_cut(text), vec![tx(text)], "{text}");
        }
        // a lookalike inside reasoning stays reasoning
        assert_eq!(same_however_cut("<think>a </thin b</think>c"), vec![th("a </thin b"), tx("c")]);
    }

    #[test]
    fn an_unclosed_tag_is_reasoning_to_the_end() {
        assert_eq!(same_however_cut("<think>\nstill thinking"), vec![th("still thinking")]);
        assert_eq!(same_however_cut("<think>abc</th"), vec![th("abc</th")]);
        assert_eq!(same_however_cut("<think>"), vec![]);
        assert_eq!(same_however_cut("<think>\n  "), vec![]);
    }

    #[test]
    fn white_space_in_the_middle_of_reasoning_is_kept_and_held_until_it_is_known() {
        let mut p = ThinkParser::new(false);
        assert_eq!(p.push("<think>one "), vec![th("one")]);
        assert_eq!(p.push("two\n"), vec![th(" two")]);
        assert_eq!(p.push("</think>\n"), vec![]);
        assert_eq!(p.push("\nDone"), vec![tx("Done")]);
    }

    #[test]
    fn a_prompt_that_opens_the_tag_starts_in_reasoning() {
        assert_eq!(parse(true, &["let me see", "</think>ok"]), vec![th("let me see"), tx("ok")]);
        assert!(prompt_opens_thinking("<|im_start|>assistant\n<think>\n"));
        assert!(prompt_opens_thinking("x<think>"));
        assert!(!prompt_opens_thinking("<|im_start|>assistant\n"));
        assert!(!prompt_opens_thinking("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "Qwen3 with thinking off closes it");
        assert!(!prompt_opens_thinking("<think>old</think><|im_start|>assistant\n"));
    }
}
