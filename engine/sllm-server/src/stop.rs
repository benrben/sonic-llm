//! Stop strings (`stop` on `/completion`, `/v1/completions` and `/v1/chat/completions`), matched on the text the engine
//! streams, in the API layer.
//!
//! [`StopFilter::push`] takes the text of one token and returns what may be sent now. A tail of the text that could still
//! become a stop string (a proper prefix of one) is held back and sent with the next token, or by [`StopFilter::finish`]
//! when the output ends without the stop string coming. The stop string itself, and everything after it, is never
//! returned. When several stop strings are given, the one that is complete first (earliest in the text) wins. The engine
//! knows nothing of stop strings: the caller cancels the request once `push` reports a match.

use serde::Deserialize;

/// The text of `buf` before the marker found at byte `at` (`marker_len` long), which is dropped with it: `buf` keeps what
/// follows the marker.
pub(crate) fn take_through(buf: &mut String, at: usize, marker_len: usize) -> String {
    let rest = buf.split_off(at + marker_len);
    buf.truncate(at);
    std::mem::replace(buf, rest)
}

/// The `stop` field: one string or a list of strings.
#[derive(Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum StopSpec {
    One(String),
    Many(Vec<String>),
}

impl StopSpec {
    /// The non-empty stop strings (`""` and `[]` ask for nothing).
    pub fn strings(&self) -> Vec<String> {
        let v = match self {
            StopSpec::One(s) => vec![s.clone()],
            StopSpec::Many(v) => v.clone(),
        };
        v.into_iter().filter(|s| !s.is_empty()).collect()
    }
}

pub fn stops_of(spec: &Option<StopSpec>) -> Vec<String> {
    spec.as_ref().map(StopSpec::strings).unwrap_or_default()
}

#[derive(Debug, Clone, Default)]
pub struct StopFilter {
    stops: Vec<String>,
    held: String,
}

impl StopFilter {
    pub fn new(stops: Vec<String>) -> StopFilter {
        StopFilter { stops: stops.into_iter().filter(|s| !s.is_empty()).collect(), held: String::new() }
    }

    pub fn stops(&self) -> &[String] {
        &self.stops
    }

    /// (text to send now, the stop string that matched). After a match nothing more may be pushed.
    pub fn push(&mut self, text: &str) -> (String, Option<String>) {
        if self.stops.is_empty() {
            return (text.to_string(), None);
        }
        self.held.push_str(text);
        let buf = std::mem::take(&mut self.held);
        let hit = self.stops.iter().filter_map(|s| buf.find(s.as_str()).map(|p| (p, s))).min_by_key(|(p, _)| *p);
        if let Some((p, s)) = hit {
            return (buf[..p].to_string(), Some(s.clone()));
        }
        // keep the longest tail that is a proper prefix of a stop string (a char boundary: the prefix is valid UTF-8)
        let keep = self.stops.iter().flat_map(|s| (1..s.len()).rev().filter(|&n| s.is_char_boundary(n)).find(|&n| buf.ends_with(&s[..n]))).max().unwrap_or(0);
        let cut = buf.len() - keep;
        self.held = buf[cut..].to_string();
        (buf[..cut].to_string(), None)
    }

    /// The output ended without a match: the held text was not a stop string after all.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], pieces: &[&str]) -> (String, Option<String>) {
        let mut f = StopFilter::new(stops.iter().map(|s| s.to_string()).collect());
        let mut out = String::new();
        for p in pieces {
            let (t, hit) = f.push(p);
            out.push_str(&t);
            if hit.is_some() {
                return (out, hit);
            }
        }
        out.push_str(&f.finish());
        (out, None)
    }

    #[test]
    fn matches_inside_a_token_across_tokens_and_overlapping() {
        assert_eq!(run(&["STOP"], &["a STOP b"]), ("a ".into(), Some("STOP".into())));
        assert_eq!(run(&["STOP"], &["a S", "T", "OP b"]), ("a ".into(), Some("STOP".into())));
        assert_eq!(run(&["ab"], &["a", "a", "b", "c"]), ("a".into(), Some("ab".into())));
        assert_eq!(run(&["aab"], &["a", "a", "a", "b"]), ("a".into(), Some("aab".into())));
        assert_eq!(run(&["x", "yz"], &["ay", "z"]), ("a".into(), Some("yz".into())));
    }

    #[test]
    fn no_match_releases_everything_in_order() {
        assert_eq!(run(&["STOP"], &["a S", "TO"]), ("a STO".into(), None));
        assert_eq!(run(&["STOP"], &["SS", "TOP"]), ("S".into(), Some("STOP".into())));
        assert_eq!(run(&[], &["ab", "c"]), ("abc".into(), None));
        // a char that is not ASCII: the held tail and the cut are on char boundaries
        assert_eq!(run(&["日本"], &["a日", "x"]), ("a日x".into(), None));
        assert_eq!(run(&["日本"], &["a日", "本b"]), ("a".into(), Some("日本".into())));
    }

    #[test]
    fn what_is_sent_never_holds_a_stop_prefix() {
        let mut f = StopFilter::new(vec!["STOP".into()]);
        assert_eq!(f.push("foo S"), ("foo ".to_string(), None));
        assert_eq!(f.push("TO"), (String::new(), None));
        assert_eq!(f.push("P bar"), (String::new(), Some("STOP".into())));
    }
}
