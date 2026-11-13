//! Facts kept word for word beside a summary: the lines of a text that carry a hard fact.

/// The most brain tokens of verbatim facts a compaction keeps beside its summary (summaries drop the numbers and names
/// an agent still needs, G3-8; a few hundred tokens of source lines cost about one extra prefill chunk).
pub const COMPACTION_FACTS_CAP: usize = 256;

/// The most a rollover keeps: the archive has every line, this is what stays in front of the model without being asked for.
const ROLL_FACTS_MAX: usize = 512;

/// A rollover keeps a sixteenth of its window of them, up to [`ROLL_FACTS_MAX`].
pub fn facts_cap(window: usize) -> usize {
    (window / 16).min(std::env::var("SLLM_ROLL_FACTS").ok().and_then(|v| v.parse().ok()).unwrap_or(ROLL_FACTS_MAX))
}

/// The header line of the facts record in the compacted context.
pub const FACTS_HEADER: &str = "Facts kept verbatim:\n";

fn is_file_name(tok: &str) -> bool {
    let t = tok.trim_matches(|c: char| !(c.is_alphanumeric() || c == '/' || c == '.' || c == '_' || c == '-'));
    if t.contains('/') {
        return t.len() > 1;
    }
    match t.rsplit_once('.') {
        Some((stem, ext)) => !stem.is_empty() && (1..=5).contains(&ext.len()) && ext.chars().all(|c| c.is_ascii_alphanumeric()) && ext.chars().any(|c| c.is_ascii_alphabetic()),
        None => false,
    }
}

/// Does this line carry a hard fact: a digit (numbers, years, IDs), a file name or path, or a `key: value` / `key=value`
/// pair? Prose without any of those does not.
pub fn line_has_fact(line: &str) -> bool {
    if line.chars().any(|c| c.is_ascii_digit()) {
        return true;
    }
    let mut toks = line.split_whitespace();
    let Some(first) = toks.next() else { return false };
    if first.len() <= 33 && first.ends_with(':') && first.len() > 1 && line.split_whitespace().count() > 1 {
        return true;
    }
    line.split_whitespace().any(|t| t.split_once('=').is_some_and(|(k, v)| !k.is_empty() && !v.is_empty()) || is_file_name(t))
}

/// The lines of `text` worth keeping word for word: the ones with a hard fact ([`line_has_fact`]), without exact
/// duplicates (the later copy is kept, so the order stays the original one of the last occurrence), without the lines
/// that `tail` still holds, in their original order. `count` is the token count of a text; when the lines are more than
/// `cap` tokens the oldest are dropped first. Deterministic, no model.
pub fn select_fact_lines(text: &str, tail: &str, cap: usize, count: &dyn Fn(&str) -> usize) -> Vec<String> {
    let kept = fact_lines(text, tail);
    let from = first_within_cap(&kept, cap, count);
    kept[from..].iter().map(|l| l.to_string()).collect()
}

/// The lines of `text` with a hard fact that `tail` does not hold, each once (its last occurrence), in order.
fn fact_lines<'a>(text: &'a str, tail: &str) -> Vec<&'a str> {
    let in_tail: std::collections::HashSet<&str> = tail.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let mut seen = std::collections::HashSet::new();
    let mut kept: Vec<&str> = Vec::new();
    for l in text.lines().rev() {
        let l = l.trim();
        if is_record_fact(l) && !in_tail.contains(l) && seen.insert(l) {
            kept.push(l);
        }
    }
    kept.reverse();
    kept
}

/// A line with a hard fact that is not part of a record: the facts header of an earlier compaction and the harness's
/// list of unread files are not facts.
fn is_record_fact(l: &str) -> bool {
    !(l.is_empty() || l.starts_with("Facts kept verbatim") || l.starts_with("Unread files:")) && line_has_fact(l)
}

/// Where the kept lines start: newest lines first until the cap, then the oldest dropped while the joined text is still
/// over it.
fn first_within_cap(kept: &[&str], cap: usize, count: &dyn Fn(&str) -> usize) -> usize {
    let mut total = 0;
    let mut from = kept.len();
    while from > 0 {
        let c = count(kept[from - 1]) + 1;
        if total + c > cap {
            break;
        }
        total += c;
        from -= 1;
    }
    while from < kept.len() && count(&kept[from..].join("\n")) > cap {
        from += 1;
    }
    from
}
