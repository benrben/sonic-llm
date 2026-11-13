//! Compress by chunk scoring: cut a tool result into chunks of at most `max_tokens` tokens, and pick the best-scoring
//! chunks within a token budget (phase 3 helper lane).
//!
//! Chunks are contiguous byte ranges that cover the text in order. Lines stay whole where they fit; a blank line closes a
//! chunk that is at least half full (paragraphs stay together); a line longer than a chunk is cut at whitespace, and a
//! single word longer than a chunk at character boundaries. Token counts come from the caller (the helper's tokenizer).

use std::ops::Range;

/// Cut `text` into chunks of at most `max_tokens` tokens by `count` (except a single character, which is never cut).
pub fn chunk_text(text: &str, max_tokens: usize, count: &dyn Fn(&str) -> usize) -> Vec<Range<usize>> {
    let max_tokens = max_tokens.max(1);
    let segs = segments(text, max_tokens, count);
    let mut out: Vec<Range<usize>> = Vec::new();
    let mut cur: Option<(Range<usize>, usize)> = None;
    for (r, n) in segs {
        let blank = text[r.clone()].trim().is_empty();
        cur = match cur {
            None => Some((r, n)),
            Some((c, cn)) if cn + n <= max_tokens && !(blank && cn * 2 >= max_tokens) => Some((c.start..r.end, cn + n)),
            Some((c, _)) => {
                out.push(c);
                Some((r, n))
            }
        };
    }
    if let Some((c, _)) = cur {
        out.push(c);
    }
    out
}

/// The segments of `text` and their token counts: lines (with their newline), long lines cut further.
fn segments(text: &str, max_tokens: usize, count: &dyn Fn(&str) -> usize) -> Vec<(Range<usize>, usize)> {
    let mut segs: Vec<(Range<usize>, usize)> = Vec::new();
    let mut start = 0;
    for (i, b) in text.bytes().enumerate() {
        if b == b'\n' {
            push_segment(text, start..i + 1, max_tokens, count, &mut segs);
            start = i + 1;
        }
    }
    if start < text.len() {
        push_segment(text, start..text.len(), max_tokens, count, &mut segs);
    }
    segs
}

fn push_segment(text: &str, r: Range<usize>, max_tokens: usize, count: &dyn Fn(&str) -> usize, out: &mut Vec<(Range<usize>, usize)>) {
    let n = count(&text[r.clone()]);
    if n <= max_tokens {
        out.push((r, n));
        return;
    }
    // cut at whitespace: words with the whitespace in front of them
    let s = &text[r.clone()];
    let mut cuts = vec![0usize];
    let mut prev_ws = false;
    for (i, ch) in s.char_indices() {
        let ws = ch.is_whitespace();
        if !ws && prev_ws {
            cuts.push(i);
        }
        prev_ws = ws;
    }
    cuts.push(s.len());
    let mut a = 0usize;
    let mut k = 1usize;
    while k < cuts.len() {
        // the longest run of words from `a` that fits
        let mut b = k;
        while b + 1 < cuts.len() && count(&s[a..cuts[b + 1]]) <= max_tokens {
            b += 1;
        }
        let piece = r.start + a..r.start + cuts[b];
        let pn = count(&text[piece.clone()]);
        if pn <= max_tokens {
            out.push((piece, pn));
        } else {
            // one word longer than a chunk: halve it at character boundaries until the parts fit
            split_word(text, piece, max_tokens, count, out);
        }
        a = cuts[b];
        k = b + 1;
    }
}

fn split_word(text: &str, r: Range<usize>, max_tokens: usize, count: &dyn Fn(&str) -> usize, out: &mut Vec<(Range<usize>, usize)>) {
    let n = count(&text[r.clone()]);
    if n <= max_tokens || text[r.clone()].chars().count() <= 1 {
        out.push((r, n));
        return;
    }
    let mut mid = r.start + (r.end - r.start) / 2;
    while !text.is_char_boundary(mid) {
        mid += 1;
    }
    if mid >= r.end {
        mid = r.start + text[r.clone()].chars().next().map_or(1, |c| c.len_utf8());
    }
    split_word(text, r.start..mid, max_tokens, count, out);
    split_word(text, mid..r.end, max_tokens, count, out);
}

/// Pick chunks within `budget` tokens: best score first, skipping a chunk that would not fit (a smaller one later still
/// may); `None` scores (empty chunks) are never kept. Returns the kept indices in text order.
pub fn select_within_budget(scores: &[Option<f32>], tokens: &[usize], budget: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).filter(|&i| scores[i].is_some()).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    let mut used = 0usize;
    let mut kept = Vec::new();
    for i in order {
        if used + tokens[i] <= budget {
            used += tokens[i];
            kept.push(i);
        }
    }
    kept.sort();
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One token per whitespace-separated word (at least one per non-empty string).
    fn words(s: &str) -> usize {
        s.split_whitespace().count().max(usize::from(!s.is_empty()))
    }

    fn check_cover(text: &str, ch: &[Range<usize>]) {
        assert!(!text.is_empty() || ch.is_empty());
        let mut at = 0;
        for c in ch {
            assert_eq!(c.start, at, "chunks are contiguous");
            assert!(c.end > c.start, "no empty chunk");
            assert!(text.is_char_boundary(c.start) && text.is_char_boundary(c.end));
            at = c.end;
        }
        assert_eq!(at, text.len(), "chunks cover the text");
    }

    #[test]
    fn lines_are_merged_up_to_the_chunk_size() {
        let text = "a b\nc d\ne f\ng h\n";
        let ch = chunk_text(text, 4, &words);
        check_cover(text, &ch);
        assert_eq!(ch.iter().map(|r| &text[r.clone()]).collect::<Vec<_>>(), vec!["a b\nc d\n", "e f\ng h\n"]);
    }

    #[test]
    fn a_blank_line_closes_a_half_full_chunk() {
        let text = "a b c\n\nd e\n";
        let ch = chunk_text(text, 6, &words); // 3 of 6 tokens: half full
        check_cover(text, &ch);
        assert_eq!(&text[ch[0].clone()], "a b c\n");
        assert!(ch.len() >= 2);
    }

    #[test]
    fn long_lines_are_cut_at_whitespace_and_every_chunk_fits() {
        let text = "one two three four five six seven eight nine ten eleven\nshort\n";
        let ch = chunk_text(text, 3, &words);
        check_cover(text, &ch);
        for c in &ch {
            assert!(words(&text[c.clone()]) <= 3, "{:?}", &text[c.clone()]);
        }
    }

    #[test]
    fn a_long_word_is_cut_at_char_boundaries() {
        let chars = |s: &str| s.chars().count();
        let text = "ééééééééééxyz";
        let ch = chunk_text(text, 4, &chars);
        check_cover(text, &ch);
        for c in &ch {
            assert!(chars(&text[c.clone()]) <= 4);
        }
    }

    #[test]
    fn random_texts_are_always_covered_within_the_limit() {
        let parts = ["alpha ", "beta\n", "\n", "gamma delta ", "épsilon ", "x", "  ", "zeta-eta-theta "];
        let mut rnd = crate::helper::seeded(99);
        for _ in 0..300 {
            let text: String = (0..rnd(40)).map(|_| parts[rnd(parts.len())]).collect();
            let max = 1 + rnd(6);
            let ch = chunk_text(&text, max, &words);
            check_cover(&text, &ch);
            for c in &ch {
                assert!(words(&text[c.clone()]) <= max, "{max} {:?}", &text[c.clone()]);
            }
        }
    }

    #[test]
    fn selection_keeps_the_best_within_the_budget_in_text_order() {
        let scores = [Some(0.1), Some(0.9), None, Some(0.5), Some(0.8)];
        let tokens = [10, 50, 5, 30, 40];
        assert_eq!(select_within_budget(&scores, &tokens, 100), vec![0, 1, 4]); // 50 + 40 + 10; 0.5 (30) does not fit
        assert_eq!(select_within_budget(&scores, &tokens, 49), vec![4]); // 0.9 (50) does not fit; then 40, and nothing else fits
        assert_eq!(select_within_budget(&scores, &tokens, 40), vec![4]); // the best that fits first (40), not 30 + 10
        assert_eq!(select_within_budget(&scores, &tokens, 39), vec![3]); // 50 and 40 do not fit, 30 does, then 10 no longer
        assert!(select_within_budget(&scores, &tokens, 0).is_empty());
    }
}
