//! Prepared calls: tokenized, validated, their pages known.

use super::*;

/// A request's optional size, or `default`, held to `min..=max`.
fn or_clamped(value: Option<usize>, default: usize, min: usize, max: usize) -> usize {
    value.unwrap_or(default).clamp(min, max)
}

/// A classify call: the text once, then the label question (and each yes/no check as a fork of the text).
pub(super) struct ClassifyIn {
    pub(super) prompt: Vec<i32>,
    pub(super) split: usize,
    pub(super) labels: Vec<String>,
    pub(super) label_ids: Vec<Vec<i32>>,
    pub(super) variants: Vec<Vec<i32>>,
    pub(super) fast: bool,
    pub(super) checks: Vec<(String, Vec<i32>)>,
    pub(super) yes: Vec<i32>,
    pub(super) no: Vec<i32>,
}

/// An extract call: the text once, then one forked question per field.
pub(super) struct ExtractIn {
    pub(super) prefix: Vec<i32>,
    pub(super) text: String,
    pub(super) n_src: usize,
    pub(super) fields: Vec<(String, Vec<i32>)>,
    pub(super) max_spans: usize,
    pub(super) max_tokens: usize,
    pub(super) none_tok: i32,
}

/// A compress call: the text's chunks, each scored against the question.
pub(super) struct CompressIn {
    pub(super) prefix: Vec<i32>,
    pub(super) suffix: Vec<i32>,
    pub(super) chunks: Vec<std::ops::Range<usize>>,
    pub(super) chunk_ids: Vec<Vec<i32>>,
    pub(super) text: String,
    pub(super) budget: usize,
    pub(super) yes: Vec<i32>,
    pub(super) no: Vec<i32>,
    pub(super) qlik: Option<QLik>,
    pub(super) wave: usize,
}

/// What preparing a call gives: its inputs and its worst-case pages, or the refusal.
type PreparedCall = Res<(Prepared, usize)>;

pub(super) enum Prepared {
    Classify(ClassifyIn),
    Extract(ExtractIn),
    Compress(CompressIn),
    Summarize { prompt: Vec<i32>, max_tokens: usize, ignore_eos: bool },
}

/// Query-likelihood scoring of a chunk: `head + chunk + mid` is read, then the question's tokens are scored.
pub(super) struct QLik {
    pub(super) head: Vec<i32>,
    pub(super) mid: Vec<i32>,
    pub(super) question: Vec<i32>,
}

const ON_TRACK: &str = "Is the agent on-track (making progress toward its goal) or off-track (repeating a failing action, looping, \
drifting to unrelated work, or ignoring its goal)?";

fn case_variants(w: &str) -> Vec<String> {
    let mut c = w.chars();
    let cap = match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    };
    let mut v = vec![w.to_string(), cap, w.to_uppercase(), w.to_lowercase()];
    v.dedup();
    v
}

/// The refusal of a call that needs `n` tokens, more than the helper context.
fn too_long(ctx: &Ctx, n: usize) -> EngineError {
    let n_ctx = ctx.cfg.n_ctx;
    EngineError::new(400, "exceed_context_size_error", format!("the helper call needs {n} tokens, more than the helper context of {n_ctx}")).with_context(n, n_ctx)
}

/// Tokenize and validate a call; returns the job's inputs and its worst-case pages.
pub(super) fn prepare(ctx: &Ctx, op: &Op) -> PreparedCall {
    match op {
        Op::Classify(r) => prepare_classify(ctx, r),
        Op::Extract(r) => prepare_extract(ctx, r),
        Op::Compress(r) => prepare_compress(ctx, r),
        Op::Summarize(r) => prepare_summarize(ctx, r),
    }
}

/// The question a classify call asks about its text.
fn classify_instruction(r: &ClassifyReq) -> String {
    match (&r.instruction, &r.labels) {
        (Some(i), _) => i.clone(),
        (None, None) => {
            let goal = r.goal.as_deref().map(|g| format!(" The agent's goal: {g}")).unwrap_or_default();
            format!("The text is the recent transcript of an AI agent.{goal} {ON_TRACK}")
        }
        (None, Some(_)) => "Which label fits the text best?".into(),
    }
}

fn prepare_classify(ctx: &Ctx, r: &ClassifyReq) -> PreparedCall {
    let labels = r.labels.clone().unwrap_or_else(|| vec!["on-track".into(), "off-track".into()]);
    if labels.len() < 2 || labels.iter().any(|l| l.trim().is_empty()) {
        return Err(invalid("labels: at least two non-empty labels"));
    }
    let instruction = classify_instruction(r);
    let mut prompt = ctx.user.clone();
    prompt.extend(ctx.tx("<text>\n"));
    prompt.extend(ctx.tx(&r.text));
    prompt.extend(ctx.tx("\n</text>\n\n"));
    // the text is read once; the label question and each check are forks of it
    let split = prompt.len();
    prompt.extend(ctx.tx(&format!("{instruction}\nAnswer with one of: {}.", labels.join(", "))));
    prompt.extend(ctx.assist.clone());
    let checks: Vec<(String, Vec<i32>)> = r
        .checks
        .iter()
        .flatten()
        .map(|c| {
            let mut t = ctx.tx(&format!("{} Answer yes or no.", c.trim()));
            t.extend(ctx.assist.clone());
            (c.clone(), t)
        })
        .collect();
    let max_check = checks.iter().map(|c| c.1.len()).max().unwrap_or(0);
    let label_ids: Vec<Vec<i32>> = labels.iter().map(|l| ctx.tx(l)).collect();
    let variants: Vec<Vec<i32>> = labels.iter().map(|l| ctx.first_tokens(&case_variants(l))).collect();
    // fast path (one read, no fork): every label is told apart by its first token, case variants included
    let mut seen = std::collections::HashSet::new();
    let disjoint = variants.iter().flatten().all(|t| seen.insert(*t));
    let fast = !r.full && disjoint;
    let max_label = label_ids.iter().map(Vec::len).max().unwrap_or(1);
    let cap = (prompt.len() + max_label).max(split + max_check) + 1;
    if cap > ctx.cfg.n_ctx {
        return Err(too_long(ctx, cap));
    }
    let forks = if checks.is_empty() { 0 } else { 1 + checks.len() };
    let need = pages(cap) + forks * (1 + pages(cap - split)) + if fast { 0 } else { labels.len() * (1 + pages(max_label)) };
    let yes = ctx.first_tokens(&case_variants("yes"));
    let no = ctx.first_tokens(&case_variants("no"));
    Ok((Prepared::Classify(ClassifyIn { prompt, split, labels, label_ids, variants, fast, checks, yes, no }), need))
}

fn prepare_extract(ctx: &Ctx, r: &ExtractReq) -> PreparedCall {
    let fields: Vec<(String, String)> = match (&r.fields, &r.what) {
        (Some(f), None) if !f.is_empty() => f.iter().map(|f| (f.name.clone(), f.what.clone().unwrap_or_else(|| f.name.clone()))).collect(),
        (None, Some(w)) => vec![("value".into(), w.clone())],
        _ => return Err(invalid("extract: give `what` or a non-empty `fields`, not both")),
    };
    let ids = ctx.tx(&r.text);
    let n_src = ids.len();
    let mut prefix = ctx.user.clone();
    prefix.extend(ctx.tx("<text>\n"));
    prefix.extend(ids);
    prefix.extend(ctx.tx("\n</text>\n\n"));
    let fields: Vec<(String, Vec<i32>)> = fields
        .into_iter()
        .map(|(name, what)| {
            let mut q = ctx.tx(&format!(
                "Copy every {what} that appears in the text above, exactly as written, one per line. If there is none, answer NONE."
            ));
            q.extend(ctx.assist.clone());
            (name, q)
        })
        .collect();
    let max_spans = or_clamped(r.max_spans, 16, 1, 256);
    let max_tokens = or_clamped(r.max_tokens, 96, 1, 1024);
    let max_q = fields.iter().map(|f| f.1.len()).max().unwrap_or(0);
    let cap = prefix.len() + max_q + max_tokens + 1;
    if cap > ctx.cfg.n_ctx {
        return Err(too_long(ctx, cap));
    }
    let need = pages(cap) + fields.len() * (1 + pages(max_q + max_tokens + 1));
    let none_tok = ctx.tx("NONE").first().copied().unwrap_or(-1);
    Ok((Prepared::Extract(ExtractIn { prefix, text: r.text.clone(), n_src, fields, max_spans, max_tokens, none_tok }), need))
}

/// The sizes of a compress call that its scoring method's pages follow from.
struct CompressShape {
    /// Cells of the yes/no base sequence and one chunk read after it.
    cap: usize,
    max_chunk: usize,
    suffix: usize,
    chunks: usize,
}

fn prepare_compress(ctx: &Ctx, r: &CompressReq) -> PreparedCall {
    let chunk_tokens = or_clamped(r.chunk_tokens, 128, 8, 1024);
    let budget = r.budget_tokens.unwrap_or(512);
    let count = |s: &str| ctx.tx(s).len();
    let chunks = chunk_text(&r.text, chunk_tokens, &count);
    let chunk_ids: Vec<Vec<i32>> = chunks.iter().map(|c| ctx.tx(&r.text[c.clone()])).collect();
    let mut prefix = ctx.user.clone();
    prefix.extend(ctx.tx(&format!("Question: {}\n\n<passage>\n", r.question)));
    let mut suffix = ctx.tx("\n</passage>\n\nDoes the passage contain information that helps answer the question? Answer yes or no.");
    suffix.extend(ctx.assist.clone());
    let max_chunk = chunk_ids.iter().map(Vec::len).max().unwrap_or(0);
    let cap = prefix.len() + max_chunk + suffix.len() + 1;
    if cap > ctx.cfg.n_ctx {
        return Err(too_long(ctx, cap));
    }
    let yes = ctx.first_tokens(&case_variants("yes"));
    let no = ctx.first_tokens(&case_variants("no"));
    let shape = CompressShape { cap, max_chunk, suffix: suffix.len(), chunks: chunks.len() };
    let (qlik, need, wave) = compress_method(ctx, r, &shape)?;
    Ok((Prepared::Compress(CompressIn { prefix, suffix, chunks, chunk_ids, text: r.text.clone(), budget, yes, no, qlik, wave }), need))
}

/// The scoring method of a compress call: its query-likelihood inputs (none for yes/no), its pages and its yes/no wave size.
fn compress_method(ctx: &Ctx, r: &CompressReq, s: &CompressShape) -> Res<(Option<QLik>, usize, usize)> {
    match r.method.as_deref().unwrap_or(DEFAULT_COMPRESS) {
        "yesno" => {
            // chunks are scored in waves of forks; a small pool gets smaller waves instead of a refusal
            let per = 1 + pages(s.max_chunk + s.suffix);
            let mut w = s.chunks.clamp(1, COMPRESS_WAVE);
            while w > 1 && pages(s.cap) + w * per > ctx.cfg.pool_pages {
                w -= 1;
            }
            Ok((None, pages(s.cap) + w * per, w))
        }
        "qlik" => {
            let mut head = ctx.user.clone();
            head.extend(ctx.tx("<passage>\n"));
            let mut mid = ctx.tx("\n</passage>\n\nWrite one question that this passage answers.");
            mid.extend(ctx.assist.clone());
            let question = ctx.tx(r.question.trim());
            if question.is_empty() || question.len() > MAX_OUT_ROWS {
                return Err(invalid(format!("compress qlik: the question must have 1 to {MAX_OUT_ROWS} tokens")));
            }
            let cap_q = head.len() + s.max_chunk + mid.len() + question.len() + 1;
            if cap_q > ctx.cfg.n_ctx {
                return Err(too_long(ctx, cap_q));
            }
            let wave = qlik_wave(question.len()).min(s.chunks.max(1));
            Ok((Some(QLik { head, mid, question }), wave * pages(cap_q), COMPRESS_WAVE))
        }
        o => Err(invalid(format!("compress: unknown method {o:?} (yesno or qlik)"))),
    }
}

fn prepare_summarize(ctx: &Ctx, r: &SummarizeReq) -> PreparedCall {
    let max_tokens = or_clamped(r.max_tokens, 400, 1, 1024);
    let words = r.words.unwrap_or(150);
    let instr = r.instruction.clone().unwrap_or_else(|| {
        format!(
            "Summarize the conversation history below for the agent that will continue the work. Keep the goal, the \
             decisions made, the facts and identifiers (IDs, names, numbers, file paths), what has been done and what is \
             left to do. Write about {words} words."
        )
    });
    let mut prompt = ctx.user.clone();
    prompt.extend(ctx.tx(&format!("{instr}\n\n<history>\n")));
    prompt.extend(ctx.tx(&r.text));
    prompt.extend(ctx.tx("\n</history>"));
    prompt.extend(ctx.assist.clone());
    let cap = prompt.len() + max_tokens + 1;
    if cap > ctx.cfg.n_ctx {
        return Err(too_long(ctx, cap));
    }
    Ok((Prepared::Summarize { prompt, max_tokens, ignore_eos: r.ignore_eos }, pages(cap)))
}
