//! The four jobs: plain async code over [`Seq`]s.

use super::prepare::{ClassifyIn, CompressIn, ExtractIn, Prepared, QLik};
use super::*;

pub(super) async fn job<'c>(ctx: &'c Ctx, p: Prepared, prio: bool) -> Res<Value> {
    match p {
        Prepared::Classify(c) => classify(ctx, c, prio).await,
        Prepared::Extract(x) => extract(ctx, x, prio).await,
        Prepared::Compress(c) => compress(ctx, c, prio).await,
        Prepared::Summarize { prompt, max_tokens, ignore_eos } => summarize(ctx, prompt, max_tokens, ignore_eos, prio).await,
    }
}

// -- classify ---------------------------------------------------------------------------------------------------

/// A yes/no check over the text read into `base`: a fork of it asked `t`, scored by the log-odds of "yes" against "no".
async fn check_p_yes(base: &Seq<'_>, q: &str, t: &[i32], yes: &[i32], no: &[i32]) -> Res<Value> {
    let f = base.fork()?;
    let l = f.logits(t.to_vec()).await?;
    let lse = logsumexp(l.iter().copied());
    let lo = logsumexp(yes.iter().map(|&t| l[t as usize] - lse)) - logsumexp(no.iter().map(|&t| l[t as usize] - lse));
    Ok(json!({"check": q, "logodds_yes": lo, "p_yes": 1.0 / (1.0 + (-lo).exp())}))
}

/// The text once into a base sequence; the label question and every check are forks of it, in one step. Returns the
/// label question's sequence, its logits and the checks' results.
async fn read_with_checks<'c>(ctx: &'c Ctx, c: &ClassifyIn, cap: usize, prio: bool) -> Res<(Seq<'c>, Vec<f32>, Vec<Value>)> {
    let base = Seq::new(ctx, cap, prio)?;
    base.prefill(c.prompt[..c.split].to_vec(), OutputPolicy::None, Sampler::Greedy).await?;
    let lab = base.fork()?;
    let label_fut = lab.logits(c.prompt[c.split..].to_vec());
    let check_futs = c.checks.iter().map(|(q, t)| check_p_yes(&base, q, t, &c.yes, &c.no));
    let (l, checks) = futures_util::future::join(label_fut, join_all(check_futs)).await;
    Ok((lab, l?, checks.into_iter().collect::<Res<Vec<Value>>>()?))
}

/// Teacher-forced log P(label): log P(l0) (`first`) from the prompt, the rest from a fork of `seq`.
async fn label_logp(ctx: &Ctx, seq: &Seq<'_>, ids: &[i32], first: f32) -> Res<f32> {
    if ids.len() <= 1 {
        return Ok(first);
    }
    let f = seq.fork()?;
    let o = f.prefill(ids[..ids.len() - 1].to_vec(), OutputPolicy::All, Sampler::HostLogits).await?;
    let l = o.logits.ok_or_else(|| server_err("no logits"))?;
    let mut s = first;
    for (k, row) in l.chunks_exact(ctx.n_vocab).enumerate() {
        let lse = logsumexp(row.iter().copied());
        s += row[ids[k + 1] as usize] - lse;
    }
    Ok(s)
}

pub(super) async fn classify(ctx: &Ctx, c: ClassifyIn, prio: bool) -> Res<Value> {
    let max_label = c.label_ids.iter().map(Vec::len).max().unwrap_or(1);
    let max_check = c.checks.iter().map(|c| c.1.len()).max().unwrap_or(0);
    let n_prompt = c.prompt.len();
    let cap = (n_prompt + max_label).max(c.split + max_check) + 1;
    let (seq, logits, checks_out) = if c.checks.is_empty() {
        let seq = Seq::new(ctx, cap, prio)?;
        let logits = seq.logits(c.prompt.clone()).await?;
        (seq, logits, Vec::new())
    } else {
        read_with_checks(ctx, &c, cap, prio).await?
    };
    let lse = logsumexp(logits.iter().copied());
    let lp = |t: i32| logits.get(t as usize).map_or(f32::NEG_INFINITY, |l| l - lse);
    let logp: Vec<f32> = if c.fast {
        // P(label) = the summed probability of its first token's case variants
        c.variants.iter().map(|v| logsumexp(v.iter().map(|&t| lp(t)))).collect()
    } else {
        // teacher-forced: log P(l0) from the prompt, the rest from one fork per label
        let futs = c.label_ids.iter().map(|ids| label_logp(ctx, &seq, ids, ids.first().map_or(f32::NEG_INFINITY, |&t| lp(t))));
        join_all(futs).await.into_iter().collect::<Res<Vec<f32>>>()?
    };
    let mass: f32 = logp.iter().map(|l| l.exp()).sum();
    Ok(classify_result(&c, &logp, mass, checks_out))
}

fn classify_result(c: &ClassifyIn, logp: &[f32], mass: f32, checks_out: Vec<Value>) -> Value {
    let labels = &c.labels;
    let z = logsumexp(logp.iter().copied());
    let probs: Vec<f32> = logp.iter().map(|l| (l - z).exp()).collect();
    let best = (0..labels.len()).max_by(|&a, &b| probs[a].partial_cmp(&probs[b]).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0);
    let mut pm = serde_json::Map::new();
    let mut lm = serde_json::Map::new();
    for (i, l) in labels.iter().enumerate() {
        pm.insert(l.clone(), json!(probs[i]));
        lm.insert(l.clone(), json!(logp[i]));
    }
    json!({
        "label": labels[best], "probs": pm, "logprobs": lm,
        // the probability the model put on any label at all (a low mass means the question or labels do not fit)
        "label_mass": mass,
        "method": if c.fast { "first-token" } else { "sequence" },
        // yes/no checks over the same read (`checks`): P(yes) of each, and the largest
        "checks": checks_out,
        "max_check_p_yes": checks_out.iter().filter_map(|c| c["p_yes"].as_f64()).fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.max(x)))),
        "prompt_tokens": c.prompt.len() + c.checks.iter().map(|c| c.1.len()).sum::<usize>(), "generated_tokens": 0,
    })
}

// -- extract ----------------------------------------------------------------------------------------------------

/// The limits of one field's copying.
struct FieldLimits {
    max_spans: usize,
    max_tokens: usize,
    /// The NONE token, when the vocabulary has one.
    none: Option<i32>,
}

/// One field's copying so far: the spans found, the span being read, and the counts that end it.
struct FieldRun<'t> {
    text: &'t str,
    spans: Vec<Span>,
    st: SpanState,
    produced: usize,
    any: bool,
    dupes: usize,
    bullets_in_row: usize,
}

impl<'t> FieldRun<'t> {
    fn new(text: &'t str) -> Self {
        FieldRun { text, spans: Vec::new(), st: SpanState::new(), produced: 0, any: false, dupes: 0, bullets_in_row: 0 }
    }

    /// Ends the span being read; true when it repeats one found before (it is not kept again).
    fn close(&mut self) -> bool {
        let mut dup = false;
        if let Some((a, b)) = self.st.run() {
            if let Some(s) = locate(self.text, a, b) {
                if self.spans.iter().any(|x| x.text == s.text) {
                    dup = true;
                } else {
                    self.spans.push(s);
                }
            }
        }
        self.st = SpanState::new();
        dup
    }

    /// NONE is a choice only before anything was copied.
    fn none_ok(&self, none: Option<i32>) -> Option<i32> {
        if !self.any && self.st.is_empty() && self.bullets_in_row == 0 { none } else { None }
    }

    /// A bullet in front of a span is formatting: fed to the model, never part of the span. The bullet to feed now, if
    /// the best one beats the pick `c`.
    fn bullet(&mut self, ctx: &Ctx, logits: &[f32], c: Choice) -> Option<i32> {
        if !(self.st.is_empty() && self.bullets_in_row < 2) {
            return None;
        }
        let b = ctx.bullets.iter().copied().max_by(|&a, &b| logits[a as usize].total_cmp(&logits[b as usize]));
        let wins = match (b, c) {
            (Some(b), Choice::Token(t) | Choice::End(t) | Choice::None(t)) => logits[b as usize] > logits[t as usize],
            _ => false,
        };
        if let (true, Some(b)) = (wins, b) {
            self.bullets_in_row += 1;
            self.produced += 1;
            return Some(b);
        }
        None
    }

    /// Takes the pick `c`: the token fed next, or None when the field is done.
    fn take(&mut self, c: Choice, ctx: &Ctx, max_spans: usize) -> Option<i32> {
        let fed = match c {
            Choice::End(_) => {
                self.close();
                return None;
            }
            Choice::None(_) => return None,
            Choice::Sep(t) => {
                self.dupes = if self.close() { self.dupes + 1 } else { 0 };
                t
            }
            Choice::Token(t) => {
                let ok = self.st.push(self.text.as_bytes(), &ctx.pieces[t as usize]);
                debug_assert!(ok);
                self.any = true;
                t
            }
        };
        self.produced += 1;
        (self.spans.len() < max_spans && self.dupes < 2).then_some(fed)
    }
}

/// One field: a fork of the text's sequence asked question `q`, its answer copied span by span from the text.
async fn extract_field(ctx: &Ctx, base: &Seq<'_>, text: &str, name: &str, q: &[i32], lim: &FieldLimits) -> Res<(String, Vec<Span>, usize)> {
    let f = base.fork()?;
    let mut logits = f.logits(q.to_vec()).await?;
    let mut run = FieldRun::new(text);
    loop {
        if run.produced >= lim.max_tokens {
            run.close();
            break;
        }
        let c = choose(&logits, &ctx.pieces, text.as_bytes(), &run.st, &ctx.seps, &ctx.eog, run.none_ok(lim.none));
        if let Some(b) = run.bullet(ctx, &logits, c) {
            logits = f.decode_logits(b).await?;
            continue;
        }
        run.bullets_in_row = 0;
        let Some(fed) = run.take(c, ctx, lim.max_spans) else { break };
        logits = f.decode_logits(fed).await?;
    }
    Ok((name.to_string(), run.spans, run.produced))
}

pub(super) async fn extract(ctx: &Ctx, x: ExtractIn, prio: bool) -> Res<Value> {
    let max_q = x.fields.iter().map(|f| f.1.len()).max().unwrap_or(0);
    let n_prefix = x.prefix.len();
    let base = Seq::new(ctx, n_prefix + max_q + x.max_tokens + 1, prio)?;
    base.prefill(x.prefix, OutputPolicy::None, Sampler::Greedy).await?;
    let lim = FieldLimits { max_spans: x.max_spans, max_tokens: x.max_tokens, none: (x.none_tok >= 0).then_some(x.none_tok) };
    let text = &x.text;
    let futs = x.fields.iter().map(|(name, q)| extract_field(ctx, &base, text, name, q, &lim));
    let res = join_all(futs).await.into_iter().collect::<Res<Vec<_>>>()?;
    let mut by_field = serde_json::Map::new();
    let mut flat = Vec::new();
    let mut generated = 0;
    let mut valid = true;
    for (name, spans, n) in res {
        generated += n;
        for s in &spans {
            valid &= text.get(s.start..s.end) == Some(s.text.as_str());
            flat.push(json!({"field": name, "text": s.text, "start": s.start, "end": s.end, "char_start": s.char_start, "char_end": s.char_end}));
        }
        by_field.insert(name, serde_json::to_value(&spans).unwrap_or(Value::Null));
    }
    Ok(json!({
        "fields": by_field, "spans": flat,
        // every span is `text[start..end]` of the source (byte offsets; char_* in characters): checked here
        "valid": valid,
        "prompt_tokens": n_prefix, "source_tokens": x.n_src, "generated_tokens": generated,
    }))
}

// -- compress ---------------------------------------------------------------------------------------------------

/// Query likelihood of chunk `i`: `read` (head, chunk, mid) in its own sequence, then the question's tokens scored
/// teacher-forced; the mean log-probability.
async fn qlik_score(ctx: &Ctx, q: &QLik, mut read: Vec<i32>, i: usize, prio: bool) -> Res<(usize, f32)> {
    let seq = Seq::new(ctx, read.len() + q.question.len() + 1, prio)?;
    let last = read.pop().unwrap();
    seq.prefill(read, OutputPolicy::None, Sampler::Greedy).await?;
    let mut scored = vec![last];
    scored.extend_from_slice(&q.question[..q.question.len() - 1]);
    let o = seq.prefill(scored, OutputPolicy::All, Sampler::HostLogits).await?;
    let l = o.logits.ok_or_else(|| server_err("no logits"))?;
    let mut s = 0f32;
    for (k, row) in l.chunks_exact(ctx.n_vocab).enumerate() {
        s += row[q.question[k] as usize] - logsumexp(row.iter().copied());
    }
    Ok((i, s / q.question.len() as f32))
}

/// Query likelihood: one sequence per chunk, in waves whose question rows fit a step. Returns the prompt tokens read.
async fn score_qlik(ctx: &Ctx, c: &CompressIn, q: &QLik, idx: &[usize], scores: &mut [Option<f32>], prio: bool) -> Res<usize> {
    let mut prompt_tokens = 0;
    for wave in idx.chunks(qlik_wave(q.question.len())) {
        let futs = wave.iter().map(|&i| {
            let mut read = q.head.clone();
            read.extend_from_slice(&c.chunk_ids[i]);
            read.extend_from_slice(&q.mid);
            qlik_score(ctx, q, read, i, prio)
        });
        record_wave(futs, scores).await?;
        prompt_tokens += wave.iter().map(|&i| q.head.len() + c.chunk_ids[i].len() + q.mid.len() + q.question.len()).sum::<usize>();
    }
    Ok(prompt_tokens)
}

/// Run one wave of chunk scorers together and keep each chunk's score.
async fn record_wave<F: std::future::Future<Output = Res<(usize, f32)>>>(wave: impl Iterator<Item = F>, scores: &mut [Option<f32>]) -> Res<()> {
    for r in join_all(wave).await {
        let (i, s) = r?;
        scores[i] = Some(s);
    }
    Ok(())
}

/// A chunk read in a fork of the question's sequence and asked whether it helps: the log-odds of "yes" against "no".
async fn yesno_score(base: &Seq<'_>, toks: Vec<i32>, i: usize, yes: &[i32], no: &[i32]) -> Res<(usize, f32)> {
    let f = base.fork()?;
    let l = f.logits(toks).await?;
    let lse = logsumexp(l.iter().copied());
    let y = logsumexp(yes.iter().map(|&t| l[t as usize] - lse));
    let n = logsumexp(no.iter().map(|&t| l[t as usize] - lse));
    Ok((i, y - n))
}

/// Yes/no: the question is read once into a base sequence; each chunk is a fork of it. Returns the prompt tokens read.
async fn score_yesno(ctx: &Ctx, c: &CompressIn, idx: &[usize], scores: &mut [Option<f32>], prio: bool) -> Res<usize> {
    let max_chunk = c.chunk_ids.iter().map(Vec::len).max().unwrap_or(0);
    let n_prefix = c.prefix.len();
    let base = Seq::new(ctx, n_prefix + max_chunk + c.suffix.len() + 1, prio)?;
    base.prefill(c.prefix.clone(), OutputPolicy::None, Sampler::Greedy).await?;
    let mut prompt_tokens = n_prefix;
    for wave in idx.chunks(c.wave.max(1)) {
        let futs = wave.iter().map(|&i| {
            let mut toks = c.chunk_ids[i].clone();
            toks.extend_from_slice(&c.suffix);
            yesno_score(&base, toks, i, &c.yes, &c.no)
        });
        record_wave(futs, scores).await?;
        prompt_tokens += wave.iter().map(|&i| c.chunk_ids[i].len() + c.suffix.len()).sum::<usize>();
    }
    Ok(prompt_tokens)
}

pub(super) async fn compress(ctx: &Ctx, c: CompressIn, prio: bool) -> Res<Value> {
    let mut scores: Vec<Option<f32>> = vec![None; c.chunks.len()];
    let idx: Vec<usize> = (0..c.chunks.len()).filter(|&i| !c.text[c.chunks[i].clone()].trim().is_empty()).collect();
    let prompt_tokens = match &c.qlik {
        Some(q) => score_qlik(ctx, &c, q, &idx, &mut scores, prio).await?,
        None => score_yesno(ctx, &c, &idx, &mut scores, prio).await?,
    };
    Ok(compress_result(&c, &scores, prompt_tokens))
}

/// The kept chunks within the budget, joined, and every chunk's score.
fn compress_result(c: &CompressIn, scores: &[Option<f32>], prompt_tokens: usize) -> Value {
    let (chunks, text) = (&c.chunks, &c.text);
    let method = if c.qlik.is_some() { "qlik" } else { "yesno" };
    let tokens: Vec<usize> = c.chunk_ids.iter().map(Vec::len).collect();
    let kept = select_within_budget(scores, &tokens, c.budget);
    let mut out = String::new();
    let mut last_end = None;
    for &i in &kept {
        if last_end.is_some_and(|e| e != chunks[i].start) {
            out.push_str("\n[...]\n");
        }
        out.push_str(&text[chunks[i].clone()]);
        last_end = Some(chunks[i].end);
    }
    let chunk_json: Vec<Value> = chunks
        .iter()
        .enumerate()
        .map(|(i, ch)| {
            // yesno: the score is the log-odds of "yes" against "no", p_relevant its logistic; qlik: the mean log-probability
            // of the question's tokens (no probability of relevance)
            let p = if c.qlik.is_none() { scores[i].map(|s| 1.0 / (1.0 + (-s).exp())) } else { None };
            json!({"start": ch.start, "end": ch.end, "tokens": tokens[i], "score": scores[i], "p_relevant": p, "kept": kept.contains(&i)})
        })
        .collect();
    json!({
        "text": out, "chunks": chunk_json, "method": method,
        "kept_tokens": kept.iter().map(|&i| tokens[i]).sum::<usize>(), "source_tokens": tokens.iter().sum::<usize>(),
        "budget_tokens": c.budget, "prompt_tokens": prompt_tokens, "generated_tokens": 0,
    })
}

// -- summarize --------------------------------------------------------------------------------------------------

fn is_eog(ctx: &Ctx, t: i32) -> bool {
    ctx.eog.binary_search(&t).is_ok()
}

/// The best token of a logits row that does not end the generation.
fn best_non_eog(ctx: &Ctx, l: &[f32]) -> i32 {
    let mut best = (f32::NEG_INFINITY, 0i32);
    for (t, &x) in l.iter().enumerate() {
        if x > best.0 && !is_eog(ctx, t as i32) {
            best = (x, t as i32);
        }
    }
    best.1
}

/// The first token after the prompt. ignore_eos (cost measurements): an end-of-generation pick is redone with host
/// logits, the EOG ids masked.
async fn first_token(ctx: &Ctx, seq: &Seq<'_>, prompt: &[i32], ignore_eos: bool) -> Res<i32> {
    let n_prompt = prompt.len();
    let tok = seq.prefill(prompt.to_vec(), OutputPolicy::Last, Sampler::Greedy).await?.token().ok_or_else(|| server_err("no token"))?;
    if !(ignore_eos && is_eog(ctx, tok)) {
        return Ok(tok);
    }
    seq.truncate(n_prompt - 1)?;
    Ok(best_non_eog(ctx, &seq.logits(vec![prompt[n_prompt - 1]]).await?))
}

/// The token after `tok` (with ignore_eos, an end-of-generation pick is redone as in [`first_token`]).
async fn next_token(ctx: &Ctx, seq: &Seq<'_>, tok: i32, ignore_eos: bool) -> Res<i32> {
    let next = seq.decode(tok, Sampler::Greedy).await?.token().ok_or_else(|| server_err("no token"))?;
    if !(ignore_eos && is_eog(ctx, next)) {
        return Ok(next);
    }
    let len = seq.len()?;
    seq.truncate(len - 1)?;
    Ok(best_non_eog(ctx, &seq.logits(vec![tok]).await?))
}

pub(super) async fn summarize(ctx: &Ctx, prompt: Vec<i32>, max_tokens: usize, ignore_eos: bool, prio: bool) -> Res<Value> {
    let n_prompt = prompt.len();
    let seq = Seq::new(ctx, n_prompt + max_tokens + 1, prio)?;
    let t0 = Instant::now();
    let mut tok = first_token(ctx, &seq, &prompt, ignore_eos).await?;
    let t_first = t0.elapsed();
    let mut out = Vec::new();
    let mut stop = "limit";
    loop {
        if is_eog(ctx, tok) {
            stop = "eos";
            break;
        }
        out.push(tok);
        if out.len() >= max_tokens {
            break;
        }
        tok = next_token(ctx, &seq, tok, ignore_eos).await?;
    }
    let decode_s = (t0.elapsed() - t_first).as_secs_f64();
    Ok(json!({
        "summary": text_of(&ctx.pieces, &out).trim(), "stop": stop,
        "prompt_tokens": n_prompt, "generated_tokens": out.len(),
        "prefill_ms": t_first.as_secs_f64() * 1e3,
        "decode_tokens_per_s": if decode_s > 0.0 { out.len().saturating_sub(1) as f64 / decode_s } else { 0.0 },
    }))
}
