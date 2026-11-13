use super::*;

fn lp(token: &str, p: f64, second: Option<f64>) -> ProbEntry {
    let mut top = vec![p.ln()];
    if let Some(s) = second {
        top.push(s.ln());
    }
    ProbEntry { token: token.into(), logprob: p.ln(), top_logprobs: top }
}

fn spelled(pieces: &[(&str, f64)]) -> (Vec<ProbEntry>, String) {
    (pieces.iter().map(|(t, p)| lp(t, *p, Some(p / 10.0))).collect(), pieces.iter().map(|(t, _)| *t).collect())
}

fn near(a: Option<f64>, b: f64) -> bool {
    a.is_some_and(|a| (a - b).abs() < 1e-9)
}

#[test]
fn the_prompts_and_caps_are_the_documented_ones() {
    assert_eq!(
        ONE,
        "Be extremely brief: one short line with only the essential step, then the final answer. If asked to write code, give only the complete code in one code block."
    );
    assert_eq!(TIGHT, "You are a helpful assistant. Be very brief: at most 3 short lines of key steps, then the final answer.");
    assert_eq!(BRIEF, "You are a helpful assistant. Be brief: show only the key steps in a few short lines, then the final answer.");
    assert_eq!(LONG, "You are a careful expert. Work step by step, check each step, then give the final answer.");
    assert_eq!(CHAT, "You are a helpful assistant.");
    assert_eq!((Stage::One.cap(), Stage::Tight.cap(), Stage::Brief.cap(), Stage::Long.cap(), Stage::Chat.cap()), (1024, 2048, 2048, 6144, 4096));
    assert_eq!((Stage::Chat.name(), Stage::Chat.prompt()), ("CHAT", "You are a helpful assistant."));
    assert!(P_SURE == 0.9 && TIE_MAX == 0.072 && TIE_GAP == 0.5 && N_PROBS == 2 && AGREE_WORDS == 6);
    let caps = |l: Level| l.stages().iter().map(|s| s.cap()).collect::<Vec<_>>();
    assert_eq!((caps(Level::Sonic), caps(Level::Fast)), (vec![1024], vec![4096]));
    assert_eq!((caps(Level::Normal), caps(Level::Deep)), (vec![1024, 2048, 6144], vec![1024, 2048, 6144]));
    assert_eq!((Level::Sonic.first_stage(), Level::Fast.first_stage(), Level::Normal.first_stage(), Level::Deep.first_stage()), (Stage::One, Stage::Chat, Stage::One, Stage::One));
    assert_eq!(Level::parse("deep"), Some(Level::Deep));
    assert_eq!((Level::parse("off"), Level::parse("superfast"), Level::parse("")), (None, None, None));
}

#[test]
fn answer_key_takes_the_last_box_after_the_thinking() {
    assert_eq!(answer_key("<think>maybe \\boxed{7}, no wait</think>\nThe total is \\boxed{18}.", &[]), "box:18");
    assert_eq!(answer_key("so \\boxed{1} and then \\boxed{ 2 }", &[]), "box:2");
    assert_eq!(answer_key("<think>x</think>\\boxed{\\frac{3}{4}}", &[]), "box:\\frac{3}{4}");
}

#[test]
fn answer_key_canonicalises_the_box() {
    let keys: std::collections::HashSet<String> = ["\\boxed{18}", "\\boxed{18.0}", "\\boxed{ 18 }", "\\boxed{\\$18}", "\\boxed{18.}"].iter().map(|x| answer_key(x, &[])).collect();
    assert_eq!(keys.len(), 1, "{keys:?}");
    assert_eq!(answer_key("\\boxed{1,200}", &[]), answer_key("\\boxed{1200}", &[]));
    assert_eq!(answer_key("\\boxed{\\text{Yes}}", &[]), answer_key("\\boxed{yes}", &[]));
    assert_ne!(answer_key("\\boxed{18}", &[]), answer_key("\\boxed{19}", &[]));
    assert_eq!(canon_box("\\dfrac{1}{2}\\,\\%"), "\\frac{1}{2}");
    assert_eq!(canon_box("007.50"), "7.5");
    assert_eq!(canon_box("100"), "100");
}

#[test]
fn answer_key_without_a_box_is_the_normalised_final_text() {
    let a = answer_key("<think>hmm</think>  The   answer is: PARIS!! ", &[]);
    let b = answer_key("the answer is paris", &[]);
    assert_eq!(a, "txt:the answer is paris");
    assert_eq!(a, b);
    assert_eq!(answer_key("x<|im_end|>junk", &["<|im_end|>"]), "txt:x");
    assert_ne!(answer_key("\\boxed{3}", &[]), answer_key("3", &[]));
}

#[test]
fn a_chat_answer_without_a_box_agrees_on_the_number_in_its_last_bold_part() {
    let a = "<think>x</think> The total is 205 minutes, so the train is moving for **184 minutes**.";
    let b = "<think>y</think> Subtracting **21 minutes** of stops, it moves for **184** minutes in all.";
    assert_eq!(answer_key(a, &[]), answer_key(b, &[]));
    assert_eq!(answer_key(a, &[]), answer_key("so \\boxed{184}", &[]));
    assert!(answer_key("<think>z</think> It is about trains.", &[]).starts_with("txt:"));
}

#[test]
fn split_thinking_follows_the_python() {
    assert_eq!(split_thinking("<think>a</think>\n  b"), "b");
    assert_eq!(split_thinking("a</think> b"), "b");
    assert_eq!(split_thinking("no thinking here"), "no thinking here");
    // harmony: analysis then the final channel
    assert_eq!(split_thinking("<|channel|>analysis<|message|>hmm<|end|><|start|>assistant<|channel|>final<|message|>Paris"), "Paris");
    // an end marker without a start (and not </think>) is just text
    assert_eq!(split_thinking("x assistant final y"), "x assistant final y");
}

#[test]
fn last_boxed_counts_nested_braces() {
    assert_eq!(last_boxed("\\boxed{\\frac{1}{2}}"), Some("\\frac{1}{2}"));
    assert_eq!(last_boxed("\\boxed{1} \\boxed {2}"), Some("2"));
    assert_eq!(last_boxed("\\boxed{1} \\boxed{2"), Some("1"));
    assert_eq!(last_boxed("no box"), None);
}

#[test]
fn boxed_pmin_is_the_least_probable_token_inside_the_last_box() {
    let (p, t) = spelled(&[("So ", 0.2), ("\\boxed{", 0.99), ("1", 0.95), ("8", 0.62), ("}", 0.3), (".", 0.5)]);
    assert!(near(boxed_pmin(Some(&p), &t), 0.62), "the tokens outside the box do not count");
    let (p, t) = spelled(&[("\\boxed{", 0.4), ("7", 0.98), ("}", 0.99), (" then \\boxed{", 0.4), ("42", 0.93), ("}", 0.2)]);
    assert!(near(boxed_pmin(Some(&p), &t), 0.93), "the LAST box");
    let (p, t) = spelled(&[("\\boxed{", 0.9), ("\\frac{", 0.8), ("1", 0.55), ("}{", 0.97), ("2", 0.99), ("}", 0.99), ("}", 0.1)]);
    assert!(near(boxed_pmin(Some(&p), &t), 0.55), "nested braces: the box ends at its own brace");
    let (p, t) = spelled(&[("The answer ", 0.5), ("is 4", 0.5)]);
    assert_eq!(boxed_pmin(Some(&p), &t), None);
    assert_eq!(boxed_pmin(None, "\\boxed{4}"), None);
    assert_eq!(boxed_pmin(Some(&[]), "\\boxed{4}"), None);
    let (p, _) = spelled(&[("a", 0.5)]);
    assert_eq!(boxed_pmin(Some(&p), "\\boxed{4}"), None, "the tokens do not spell the box");
}

#[test]
fn tie_share_counts_tokens_whose_best_two_are_close() {
    let mut probs = vec![lp("a", 0.5, Some(0.45)), lp("b", 0.9, Some(0.01)), lp("c", 0.4, Some(0.3)), lp("d", 0.99, Some(0.001))];
    assert!(near(tie_share(Some(&probs)), 0.5));
    probs.push(lp("e", 0.9, None));
    assert!(near(tie_share(Some(&probs)), 0.5), "one best token only: not counted");
    assert_eq!(tie_share(Some(&[lp("a", 0.9, None)])), None);
    assert_eq!((tie_share(Some(&[])), tie_share(None)), (None, None));
    let edge = |gap: f64| ProbEntry { token: String::new(), logprob: 0.0, top_logprobs: vec![0.0, -gap] };
    assert_eq!(tie_share(Some(&[edge(0.5)])), Some(0.0), "exactly half a log-probability apart is not a tie");
    assert_eq!(tie_share(Some(&[edge(0.5 - 1e-9)])), Some(1.0));
}

fn plain(n: usize, tie_every: usize) -> Vec<ProbEntry> {
    (0..n).map(|i| if tie_every > 0 && i % tie_every == 0 { lp("w ", 0.5, Some(0.45)) } else { lp("w ", 0.9, Some(0.01)) }).collect()
}

#[test]
fn needs_second_normal_with_a_box() {
    let (sure, text) = spelled(&[("so ", 0.5), ("\\boxed{", 0.9), ("18", 0.95), ("}", 0.9)]);
    let (again, why) = needs_second(&text, Some(&sure));
    assert!(!again && why.contains("sure") && why.contains("0.95"), "{why}");
    assert_eq!(why, "sure (least sure answer token 0.95 >= 0.9)");
    let (unsure, text) = spelled(&[("so ", 0.5), ("\\boxed{", 0.9), ("18", 0.62), ("}", 0.9)]);
    let (again, why) = needs_second(&text, Some(&unsure));
    assert!(again);
    assert_eq!(why, "not sure (least sure answer token 0.62 < 0.9): answering again step by step");
    // a box answer is judged by the box, not by the near ties around it
    let (noisy, text) = spelled(&[("so ", 0.5), ("\\boxed{", 0.9), ("18", 0.95), ("}", 0.9)]);
    let mut all = plain(50, 1);
    all.extend(noisy);
    assert!(!needs_second(&text, Some(&all)).0);
}

#[test]
fn needs_second_normal_without_a_box_uses_the_near_tie_share() {
    let (again, why) = needs_second(&"x".repeat(100), Some(&plain(100, 20)));
    assert!(!again && why.contains("5.0%"), "{why}");
    let (again, why) = needs_second(&"x".repeat(100), Some(&plain(100, 8)));
    assert!(again);
    assert_eq!(why, "not sure (13.0% near-tie words > 7.2%): answering again step by step");
    let mut just = plain(1000, 0);
    for p in just.iter_mut().take(72) {
        *p = lp("w ", 0.5, Some(0.45));
    }
    assert!(!needs_second("x", Some(&just)).0, "7.2% is not above 7.2%");
    just[72] = lp("w ", 0.5, Some(0.45));
    assert!(needs_second("x", Some(&just)).0);
    assert_eq!(needs_second("x", Some(&[lp("a", 0.9, None)])), (false, "no token had two best tokens: kept the first answer".into()));
}

#[test]
fn needs_second_without_probabilities_never_asks_again() {
    assert!(!Level::Sonic.gated() && !Level::Fast.gated() && !Level::Normal.gated() && !Level::Deep.gated());
    assert_eq!(needs_second("t", None), (false, "no token probabilities on this engine".into()));
    assert_eq!(needs_second("t", Some(&[])), (false, "no token probabilities on this engine".into()));
}

#[test]
fn answers_agree_compares_the_answer_keys() {
    assert_eq!(answers_agree("\\boxed{18}", "6 * 3 = 18, so \\boxed{18.0}"), (true, "the two short answers agree (18.0)".to_string()));
    assert!(answers_agree("it is **18 km**", "a\nb\n\\boxed{18}").0, "a bold number is a key too");
    assert_eq!(answers_agree("\\boxed{18}", "\\boxed{19}"), (false, "the two short answers do not agree: answering again step by step".to_string()));
    assert!(answers_agree("Paris.", "paris").0);
    // no equal keys: the last number of each result
    assert!(answers_agree("240 / 4 = 60 km/h", "Total 240 km in 4 h.\n**60 km/h**").0);
    assert!(!answers_agree("240 / 4 = 60 km/h", "Total 240 km.\n**64 km/h**").0);
    assert!(answers_agree("so 1,200 dollars", "a\n**$1200.00**").0);
    assert!(!answers_agree("it is 14", "four").0);
    // worded results: a short one that is part of the other, word for word
    assert!(answers_agree("Canberra", "Sydney is the largest city.\nThe capital is **Canberra**.").0);
    assert!(answers_agree("The capital of Australia is Canberra.", "**Canberra**").0);
    assert!(!answers_agree("Sydney", "The capital is **Canberra**.").0);
    assert!(!answers_agree("no", "**not known**").0);
    assert!(!answers_agree("one two three four five six seven", "x\none two three four five six seven eight").0);
    assert_eq!(answer_core("a\n\nlast line\n"), "last line");
    assert_eq!(answer_core(""), "");
    assert_eq!(answer_core("x\nFinal Answer: Canberra"), "Canberra");
    assert_eq!(answer_core("The answer is 4"), "4");
    assert!(answers_agree("The capital of Australia is Canberra.", "The capital is Canberra.\nIt was a compromise.\nFinal Answer: Canberra").0);
    assert!(!answers_agree("", "").0);
    assert!(!answers_agree("  ", " ").0);
}

#[test]
fn stage_system_keeps_the_callers_instructions_unless_it_is_the_default() {
    let default = "You are a helpful assistant. Answer clearly and briefly.";
    assert_eq!(stage_system(TIGHT, default, default), TIGHT);
    assert_eq!(stage_system(TIGHT, "  ", default), TIGHT);
    assert_eq!(stage_system(TIGHT, "", ""), TIGHT);
    assert_eq!(stage_system(LONG, "You manage agents. Reply in JSON.", default), format!("{LONG}\n\nYou manage agents. Reply in JSON."));
}

#[test]
fn the_text_helpers_follow_the_python_regular_expressions() {
    assert_eq!(bold_parts("a **b** c **d e** **"), vec!["b", "d e"]);
    assert_eq!(bold_parts("**a\nb**"), Vec::<&str>::new());
    assert_eq!(numbers("x -12,5.5 y 3. 7,"), vec!["-12,5.5", "3", "7,"]);
    assert_eq!(normalize_text("  The   answer is: PARIS!! "), "the answer is paris");
    assert_eq!(strip_answer_label("The final answer is: 42"), "42");
    assert_eq!(strip_answer_label("Answer - x"), "x");
    assert_eq!(strip_answer_label("theanswer 3"), "theanswer 3");
    assert_eq!(splitlines("a\r\nb\rc\n"), vec!["a", "b", "c"]);
}
