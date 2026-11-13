//! GBNF constrained output (`grammar` on /completion, /v1/completions and agent turns) on a real model (libllama). Metal:
//! `#[ignore]`d; run under the GPU lock, one test at a time:
//!
//!     engine/scripts/run_quiet.sh <owner> bash <script that runs>
//!         cargo test --release -p sllm-server --test grammar_real -- --ignored --test-threads=1 --nocapture
//!
//! Skips (passes) when the model file is absent.

mod common;
use common::*;
use serde_json::{json, Value};
use sllm_server::engine::Engine;
use sllm_server::llama_engine::{EngineConfig, KvType, LlamaEngine, RowExact};
use sllm_server::sched::SchedConfig;
use sllm_server::spec::SpecMode;
use sllm_server::tokenizer::Tokenizer;
use sllm_server::{start, ServerConfig};
use std::path::PathBuf;
use std::time::Duration;

const MODEL: &str = "/Users/benreich/Projects/crazyidea/research/inference_runtime/converted/qwen3-0.6b-Q8_0.gguf";

/// A JSON object of a fixed shape with bounded strings, so a constrained run always ends.
const PERSON: &str = r#"root ::= "{" ws "\"name\"" ws ":" ws str ws "," ws "\"age\"" ws ":" ws num ws "," ws "\"tags\"" ws ":" ws "[" ws str (ws "," ws str){0,3} ws "]" ws "}"
str ::= "\"" [a-zA-Z0-9 ]{1,16} "\""
num ::= [1-9] [0-9]{0,2}
ws ::= [ \n]{0,2}"#;

/// Never ends by itself: "1, 1, 1, ..." (every n-gram repeats, which is what prompt lookup drafts from).
const ENDLESS: &str = r#"root ::= "[" ("1, ")+"#;

#[derive(Clone, Copy)]
struct Opts {
    unified: bool,
    q4_read_f16: bool,
    spec: SpecMode,
    /// pool pages of 256 cells (small: requests pause for memory)
    pages: usize,
}

const DEFAULT: Opts = Opts { unified: false, q4_read_f16: false, spec: SpecMode::Off, pages: 64 };

fn spawn_real(o: Opts) -> Option<Srv> {
    if !PathBuf::from(MODEL).is_file() {
        eprintln!("SKIP: {MODEL} not found");
        return None;
    }
    let _ = sllm_core::backend::registry();
    let mut cfg = EngineConfig { n_ctx: 4096, pool_pages: Some(o.pages), ..EngineConfig::new(PathBuf::from(MODEL)) };
    if o.q4_read_f16 {
        cfg.kv_type = KvType::Q4_0;
        cfg.kv_read_f16 = true;
    }
    cfg.spec.mode = o.spec;
    cfg.kv_unified = o.unified;
    cfg.row_exact = RowExact::On; // Q8_0 is not the exact kernel's fast type, so auto would leave it off
    let sched = SchedConfig::default();
    cfg.max_rows = sched.max_rows();
    cfg.step_tokens = sched.step_tokens;
    let tok = Tokenizer::gguf(cfg.model.clone());
    let vocab = tok.vocab().unwrap();
    let factory: sllm_server::executor::EngineFactory = Box::new(move || LlamaEngine::new(cfg, vocab).map(|e| Box::new(e) as Box<dyn Engine>));
    let run = start(ServerConfig { port: 0, sched, ..Default::default() }, tok, factory).expect("start");
    run.wait_ready(Duration::from_secs(120)).expect("engine ready");
    Some(Srv { run, http: reqwest::Client::builder().no_proxy().build().unwrap() })
}

/// One request; (status, body).
async fn gen(srv: &Srv, prompt: &str, n: i64, ignore_eos: bool, grammar: Option<&str>) -> (u16, Value) {
    let mut b = json!({"prompt": prompt, "n_predict": n, "temperature": 0.0, "ignore_eos": ignore_eos, "return_tokens": true, "stream": false});
    if let Some(g) = grammar {
        b["grammar"] = json!(g);
    }
    let r = srv.post("/completion", &b).await;
    let st = r.status().as_u16();
    (st, r.json().await.unwrap())
}

async fn ok(srv: &Srv, prompt: &str, n: i64, ignore_eos: bool, grammar: Option<&str>) -> Value {
    let (st, v) = gen(srv, prompt, n, ignore_eos, grammar).await;
    assert_eq!(st, 200, "{v}");
    v
}

fn toks(v: &Value) -> Vec<i64> {
    v["tokens"].as_array().unwrap().iter().map(|t| t.as_i64().unwrap()).collect()
}

fn text(v: &Value) -> &str {
    v["content"].as_str().unwrap()
}

/// Strict parse: the whole text is one JSON value (serde_json rejects trailing text and truncated input).
fn parses(s: &str) -> bool {
    serde_json::from_str::<Value>(s).is_ok()
}

const PROMPTS: [&str; 5] = [
    "Describe a person as a JSON object with a name, an age and tags:\n",
    "Here is a record for a customer in JSON format:\n",
    "Write the profile of a software engineer.\n",
    "The following is a JSON document about a cat named Tom:\n",
    "Name: Alice. Age: 41. Hobbies: chess, climbing. As JSON:\n",
];

/// `value` of the first key named `key` anywhere in `v` (the metrics tree).
fn find(v: &Value, key: &str) -> Option<Value> {
    match v {
        Value::Object(o) => o.get(key).cloned().or_else(|| o.values().find_map(|x| find(x, key))),
        Value::Array(a) => a.iter().find_map(|x| find(x, key)),
        _ => None,
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn json_grammar_output_parses_and_the_control_does_not() {
    let Some(srv) = spawn_real(DEFAULT) else { return };
    let (mut constrained_ok, mut control_ok) = (0, 0);
    for p in PROMPTS {
        let g = ok(&srv, p, 160, false, Some(PERSON)).await;
        let c = ok(&srv, p, 160, false, None).await;
        eprintln!("PROMPT {p:?}\n  grammar: {:?} ({})\n  control: {:?}", text(&g), g["stop_type"], text(&c));
        assert!(parses(text(&g)), "grammar output is not JSON: {:?}", text(&g));
        assert_eq!(g["stop_type"], "eos", "the grammar ended the request: {:?}", text(&g));
        assert!(serde_json::from_str::<Value>(text(&g)).unwrap()["tags"].is_array());
        constrained_ok += 1;
        control_ok += parses(text(&c)) as usize;
    }
    eprintln!("RESULT json_grammar: constrained {constrained_ok}/5 parse, control {control_ok}/5 parse");
    assert_eq!(constrained_ok, 5);
    assert!(control_ok < 5, "negative control: the unconstrained output parsed on every prompt, the test proves nothing");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn a_grammar_job_is_the_same_alone_and_beside_other_jobs() {
    let Some(srv) = spawn_real(DEFAULT) else { return };
    let gp: Vec<&str> = PROMPTS[..4].to_vec();
    let np = ["The quick brown fox", "Once upon a time, in a land far away,", "def fibonacci(n):\n", "The three laws of thermodynamics are"];
    let mut solo_g = Vec::new();
    for p in &gp {
        solo_g.push(toks(&ok(&srv, p, 120, false, Some(PERSON)).await));
    }
    let mut solo_n = Vec::new();
    for p in &np[..3] {
        solo_n.push(toks(&ok(&srv, p, 60, true, None).await));
    }
    assert!(solo_g.iter().all(|t| t.len() > 10));
    // grammar job 0 beside 3 non-grammar jobs (which outlive it)
    let before = srv.metrics().await;
    let mut hs = vec![];
    for (i, p) in std::iter::once(&gp[0]).chain(np[..3].iter()).enumerate() {
        let c = srv.http.clone();
        let url = srv.url("/completion");
        let mut b = json!({"prompt": p, "n_predict": if i == 0 { 120 } else { 60 }, "ignore_eos": i != 0, "return_tokens": true, "stream": false});
        if i == 0 {
            b["grammar"] = json!(PERSON);
        }
        hs.push(tokio::spawn(async move { c.post(url).json(&b).send().await.unwrap().json::<Value>().await.unwrap() }));
    }
    let mut out = vec![];
    for h in hs {
        out.push(toks(&h.await.unwrap()));
    }
    assert_eq!(out[0], solo_g[0], "the grammar job beside 3 non-grammar jobs");
    for i in 0..3 {
        assert_eq!(out[1 + i], solo_n[i], "non-grammar job {i} beside a grammar job");
    }
    // the four grammar jobs together
    let mut hs = vec![];
    for p in &gp {
        let c = srv.http.clone();
        let url = srv.url("/completion");
        let b = json!({"prompt": p, "n_predict": 120, "ignore_eos": false, "return_tokens": true, "stream": false, "grammar": PERSON});
        hs.push(tokio::spawn(async move { c.post(url).json(&b).send().await.unwrap().json::<Value>().await.unwrap() }));
    }
    for (i, h) in hs.into_iter().enumerate() {
        assert_eq!(toks(&h.await.unwrap()), solo_g[i], "grammar job {i} beside 3 other grammar jobs");
    }
    let after = srv.metrics().await;
    // the jobs really were in flight together
    let inflight = find(&after, "max_inflight_jobs").and_then(|v| v.as_u64()).unwrap();
    eprintln!("RESULT solo_vs_batch: ok; most jobs in flight {inflight} (before the batches: {:?})", find(&before, "max_inflight_jobs"));
    assert!(inflight >= 4, "the jobs did not overlap");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn yes_or_no_ends_cleanly_with_and_without_ignore_eos() {
    let Some(srv) = spawn_real(DEFAULT) else { return };
    let g = r#"root ::= "yes" | "no""#;
    for p in ["Is the sky blue? Answer yes or no: ", "Is fire cold? Answer yes or no: "] {
        for ignore_eos in [false, true] {
            let v = ok(&srv, p, 20, ignore_eos, Some(g)).await;
            eprintln!("{p:?} ignore_eos={ignore_eos}: {:?} tokens {:?} stop {}", text(&v), toks(&v), v["stop_type"]);
            assert!(text(&v) == "yes" || text(&v) == "no", "{:?}", text(&v));
            assert_eq!(v["stop_type"], "eos", "ignore_eos={ignore_eos}");
            // it is the tokenization of the word: one token for "yes"/"no" in this vocabulary
            let t: Value = srv.post("/tokenize", &json!({"content": text(&v)})).await.json().await.unwrap();
            assert_eq!(toks(&v), t["tokens"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect::<Vec<_>>());
            assert_eq!(toks(&v).len(), 1);
        }
    }
    // a later request on the same server is unaffected (nothing leaked)
    let free = ok(&srv, "The quick brown fox", 8, true, None).await;
    assert_eq!(toks(&free).len(), 8);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn invalid_grammar_is_a_400_and_an_empty_one_is_no_grammar() {
    let Some(srv) = spawn_real(DEFAULT) else { return };
    let free = ok(&srv, "The quick brown fox", 12, true, None).await;
    for bad in ["this is not a grammar", "root ::= foo", "start ::= \"a\"", "root ::= root \"a\"", "root ::= \"a", "root ::= [a-"] {
        let (st, v) = gen(&srv, "The quick brown fox", 12, true, Some(bad)).await;
        assert_eq!(st, 400, "{bad:?}: {v}");
        assert_eq!(v["error"]["code"], 400);
        assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
        assert!(v["error"]["message"].as_str().unwrap().contains("grammar"), "{v}");
    }
    // a grammar of only the empty string is valid: the first token already finds it complete (the first-token dead end)
    for ignore_eos in [false, true] {
        let v = ok(&srv, "The quick brown fox", 12, ignore_eos, Some("root ::=")).await;
        assert_eq!((text(&v), toks(&v).len(), v["stop_type"].as_str().unwrap()), ("", 0, "eos"), "ignore_eos={ignore_eos}");
    }
    // the same on the other routes
    let r = srv.post("/v1/completions", &json!({"prompt": "hi", "max_tokens": 3, "grammar": "nope"})).await;
    assert_eq!(r.status(), 400);
    let a = srv.post("/v1/agents", &json!({})).await.json::<Value>().await.unwrap();
    let id = a["id"].as_str().unwrap();
    let r = srv.post(&format!("/v1/agents/{id}/turn"), &json!({"prompt": "hi", "n_predict": 3, "grammar": "nope", "stream": false})).await;
    assert_eq!(r.status(), 400);
    // the agent is untouched: its first turn still works and starts from an empty context
    let r = srv.post(&format!("/v1/agents/{id}/turn"), &json!({"prompt": "hi", "n_predict": 3, "stream": false})).await;
    assert_eq!(r.status(), 200);
    // empty string and null: no grammar
    let (st, e) = gen(&srv, "The quick brown fox", 12, true, Some("")).await;
    assert_eq!(st, 200);
    assert_eq!(toks(&e), toks(&free));
    let mut b = json!({"prompt": "The quick brown fox", "n_predict": 12, "ignore_eos": true, "return_tokens": true, "grammar": null});
    b["stream"] = json!(false);
    let n: Value = srv.post("/completion", &b).await.json().await.unwrap();
    assert_eq!(toks(&n), toks(&free));
    // json_schema and the rest stay refused
    let (st, _) = gen(&srv, "x", 4, true, None).await;
    assert_eq!(st, 200);
    let r = srv.post("/completion", &json!({"prompt": "x", "n_predict": 4, "json_schema": {"type": "object"}})).await;
    assert_eq!(r.status(), 400);
    // the server still serves grammar requests after the failures
    let v = ok(&srv, "Is the sky blue? ", 8, false, Some(r#"root ::= "yes" | "no""#)).await;
    assert!(text(&v) == "yes" || text(&v) == "no");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn speculation_drafts_nothing_for_a_grammar_job_and_the_output_is_the_same() {
    let prompt = "1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,";
    let proposed = |m: &Value| m["spec"]["proposed"].as_u64().unwrap();
    let Some(srv) = spawn_real(Opts { spec: SpecMode::Auto, ..DEFAULT }) else { return };
    let caps = srv.get_json("/caps").await;
    assert_eq!(caps["engine"]["spec"]["mode"], "auto");
    assert!(caps["engine"]["spec"]["unavailable"].is_null(), "{}", caps["engine"]["spec"]);
    // negative control: without a grammar this prompt drafts
    let p0 = proposed(&srv.metrics().await);
    let free = ok(&srv, prompt, 48, true, None).await;
    let p1 = proposed(&srv.metrics().await);
    assert!(p1 > p0, "the control does not draft ({p0} -> {p1}): the test proves nothing");
    // a grammar job with the same prompt proposes nothing
    let g = ok(&srv, prompt, 48, true, Some(ENDLESS)).await;
    let p2 = proposed(&srv.metrics().await);
    assert_eq!(p2, p1, "a grammar job proposed drafts");
    assert_eq!(toks(&g).len(), 48);
    // ...and neither does the step it shares with other jobs change theirs: a non-grammar job beside it still drafts alone
    // (one decoding sequence) and keeps its tokens
    let free2 = ok(&srv, prompt, 48, true, None).await;
    assert_eq!(toks(&free2), toks(&free));
    drop(srv);
    // spec off: the grammar job's tokens are the same
    let Some(off) = spawn_real(DEFAULT) else { return };
    let g_off = ok(&off, prompt, 48, true, Some(ENDLESS)).await;
    let free_off = ok(&off, prompt, 48, true, None).await;
    assert_eq!(toks(&g), toks(&g_off), "grammar job, --spec auto vs off");
    assert_eq!(toks(&free), toks(&free_off), "control, --spec auto vs off");
    eprintln!("RESULT spec: control proposed {} drafts, grammar job 0; tokens equal to --spec off", p1 - p0);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn first_token_obeys_the_grammar_with_a_q4_cache_read_in_f16() {
    let Some(srv) = spawn_real(Opts { q4_read_f16: true, ..DEFAULT }) else { return };
    let caps = srv.get_json("/caps").await;
    assert_eq!(caps["engine"]["kv_read"]["mode"], "f16", "{}", caps["engine"]["kv_read"]);
    for p in ["The capital of France is", "2 + 2 =", "Say hello to the world:\n"] {
        let free = ok(&srv, p, 8, true, None).await;
        let first = text(&free).trim_start().chars().next().unwrap();
        assert_ne!(first, '{', "the unconstrained first token already obeys: {:?}", text(&free));
        let g = ok(&srv, p, 160, false, Some(PERSON)).await;
        eprintln!("{p:?}: free {:?} -> grammar {:?}", text(&free), text(&g));
        assert!(text(&g).starts_with('{'), "{:?}", text(&g));
        assert!(parses(text(&g)), "{:?}", text(&g));
    }
    let reads = find(&srv.metrics().await, "reads").and_then(|v| v.as_u64()).unwrap_or(0);
    eprintln!("RESULT kv_read f16: reader reads {reads}");
    assert!(reads > 0, "the prompts were not read in the f16 reader");
    // a one-token grammar through the reader
    let v = ok(&srv, "Is the sky blue? Answer yes or no: ", 8, true, Some(r#"root ::= "yes" | "no""#)).await;
    assert!(text(&v) == "yes" || text(&v) == "no", "{:?}", text(&v));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn an_agent_turn_takes_the_grammar_for_that_turn_only() {
    let Some(srv) = spawn_real(DEFAULT) else { return };
    let a = srv.post("/v1/agents", &json!({})).await.json::<Value>().await.unwrap();
    let id = a["id"].as_str().unwrap().to_string();
    let turn = |body: Value| {
        let (srv, id) = (&srv, id.clone());
        async move {
            let r = srv.post(&format!("/v1/agents/{id}/turn"), &body).await;
            assert_eq!(r.status(), 200);
            r.json::<Value>().await.unwrap()
        }
    };
    let t1 = turn(json!({"prompt": "Is the sky blue? Answer yes or no: ", "n_predict": 20, "grammar": r#"root ::= "yes" | "no""#, "ignore_eos": true, "stream": false, "return_tokens": true})).await;
    assert!(text(&t1) == "yes" || text(&t1) == "no", "{:?}", text(&t1));
    // the next turn has no grammar: it is not held to yes/no
    let t2 = turn(json!({"prompt": " Why? Explain in a sentence: ", "n_predict": 24, "ignore_eos": true, "stream": false, "return_tokens": true})).await;
    assert_eq!(toks(&t2).len(), 24);
    assert!(text(&t2).len() > 6, "{:?}", text(&t2));
    // another grammar on the third turn
    let t3 = turn(json!({"prompt": " Now give a JSON profile of the answerer: ", "n_predict": 160, "grammar": PERSON, "stream": false, "return_tokens": true})).await;
    assert!(parses(text(&t3)), "{:?}", text(&t3));
    // a turn cancelled mid-way leaves the agent usable
    let mut r = srv.post(&format!("/v1/agents/{id}/turn"), &json!({"prompt": " Count: ", "n_predict": 100000, "grammar": ENDLESS, "ignore_eos": true})).await;
    let _ = r.chunk().await;
    drop(r);
    tokio::time::sleep(Duration::from_millis(500)).await;
    let t4 = turn(json!({"prompt": " Again yes or no: ", "n_predict": 20, "grammar": r#"root ::= "yes" | "no""#, "ignore_eos": true, "stream": false})).await;
    assert!(text(&t4) == "yes" || text(&t4) == "no", "{:?}", text(&t4));
}

/// Requests paused for memory keep their grammar state: the tokens are those of the same request alone. (The cache is one
/// unified pool here, the only layout that pauses; background jobs fill it and an interactive grammar job takes cells from them.)
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Metal and the GPU lock"]
async fn paused_and_cancelled_grammar_jobs_keep_their_state() {
    let Some(srv) = spawn_real(Opts { pages: 4, unified: true, ..DEFAULT }) else { return };
    let cells = srv.get_json("/caps").await["engine"]["kv"]["total_cells"].as_u64().unwrap() as usize;
    let n = 1000;
    let prompts: Vec<String> = (0..4).map(|i| format!("Item {i}: {}", "1, ".repeat(10 + i))).collect();
    let per = 60 + n;
    let n_bg = (cells * 85 / 100 / per).clamp(1, 3);
    eprintln!("unified pool of {cells} cells, {n_bg} background jobs of about {per} cells, then an interactive one");
    let mut solo = Vec::new();
    for p in &prompts {
        solo.push(toks(&ok(&srv, p, n as i64, true, Some(ENDLESS)).await));
    }
    let send = |p: String, lane: &'static str| {
        let (c, url) = (srv.http.clone(), srv.url("/completion"));
        let b = json!({"prompt": p, "n_predict": n, "ignore_eos": true, "return_tokens": true, "stream": false, "grammar": ENDLESS, "lane": lane});
        tokio::spawn(async move { c.post(url).json(&b).send().await.unwrap().json::<Value>().await.unwrap() })
    };
    let mut hs: Vec<_> = (0..n_bg).map(|i| send(prompts[i].clone(), "background")).collect();
    // a background job that is cancelled mid-way
    let mut r = srv.post("/completion", &json!({"prompt": "Cancel me: ", "n_predict": 100000, "grammar": ENDLESS, "ignore_eos": true, "stream": true, "lane": "background"})).await;
    let _ = r.chunk().await;
    drop(r);
    tokio::time::sleep(Duration::from_millis(400)).await;
    let mut jobs: Vec<usize> = (0..n_bg).collect();
    for k in 0..(cells / per) {
        // interactive jobs claim cells: the youngest background job pauses and resumes later
        let i = n_bg + k % (4 - n_bg).max(1);
        hs.push(send(prompts[i.min(3)].clone(), "interactive"));
        jobs.push(i.min(3));
    }
    for (h, i) in hs.into_iter().zip(jobs) {
        let v = h.await.unwrap();
        assert!(v.get("error").is_none(), "{v}");
        assert_eq!(toks(&v), solo[i], "grammar job {i} under memory pressure");
    }
    let m = srv.metrics().await;
    let pauses = find(&m, "preempt_pauses").and_then(|v| v.as_u64()).unwrap_or(0);
    eprintln!("RESULT pause: preempt_pauses {pauses} resumed {:?}", find(&m, "resumed"));
    assert!(pauses > 0, "nothing was paused: raise the load");
    // the server still serves a grammar request
    let v = ok(&srv, "Is the sky blue? ", 8, false, Some(r#"root ::= "yes" | "no""#)).await;
    assert!(text(&v) == "yes" || text(&v) == "no");
}
