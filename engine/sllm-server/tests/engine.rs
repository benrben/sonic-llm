//! The FakeEngine on its own (no HTTP): determinism, n_predict, ignore_eos, agent context equivalence, cancellation.

use sllm_server::engine::*;
use sllm_server::fake::{FakeConfig, FakeEngine};
use std::collections::HashMap;

/// A minimal driver: every admitted request prefills its whole feed in one item, then decodes (what the scheduler does
/// with an unlimited budget).
struct Driver {
    left: HashMap<ReqId, usize>,
}

impl Driver {
    fn new() -> Driver {
        Driver { left: HashMap::new() }
    }
    fn start(&mut self, e: &mut FakeEngine, req: StartRequest) -> Result<Started, EngineError> {
        let id = req.id;
        let s = e.start_request(req)?;
        assert!(e.try_begin(id));
        self.left.insert(id, s.feed_n);
        Ok(s)
    }
    fn cancel(&mut self, e: &mut FakeEngine, id: ReqId) {
        e.cancel(id);
        self.left.remove(&id);
    }
    fn has_work(&self) -> bool {
        !self.left.is_empty()
    }
    fn step(&mut self, e: &mut FakeEngine) -> Vec<EngineEvent> {
        let mut ids: Vec<ReqId> = self.left.keys().copied().collect();
        ids.sort();
        let items: Vec<WorkItem> = ids
            .iter()
            .map(|&req| WorkItem { req, work: if self.left[&req] > 0 { Work::Prefill { n: self.left[&req] } } else { Work::Decode } })
            .collect();
        for v in self.left.values_mut() {
            *v = 0;
        }
        let ev = e.step(&items);
        for x in &ev {
            if let EngineEvent::Finished { req, .. } = x {
                self.left.remove(req);
            }
        }
        ev
    }
}

fn start(e: &mut FakeEngine, id: ReqId, agent: Option<&str>, ids: &[i32], p: SamplingParams) -> Result<Started, EngineError> {
    e.start_request(StartRequest { id, agent: agent.map(String::from), ids: ids.to_vec(), params: p, lane: Lane::Background })
}

fn cfg() -> FakeConfig {
    FakeConfig { prefill_ms: 0.0, prefill_ms_per_token: 0.0, token_ms: 0.0, ..Default::default() }
}

fn params(n: i64, ignore_eos: bool) -> SamplingParams {
    SamplingParams { n_predict: n, temperature: 0.0, ignore_eos, cache_prompt: true, grammar: None, latent_steps: 0, n_probs: 0 }
}

/// Run one request alone to completion: (tokens, reason).
fn run(e: &mut FakeEngine, id: ReqId, agent: Option<&str>, ids: &[i32], p: SamplingParams) -> (Vec<i32>, StopReason) {
    let mut d = Driver::new();
    d.start(e, StartRequest { id, agent: agent.map(String::from), ids: ids.to_vec(), params: p, lane: Lane::Background }).unwrap();
    let (mut toks, mut why) = (vec![], None);
    while d.has_work() {
        for ev in d.step(e) {
            match ev {
                EngineEvent::Token { req, id: t, .. } if req == id => toks.push(t),
                EngineEvent::Finished { req, reason } if req == id => why = Some(reason),
                _ => {}
            }
        }
    }
    (toks, why.expect("finished"))
}

#[test]
fn same_prompt_same_tokens_across_engine_instances() {
    let p: Vec<i32> = (0..50).collect();
    let a = run(&mut FakeEngine::new(cfg()), 1, None, &p, params(32, true));
    let b = run(&mut FakeEngine::new(cfg()), 9, None, &p, params(32, true));
    assert_eq!(a, b);
    assert_eq!(a.0.len(), 32);
    assert_eq!(a.1, StopReason::Limit);
}

#[test]
fn tokens_are_golden_for_the_default_seed() {
    // Pin the generator: a change here changes every fixture and recorded fake run.
    let (t, _) = run(&mut FakeEngine::new(cfg()), 1, None, &[1, 2, 3], params(3, true));
    assert_eq!(t, vec![22347, 38311, 9508]);
}

#[test]
fn eos_only_by_natural_length_and_never_as_a_token() {
    let mut e = FakeEngine::new(FakeConfig { natural_len: Some(9), ..cfg() });
    let (t, why) = run(&mut e, 1, None, &[5, 6], params(100, false));
    assert_eq!((t.len(), why), (9, StopReason::Eos));
    let (t2, why2) = run(&mut e, 2, None, &[5, 6], params(30, true));
    assert_eq!((t2.len(), why2), (30, StopReason::Limit));
    assert!(!t2.contains(&e.info().eos_id));
}

#[test]
fn context_overflow_is_refused_and_context_full_stops() {
    let mut e = FakeEngine::new(FakeConfig { n_ctx: 16, ..cfg() });
    let err = start(&mut e, 1, None, &[1; 16], params(1, true)).unwrap_err();
    assert_eq!((err.status, err.kind), (400, "exceed_context_size_error"));
    let (t, why) = run(&mut e, 2, None, &[1; 10], params(-1, true));
    assert_eq!((t.len(), why), (6, StopReason::Context));
}

#[test]
fn agent_context_equals_the_concatenated_prompt() {
    let mut e = FakeEngine::new(cfg());
    e.agent_create("a", Lane::Interactive, &[7, 8, 9]).unwrap();
    assert_eq!(e.agent_context_tokens("a"), Some(3));
    let (g1, _) = run(&mut e, 1, Some("a"), &[1, 2], params(5, true));
    assert_eq!(e.agent_context_tokens("a"), Some(10));
    let (g2, _) = run(&mut e, 2, Some("a"), &[3], params(5, true));
    let mut plain = FakeEngine::new(cfg());
    let mut all = vec![7, 8, 9, 1, 2];
    assert_eq!(run(&mut plain, 1, None, &all, params(5, true)).0, g1);
    all.extend(&g1);
    all.push(3);
    assert_eq!(run(&mut plain, 2, None, &all, params(5, true)).0, g2);
    e.agent_free("a");
    assert_eq!(e.agent_context_tokens("a"), None);
    let err = start(&mut e, 3, Some("a"), &[1], params(1, true)).unwrap_err();
    assert_eq!(err.status, 404);
}

#[test]
fn cancel_removes_the_request_and_batching_does_not_change_tokens() {
    let mut e = FakeEngine::new(cfg());
    let solo = run(&mut e, 1, None, &[1, 2, 3], params(6, true)).0;
    let mut d = Driver::new();
    for id in [10, 11, 12] {
        d.start(&mut e, StartRequest { id, agent: None, ids: vec![1, 2, 3], params: params(6, true), lane: Lane::Background }).unwrap();
    }
    d.cancel(&mut e, 11);
    let mut got: std::collections::HashMap<ReqId, Vec<i32>> = Default::default();
    while d.has_work() {
        for ev in d.step(&mut e) {
            if let EngineEvent::Token { req, id, .. } = ev {
                got.entry(req).or_default().push(id);
            }
        }
    }
    assert_eq!(got[&10], solo);
    assert_eq!(got[&12], solo);
    assert!(!got.contains_key(&11));
}

#[test]
fn latency_is_configurable() {
    let mut e = FakeEngine::new(FakeConfig { prefill_ms: 30.0, prefill_ms_per_token: 1.0, token_ms: 10.0, ..cfg() });
    let t = std::time::Instant::now();
    run(&mut e, 1, None, &[1; 20], params(4, true));
    let ms = t.elapsed().as_secs_f64() * 1e3;
    // 30 + 20 prefill, 4 steps of 10 ms, plus the step that ends the request; never less than 90
    assert!(ms >= 89.0 && ms < 400.0, "{ms}");
}

/// Chunked prefill gives the same tokens as one chunk, and the first token comes with the chunk that ends the prompt.
#[test]
fn chunked_prefill_equals_one_chunk_and_samples_at_the_end() {
    let p: Vec<i32> = (0..40).collect();
    let solo = run(&mut FakeEngine::new(cfg()), 1, None, &p, params(4, true)).0;
    let mut e = FakeEngine::new(cfg());
    let s = start(&mut e, 2, None, &p, params(4, true)).unwrap();
    assert_eq!((s.prompt_n, s.feed_n), (40, 40));
    assert!(e.try_begin(2));
    let mut toks = vec![];
    for n in [16, 16] {
        let ev = e.step(&[WorkItem { req: 2, work: Work::Prefill { n } }]);
        assert!(ev.is_empty(), "no token before the prompt is in: {ev:?}");
    }
    for ev in e.step(&[WorkItem { req: 2, work: Work::Prefill { n: 8 } }]) {
        if let EngineEvent::Token { id, .. } = ev {
            toks.push(id);
        }
    }
    while toks.len() < 4 {
        for ev in e.step(&[WorkItem { req: 2, work: Work::Decode }]) {
            if let EngineEvent::Token { id, .. } = ev {
                toks.push(id);
            }
        }
    }
    assert_eq!(toks, solo);
}

/// A turn cancelled before its first token leaves the agent as it was; the fake pool reserves worst cases.
#[test]
fn turn_cancelled_before_its_first_token_leaves_the_agent_unchanged_and_the_pool_reserves() {
    let mut e = FakeEngine::new(FakeConfig { pool_pages: Some(4), ..cfg() });
    e.agent_create("a", Lane::Background, &[1; 10]).unwrap();
    start(&mut e, 1, Some("a"), &[2; 100], params(10, true)).unwrap();
    assert!(e.try_begin(1));
    e.step(&[WorkItem { req: 1, work: Work::Prefill { n: 50 } }]);
    e.cancel(1);
    assert_eq!(e.agent_context_tokens("a"), Some(10));
    // 4 pages of 256: a 600-token completion (3 pages worst case) fits, a second one does not while the first is admitted
    start(&mut e, 2, None, &[3; 500], params(100, true)).unwrap();
    let err = start(&mut e, 3, None, &[3; 500], params(100, true)).unwrap_err();
    assert_eq!((err.status, err.kind), (503, "kv_pool_exhausted_error"));
    e.cancel(2);
    start(&mut e, 3, None, &[3; 500], params(100, true)).unwrap();
}
