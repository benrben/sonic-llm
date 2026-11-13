use super::*;

fn done(prompt_n: usize, cached_n: usize, shared_n: usize) -> Done {
    Done { stop: StopReason::Eos, prompt_n, predicted_n: 7, cached_n, prompt_ms: 0.0, predicted_ms: 0.0, queued_ms: 0.0, context_tokens: None, shared_n, wait_ms: 0.0, wake: None, rollovers: 0, rollover_ms: 0.0 }
}

fn meta(cache_control: bool) -> Meta {
    Meta { id: "msg_1".into(), req_id: 1, model: "m".into(), input_tokens: 100, cache_control }
}

#[test]
fn cache_reads_come_from_cached_and_shared_tokens_and_leave_input_tokens_only_for_cache_control_requests() {
    let d = done(100, 30, 50);
    // no cache_control: input_tokens is the whole prompt, as count_tokens counts it
    assert_eq!(usage_json(&meta(false), &d), json!({"input_tokens": 100, "output_tokens": 7, "cache_read_input_tokens": 80, "cache_creation_input_tokens": 0}));
    // cache_control (Claude Code): the three counts are disjoint and add up to the prompt
    assert_eq!(usage_json(&meta(true), &d), json!({"input_tokens": 20, "output_tokens": 7, "cache_read_input_tokens": 80, "cache_creation_input_tokens": 0}));
    assert_eq!(usage_json(&meta(true), &done(100, 0, 0))["input_tokens"], 100);
}

#[test]
fn stop_reasons() {
    let mut d = done(1, 0, 0);
    assert_eq!(stop_reason(&d, false, 0), "end_turn");
    assert_eq!(stop_reason(&d, false, 2), "tool_use");
    assert_eq!(stop_reason(&d, true, 0), "stop_sequence");
    d.stop = StopReason::Limit;
    assert_eq!(stop_reason(&d, false, 0), "max_tokens");
    assert_eq!(stop_reason(&d, true, 0), "stop_sequence", "a stop sequence that matched ended it, not the limit");
}

#[test]
fn errors_map_to_anthropics_types() {
    let e = |status, kind| wire_error(&EngineError::new(status, kind, "m"));
    assert_eq!(e(422, "invalid_request_error").0, 400);
    assert_eq!(e(501, "not_supported_error").1, "invalid_request_error");
    assert_eq!((e(503, "unavailable_error").0, e(503, "unavailable_error").1), (529, "overloaded_error"));
    assert_eq!(e(500, "server_error").1, "api_error");
    assert_eq!(e(404, "not_found_error").1, "not_found_error");
    let ctx = EngineError::new(400, "exceed_context_size_error", "m").with_context(5000, 4096);
    assert_eq!(wire_error(&ctx).2, "prompt is too long: 5000 tokens > 4096 maximum");
}

#[test]
fn rfc3339_dates() {
    assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
    assert_eq!(rfc3339(951_782_400 + 3661), "2000-02-29T01:01:01Z");
    assert_eq!(rfc3339(1_793_318_400), "2026-10-30T00:00:00Z");
}
