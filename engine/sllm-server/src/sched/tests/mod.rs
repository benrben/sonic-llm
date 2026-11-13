use super::*;

fn r(req: ReqId, lane: Lane, left: usize) -> Running {
    Running { req, lane, prefill_left: left, order: req, chunk_cap: usize::MAX }
}

fn work_of(p: &Plan, req: ReqId) -> Option<Work> {
    p.items.iter().find(|w| w.req == req).map(|w| w.work)
}

#[test]
fn decode_rows_go_first_interactive_then_helper_then_background() {
    let mut pl = Planner::new(SchedConfig::default());
    let p = pl.plan(&[r(5, Lane::Background, 0), r(1, Lane::Interactive, 0), r(3, Lane::Helper, 0), r(2, Lane::Background, 0)]);
    let order: Vec<ReqId> = p.items.iter().map(|w| w.req).collect();
    assert_eq!(order, vec![1, 3, 2, 5]);
    assert!(p.items.iter().all(|w| w.work == Work::Decode));
    assert_eq!((p.decode_rows, p.prefill_tokens), (4, 0));
}

#[test]
fn prefill_is_chunked_to_the_step_budget_one_chunk_per_request() {
    let cfg = SchedConfig { chunk: 256, step_tokens: 600, prefill_fair: false, ..Default::default() };
    let mut pl = Planner::new(cfg);
    let p = pl.plan(&[r(1, Lane::Background, 1000), r(2, Lane::Background, 100), r(3, Lane::Background, 900)]);
    assert_eq!(work_of(&p, 1), Some(Work::Prefill { n: 256 }));
    assert_eq!(work_of(&p, 2), Some(Work::Prefill { n: 100 }));
    assert_eq!(work_of(&p, 3), Some(Work::Prefill { n: 244 }), "the rest of the budget");
    assert_eq!(p.prefill_tokens, 600);
}

#[test]
fn fair_prefill_splits_the_budget_so_prompts_arriving_together_finish_together() {
    // water-filling: the short prompt takes what it needs, the other two share the rest evenly
    let cfg = SchedConfig { chunk: 256, step_tokens: 600, fair_min: 32, fair_granule: false, ..Default::default() };
    let mut pl = Planner::new(cfg);
    let p = pl.plan(&[r(1, Lane::Background, 1000), r(2, Lane::Background, 100), r(3, Lane::Background, 900)]);
    assert_eq!(work_of(&p, 1), Some(Work::Prefill { n: 250 }));
    assert_eq!(work_of(&p, 2), Some(Work::Prefill { n: 100 }));
    assert_eq!(work_of(&p, 3), Some(Work::Prefill { n: 250 }));
    assert_eq!(p.prefill_tokens, 600);
    // 32 agents of 512 prompt tokens each, a 2048-token budget and a 32-token minimum: 64 tokens each per step, so
    // all 32 finish their prefill in the same (8th) step
    let mut pl = Planner::new(SchedConfig { step_tokens: 2048, fair_min: 32, fair_granule: false, ..Default::default() });
    let mut left = vec![512usize; 32];
    let mut finished_at = vec![0usize; 32];
    for step in 1..=8 {
        let running: Vec<Running> = (0..32).map(|i| r(i as ReqId, Lane::Background, left[i])).collect();
        let p = pl.plan(&running);
        assert_eq!(p.prefill_tokens, 2048);
        for w in &p.items {
            if let Work::Prefill { n } = w.work {
                assert_eq!(n, 64);
                left[w.req as usize] -= n;
                if left[w.req as usize] == 0 {
                    finished_at[w.req as usize] = step;
                }
            }
        }
    }
    assert!(finished_at.iter().all(|&s| s == 8), "{finished_at:?}");
    // the remainder of the division goes to the earliest requests
    let p = Planner::new(SchedConfig { step_tokens: 100, fair_min: 32, fair_granule: false, ..Default::default() }).plan(&[r(1, Lane::Background, 512), r(2, Lane::Background, 512), r(3, Lane::Background, 512)]);
    assert_eq!((work_of(&p, 1), work_of(&p, 2), work_of(&p, 3)), (Some(Work::Prefill { n: 34 }), Some(Work::Prefill { n: 33 }), Some(Work::Prefill { n: 33 })));
    // a budget that cannot give everyone MIN_FAIR_CHUNK: the earliest requests share it (a wave), the rest wait
    let running: Vec<Running> = (0..100).map(|i| r(i, Lane::Background, 512)).collect();
    let p = Planner::new(SchedConfig::default()).plan(&running);
    let wave = (DEFAULT_STEP_TOKENS / MIN_FAIR_CHUNK) as u64;
    assert_eq!(p.prefill_tokens, DEFAULT_STEP_TOKENS);
    assert_eq!(p.items.len() as u64, wave);
    assert!(p.items.iter().all(|w| w.req < wave && w.work == Work::Prefill { n: MIN_FAIR_CHUNK }));
    // agents-32 (31 prompts of 512 own tokens): waves of `wave` prompts, two steps each, no share under 256 tokens
    // unless it finishes the prompt (fair_shares; granular shares are tested below)
    let mut pl = Planner::new(SchedConfig { fair_granule: false, ..Default::default() });
    let mut left = vec![512usize; 31];
    let mut finished_at = vec![0usize; 31];
    for step in 1..=8 {
        let running: Vec<Running> = (0..31).map(|i| r(i as ReqId, Lane::Background, left[i])).collect();
        for w in &pl.plan(&running).items {
            if let Work::Prefill { n } = w.work {
                assert!(n >= MIN_FAIR_CHUNK || n == left[w.req as usize], "{n}");
                left[w.req as usize] -= n;
                if left[w.req as usize] == 0 {
                    finished_at[w.req as usize] = step;
                }
            }
        }
    }
    assert!(finished_at.iter().enumerate().all(|(i, &s)| s == 2 * (i / wave as usize + 1)), "{finished_at:?}");
    // a budget that covers every request: no split
    let p = Planner::new(SchedConfig::default()).plan(&[r(1, Lane::Background, 512), r(2, Lane::Background, 300)]);
    assert_eq!((work_of(&p, 1), work_of(&p, 2)), (Some(Work::Prefill { n: 512 }), Some(Work::Prefill { n: 300 })));
}

#[test]
fn background_decode_rows_sit_out_a_step_of_many_prefill_chunks() {
    // the defaults give at most DEFAULT_STEP_TOKENS / MIN_FAIR_CHUNK = 8 chunks, so the guard needs a smaller minimum
    assert!(DEFAULT_STEP_TOKENS / MIN_FAIR_CHUNK <= MAX_MIXED_CHUNKS);
    let mut pl = Planner::new(SchedConfig { fair_min: 32, burst_yield: 0, ..Default::default() });
    let mut running: Vec<Running> = (0..32).map(|i| r(i, Lane::Background, 512)).collect();
    running.push(r(100, Lane::Background, 0));
    running.push(r(101, Lane::Helper, 0));
    let p = pl.plan(&running);
    assert_eq!(work_of(&p, 101), Some(Work::Decode), "the helper row stays");
    assert_eq!(work_of(&p, 100), None, "the background row sits out a 32-chunk step");
    assert_eq!(p.decode_rows, 1);
    // few chunks: every row rides along
    let running = [r(1, Lane::Background, 512), r(2, Lane::Background, 512), r(100, Lane::Background, 0)];
    assert_eq!(work_of(&pl.plan(&running), 100), Some(Work::Decode));
}

#[test]
fn interactive_prefill_goes_before_background_prefill() {
    let mut pl = Planner::new(SchedConfig { step_tokens: 512, ..Default::default() });
    let p = pl.plan(&[r(1, Lane::Background, 512), r(2, Lane::Interactive, 512)]);
    assert_eq!(work_of(&p, 2), Some(Work::Prefill { n: 512 }));
    assert_eq!(work_of(&p, 1), None, "the budget is spent on the interactive prompt");
    assert!(!p.bg_deferred, "no interactive request is decoding yet: the budget, not the guard, held it back");
}

#[test]
fn background_prefill_is_deferred_while_the_user_waits_then_let_through_by_period() {
    let cfg = SchedConfig { bg_prefill_period: 4, ..Default::default() };
    let mut pl = Planner::new(cfg);
    let running = [r(1, Lane::Interactive, 0), r(2, Lane::Background, 512), r(3, Lane::Background, 0)];
    for _ in 0..3 {
        let p = pl.plan(&running);
        assert_eq!(work_of(&p, 1), Some(Work::Decode));
        assert_eq!(work_of(&p, 3), Some(Work::Decode), "background decode rows still ride along");
        assert_eq!(work_of(&p, 2), None);
        assert!(p.bg_deferred && p.interactive);
    }
    let running = [running[0], running[1], running[2], r(4, Lane::Background, 512)];
    let p = pl.plan(&running);
    assert_eq!(p.prefill_tokens, 512, "the 4th deferred step lets one chunk through, not the whole 2048-token budget");
    assert_eq!((work_of(&p, 2), work_of(&p, 4)), (Some(Work::Prefill { n: 512 }), None), "in arrival order");
    assert!(!p.bg_deferred);
    assert_eq!(work_of(&pl.plan(&running), 2), None, "and the count starts again");
    // with no interactive request decoding, background prefill runs at once
    let mut pl = Planner::new(SchedConfig::default());
    assert_eq!(work_of(&pl.plan(&[r(2, Lane::Background, 512)]), 2), Some(Work::Prefill { n: 512 }));
}

#[test]
fn a_small_background_prefill_share_while_interactive() {
    let cfg = SchedConfig { bg_prefill_interactive: 64, bg_prefill_period: 0, ..Default::default() };
    let mut pl = Planner::new(cfg);
    for _ in 0..100 {
        let p = pl.plan(&[r(1, Lane::Interactive, 0), r(2, Lane::Background, 512)]);
        assert_eq!(work_of(&p, 2), Some(Work::Prefill { n: 64 }), "period 0 never forces a full chunk");
    }
}

#[test]
fn helper_keeps_its_reserve_against_interactive_prefill() {
    let cfg = SchedConfig { step_tokens: 512, chunk: 512, helper_tokens: 128, ..Default::default() };
    let mut pl = Planner::new(cfg);
    let p = pl.plan(&[r(1, Lane::Interactive, 2000), r(2, Lane::Helper, 300), r(3, Lane::Background, 512)]);
    assert_eq!(work_of(&p, 1), Some(Work::Prefill { n: 384 }));
    assert_eq!(work_of(&p, 2), Some(Work::Prefill { n: 128 }));
    assert_eq!(work_of(&p, 3), None);
    // without a helper the interactive prompt takes the whole budget
    let p = pl.plan(&[r(1, Lane::Interactive, 2000)]);
    assert_eq!(work_of(&p, 1), Some(Work::Prefill { n: 512 }));
    // a helper is not held back by background prefill, and is not deferred by the interactive guard
    let p = pl.plan(&[r(4, Lane::Interactive, 0), r(2, Lane::Helper, 300), r(3, Lane::Background, 512)]);
    assert_eq!(work_of(&p, 2), Some(Work::Prefill { n: 300 }));
    assert_eq!(work_of(&p, 3), None);
}

#[test]
fn background_rows_can_be_capped_round_robin_while_interactive() {
    let cfg = SchedConfig { bg_rows_interactive: Some(2), ..Default::default() };
    let mut pl = Planner::new(cfg);
    let running: Vec<Running> =
        std::iter::once(r(0, Lane::Interactive, 0)).chain((1..=5).map(|i| r(i, Lane::Background, 0))).collect();
    let mut served = [0usize; 6];
    for _ in 0..10 {
        let p = pl.plan(&running);
        assert_eq!(p.items[0].req, 0);
        assert_eq!(p.decode_rows, 3);
        for w in &p.items {
            served[w.req as usize] += 1;
        }
    }
    assert_eq!(served[0], 10);
    assert!(served[1..].iter().all(|&n| n == 4), "every background agent gets its share: {served:?}");
    // no interactive decode: no cap
    assert_eq!(pl.plan(&running[1..]).decode_rows, 5);
}

#[test]
fn default_cap_leaves_small_groups_alone() {
    let mut pl = Planner::new(SchedConfig::default());
    let n = DEFAULT_BG_ROWS_INTERACTIVE as ReqId;
    let front = std::iter::once(r(0, Lane::Interactive, 0));
    let small: Vec<Running> = front.clone().chain((1..=n).map(|i| r(i, Lane::Background, 0))).collect();
    assert_eq!(pl.plan(&small).decode_rows, 1 + DEFAULT_BG_ROWS_INTERACTIVE);
    let big: Vec<Running> = front.chain((1..=n + 8).map(|i| r(i, Lane::Background, 0))).collect();
    assert_eq!(pl.plan(&big).decode_rows, 1 + DEFAULT_BG_ROWS_INTERACTIVE);
    assert_eq!(pl.plan(&big[1..]).decode_rows, big.len() - 1, "no interactive decode: no cap");
}

#[test]
fn a_burst_of_background_prompts_yields_the_interactive_guard_but_the_front_row_stays() {
    // the front agent decodes, 24 background prompts prefill: with burst yield on (batch mode) the burst gets the whole
    // step budget, the front keeps its row, background rows sit out; with burst_yield 0 (the default) the guard defers
    // background prefill
    let mut running = vec![r(1, Lane::Interactive, 0)];
    running.extend((2..26).map(|i| r(i, Lane::Background, 512)));
    running.extend((26..30).map(|i| r(i, Lane::Background, 0)));
    let on = SchedConfig { burst_yield: BURST_YIELD_MIN, ..Default::default() };
    let p = Planner::new(on.clone()).plan(&running);
    assert_eq!(work_of(&p, 1), Some(Work::Decode));
    assert_eq!(p.decode_rows, 1);
    assert_eq!(p.prefill_tokens, DEFAULT_STEP_TOKENS);
    assert!(p.burst_skip && !p.bg_deferred);
    let off = Planner::new(SchedConfig { burst_yield: 0, ..Default::default() }).plan(&running);
    assert_eq!(off.prefill_tokens, 0);
    assert!(off.bg_deferred);
    // the tail of the burst (fewer than burst_yield left) still goes: no wait for the forced chunk
    let mut pl = Planner::new(on.clone());
    pl.plan(&running);
    let tail: Vec<Running> = running.iter().copied().filter(|x| x.req < 5 || x.req >= 26).collect();
    assert_eq!(pl.plan(&tail).prefill_tokens, 3 * 512);
    // fewer than burst_yield prefilling to begin with: the guard holds
    let few: Vec<Running> = running.iter().copied().filter(|x| x.req < 8 || x.req >= 26).collect();
    assert_eq!(Planner::new(on).plan(&few).prefill_tokens, 0);
}

#[test]
fn a_writer_share_outside_zero_to_one_is_refused() {
    for bad in [1.0, 2.0, -1.0, f64::NAN] {
        let cfg = SchedConfig { bg_write_share: bad, ..SchedConfig::default() };
        assert!(cfg.validate().is_err(), "share {bad} must be refused");
    }
    let cfg = SchedConfig { bg_gap_ms: -5.0, ..SchedConfig::default() };
    assert!(cfg.validate().is_err());
    for good in [0.0, 0.25, 0.5, 0.99] {
        let cfg = SchedConfig { bg_write_share: good, ..SchedConfig::default() };
        assert!(cfg.validate().is_ok(), "share {good} is valid");
    }
}

#[test]
fn background_writers_keep_their_share_beside_background_readers() {
    // one writer, one reader with a long prompt; a read token costs 1 ms, a decode-only step 20 ms
    let mut left = 6000usize;
    let mut pl = Planner::new(SchedConfig::default());
    // the writer is writing before the reader arrives
    let alone = pl.plan(&[r(1, Lane::Background, 0)]);
    pl.observe(20.0, &alone);
    let (mut read_ms, mut write_ms, mut longest, mut first) = (0.0, 0.0, 0.0f64, None);
    while left > 0 {
        let p = pl.plan(&[r(1, Lane::Background, 0), r(2, Lane::Background, left)]);
        assert_eq!(p.decode_rows, 1, "the writer writes in every step");
        let ms = if p.prefill_tokens > 0 { p.prefill_tokens as f64 } else { 20.0 };
        first.get_or_insert(p.prefill_tokens);
        if first != Some(p.prefill_tokens) || left < 6000 - 512 {
            longest = longest.max(ms);
        }
        if p.prefill_tokens > 0 { read_ms += ms } else { write_ms += ms }
        left -= p.bg_prefill;
        pl.observe(ms, &p);
    }
    assert_eq!(first, Some(512), "nothing measured yet: a whole chunk");
    assert!(longest <= 256.0, "a read step beside a writer fits the gap, got {longest} ms");
    assert!(write_ms >= 0.8 * read_ms, "the writer keeps about half of the time: {write_ms} of {read_ms} ms");
    // off: every step reads a whole chunk
    let mut off = Planner::new(SchedConfig { bg_write_share: 0.0, ..SchedConfig::default() });
    for _ in 0..4 {
        let p = off.plan(&[r(1, Lane::Background, 0), r(2, Lane::Background, 6000)]);
        assert_eq!(p.prefill_tokens, 512);
        off.observe(512.0, &p);
    }
}

#[test]
fn readers_of_the_writers_own_burst_keep_whole_read_steps_and_all_of_the_time() {
    // three agents sent together: agent 1 has finished reading, 2 and 3 still read; a read token costs 1 ms
    let mut pl = Planner::new(SchedConfig::default());
    let (mut read_ms, mut write_ms, mut left) = (0.0, 0.0, 6000usize);
    while left > 0 {
        let p = pl.plan(&[r(1, Lane::Background, 0), r(2, Lane::Background, left), r(3, Lane::Background, left)]);
        assert_eq!(p.decode_rows, 1, "the writer writes in every step");
        if p.prefill_tokens > 0 {
            assert!(p.prefill_tokens >= 1024.min(2 * left), "a whole read step, got {}", p.prefill_tokens);
        }
        let ms = if p.prefill_tokens > 0 { p.prefill_tokens as f64 } else { 20.0 };
        if p.prefill_tokens > 0 { read_ms += ms } else { write_ms += ms }
        left -= p.bg_prefill / 2;
        pl.observe(ms, &p);
    }
    assert_eq!(write_ms, 0.0, "no step without reading: the writer has its token of every read step ({read_ms} ms read)");
    // a reader that arrives while they write is beside the writers: the gap holds again
    let p = pl.plan(&[r(1, Lane::Background, 0), r(2, Lane::Background, 0), r(9, Lane::Background, 6000)]);
    assert!(p.prefill_tokens <= 256, "a late reader's step fits the gap, got {}", p.prefill_tokens);
}

#[test]
fn readers_alone_and_readers_beside_a_chat_are_not_slowed_for_background_writers() {
    let mut pl = Planner::new(SchedConfig::default());
    for _ in 0..4 {
        let p = pl.plan(&[r(1, Lane::Background, 6000), r(2, Lane::Background, 6000)]);
        assert_eq!(p.prefill_tokens, 1024, "no writer: the whole chunks");
        pl.observe(1024.0, &p);
    }
    // a writer that left owes nothing
    let p = pl.plan(&[r(1, Lane::Background, 0), r(2, Lane::Background, 6000)]);
    pl.observe(250.0, &p);
    let p = pl.plan(&[r(2, Lane::Background, 6000)]);
    assert_eq!(p.prefill_tokens, 512);
}

#[test]
fn slow_background_rows_leave_the_chat_a_beat_and_quick_ones_do_not() {
    let running = [r(1, Lane::Interactive, 0), r(2, Lane::Background, 0), r(3, Lane::Background, 0)];
    let rows = |pl: &mut Planner, ms: f64, n: usize| -> Vec<usize> {
        (0..n)
            .map(|_| {
                let p = pl.plan(&running);
                pl.observe(if p.decode_rows > p.front_rows { ms } else { 13.0 }, &p);
                p.decode_rows
            })
            .collect()
    };
    // quick steps: everybody in every step
    let mut quick = Planner::new(SchedConfig::default());
    assert_eq!(rows(&mut quick, 30.0, 8), vec![3; 8]);
    // 100 ms with the agents, 13 alone: (100 - 13) / (40 - 13) rounds up to 4
    let mut slow = Planner::new(SchedConfig::default());
    rows(&mut slow, 100.0, 12);
    let beat = rows(&mut slow, 100.0, 12);
    assert_eq!(beat.iter().filter(|&&n| n == 3).count(), 3, "{beat:?}");
    // off
    let mut off = Planner::new(SchedConfig { chat_gap_ms: 0.0, ..Default::default() });
    assert_eq!(rows(&mut off, 100.0, 8), vec![3; 8]);
    // no chat: nothing sits out
    let bg = [r(2, Lane::Background, 0), r(3, Lane::Background, 0)];
    assert_eq!(slow.plan(&bg).decode_rows, 2);
}

#[test]
fn a_burst_prefills_before_its_background_rows_ride_along() {
    let cfg = SchedConfig { burst_min: 4, burst_yield: 0, ..Default::default() };
    let mut pl = Planner::new(cfg);
    // 8 background agents decoding, 24 still prefilling: the rows sit out the prefill step; interactive and helper
    // rows stay
    let mut running: Vec<Running> = (0..8).map(|i| r(i, Lane::Background, 0)).collect();
    running.extend((8..32).map(|i| r(i, Lane::Background, 512)));
    running.push(r(100, Lane::Helper, 0));
    let p = pl.plan(&running);
    assert!(p.burst_skip);
    assert_eq!(p.decode_rows, 1);
    assert_eq!(work_of(&p, 100), Some(Work::Decode));
    assert_eq!(work_of(&p, 0), None);
    assert_eq!(p.prefill_tokens, DEFAULT_STEP_TOKENS, "the prefill itself is unchanged");
    // the chunks are the same as without the flag (fair shares of at least MIN_FAIR_CHUNK)
    let q = Planner::new(SchedConfig { burst_yield: 0, ..Default::default() }).plan(&running);
    let chunks = |p: &Plan| p.items.iter().filter(|w| matches!(w.work, Work::Prefill { .. })).copied().collect::<Vec<_>>();
    assert_eq!(chunks(&p), chunks(&q));
    assert_eq!(q.decode_rows, 9, "off: every row rides along");
    // fewer than burst_min prefilling: the rows ride along
    let few: Vec<Running> = (0..8).map(|i| r(i, Lane::Background, 0)).chain((8..11).map(|i| r(i, Lane::Background, 512))).collect();
    let p = pl.plan(&few);
    assert!(!p.burst_skip);
    assert_eq!(p.decode_rows, 8);
    // an interactive request decoding: background prefill is deferred, so the background rows ride along with it
    let mut with_front = running.clone();
    with_front.push(r(200, Lane::Interactive, 0));
    let p = pl.plan(&with_front);
    assert_eq!(p.prefill_tokens, 0);
    assert_eq!(p.decode_rows, 10);
    // no starvation: after BURST_MAX_SKIP sit-outs in a row the rows ride along once
    let mut pl = Planner::new(SchedConfig { burst_min: 4, burst_yield: 0, ..Default::default() });
    for _ in 0..BURST_MAX_SKIP {
        assert!(pl.plan(&running).burst_skip);
    }
    let p = pl.plan(&running);
    assert!(!p.burst_skip);
    assert_eq!(p.decode_rows, 9);
    assert!(pl.plan(&running).burst_skip, "and the count starts again");
}

#[test]
fn the_forced_background_chunk_can_be_capped() {
    let cfg = SchedConfig { bg_prefill_period: 2, bg_prefill_forced: Some(256), ..Default::default() };
    let mut pl = Planner::new(cfg);
    let running = [r(1, Lane::Interactive, 0), r(2, Lane::Background, 512)];
    assert_eq!(work_of(&pl.plan(&running), 2), None);
    assert_eq!(work_of(&pl.plan(&running), 2), Some(Work::Prefill { n: 256 }), "the forced chunk, capped");
    // without an interactive request decoding the cap does not apply
    let p = Planner::new(SchedConfig { bg_prefill_forced: Some(256), ..Default::default() }).plan(&running[1..]);
    assert_eq!(work_of(&p, 2), Some(Work::Prefill { n: 512 }));
    assert!(SchedConfig { bg_prefill_forced: Some(0), ..Default::default() }.validate().is_err());
}

#[test]
fn front_prefill_solo_defers_background_prefill_while_the_front_prefills() {
    let running = [r(1, Lane::Interactive, 512), r(2, Lane::Background, 512), r(3, Lane::Background, 0)];
    // off: the background takes what the interactive prompt leaves
    let p = Planner::new(SchedConfig { front_prefill_solo: false, ..Default::default() }).plan(&running);
    assert_eq!((work_of(&p, 1), work_of(&p, 2)), (Some(Work::Prefill { n: 512 }), Some(Work::Prefill { n: 512 })));
    // on: the front's prompt step carries no background chunk; background rows still ride along
    let mut pl = Planner::new(SchedConfig { front_prefill_solo: true, bg_prefill_period: 3, ..Default::default() });
    let p = pl.plan(&running);
    assert_eq!((work_of(&p, 1), work_of(&p, 2), work_of(&p, 3)), (Some(Work::Prefill { n: 512 }), None, Some(Work::Decode)));
    assert!(p.bg_deferred && !p.interactive);
    assert_eq!(work_of(&pl.plan(&running), 2), None);
    assert_eq!(work_of(&pl.plan(&running), 2), Some(Work::Prefill { n: 512 }), "the period still lets a chunk through");
}

#[test]
fn granular_fair_shares_are_whole_units() {
    let w = |v: &[usize]| v.iter().enumerate().map(|(i, &n)| (i as ReqId, n)).collect::<Vec<_>>();
    // 7 prompts of 512 left, 2,048 tokens: 6 x 256 and one 512 (not 7 x 292), then 6 x 256 (not 7 x 220)
    assert_eq!(granular_shares(&w(&[512; 7]), 2048, 256), vec![512, 256, 256, 256, 256, 256, 256]);
    assert_eq!(granular_shares(&w(&[0, 256, 256, 256, 256, 256, 256]), 2048, 256), vec![0, 256, 256, 256, 256, 256, 256]);
    // a short end of a prompt is taken whole; the budget then goes on in units, the rest under a unit to the earliest
    assert_eq!(granular_shares(&w(&[100, 512, 512, 512, 512]), 1024, 256), vec![100, 412, 256, 256, 0]);
    // more requests than units: a wave of the earliest, as fair_shares
    assert_eq!(granular_shares(&w(&[512; 10]), 2048, 256), vec![256, 256, 256, 256, 256, 256, 256, 256, 0, 0]);
    // everything fits: no split
    assert_eq!(granular_shares(&w(&[512, 300]), 2048, 256), vec![512, 300]);
    // a budget under one unit, or a rest under one unit: to the earliest
    assert_eq!(granular_shares(&w(&[512, 512]), 100, 256), vec![100, 0]);
    assert_eq!(granular_shares(&w(&[512; 7]), 1920, 256), vec![384, 256, 256, 256, 256, 256, 256]);
    assert_eq!(granular_shares(&w(&[512]), 384, 256), vec![384]);
    // in the planner: every share a multiple of 256 or the end of its prompt
    let mut pl = Planner::new(SchedConfig { fair_granule: true, ..Default::default() });
    let mut left = vec![512usize; 31];
    while left.iter().any(|&n| n > 0) {
        let running: Vec<Running> = (0..31).filter(|&i| left[i] > 0).map(|i| r(i as ReqId, Lane::Background, left[i])).collect();
        for w in &pl.plan(&running).items {
            if let Work::Prefill { n } = w.work {
                assert!(n % MIN_FAIR_CHUNK == 0 || n == left[w.req as usize], "{n}");
                left[w.req as usize] -= n;
            }
        }
    }
}

#[test]
fn config_validation() {
    assert!(SchedConfig::default().validate().is_ok());
    assert!(SchedConfig { caps: [MAX_INTERACTIVE + 1, 8, 1], ..Default::default() }.validate().is_err());
    assert!(SchedConfig { helper_tokens: DEFAULT_STEP_TOKENS + 1, ..Default::default() }.validate().is_err());
    assert_eq!(SchedConfig::default().max_rows(), LANE_ROWS + 1); // 261, as with the old interactive cap of 4
}
