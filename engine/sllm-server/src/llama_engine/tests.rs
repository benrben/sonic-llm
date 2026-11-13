
/// Applies a plan to the running ids (what `pack_rows` does to the rows).
fn apply_plan(rows: &[i32], plan: &[(i32, i32)]) -> Vec<i32> {
    let mut r = rows.to_vec();
    for &(from, to) in plan {
        let i = r.iter().position(|&x| x == from).expect("from is running");
        r[i] = to;
    }
    r.sort_unstable();
    r
}

#[test]
fn plan_pack_closes_holes_under_the_running_ids() {
    // 24 of 48 agents left, every other id running: the free ids are the odd ones
    let rows: Vec<i32> = (0..48).step_by(2).collect();
    let free: Vec<i32> = (0..48).skip(1).step_by(2).collect();
    let mut cur = rows.clone();
    let mut free = free;
    let mut steps = 0;
    while id_runs(&cur) > 1 {
        let plan = plan_pack(&free, &cur, MAX_PACK_MOVES);
        assert!(!plan.is_empty() && plan.len() <= MAX_PACK_MOVES, "progress, bounded");
        for &(from, to) in &plan {
            free.retain(|&f| f != to);
            free.push(from);
        }
        let before = id_runs(&cur);
        cur = apply_plan(&cur, &plan);
        assert!(id_runs(&cur) < before, "every bounded batch lowers the runs");
        steps += 1;
        assert!(steps < 20);
    }
    assert_eq!(cur, (0..24).collect::<Vec<_>>());
}

#[test]
fn plan_pack_leaves_packed_and_single_run_steps_alone() {
    assert!(plan_pack(&[0, 1, 2], &[3, 4, 5], 4).is_empty(), "one run: nothing to do");
    assert!(plan_pack(&[5, 6], &[0, 1, 2], 4).is_empty(), "free ids above the running ones");
    assert!(plan_pack(&[1, 2], &[0], 4).is_empty());
    assert!(plan_pack(&[], &[0, 2], 4).is_empty());
    assert!(plan_pack(&[1], &[0, 2], 0).is_empty());
}

#[test]
fn plan_pack_moves_the_highest_into_the_lowest_free_id() {
    assert_eq!(plan_pack(&[3, 1], &[0, 2, 7], 4), vec![(7, 1)]);
    assert_eq!(plan_pack(&[1, 3, 4], &[0, 2, 5, 9], 4), vec![(9, 1), (5, 3)]);
    // the cap
    assert_eq!(plan_pack(&[1, 3, 4], &[0, 2, 5, 9], 1), vec![(9, 1)]);
}

#[test]
fn plan_pack_ignores_gaps_held_by_others() {
    // ids 1..=3 are idle agents or holders (not free): moving 9 into 4 does not make the rows consecutive
    assert!(plan_pack(&[4, 5, 6], &[0, 9], 4).is_empty());
    // a partly closable gap: 9 -> 3 gives {0,1,3}... no better than {0,1,9}? runs 2 -> 2: left alone
    assert!(plan_pack(&[3], &[0, 1, 9], 4).is_empty());
    // 4 fills the hole between 3 and 5
    assert_eq!(plan_pack(&[4], &[3, 5, 9], 4), vec![(9, 4)]);
    assert_eq!(apply_plan(&[3, 5, 9], &[(9, 4)]), vec![3, 4, 5]);
}
use super::*;

fn naive(row: &[f32]) -> i32 {
    let (mut best, mut best_v) = (0usize, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as i32
}

#[test]
fn fast_argmax_matches_the_plain_loop_including_ties_and_nan() {
    let mut x: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for len in [1usize, 2, 15, 16, 17, 31, 33, 1000, 4099] {
        for case in 0..40 {
            let mut row: Vec<f32> = (0..len).map(|_| (next() % 64) as f32 - 32.0).collect(); // many ties
            match case % 5 {
                1 => row.iter_mut().for_each(|v| *v = f32::NEG_INFINITY),
                2 => row[(next() as usize) % len] = f32::NAN,
                3 => row.iter_mut().for_each(|v| *v = f32::NAN),
                4 => row[(next() as usize) % len] = 0.0f32 * -1.0,
                _ => {}
            }
            assert_eq!(argmax(&row), naive(&row), "len {len} case {case}");
        }
    }
    // the threaded path gives the same tokens in the same order
    let rows: Vec<Vec<f32>> = (0..40).map(|_| (0..1000).map(|_| (next() % 100) as f32).collect()).collect();
    let picks: Vec<(ReqId, bool, &[f32])> = rows.iter().enumerate().map(|(i, r)| (i as ReqId, false, r.as_slice())).collect();
    let mut out = Vec::new();
    sample_rows(&picks, &[], &mut out);
    let want: Vec<(ReqId, i32)> = rows.iter().enumerate().map(|(i, r)| (i as ReqId, naive(r))).collect();
    assert_eq!(out, want);
}

#[test]
fn the_copy_queue_takes_the_most_rows_and_one_unknown_copy_makes_it_whole() {
    use CopyQueue::*;
    assert_eq!(Idle.hint(), 0, "nothing queued: no hint");
    assert_eq!(Idle.queue(Some(300)), Rows(300));
    assert_eq!(Rows(300).queue(Some(900)).queue(Some(100)), Rows(900), "the most rows of any queued copy");
    assert_eq!(Rows(900).hint(), 900);
    // a copy that cannot say how many rows it needs, in any order, turns the whole queue whole
    assert_eq!(Idle.queue(None), Whole);
    assert_eq!(Rows(900).queue(None), Whole);
    assert_eq!(Whole.queue(Some(10)), Whole, "and it stays whole until a decode applies the queue");
    assert_eq!(Idle.queue(Some(0)), Whole, "an empty stream is not a row count");
    assert_eq!(Whole.hint(), 0, "whole: ggml gets no hint");
}

#[test]
fn every_main_stream_copy_is_queued() {
    // `llama_memory_seq_cp` of the main context may only be called in `stream_copy` (which queues its rows) and for the
    // unified pool (one stream: no data copy). The reader's calls use `rmem`, another context.
    // every source file of the module (src/llama_engine/*.rs)
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/llama_engine");
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir).expect("the module's directory").map(|e| e.expect("a directory entry").path()).collect();
    files.sort();
    let src: String = files.iter().filter(|p| p.extension().is_some_and(|e| e == "rs")).map(|p| std::fs::read_to_string(p).expect("a source file")).collect::<Vec<_>>().join("\n");
    let call = concat!("unsafe { ffi::llama_memory_seq_", "cp(self.mem()"); // not this line's own text
    let calls: Vec<&str> = src.lines().filter(|l| l.contains(call)).collect();
    assert_eq!(calls.len(), 2, "a new stream copy must go through stream_copy: {calls:?}");
    assert!(calls.iter().any(|l| l.contains("0, -1)")), "{calls:?}");
    assert!(calls.iter().any(|l| l.contains("0, k as i32)")), "{calls:?}");
}

#[cfg(test)]
mod reader_tests {
    use super::*;

    #[test]
    fn readers_are_a_chat_reader_and_a_reader_of_the_whole_context() {
        assert_eq!(reader_cells(40_960), [4096, 40_960]);
        assert_eq!(reader_cells(768), [768, 768], "a context under the chat reader's size gets two readers of it");
    }

    #[test]
    fn a_chat_that_fits_goes_to_the_chat_reader_and_the_rest_to_the_long_one() {
        assert_eq!(own_reader(Lane::Interactive, 300, 4096, 2), 0);
        assert_eq!(own_reader(Lane::Interactive, 4096, 4096, 2), 0, "the chat reader holds its cells");
        assert_eq!(own_reader(Lane::Interactive, 4097, 4096, 2), 1, "outgrown: the long reader");
        assert_eq!(own_reader(Lane::Background, 300, 4096, 2), 1, "a background read never takes the chat's");
    }

    /// The compute buffers come from libllama (a `no_alloc` probe, which still starts Metal: run under the GPU lock).
    #[test]
    #[ignore = "starts Metal (no_alloc probe); run under the GPU lock"]
    fn the_reserve_adds_each_readers_compute_buffer() {
        let path = std::path::Path::new("/Users/benreich/Projects/crazyidea/research/inference_runtime/converted/qwen3-0.6b-Q4_0.gguf");
        let Ok(p) = crate::profile::load_profile(path) else { return eprintln!("skipping: {} not on this machine", path.display()) };
        let (chat, long) = (reader_compute_bytes(&p, 4096, 512, false), reader_compute_bytes(&p, 40_960, 512, false));
        let kv = reader_kv_bytes(&p, 40_960, 512, false);
        eprintln!("READER compute: chat {chat} long {long}; kv {kv}; reserve {}", reader_reserve_bytes(&p, 40_960, 512, false));
        assert!(chat > 0 && long >= chat && chat != READER_COMPUTE_FALLBACK, "read from libllama, not the fallback");
        assert_eq!(reader_reserve_bytes(&p, 40_960, 512, false), kv + chat + long);
    }

    /// The readers are reserved as libllama allocates them: the sliding-window layers of Gemma-3-4B keep a ring, not 40k cells
    /// (one reader of 40,960 cells took 5.7 GB in the old reserve and allocates about 1.16 GB).
    #[test]
    fn the_reserve_counts_the_ring_of_a_sliding_window_model() {
        let path = std::path::Path::new("/Users/benreich/Projects/crazyidea/models/gguf/gemma-3-4b-it-qat-Q4_0.gguf");
        let Ok(mut p) = crate::profile::Profile::from_gguf(path) else { return eprintln!("skipping: {} not on this machine", path.display()) };
        // the layer map libllama resolves for Gemma-3: five sliding-window layers, then a full one
        for (il, l) in p.layers.iter_mut().enumerate() {
            l.swa = il % 6 < 5;
        }
        let (full, swa) = p.kv_bytes_per_cell(crate::profile::KvType::F16);
        let ring = |cells: u64| p.swa_cells(cells, 2, 512, false);
        assert_eq!(ring(40_960), 2560, "n_swa x the two sequences + one micro-batch, padded");
        let long = p.kv_alloc_bytes(crate::profile::KvType::F16, 40_960, 2, 512, false);
        assert!((long as f64 / 1e9 - 1.16).abs() < 0.05, "{long}");
        assert!(2 * 40_960 * (full + swa) > 9 * long, "the old reserve counted every layer at full context");
        assert_eq!(reader_kv_bytes(&p, 40_960, 512, false), full * 4096 + swa * ring(4096) + long);
        // swa_full keeps the whole context in every layer: then the old figure was right
        assert_eq!(reader_kv_bytes(&p, 40_960, 512, true), (4096 + 40_960) * (full + swa));
    }
}
