use super::*;

pub const MODELS: &str = "/Users/benreich/Projects/crazyidea/models/gguf";
pub const QWEN06: &str = "/Users/benreich/Projects/crazyidea/research/inference_runtime/converted/qwen3-0.6b-Q4_0.gguf";

pub fn model(name: &str) -> Option<PathBuf> {
    let p = if name.starts_with('/') { PathBuf::from(name) } else { Path::new(MODELS).join(name) };
    p.is_file().then_some(p).or_else(|| {
        eprintln!("skipping: {name} not on this machine");
        None
    })
}

#[test]
fn machine_config_parses() {
    let t = "schema = 1\n[machine.\"Apple M5 Pro\"]\nbandwidth_gbs = 291 # GB/s\n[machine.\"Other\"]\nbandwidth_gbs = 100\n";
    assert_eq!(parse_machine_bandwidth(t, "Apple M5 Pro"), Some(291.0));
    assert_eq!(parse_machine_bandwidth(t, "Other"), Some(100.0));
    assert_eq!(parse_machine_bandwidth(t, "Nope"), None);
    let f = std::fs::read_to_string(default_machine_config_path()).expect("engine/config/machine.toml");
    assert_eq!(parse_machine_bandwidth(&f, "Apple M5 Pro"), Some(291.0));
}

#[test]
fn kv_log_line_parses() {
    let l = "llama_kv_cache: size =  224.00 MiB (  4096 cells,  28 layers,  1/1 seqs), K (f16):  112.00 MiB, V (f16):  112.00 MiB\n";
    let v = parse_kv_lines(l);
    assert_eq!(v.len(), 1);
    assert_eq!((v[0].cells, v[0].layers, v[0].n_seq), (4096, 28, 1));
    assert_eq!(v[0].mib_total, 224.0);
    assert_eq!(v[0].type_k, "f16");
}

#[test]
fn period_rule_matches_llama_cpp() {
    // gemma-3: 5 SWA layers then a full one
    assert_eq!(period_pattern(12, 6), vec![true, true, true, true, true, false, true, true, true, true, true, false]);
    // gpt-oss: alternate, full on odd layers
    assert_eq!(period_pattern(4, 2), vec![true, false, true, false]);
    assert!(period_pattern(3, 0).iter().all(|&s| s));
}

#[test]
fn expert_fraction_and_weights() {
    let mut p = synthetic();
    p.n_expert = 8;
    p.n_expert_used = 2;
    p.weights_experts = 800;
    p.weights_dense = 200;
    p.embd_row_bytes = 0;
    assert!((p.expert_fraction(1) - 0.25).abs() < 1e-9);
    assert!(p.expert_fraction(8) > 0.85 && p.expert_fraction(8) < 1.0);
    assert_eq!(p.weights_per_step(1), 200 + 200);
}

pub fn synthetic() -> Profile {
    Profile {
        path: PathBuf::new(),
        arch: "x".into(),
        name: None,
        file_bytes: 0,
        fingerprint: String::new(),
        n_layer: 2,
        n_embd: 64,
        n_head: 4,
        n_head_kv: 2,
        head_dim_k: 16,
        head_dim_v: 16,
        n_vocab: 100,
        n_ctx_train: 4096,
        n_swa: 0,
        layers: vec![LayerKv { n_head_kv: 2, k_dim: 16, v_dim: 16, swa: false }; 2],
        swa_source: SwaSource::None,
        n_expert: 0,
        n_expert_used: 0,
        n_expert_shared: 0,
        recurrent: false,
        hybrid: false,
        mla: false,
        weights_total: 1000,
        weights_dense: 1000,
        weights_experts: 0,
        weights_gather: 0,
        embd_row_bytes: 0,
        tied_head: true,
        notes: vec![],
    }
}

/// The three real GGUFs of the brief; a missing file skips its check.
fn real_models() -> Vec<PathBuf> {
    [QWEN06, "gemma-3-4b-it-qat-Q4_0.gguf", "gpt-oss-20b-MXFP4.gguf"].iter().filter_map(|m| model(m)).collect()
}

#[test]
fn real_profiles_read_without_gpu() {
    for path in real_models() {
        let p = Profile::from_gguf(&path).unwrap();
        println!("{}", p.summary_line());
        assert!(p.n_layer > 0 && p.n_vocab > 1000 && p.weights_total > 0);
        // the tensor table adds up to the file apart from the header
        assert!(p.weights_total < p.file_bytes && p.weights_total as f64 > 0.9 * p.file_bytes as f64, "{}", p.arch);
        assert!(p.weights_per_step(1) <= p.weights_total);
    }
}

/// KV bytes per cell against libllama's own accounting (its `llama_kv_cache: size = ...` log lines of a `no_alloc`
/// context): same cells, same layers, same MiB, for the unified cache and the SWA cache.
#[test]
fn kv_bytes_match_llama_cpp_log() {
    for path in real_models() {
        let mut p = Profile::from_gguf(&path).unwrap();
        for (cells, n_seq, ubatch, swa_full) in [(4096u32, 1u32, 512u32, true), (4096, 1, 512, false), (2048, 8, 64, false)] {
            let pr = ProbeParams { n_gpu_layers: 0, cells, n_seq, n_ubatch: ubatch, n_batch: ubatch, swa_full, ..Default::default() };
            let probe = probe_memory(&path, &pr).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            p.resolve_with_libllama();
            println!("{} cells {cells} seqs {n_seq}: swa map from {}; libllama: {:?}", p.arch, p.swa_source.name(), probe.kv);
            let (full_bpc, swa_bpc) = p.kv_bytes_per_cell(KvType::F16);
            let swa_cells = p.swa_cells(cells as u64, n_seq as u64, ubatch as u64, swa_full);
            let mut want: Vec<(u64, u64, usize)> = vec![];
            if full_bpc > 0 {
                want.push((cells as u64, full_bpc, p.n_full_layers()));
            }
            if swa_bpc > 0 {
                want.push((swa_cells, swa_bpc, p.n_swa_layers()));
            }
            assert_eq!(probe.kv.len(), want.len(), "{}: caches", p.arch);
            for (l, (c, bpc, layers)) in probe.kv.iter().zip(want) {
                assert_eq!((l.cells, l.layers), (c, layers), "{} cells/layers", p.arch);
                let mine = (bpc * c) as f64 / 1048576.0;
                assert!((l.mib_total - mine).abs() <= 0.011, "{}: libllama {} MiB, profile {mine} MiB", p.arch, l.mib_total);
            }
            let total: f64 = probe.kv.iter().map(|l| l.bytes()).sum();
            let mine = p.kv_alloc_bytes(KvType::F16, cells as u64, n_seq as u64, ubatch as u64, swa_full) as f64;
            assert!((total - mine).abs() < 0.03 * 1048576.0, "{}: total libllama {total}, profile {mine}", p.arch);
        }
    }
}

#[test]
fn kv_math_synthetic() {
    let p = synthetic();
    // 2 layers x (2 heads x 16) x (K + V) x 2 bytes
    assert_eq!(p.kv_bytes_per_cell(KvType::F16), (2 * 32 * 2 * 2, 0));
    assert_eq!(p.kv_alloc_bytes(KvType::F16, 1024, 4, 512, true), 2 * 32 * 2 * 2 * 1024);
    let b = Budget::plan(
        &p,
        &BudgetInputs {
            swa_full: true,
            stream_cells: None,
            max_seq_ids: 256,
            working_set: 1 << 30,
            util: 1.0,
            headroom: 0,
            kv_type: KvType::F16,
            n_seq: 4,
            n_ubatch: 512,
            compute_bytes: 0,
            compute_source: "test".into(),
            want_cells: 1 << 20,
            asked_by: "test".into(),
            min_cells: 256,
            max_total_cells: 1 << 20,
            alloc_share: 1.0,
            bandwidth_gbs: Some(100.0),
        },
    );
    assert!(p.kv_alloc_bytes(KvType::F16, b.max_cells, 4, 512, true) <= b.kv_budget);
    assert!(p.kv_alloc_bytes(KvType::F16, b.max_cells + 256, 4, 512, true) > b.kv_budget);
    assert_eq!(b.cells, b.max_cells.min(1 << 20) / 256 * 256);
}

/// Budget inputs of the synthetic model: `fits` streams (or contexts) of 4096 cells of f16 fit the KV budget exactly.
fn fit_inputs(p: &Profile, fits: u64, per_sequence: bool, alloc_share: f64) -> BudgetInputs {
    let per_stream = p.kv_alloc_bytes(KvType::F16, 4096, 1, 512, false);
    BudgetInputs {
        swa_full: false,
        working_set: p.weights_total + fits * per_stream,
        util: 1.0,
        headroom: 0,
        kv_type: KvType::F16,
        n_seq: 8,
        n_ubatch: 512,
        compute_bytes: 0,
        compute_source: "synthetic".into(),
        want_cells: 8 * 4096,
        asked_by: "test".into(),
        min_cells: 4096,
        max_total_cells: MAX_TOTAL_CELLS,
        stream_cells: per_sequence.then_some(4096),
        max_seq_ids: 256,
        alloc_share,
        bandwidth_gbs: None,
    }
}

/// A budget that fits 0, 1, 2, 4 or 8 contexts never allocates more than it has, in both layouts: the two-context
/// floor of the share and the one-context minimum give way to the bytes. 0 contexts allocates nothing and is refused.
#[test]
fn allocation_never_exceeds_the_kv_budget() {
    let p = synthetic();
    for per_sequence in [true, false] {
        for (share, shared_floor) in [(1.0, false), (0.25, true)] {
            for fits in [0u64, 1, 2, 4, 8] {
                let b = Budget::plan(&p, &fit_inputs(&p, fits, per_sequence, share));
                let layout = if per_sequence { "streams" } else { "unified" };
                let ctx = format!("{layout}, share {share}, fits {fits}: {b:?}");
                assert!(b.kv_bytes(&p, b.cells) <= b.kv_budget, "{ctx}");
                assert_eq!(b.max_cells, fits * 4096, "{ctx}");
                // 8 wanted; a quarter share holds the two-context floor (a quarter of 8 is 2), and never more than fit
                let want = if shared_floor { fits.min(2) } else { fits };
                assert_eq!(b.cells, want * 4096, "{ctx}");
                if per_sequence {
                    assert_eq!(b.n_seq, want, "{ctx}");
                }
                assert_eq!(b.check_fits(&p, 4096).is_ok(), fits > 0, "{ctx}");
                if fits == 0 {
                    assert!(b.cells_reason.starts_with("not enough memory for one"), "{ctx}");
                }
            }
        }
    }
}

/// The same, on a model with sliding-window layers and across budgets that are not a whole number of contexts.
#[test]
fn allocation_stays_inside_the_budget_with_sliding_window_layers() {
    let mut p = synthetic();
    p.n_swa = 1024;
    p.layers[1].swa = true;
    for swa_full in [false, true] {
        let per_stream = p.kv_alloc_bytes(KvType::Q8_0, 4096, 1, 512, swa_full);
        let mut last = [0u64; 2];
        for step in 0..60u64 {
            let kv = per_stream * step / 7;
            for (k, per_sequence) in [true, false].into_iter().enumerate() {
                let mut i = fit_inputs(&p, 0, per_sequence, 0.25);
                (i.kv_type, i.swa_full, i.working_set) = (KvType::Q8_0, swa_full, p.weights_total + kv);
                let b = Budget::plan(&p, &i);
                assert!(b.kv_bytes(&p, b.cells) <= b.kv_budget, "swa_full {swa_full}, kv {kv}, streams {per_sequence}: {b:?}");
                assert!(b.cells >= last[k], "more memory never allocates less (swa_full {swa_full}, streams {per_sequence}, kv {kv})");
                last[k] = b.cells;
                assert_eq!(b.check_fits(&p, 4096).is_ok(), b.max_cells >= 4096 && b.cells >= 4096);
            }
        }
    }
}

#[test]
fn refusal_names_the_context_and_the_ways_out() {
    let p = synthetic();
    let b = Budget::plan(&p, &fit_inputs(&p, 0, true, 1.0));
    let e = b.check_fits(&p, 4096).unwrap_err();
    assert!(e.contains("one context of 4096 tokens") && e.contains("--ctx") && e.contains("--kv-type"), "{e}");
}

/// No `--kv-pages`: the default job, whatever the lane slots are; `--kv-pages` is taken as it is.
#[test]
fn default_size_follows_the_default_job() {
    let (cells, why) = wanted(None, 4096);
    assert_eq!(cells, 4096 * requests_for(DEFAULT_AGENTS));
    assert!(why.contains("default job") && why.contains(&DEFAULT_AGENTS.to_string()), "{why}");
    assert_eq!(wanted(Some(96), 4096), (96 * 256, "--kv-pages".to_string()));
    // the default reaches the plan and stays inside a roomy budget, as the reason says
    let p = synthetic();
    let mut i = fit_inputs(&p, 64, true, 0.25);
    (i.want_cells, i.asked_by) = wanted(None, 4096);
    let b = Budget::plan(&p, &i);
    assert_eq!((b.n_seq, b.cells), (requests_for(DEFAULT_AGENTS), requests_for(DEFAULT_AGENTS) * 4096));
    assert!(b.cells_reason.contains("default job"), "{}", b.cells_reason);
    // a tighter budget limits it and says so
    let mut i = fit_inputs(&p, 8, true, 0.25);
    (i.want_cells, i.asked_by) = wanted(None, 4096);
    let b = Budget::plan(&p, &i);
    assert_eq!(b.n_seq, 2);
    assert!(b.cells_reason.starts_with("limited to 25% of the KV budget"), "{}", b.cells_reason);
}

/// The real planner on the small model: a plain run no longer reserves a quarter of the budget, an explicit
/// `--kv-pages` still gets what it asks for, and a context that cannot fit is refused. Loads a no_alloc model on the
/// Metal device: run it through the GPU lock.
#[test]
#[ignore = "loads a model on Metal: run through engine/scripts/with_gpu_lock.py"]
fn plan_engine_default_job_on_the_small_model() {
    let Some(path) = model(QWEN06) else { return };
    let p = Profile::from_gguf(&path).unwrap();
    let args = |pool_pages, n_ctx, working_set| PlanArgs {
        n_ctx,
        slots: 64,
        pool_pages,
        n_ubatch: 512,
        step_tokens: 2048,
        kv_type: KvType::F16,
        unified: false,
        max_seq_ids: 256,
        swa_full: false,
        working_set,
        bandwidth_gbs: None,
    };
    let ws = Some(64u64 << 30);
    let (plain, n_seq, _) = plan_engine(&p, &args(None, 4096, ws));
    assert_eq!(n_seq as u64, requests_for(DEFAULT_AGENTS));
    assert_eq!(plain.cells, requests_for(DEFAULT_AGENTS) * 4096, "{plain:?}");
    assert!(plain.max_cells > 4 * plain.cells, "a few contexts, not a share of the budget: {plain:?}");
    assert!(plain.check_fits(&p, 4096).is_ok());
    let (pages, _, _) = plan_engine(&p, &args(Some(96), 4096, ws));
    assert_eq!(pages.cells, 96 * 256, "{pages:?}");
    // a working set that holds the weights and nothing else: refused, not over-allocated
    let (none, _, _) = plan_engine(&p, &args(None, 4096, Some(p.weights_total + (1 << 30) + (64 << 20))));
    assert!(none.kv_bytes(&p, none.cells) <= none.kv_budget, "{none:?}");
    assert!(none.check_fits(&p, 4096).is_err(), "{none:?}");
}
