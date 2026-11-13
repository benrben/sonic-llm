//! `llama-ref`: the llama.cpp reference side of the parity harness.
//!
//! ```text
//! llama-ref logits   <gguf> <ids.json> <out.npy>  [--n-ctx N] [--n-ubatch N] [--cache-type-k T] [--cache-type-v T]
//! llama-ref greedy   <gguf> <ids.json> <out.json> [--n 64] [--index I]
//! llama-ref tokenize <gguf> <text-file|-> <out.json> [--add-special] [--parse-special]
//! ```
//!
//! `logits` writes an f32 `.npy` of shape `[n_tokens, n_vocab]`; it refuses to overwrite an existing file.
//! `greedy` runs every prompt of the file (or only `--index I`) for exactly `--n` tokens and writes ids and the top-2
//! logits of each step. `tokenize` writes `{"ids": [...]}`. GPU commands must run under `engine/scripts/with_gpu_lock.py`.

use llama_ref as lr;
use serde_json::json;
use std::fs::OpenOptions;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("llama-ref: {msg}");
    std::process::exit(1)
}

struct Args {
    pos: Vec<String>,
    opts: Vec<(String, Option<String>)>,
}

impl Args {
    fn parse(argv: &[String], with_value: &[&str], flags: &[&str]) -> Args {
        let (mut pos, mut opts) = (vec![], vec![]);
        let mut it = argv.iter();
        while let Some(a) = it.next() {
            if with_value.contains(&a.as_str()) {
                let v = it.next().unwrap_or_else(|| die(format!("{a} needs a value")));
                opts.push((a.clone(), Some(v.clone())));
            } else if flags.contains(&a.as_str()) {
                opts.push((a.clone(), None));
            } else if a.starts_with("--") {
                die(format!("unknown option {a}"));
            } else {
                pos.push(a.clone());
            }
        }
        Args { pos, opts }
    }
    fn value<T: std::str::FromStr>(&self, name: &str) -> Option<T> {
        self.opts.iter().find(|(k, _)| k == name).map(|(_, v)| {
            v.as_deref().unwrap().parse().unwrap_or_else(|_| die(format!("bad value for {name}")))
        })
    }
    fn flag(&self, name: &str) -> bool {
        self.opts.iter().any(|(k, _)| k == name)
    }
}

fn create_new(path: &Path) -> std::fs::File {
    OpenOptions::new().write(true).create_new(true).open(path).unwrap_or_else(|e| die(format!("{}: {e}", path.display())))
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = argv.first() else { die("usage: llama-ref logits|greedy|tokenize ... (see the module docs)") };
    let rest = &argv[1..];
    match cmd.as_str() {
        "logits" => logits(rest),
        "greedy" => greedy(rest),
        "tokenize" => tokenize(rest),
        other => die(format!("unknown command {other:?}; expected logits, greedy or tokenize")),
    }
}

/// KV cache types (f16, q8_0, q4_0); one given alone applies to K and V alike, none keeps the f16 default.
fn cache_types(a: &Args) -> Option<(lr::sys::ggml_type, lr::sys::ggml_type)> {
    let (tk, tv): (Option<String>, Option<String>) = (a.value("--cache-type-k"), a.value("--cache-type-v"));
    match (&tk, &tv) {
        (None, None) => None,
        _ => {
            let k = lr::parse_cache_type(tk.as_deref().or(tv.as_deref()).unwrap()).unwrap_or_else(|e| die(e));
            let v = lr::parse_cache_type(tv.as_deref().or(tk.as_deref()).unwrap()).unwrap_or_else(|e| die(e));
            Some((k, v))
        }
    }
}

/// The text to tokenize: a file, or stdin for `-`.
fn read_text(src: &str) -> String {
    let mut text = String::new();
    if src == "-" {
        std::io::stdin().read_to_string(&mut text).unwrap_or_else(|e| die(e));
    } else {
        text = std::fs::read_to_string(src).unwrap_or_else(|e| die(format!("{src}: {e}")));
    }
    text
}

/// `logits <gguf> <ids.json> <out.npy>`
fn logits(rest: &[String]) {
    let a = Args::parse(rest, &["--n-ctx", "--n-ubatch", "--cache-type-k", "--cache-type-v"], &[]);
    if a.pos.len() != 3 {
        die("usage: llama-ref logits <gguf> <ids.json> <out.npy> [--n-ctx N] [--n-ubatch N] [--cache-type-k T] [--cache-type-v T]");
    }
    let lists = lr::inputs::read_id_lists(Path::new(&a.pos[1])).unwrap_or_else(|e| die(e));
    if lists.len() != 1 {
        die("logits needs a file with exactly one id list");
    }
    let ids = &lists[0];
    let n_ubatch: u32 = a.value("--n-ubatch").unwrap_or(512);
    let n_ctx: u32 = a.value("--n-ctx").unwrap_or((ids.len() as u32).div_ceil(256) * 256);
    let kv = cache_types(&a);
    let out = PathBuf::from(&a.pos[2]);
    let mut w = BufWriter::with_capacity(1 << 20, create_new(&out));
    let n = ids.len();
    let r = lr::reference_logits_stream_kv(Path::new(&a.pos[0]), ids, n_ctx, n_ubatch, kv, |c| {
        if c.start == 0 {
            lr::npy::write_header_f32_2d(&mut w, n, c.n_vocab).map_err(|e| lr::Error(e.to_string()))?;
        }
        lr::npy::write_f32s(&mut w, c.data).map_err(|e| lr::Error(e.to_string()))
    });
    match r {
        Ok(s) => {
            w.flush().unwrap_or_else(|e| die(e));
            println!("{}", json!({"out": out, "n_tokens": n, "settings": s}));
        }
        Err(e) => {
            drop(w);
            let _ = std::fs::remove_file(&out); // our own partial file
            die(e)
        }
    }
}

/// `greedy <gguf> <ids.json> <out.json>`
fn greedy(rest: &[String]) {
    let a = Args::parse(rest, &["--n", "--index"], &[]);
    if a.pos.len() != 3 {
        die("usage: llama-ref greedy <gguf> <ids.json> <out.json> [--n 64] [--index I]");
    }
    let n: usize = a.value("--n").unwrap_or(64);
    let lists = lr::inputs::read_id_lists(Path::new(&a.pos[1])).unwrap_or_else(|e| die(e));
    let index: Option<usize> = a.value("--index");
    let selected: Vec<(usize, Vec<i32>)> = match index {
        Some(i) => vec![(i, lists.get(i).cloned().unwrap_or_else(|| die(format!("no prompt {i}"))))],
        None => lists.into_iter().enumerate().collect(),
    };
    let prompts: Vec<Vec<i32>> = selected.iter().map(|(_, p)| p.clone()).collect();
    let res = lr::reference_greedy_many(Path::new(&a.pos[0]), &prompts, n).unwrap_or_else(|e| die(e));
    let results: Vec<_> = selected
        .iter()
        .zip(&res)
        .map(|((i, p), g)| json!({"index": i, "prompt_len": p.len(), "ids": g.ids, "top2": g.top2}))
        .collect();
    let doc = json!({
        "gguf": a.pos[0], "ids_file": a.pos[1], "n": n, "settings": res[0].settings,
        "sampling": "greedy argmax (lowest id on a tie), EOS ignored", "results": results,
    });
    let mut f = create_new(Path::new(&a.pos[2]));
    f.write_all(serde_json::to_string_pretty(&doc).unwrap().as_bytes()).unwrap_or_else(|e| die(e));
    println!("{} prompts, {} tokens each -> {}", res.len(), n, a.pos[2]);
}

/// `tokenize <gguf> <text-file|-> <out.json>`
fn tokenize(rest: &[String]) {
    let a = Args::parse(rest, &[], &["--add-special", "--parse-special"]);
    if a.pos.len() != 3 {
        die("usage: llama-ref tokenize <gguf> <text-file|-> <out.json> [--add-special] [--parse-special]");
    }
    let text = read_text(&a.pos[1]);
    let (add, parse) = (a.flag("--add-special"), a.flag("--parse-special"));
    let ids = lr::tokenize(Path::new(&a.pos[0]), &text, add, parse).unwrap_or_else(|e| die(e));
    let (n_vocab, bos, model_adds_bos) = lr::vocab_info(Path::new(&a.pos[0])).unwrap_or_else(|e| die(e));
    let doc = json!({
        "gguf": a.pos[0], "text_bytes": text.len(), "add_special": add, "parse_special": parse,
        "n_vocab": n_vocab, "bos_id": bos, "model_adds_bos": model_adds_bos, "n": ids.len(), "ids": ids,
    });
    let mut f = create_new(Path::new(&a.pos[2]));
    f.write_all(serde_json::to_string(&doc).unwrap().as_bytes()).unwrap_or_else(|e| die(e));
    println!("{} tokens -> {}", ids.len(), a.pos[2]);
}
