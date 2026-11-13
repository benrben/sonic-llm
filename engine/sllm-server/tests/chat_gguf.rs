//! Golden parity of SLLM's chat rendering ([`ChatTemplate::render_chat`]) and tool-call parsing ([`sllm_server::toolcall`])
//! against the pinned `llama-server --jinja` (build 11669), on each family's own GGUF.
//!
//! Fixtures: `tests/fixtures/chat_golden/<family>/*.json`, written by `engine/scripts/capture_chat_golden.py`:
//! * one file per conversation: the OpenAI-wire `messages`/`tools`, llama-server's rendered `prompt` (`/apply-template`,
//!   `enable_thinking` false, generation prompt on) and its token ids; or `error` when llama-server refused the messages;
//! * `generations.json`: RAW model text recorded next to the server's parsed `message` (content, tool_calls).
//!
//! A conversation passes when SLLM's prompt is the same string and tokenizes to the same ids (vocab-only llama.cpp
//! tokenizer, no GPU; BOS handled as the chat route does: a BOS the template wrote is cut when the vocabulary adds one).
//! A family/conversation that is known not to match is listed in [`KNOWN`] with the reason; it must keep failing the way
//! it is listed (a listed case that starts to pass fails the test, so the list stays honest). Skips cleanly when the
//! GGUF (`SLLM_MODELS_DIR`, default the repo's models/gguf) is absent.

use serde_json::Value;
use sllm_server::chat::ChatTemplate;
use sllm_server::toolcall::{Part, ToolParser};
use sllm_server::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};

const DEFAULT_MODELS: &str = "/Users/benreich/Projects/crazyidea/models/gguf";

/// (family, conversation, reason). Tool-call parsing cases use "parse:<name>".
const KNOWN: &[(&str, &str, &str)] = &[
    // The recorded generation hit max_tokens (finish_reason length) in the middle of a call, with no `</function>` and no
    // `</tool_call>`, and the model repeating itself. llama-server returns the half call it has read (duplicate keys and
    // all); SLLM's rule is that a call that is never closed comes back as content, verbatim (a cut-off call is not run).
    ("qwen35", "parse:search_nested", "call cut off by max_tokens: llama-server returns the partial call, SLLM returns content"),
];

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chat_golden")
}

fn models_dir() -> PathBuf {
    std::env::var_os("SLLM_MODELS_DIR").map(PathBuf::from).unwrap_or_else(|| DEFAULT_MODELS.into())
}

fn read(p: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn families() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(fixtures_dir())
        .map(|d| d.flatten().filter(|e| e.path().is_dir()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    v.sort();
    v
}

fn known(family: &str, case: &str) -> Option<&'static str> {
    KNOWN.iter().find(|(f, c, _)| *f == family && (*c == "*" || *c == case)).map(|(_, _, why)| *why)
}

struct Family {
    name: String,
    tmpl: ChatTemplate,
    tok: Tokenizer,
}

/// The family's template and tokenizer from its GGUF; `None` (skip) when the file is absent.
fn load(family: &str) -> Option<Family> {
    let dir = fixtures_dir().join(family);
    let meta = read(&dir.join("meta.json"));
    let gguf = models_dir().join(meta["gguf"].as_str().unwrap());
    if !gguf.exists() {
        eprintln!("{family}: {} absent, skipped", gguf.display());
        return None;
    }
    let tmpl = ChatTemplate::from_gguf(&gguf).unwrap_or_else(|e| panic!("{family}: {e}"));
    // templates that print the date (Llama 3.1, SmolLM3, gpt-oss) get the day the fixtures were captured
    tmpl.set_clock(meta["now_as_utc"].as_i64().unwrap());
    Some(Family { name: family.into(), tmpl, tok: Tokenizer::gguf(gguf) })
}

/// Index and context of the first byte where `a` and `b` differ.
fn first_diff(a: &str, b: &str) -> String {
    let n = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
    let ctx = |s: &str| {
        let lo = (0..=n.saturating_sub(12)).rev().find(|&i| s.is_char_boundary(i)).unwrap_or(0);
        let hi = (n + 24..=s.len().max(n + 24)).find(|&i| i >= s.len() || s.is_char_boundary(i)).unwrap_or(s.len()).min(s.len());
        format!("{:?}", &s[lo..hi.max(lo)])
    };
    format!("first difference at byte {n} (sllm {} bytes, llama-server {} bytes): sllm {} | llama-server {}", a.len(), b.len(), ctx(a), ctx(b))
}

/// SLLM's text and ids for a fixture's conversation, the way the chat route makes them (`chat_api::prompt_ids`).
fn sllm_prompt(f: &Family, fx: &Value) -> Result<(String, Vec<i32>), String> {
    let msgs = fx["messages"].as_array().unwrap();
    let tools = fx["tools"].as_array().cloned().unwrap_or_default();
    let text = f.tmpl.render_chat(msgs, &tools, true, false).map_err(|e| e.to_string())?;
    let add_bos = f.tok.vocab().is_some_and(|v| v.add_bos());
    let text = match text.strip_prefix(f.tmpl.bos()) {
        Some(rest) if add_bos && !f.tmpl.bos().is_empty() => rest.to_string(),
        _ => text,
    };
    let ids = f.tok.encode(&text, add_bos, true)?;
    Ok((text, ids))
}

#[derive(Debug, PartialEq)]
struct Outcome {
    exact: bool,
    tokens: bool,
    /// The prompts hold the same bytes (in any order).
    same_bytes: bool,
    detail: String,
}

/// Compare one fixture with SLLM.
fn check(f: &Family, fx: &Value) -> Outcome {
    let want_err = fx.get("error").is_some_and(|e| !e.is_null());
    match (sllm_prompt(f, fx), want_err) {
        (Err(_), true) => Outcome { exact: true, tokens: true, same_bytes: true, detail: String::new() },
        (Err(e), false) => Outcome { exact: false, tokens: false, same_bytes: false, detail: format!("sllm refused: {e}") },
        (Ok(_), true) => Outcome { exact: false, tokens: false, same_bytes: false, detail: format!("llama-server refused ({}), sllm rendered", fx["error"]["body"]) },
        (Ok((text, ids)), false) => {
            let want = fx["prompt"].as_str().unwrap();
            let want_ids: Vec<i32> = fx["tokens"].as_array().unwrap().iter().map(|t| t.as_i64().unwrap() as i32).collect();
            let exact = text == want;
            let sorted = |s: &str| {
                let mut b = s.as_bytes().to_vec();
                b.sort_unstable();
                b
            };
            let same_bytes = sorted(&text) == sorted(want);
            let tokens = ids == want_ids;
            let mut detail = String::new();
            if !exact {
                detail = first_diff(&text, want);
            }
            if !tokens {
                let n = ids.iter().zip(&want_ids).take_while(|(a, b)| a == b).count();
                detail.push_str(&format!("; tokens differ at {n} (sllm {} ids, llama-server {})", ids.len(), want_ids.len()));
            }
            Outcome { exact, tokens, same_bytes, detail }
        }
    }
}

#[test]
fn rendered_prompts_match_llama_server() {
    let mut table = Vec::new();
    let mut bad = Vec::new();
    for fam in families() {
        let Some(f) = load(&fam) else { continue };
        let mut files: Vec<_> = std::fs::read_dir(fixtures_dir().join(&fam)).unwrap().flatten().map(|e| e.path()).collect();
        files.sort();
        let (mut n, mut exact, mut toks, mut first) = (0, 0, 0, String::new());
        for p in files {
            let name = p.file_stem().unwrap().to_string_lossy().into_owned();
            if name == "meta" || name == "generations" {
                continue;
            }
            let fx = read(&p);
            // the fixture's own prompt must tokenize to the fixture's ids, or the tokenizer (not the rendering) differs
            if let Some(prompt) = fx.get("prompt").and_then(Value::as_str) {
                let add_bos = f.tok.vocab().is_some_and(|v| v.add_bos());
                let ids = f.tok.encode(prompt, add_bos, true).unwrap();
                let want: Vec<i64> = fx["tokens"].as_array().unwrap().iter().map(|t| t.as_i64().unwrap()).collect();
                assert_eq!(ids.iter().map(|&i| i as i64).collect::<Vec<_>>(), want, "{fam}/{name}: SLLM's tokenizer disagrees with llama-server's on the same text");
            }
            let o = check(&f, &fx);
            n += 1;
            exact += o.exact as usize;
            toks += o.tokens as usize;
            let ok = o.exact && o.tokens;
            match (ok, known(&fam, &name)) {
                (true, None) => {}
                (false, Some(_)) => {
                    if first.is_empty() {
                        first = format!("{name}: {}", o.detail);
                    }
                }
                (false, None) => {
                    if first.is_empty() {
                        first = format!("{name}: {}", o.detail);
                    }
                    bad.push(format!("{fam}/{name}: {}", o.detail));
                }
                (true, Some(why)) => bad.push(format!("{fam}/{name}: listed as a known failure ({why}) but now matches; remove it from KNOWN")),
            }
        }
        table.push(format!("{:<9} conversations {n:>2}  exact {exact:>2}  token-exact {toks:>2}  {first}", f.name));
    }
    eprintln!("{}", table.join("\n"));
    assert!(bad.is_empty(), "prompts differ from llama-server --jinja:\n{}", bad.join("\n"));
}

/// Negative control: one changed character in a fixture's prompt (and, separately, in its token ids) must be seen.
#[test]
fn a_one_character_change_in_a_fixture_is_caught() {
    let mut tried = 0;
    for fam in families() {
        let Some(f) = load(&fam) else { continue };
        let fx = read(&fixtures_dir().join(&fam).join("system_user.json"));
        if fx.get("prompt").is_none() || check(&f, &fx) != (Outcome { exact: true, tokens: true, same_bytes: true, detail: String::new() }) {
            continue; // control only where the unmodified fixture matches
        }
        let mut bad = fx.clone();
        let p = bad["prompt"].as_str().unwrap().to_string();
        let i = p.char_indices().nth(p.chars().count() / 2).unwrap().0;
        let c = p[i..].chars().next().unwrap();
        bad["prompt"] = Value::String(format!("{}{}{}", &p[..i], if c == 'x' { 'y' } else { 'x' }, &p[i + c.len_utf8()..]));
        let o = check(&f, &bad);
        assert!(!o.exact, "{fam}: a changed character in the expected prompt went unnoticed");
        assert!(o.detail.contains("first difference at byte"), "{}", o.detail);
        let mut bad = fx.clone();
        let t = bad["tokens"][0].as_i64().unwrap();
        bad["tokens"][0] = Value::from(t + 1);
        assert!(!check(&f, &bad).tokens, "{fam}: a changed token id went unnoticed");
        tried += 1;
    }
    eprintln!("negative control ran on {tried} families");
}

/// Per family: the format name `ToolFormat::detect` must find (`None`: the template has no tool syntax, so llama-server shows
/// the model no tools and parses no calls), and how many recorded generations hold a real call on llama-server's side.
/// Written down so that a fixture without calls cannot make the parsing test pass for nothing.
const FORMATS: &[(&str, Option<&str>, usize)] = &[
    ("qwen3", Some("qwen-json"), 3),
    ("qwen25", Some("qwen-json"), 3),
    // the template holds the `<tool_call>` tags of Qwen's JSON format too; the `<function=` / `<parameter=` tags decide
    ("qwen35", Some("qwen-xml"), 3),
    // the GGUF's template names `<tool_call>` but renders tools only from `xml_tools` / `python_tools`, which llama-server (and
    // SLLM) never pass: the model is shown no tools and made no call
    ("smollm3", Some("qwen-json"), 0),
    ("gemma4", Some("gemma4"), 3),
    ("gptoss", Some("gpt-oss"), 3),
    ("llama31", Some("llama31-json"), 3),
    // this GGUF's Mistral template is the old one (no tools, no system role): `[TOOL_CALLS]` is in the vocabulary, not the template
    ("mistral", None, 0),
    ("gemma3", None, 0),
    ("phi3", None, 0),
];

fn want_calls(g: &Value) -> Vec<(String, Value)> {
    g["message"]["tool_calls"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|c| {
                    let args = c["function"]["arguments"].as_str().unwrap_or("{}");
                    (c["function"]["name"].as_str().unwrap().to_string(), serde_json::from_str(args).unwrap_or(Value::String(args.into())))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Every GGUF family resolves to the expected tool-call format.
#[test]
fn every_family_resolves_to_its_tool_format() {
    let mut seen = 0;
    for fam in families() {
        let Some(f) = load(&fam) else { continue };
        let (_, want, _) = FORMATS.iter().find(|(n, _, _)| *n == fam).unwrap_or_else(|| panic!("{fam}: add it to FORMATS"));
        assert_eq!(f.tmpl.tool_format().map(|t| t.name()), *want, "{fam}");
        seen += 1;
    }
    eprintln!("formats checked on {seen} families");
}

/// How a format's marker tokens print. Most print as their own text from the engine (`llama_token_to_piece` special = false);
/// a true control token (Mistral's `[TOOL_CALLS]`) prints as nothing there, and `Vocab::piece_special` gives its text: the chat
/// routes use that for the tokens a format preserves. Checked on each family's vocabulary, and Mistral's for its format.
#[test]
fn a_formats_marker_tokens_print_as_their_text_or_through_piece_special() {
    use sllm_server::toolcall::ToolFormat;
    let (mut plain, mut special_only) = (Vec::new(), Vec::new());
    let mut cases: Vec<(String, &[&str])> = Vec::new();
    for fam in families() {
        if let Some(format) = load(&fam).and_then(|f| f.tmpl.tool_format()) {
            cases.push((fam, format.spec().preserved));
        }
    }
    cases.push(("mistral".into(), ToolFormat::MISTRAL.spec().preserved));
    for (fam, preserved) in cases {
        let Some(f) = load(&fam) else { continue };
        let vocab = f.tok.vocab().unwrap();
        for marker in preserved {
            let ids = f.tok.encode(marker, false, true).unwrap();
            if let [id] = ids[..] {
                let (piece, special) = (String::from_utf8_lossy(&vocab.piece(id)).into_owned(), String::from_utf8_lossy(&vocab.piece_special(id)).into_owned());
                assert_eq!(special, *marker, "{fam}: {marker}");
                match (piece.as_str(), vocab.is_eog(id)) {
                    (p, _) if p == *marker => plain.push(format!("{fam}:{marker}")),
                    ("", true) => {} // an end-of-generation token (`<|call|>`): never streamed
                    ("", false) => special_only.push(format!("{fam}:{marker}")),
                    (p, _) => panic!("{fam}: {marker} prints as {p:?}"),
                }
            }
        }
    }
    eprintln!("print as their text: {plain:?}\nneed piece_special: {special_only:?}");
    assert!(plain.len() >= 10);
    assert_eq!(special_only, ["mistral:[TOOL_CALLS]"]);
}

/// Tool calls out of the recorded raw text, against what llama-server parsed from the same text: the content and the
/// calls (arguments compared as parsed JSON).
#[test]
fn parsed_tool_calls_match_llama_server() {
    let mut table = Vec::new();
    let mut bad = Vec::new();
    for fam in families() {
        let Some(f) = load(&fam) else { continue };
        let gens = read(&fixtures_dir().join(&fam).join("generations.json"));
        let format = f.tmpl.tool_format();
        let expected_with_calls = FORMATS.iter().find(|(n, _, _)| *n == fam).map(|t| t.2).unwrap_or_else(|| panic!("{fam}: add it to FORMATS"));
        let (mut n, mut ok_n, mut theirs, mut ours, mut first) = (0, 0, 0, 0, String::new());
        for g in gens.as_array().unwrap() {
            let name = g["name"].as_str().unwrap();
            let Some(raw) = g.get("raw").and_then(Value::as_str) else { continue };
            n += 1;
            // no format: the model was shown no tools; the reply is content, as llama-server returned it
            let parts = match format {
                Some(format) => {
                    let mut p = ToolParser::new(format).with_tools(&g["tools"].as_array().cloned().unwrap_or_default());
                    let mut parts = p.push(raw);
                    parts.extend(p.finish());
                    parts
                }
                None => vec![Part::Content(raw.to_string())],
            };
            let (mut content, mut calls) = (String::new(), Vec::new());
            for part in parts {
                match part {
                    Part::Content(t) => content.push_str(&t),
                    Part::Call(c) => calls.push((c.name, serde_json::from_str::<Value>(&c.arguments).unwrap_or(Value::String(c.arguments)))),
                    Part::Thinking(_) => {}
                }
            }
            let want_calls = want_calls(g);
            theirs += !want_calls.is_empty() as usize;
            ours += !calls.is_empty() as usize;
            let want_content = g["message"]["content"].as_str().unwrap_or("");
            let case = format!("parse:{name}");
            let diff = if calls != want_calls {
                Some(format!("tool_calls: sllm {calls:?} vs llama-server {want_calls:?} (raw {raw:?})"))
            } else if content.trim() != want_content.trim() {
                Some(format!("content: sllm {:?} vs llama-server {want_content:?}", content.trim()))
            } else {
                None
            };
            match (diff, known(&fam, &case)) {
                (None, None) => ok_n += 1,
                (None, Some(why)) => bad.push(format!("{fam}/{case}: listed as known ({why}) but now matches")),
                (Some(d), Some(_)) => {
                    if first.is_empty() {
                        first = format!("{name}: {d}");
                    }
                }
                (Some(d), None) => {
                    if first.is_empty() {
                        first = format!("{name}: {d}");
                    }
                    bad.push(format!("{fam}/{case}: {d}"));
                }
            }
        }
        // not vacuous: the fixtures hold the calls we say they hold, and sllm found as many as are not listed as known
        if theirs != expected_with_calls {
            bad.push(format!("{fam}: {theirs} recorded generations hold a call on llama-server's side, FORMATS says {expected_with_calls}"));
        }
        let known_calls = KNOWN.iter().filter(|(f, c, _)| *f == fam && c.starts_with("parse:")).count();
        if format.is_some() && ours + known_calls < expected_with_calls {
            bad.push(format!("{fam}: sllm found calls in {ours} generations, llama-server in {theirs} ({known_calls} listed as known)"));
        }
        table.push(format!(
            "{fam:<9} format {:<13} generations {n}  match {ok_n}  with a call: llama-server {theirs}, sllm {ours}  {first}",
            format.map_or("none", |f| f.name())
        ));
    }
    eprintln!("{}", table.join("\n"));
    assert!(bad.is_empty(), "parsed output differs from llama-server:\n{}", bad.join("\n"));
}
