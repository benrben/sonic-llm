#![allow(dead_code)]
//! Shared test harness: an in-process sllm-server on a free port with a FakeEngine, a reqwest client, SSE parsing.

use serde_json::Value;
use sllm_server::engine::Engine;
use sllm_server::fake::{FakeConfig, FakeEngine};
use sllm_server::tokenizer::Tokenizer;
use sllm_server::{start, Running, ServerConfig};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub struct Srv {
    pub run: Running,
    pub http: reqwest::Client,
}

pub fn fast() -> FakeConfig {
    FakeConfig { prefill_ms: 0.0, prefill_ms_per_token: 0.0, token_ms: 0.0, ..Default::default() }
}

pub fn spawn(fake: FakeConfig) -> Srv {
    spawn_with(fake, Tokenizer::Bytes, ServerConfig { port: 0, ..Default::default() })
}

pub fn spawn_with(fake: FakeConfig, tok: Tokenizer, cfg: ServerConfig) -> Srv {
    let factory: sllm_server::executor::EngineFactory = Box::new(move || Ok(Box::new(FakeEngine::new(fake)) as Box<dyn Engine>));
    let run = start(cfg, tok, factory).expect("start");
    run.wait_ready(Duration::from_secs(10)).expect("ready");
    Srv { run, http: reqwest::Client::builder().no_proxy().build().unwrap() }
}

impl Srv {
    pub fn url(&self, path: &str) -> String {
        format!("{}{}", self.run.url(), path)
    }

    pub async fn post(&self, path: &str, body: &Value) -> reqwest::Response {
        self.http.post(self.url(path)).json(body).send().await.unwrap()
    }

    pub async fn post_raw(&self, path: &str, body: &str) -> reqwest::Response {
        self.http.post(self.url(path)).header("content-type", "application/json").body(body.to_string()).send().await.unwrap()
    }

    pub async fn get(&self, path: &str) -> reqwest::Response {
        self.http.get(self.url(path)).send().await.unwrap()
    }

    pub async fn get_json(&self, path: &str) -> Value {
        self.get(path).await.json().await.unwrap()
    }

    /// POST /completion with stream false; returns the parsed JSON.
    pub async fn complete(&self, ids: &[i32], n: i64, extra: Value) -> Value {
        let mut b = serde_json::json!({"prompt": ids, "n_predict": n, "temperature": 0.0, "ignore_eos": true, "return_tokens": true});
        for (k, v) in extra.as_object().into_iter().flatten() {
            b[k] = v.clone();
        }
        let r = self.post("/completion", &b).await;
        assert_eq!(r.status(), 200);
        r.json().await.unwrap()
    }

    /// POST streamed, read everything; returns the raw body and the parsed chunks.
    pub async fn stream(&self, path: &str, body: &Value) -> (String, Vec<Value>) {
        let r = self.post(path, body).await;
        assert_eq!(r.status(), 200, "{path}");
        let raw = r.text().await.unwrap();
        let chunks = sse_json(&raw);
        (raw, chunks)
    }

    pub async fn metrics(&self) -> Value {
        self.get_json("/metrics").await
    }

    /// Poll /metrics until `f` is true.
    pub async fn wait_metrics(&self, what: &str, f: impl Fn(&Value) -> bool) -> Value {
        let t0 = Instant::now();
        loop {
            let m = self.metrics().await;
            if f(&m) {
                return m;
            }
            assert!(t0.elapsed() < Duration::from_secs(5), "timed out waiting for {what}: {m}");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

/// Split an SSE body into its `data:` payloads; asserts the framing (`data: ...\n\n`, nothing else).
pub fn sse_events(raw: &str) -> Vec<String> {
    assert!(raw.is_empty() || raw.ends_with("\n\n"), "SSE body must end with a blank line: {raw:?}");
    raw.split("\n\n")
        .filter(|e| !e.is_empty())
        .map(|e| {
            let p = e.strip_prefix("data: ").unwrap_or_else(|| panic!("event without `data: ` prefix: {e:?}"));
            assert!(!p.contains('\n'), "multi-line event: {e:?}");
            p.to_string()
        })
        .collect()
}

pub fn sse_json(raw: &str) -> Vec<Value> {
    sse_events(raw).iter().filter(|p| p.as_str() != "[DONE]").map(|p| serde_json::from_str(p).expect("event is JSON")).collect()
}

pub fn tokens_of(chunks: &[Value]) -> Vec<i64> {
    chunks.iter().flat_map(|c| c["tokens"].as_array().unwrap().iter().map(|t| t.as_i64().unwrap())).collect()
}

pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
}

pub fn qwen_gguf() -> Option<PathBuf> {
    let p = PathBuf::from("/Users/benreich/Projects/crazyidea/research/inference_runtime/converted/qwen3-0.6b-Q4_0.gguf");
    if p.is_file() {
        Some(p)
    } else {
        eprintln!("SKIP: qwen3-0.6b-Q4_0.gguf not found");
        None
    }
}
