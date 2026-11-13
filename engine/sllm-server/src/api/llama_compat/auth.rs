//! `--api-key`: a key check in front of every route but the health checks, as llama-server does it.

use super::*;
use axum::extract::Request;
use axum::http::header;
use axum::middleware::Next;

/// The routes that need no key (llama-server's public endpoints).
const PUBLIC: &[&str] = &["/health", "/v1/health"];

/// The key a request carries: `Authorization` (with or without `Bearer `), else `X-Api-Key` (the Anthropic header).
fn presented_key(req: &Request) -> String {
    let get = |name: header::HeaderName| req.headers().get(name).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    let mut key = get(header::AUTHORIZATION);
    if key.is_empty() {
        key = get(header::HeaderName::from_static("x-api-key"));
    }
    key.strip_prefix("Bearer ").map(str::to_string).unwrap_or(key)
}

/// Whether `req` may pass: no keys configured, a public route, or one of the keys.
fn allowed(keys: &[String], req: &Request) -> bool {
    keys.is_empty() || PUBLIC.contains(&req.uri().path()) || keys.contains(&presented_key(req))
}

pub(in crate::api) async fn require_key(State(st): State<St>, req: Request, next: Next) -> Response {
    if allowed(&st.compat.api_keys, &req) {
        return next.run(req).await;
    }
    ApiError::new(401, "authentication_error", "Invalid API Key").into_response()
}

/// The keys of `--api-key` (comma separated) and the lines of `--api-key-file` (one per line; `#` starts a comment line).
pub fn parse_keys(inline: Option<&str>, file_text: Option<&str>) -> Vec<String> {
    let from_flag = inline.into_iter().flat_map(|s| s.split(',')).map(str::trim).filter(|k| !k.is_empty());
    let from_file = file_text.into_iter().flat_map(str::lines).map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#'));
    from_flag.chain(from_file).map(str::to_string).collect()
}
