//! HTTP routes of the helper lane: `POST /v1/helper/{classify,extract,compress,summarize}` and `GET /v1/helper`.
//!
//! Without `--helper-model` they answer 404 with a message saying so. Errors have the server's usual shape.

use super::*;
use crate::api::{ApiError, St};
use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

pub fn routes() -> Router<St> {
    Router::new()
        .route("/v1/helper", get(status))
        .route("/v1/helper/classify", post(run_request::<ClassifyReq>))
        .route("/v1/helper/extract", post(run_request::<ExtractReq>))
        .route("/v1/helper/compress", post(run_request::<CompressReq>))
        .route("/v1/helper/summarize", post(run_request::<SummarizeReq>))
}

fn helper(st: &St) -> Result<&Helper, ApiError> {
    st.helper.as_deref().ok_or_else(|| ApiError::new(404, "not_found_error", "no helper model: start sllm-server with --helper-model GGUF"))
}

fn parse<T: serde::de::DeserializeOwned>(body: &Bytes) -> Result<T, ApiError> {
    let v: Value = serde_json::from_slice(body).map_err(|e| ApiError::new(400, "invalid_request_error", format!("invalid JSON: {e}")))?;
    serde_json::from_value(v).map_err(|e| ApiError::invalid(e.to_string()))
}

async fn run_op(st: &St, op: Op) -> Response {
    let h = match helper(st) {
        Ok(h) => h,
        Err(e) => return e.into_response(),
    };
    match h.call(op).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => ApiError(e).into_response(),
    }
}

async fn status(State(st): State<St>) -> Response {
    match helper(&st) {
        Ok(h) => Json(json!({"info": h.shared.info.get(), "brain_lane_info": h.brain_shared.as_ref().and_then(|b| b.info.get()),
            "metrics": h.metrics(), "config": h.cfg.to_json()})).into_response(),
        Err(e) => e.into_response(),
    }
}

/// A helper operation's route: the body read as the operation's request, then run.
async fn run_request<T: serde::de::DeserializeOwned + Into<Op>>(State(st): State<St>, body: Bytes) -> Response {
    match parse::<T>(&body) {
        Ok(r) => run_op(&st, r.into()).await,
        Err(e) => e.into_response(),
    }
}
