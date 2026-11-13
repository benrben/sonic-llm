//! llama-server interface compatibility: the routes, names and fields a client written for `llama-server` expects, over the same
//! handlers as the native ones. The gap report that drove it is `sllm/results/interface-2026-11-13/report-before.md`.
//!
//! * Aliases with the same handler: `GET /v1/health`, `POST /completions`, `POST /chat/completions`, `GET /models`.
//! * Read-only endpoints: `GET /props`, `GET /slots`, `GET /lora-adapters` (always `[]`), `POST /detokenize`, `POST /apply-template`,
//!   and `with_pieces` on `POST /tokenize` ([`props`], [`text`]).
//! * What this server does not do answers llama-server's 501 `not_supported_error`: embeddings, reranking, changing `/props`.
//! * `--api-key` ([`auth`]), `--alias` and `--reasoning-format` ([`CompatConfig`]); `system_fingerprint`, `timings`,
//!   `usage.prompt_tokens_details` and `logprobs` on the replies ([`wire`]).
//!
//! Not implemented, still 404: `POST /infill`, `POST /v1/responses`, `POST /slots/{id}?action=save|restore|erase`, and images.

use super::*;

pub mod auth;
mod config;
mod constrain;
mod props;
mod text;
mod wire;

pub use config::{CompatConfig, ReasoningFormat};
pub(super) use constrain::grammar_for;
pub(super) use props::{model_name, model_object, models_array};
pub(super) use wire::*;

pub(super) fn routes() -> Router<St> {
    Router::new()
        .route("/v1/health", get(health))
        .route("/completions", post(completion))
        .route("/models", get(anthropic_api::models))
        .route("/lora-adapters", get(props::lora_adapters))
        .route("/props", get(props::props).post(props::set_props))
        .route("/slots", get(props::slots))
        .route("/tokenize", post(text::tokenize))
        .route("/detokenize", post(text::detokenize))
        .route("/apply-template", post(text::apply_template))
        .route("/embedding", post(embeddings))
        .route("/embeddings", post(embeddings))
        .route("/v1/embeddings", post(embeddings))
        .route("/rerank", post(rerank))
        .route("/reranking", post(rerank))
        .route("/v1/rerank", post(rerank))
        .route("/v1/reranking", post(rerank))
}

fn not_supported(what: &str) -> ApiError {
    ApiError::new(501, "not_supported_error", format!("This server does not support {what}: sllm-server generates text only"))
}

async fn embeddings() -> ApiError {
    not_supported("embeddings")
}

async fn rerank() -> ApiError {
    not_supported("reranking")
}

#[cfg(test)]
mod tests {
    use super::auth::parse_keys;

    #[test]
    fn keys_come_from_the_flag_and_the_file() {
        assert_eq!(parse_keys(Some("a, b,,c"), Some("# comment\nd\n\n  e  \n")), ["a", "b", "c", "d", "e"]);
        assert!(parse_keys(None, None).is_empty());
    }
}
