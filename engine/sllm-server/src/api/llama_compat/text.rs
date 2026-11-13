//! `POST /tokenize` (with `with_pieces`), `POST /detokenize` and `POST /apply-template`, as llama-server has them.

use super::*;

#[derive(Deserialize)]
struct TokenizeReq {
    content: String,
    #[serde(default)]
    add_special: bool,
    #[serde(default)]
    parse_special: bool,
    #[serde(default)]
    with_pieces: bool,
}

/// A token's piece as llama-server writes it: the text when it is valid UTF-8, else the list of its byte values.
fn piece_json(bytes: Vec<u8>) -> Value {
    match String::from_utf8(bytes) {
        Ok(s) => json!(s),
        Err(e) => json!(e.into_bytes()),
    }
}

pub(super) async fn tokenize(State(st): State<St>, body: Bytes) -> Result<Json<Value>, ApiError> {
    let r: TokenizeReq = parse_body(&body, false)?;
    let ids = encode(&st, r.content, r.add_special, r.parse_special).await?;
    if !r.with_pieces {
        return Ok(Json(json!({"tokens": ids})));
    }
    let tokens: Vec<Value> = ids.iter().map(|&id| json!({"id": id, "piece": piece_json(st.tokenizer.piece(id, true))})).collect();
    Ok(Json(json!({"tokens": tokens})))
}

#[derive(Deserialize)]
struct DetokenizeReq {
    #[serde(default)]
    tokens: Vec<i32>,
}

/// The text of the tokens, control tokens printed (llama-server's `tokens_to_str`).
pub(super) async fn detokenize(State(st): State<St>, body: Bytes) -> Result<Json<Value>, ApiError> {
    let r: DetokenizeReq = parse_body(&body, false)?;
    let bytes: Vec<u8> = r.tokens.iter().flat_map(|&id| st.tokenizer.piece(id, true)).collect();
    Ok(Json(json!({"content": String::from_utf8_lossy(&bytes)})))
}

/// The prompt the chat route would feed the model for `messages` (and `tools`), as text: the model's chat template with the
/// generation prompt, with thinking off as on `/v1/chat/completions` unless `chat_template_kwargs` says `enable_thinking` true.
pub(super) async fn apply_template(State(st): State<St>, body: Bytes) -> Result<Json<Value>, ApiError> {
    let v: Value = parse_body(&body, false)?;
    let messages = match v.get("messages") {
        None | Some(Value::Null) => return Err(ApiError::new(400, "invalid_request_error", "'messages' is required")),
        Some(Value::Array(a)) => a.clone(),
        Some(_) => return Err(ApiError::new(400, "invalid_request_error", "Expected 'messages' to be an array")),
    };
    let tools = v.get("tools").and_then(Value::as_array).cloned().unwrap_or_default();
    let tmpl = chat_template_cached(&st).map_err(|e| ApiError::new(501, "not_supported_error", format!("the model's chat template is not available: {e}")))?;
    let thinking = enable_thinking(&v.get("chat_template_kwargs").cloned())?;
    let prompt = tmpl.render_chat(&messages, &tools, true, thinking).map_err(|e| ApiError::new(400, "invalid_request_error", e.to_string()))?;
    Ok(Json(json!({"prompt": prompt})))
}
