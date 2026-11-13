//! `GET /props`, `GET /slots` and the model object of `GET /v1/models` / `GET /models`, in llama-server's shape.

use super::*;

/// The keys of `default_generation_settings.params` (llama-server's `to_json(only_metrics)`): the sampler settings and the
/// generation limits, without the per-request grammar and stop lists.
const PARAMS_OMITTED: &[&str] = &["dry_sequence_breakers", "stop", "logit_bias", "grammar", "grammar_lazy", "grammar_triggers", "preserved_tokens"];

fn default_params() -> Value {
    let params = GenReq::default().params().expect("the default request is valid");
    let mut v = completion_api::generation_settings(&params, false);
    if let Some(o) = v.as_object_mut() {
        PARAMS_OMITTED.iter().for_each(|k| {
            o.remove(*k);
        });
    }
    v
}

fn info_n_ctx(st: &St) -> usize {
    st.shared.info.get().map_or(0, |i| i.n_ctx)
}

fn profile_u64(st: &St, path: &str) -> u64 {
    st.shared.caps.get().and_then(|c| c.pointer(path)).and_then(Value::as_u64).unwrap_or(0)
}

/// The chat template's text, BOS and EOS (empty without a GGUF template).
fn template_parts(st: &St) -> (String, String, String) {
    match chat_template_cached(st) {
        Ok(t) if st.tokenizer.gguf_path().is_some() => (t.source.clone(), t.bos().to_string(), t.eos().to_string()),
        _ => Default::default(),
    }
}

fn build_info() -> String {
    system_fingerprint()
}

pub(super) async fn props(State(st): State<St>) -> Result<Json<Value>, ApiError> {
    let info = st.shared.info.get().ok_or_else(|| ApiError::new(503, "unavailable_error", "Loading model"))?;
    let (template, bos, eos) = template_parts(&st);
    Ok(Json(json!({
        "default_generation_settings": {"params": default_params(), "n_ctx": info.n_ctx},
        "total_slots": st.slots,
        "model_alias": default_model(&st),
        "model_ftype": "",
        "model_path": info.model,
        "modalities": {"vision": false, "video": false, "audio": false},
        "media_marker": "<__media__>",
        "endpoint_slots": true,
        // GET /metrics is JSON here, not Prometheus text: a client that scrapes llama-server's endpoint must not take it for that
        "endpoint_props": false,
        "endpoint_metrics": false,
        "ui": false,
        "ui_settings": {},
        "chat_template": template,
        "chat_template_caps": {},
        "bos_token": bos,
        "eos_token": eos,
        "build_info": build_info(),
        "is_sleeping": false,
        "cors_proxy_enabled": false,
    })))
}

/// `POST /props` changes llama-server's global properties when started with `--props`; there are none to change here.
pub(super) async fn set_props() -> ApiError {
    ApiError::new(501, "not_supported_error", "This server does not support changing global properties")
}

/// One entry per request slot the scheduler runs at once; the first ones are the requests in flight.
pub(super) async fn slots(State(st): State<St>) -> Result<Json<Value>, ApiError> {
    if st.shared.info.get().is_none() {
        return Err(ApiError::new(503, "unavailable_error", "Loading model"));
    }
    let busy = st.shared.active_requests.load(Relaxed) as usize;
    let n_ctx = info_n_ctx(&st);
    Ok(Json(Value::Array((0..st.slots).map(|id| json!({"id": id, "n_ctx": n_ctx, "speculative": false, "is_processing": id < busy})).collect())))
}

pub(super) async fn lora_adapters() -> Json<Value> {
    Json(json!([]))
}

/// The name this server lists its model under: `--alias`, else the GGUF's file name without `.gguf`.
pub(in crate::api) fn model_name(st: &St) -> String {
    if let Some(a) = &st.compat.alias {
        return a.clone();
    }
    let m = st.shared.info.get().map(|i| i.model.clone()).unwrap_or_default();
    std::path::Path::new(&m).file_stem().and_then(|s| s.to_str()).map_or(m.clone(), str::to_string)
}

/// llama-server's model object (`data[]` of `/v1/models`), with what is known about the model; `extra` is merged in
/// (the Anthropic fields the same route has always had).
pub(in crate::api) fn model_object(st: &St, extra: Value) -> Value {
    let info = st.shared.info.get().cloned();
    let (path, n_vocab) = info.as_ref().map_or((String::new(), 0), |i| (i.model.clone(), i.n_vocab));
    let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()).saturating_sub(st.started.elapsed().as_secs());
    let aliases: Vec<String> = std::iter::once(path.clone()).chain(st.compat.aliases.iter().cloned()).collect();
    let mut v = json!({
        "id": model_name(st), "aliases": aliases, "tags": [], "object": "model",
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "created": created, "owned_by": "sllm",
        "meta": {
            "vocab_type": 0, "n_vocab": n_vocab, "n_ctx": info_n_ctx(st), "n_ctx_train": profile_u64(st, "/profile/ctx_train"),
            "n_embd": profile_u64(st, "/profile/embd"), "n_params": 0, "size": profile_u64(st, "/profile/weights/total_bytes"), "ftype": "",
        },
    });
    if let (Some(o), Some(e)) = (v.as_object_mut(), extra.as_object()) {
        e.iter().for_each(|(k, x)| {
            o.insert(k.clone(), x.clone());
        });
    }
    v
}

/// The Ollama-style `models` array llama-server puts beside `data`.
pub(in crate::api) fn models_array(st: &St) -> Value {
    let name = default_model(st);
    json!([{
        "name": name, "model": name, "modified_at": "", "size": "", "digest": "", "type": "model", "description": "", "tags": [""],
        "capabilities": ["completion"], "parameters": "",
        "details": {"parent_model": "", "format": "gguf", "family": "", "families": [""], "parameter_size": "", "quantization_level": ""},
    }])
}
