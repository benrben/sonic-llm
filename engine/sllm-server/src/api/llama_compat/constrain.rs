//! `json_schema` and `response_format`: constrained output through the grammar the engine already has.
//!
//! The schema becomes a GBNF grammar ([`crate::json_schema`], a port of llama.cpp's converter) and rides the request's `grammar`
//! field, so the engines enforce it exactly as they enforce a grammar sent by hand.

use super::*;
use crate::json_schema::schema_to_grammar;

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError::new(400, "invalid_request_error", msg)
}

/// null, "", [], {}, false and 0 are "not asked for" (as for every field of [`REFUSED_FIELDS`]).
fn is_unset(v: &Option<Value>) -> bool {
    v.as_ref().is_none_or(|x| is_not_asked_for("json_schema", x))
}

/// The schema `response_format` asks for: `None` for text (or nothing). `json_object` is any object unless it carries a `schema`;
/// `json_schema` is `json_schema.schema` (llama-server also takes the schema beside the type).
fn schema_of_format(rf: &Value) -> Result<Option<Value>, ApiError> {
    match rf.get("type").and_then(Value::as_str) {
        None | Some("") | Some("text") => Ok(None),
        Some("json_object") => Ok(Some(rf.get("schema").cloned().unwrap_or_else(|| json!({})))),
        Some("json_schema") => rf
            .pointer("/json_schema/schema")
            .or_else(|| rf.get("schema"))
            .cloned()
            .map(Some)
            .ok_or_else(|| bad("response_format of type json_schema needs `json_schema.schema`")),
        Some(other) => Err(bad(format!("response_format type must be one of \"text\", \"json_object\" or \"json_schema\", but got: {other}"))),
    }
}

/// The grammar of a request: its `grammar`, or the one `json_schema` / `response_format` stands for (`None`: unconstrained).
/// Both a grammar and a schema is 400, as in llama-server; so is a schema the converter cannot express.
pub(in crate::api) fn grammar_for(grammar: &Option<String>, json_schema: &Option<Value>, response_format: &Option<Value>) -> Result<Option<String>, ApiError> {
    let grammar = grammar.clone().filter(|g| !g.is_empty());
    let mut schema = if is_unset(json_schema) { None } else { json_schema.clone() };
    if let Some(rf) = response_format.as_ref().filter(|v| v.is_object()) {
        schema = schema_of_format(rf)?.or(schema);
    }
    match (grammar, schema) {
        (Some(_), Some(_)) => Err(bad("Cannot use both json_schema (or response_format) and grammar")),
        (g, None) => Ok(g),
        (None, Some(s)) => schema_to_grammar(&s).map(Some).map_err(|e| bad(format!("JSON schema conversion failed: {e}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(grammar: Option<&str>, js: Option<Value>, rf: Option<Value>) -> Result<Option<String>, ApiError> {
        grammar_for(&grammar.map(str::to_string), &js, &rf)
    }

    #[test]
    fn nothing_asked_is_unconstrained() {
        assert_eq!(g(None, None, None).unwrap(), None);
        assert_eq!(g(Some(""), Some(json!({})), Some(json!({"type": "text"}))).unwrap(), None);
        assert_eq!(g(Some("root ::= \"a\""), None, Some(json!({}))).unwrap().as_deref(), Some("root ::= \"a\""));
    }

    #[test]
    fn response_format_kinds() {
        let obj = g(None, None, Some(json!({"type": "json_object"}))).unwrap().unwrap();
        assert!(obj.contains("root ::= object"), "{obj}");
        let schema = json!({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]});
        let wrapped = g(None, None, Some(json!({"type": "json_schema", "json_schema": {"name": "x", "schema": schema.clone()}}))).unwrap();
        assert_eq!(wrapped, g(None, Some(schema), None).unwrap());
        assert!(g(None, None, Some(json!({"type": "json_schema"}))).is_err());
        assert!(g(None, None, Some(json!({"type": "xml"}))).is_err());
    }

    #[test]
    fn a_grammar_and_a_schema_together_are_refused() {
        let e = g(Some("root ::= \"a\""), Some(json!({"type": "string"})), None).unwrap_err();
        assert_eq!(e.0.status, 400);
        assert!(g(Some("root ::= \"a\""), None, Some(json!({"type": "json_object"}))).is_err());
    }

    #[test]
    fn an_unsupported_schema_is_400() {
        let e = g(None, Some(json!({"type": "string", "pattern": "^a$"})), None).unwrap_err();
        assert_eq!(e.0.status, 400);
        assert!(e.0.message.contains("pattern"));
    }
}
