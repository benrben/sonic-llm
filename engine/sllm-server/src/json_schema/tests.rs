use super::*;
use serde_json::json;

fn rule<'a>(g: &'a str, name: &str) -> &'a str {
    g.lines().find_map(|l| l.strip_prefix(&format!("{name} ::= "))).unwrap_or_else(|| panic!("no rule {name} in:\n{g}"))
}

#[test]
fn an_object_with_required_properties_is_llama_cpps_grammar() {
    let g = schema_to_grammar(&json!({"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]})).unwrap();
    assert_eq!(rule(&g, "root"), "\"{\" space name-kv \",\" space age-kv space \"}\"");
    assert_eq!(rule(&g, "name-kv"), "\"\\\"name\\\"\" space \":\" space string");
    assert_eq!(rule(&g, "age-kv"), "\"\\\"age\\\"\" space \":\" space integer");
    assert_eq!(rule(&g, "integer"), "(\"-\"? integral-part)");
    assert_eq!(rule(&g, "space"), "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}");
    assert!(!g.contains("decimal-part"), "an integer needs no decimal part");
}

#[test]
fn optional_properties_and_additional_properties() {
    let g = schema_to_grammar(&json!({"type": "object", "properties": {"a": {"type": "boolean"}, "b": {"type": "null"}}, "required": ["a"]})).unwrap();
    assert_eq!(rule(&g, "root"), "\"{\" space a-kv ( \",\" space ( b-kv ) )? space \"}\"");
    let g = schema_to_grammar(&json!({"type": "object", "properties": {"a": {"type": "boolean"}}, "additionalProperties": {"type": "integer"}})).unwrap();
    assert!(rule(&g, "additional-k").starts_with("[\"] ( [a] char+ | [^\"a] char* )? [\"]"), "{g}");
    assert_eq!(rule(&g, "additional-kv"), "additional-k \":\" space integer");
}

#[test]
fn an_empty_schema_or_a_bare_object_is_any_object() {
    for s in [json!({}), json!({"type": "object"})] {
        let g = schema_to_grammar(&s).unwrap();
        assert_eq!(rule(&g, "root"), "object", "{s}");
        assert!(rule(&g, "object").starts_with("\"{\" space ( string"));
    }
}

#[test]
fn arrays_strings_enums_unions_and_refs() {
    let g = schema_to_grammar(&json!({"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3})).unwrap();
    assert_eq!(rule(&g, "root"), "\"[\" space string (\",\" space string){0,2} space \"]\"");
    let g = schema_to_grammar(&json!({"enum": ["yes", "no", 3, null]})).unwrap();
    assert_eq!(rule(&g, "root"), "(\"\\\"yes\\\"\" | \"\\\"no\\\"\" | \"3\" | \"null\")");
    let g = schema_to_grammar(&json!({"type": ["string", "null"]})).unwrap();
    assert_eq!(rule(&g, "root"), "string | null");
    let g = schema_to_grammar(&json!({"type": "string", "minLength": 2, "maxLength": 4})).unwrap();
    assert_eq!(rule(&g, "root"), "\"\\\"\" char{2,4} \"\\\"\"");
    let g = schema_to_grammar(&json!({"$defs": {"n": {"type": "integer"}}, "type": "array", "items": {"$ref": "#/$defs/n"}})).unwrap();
    assert_eq!(rule(&g, "item"), "integer");
}

#[test]
fn recursive_refs_terminate() {
    let g = schema_to_grammar(&json!({"$defs": {"t": {"type": "object", "properties": {"kids": {"type": "array", "items": {"$ref": "#/$defs/t"}}}}}, "$ref": "#/$defs/t"})).unwrap();
    assert!(rule(&g, "root").starts_with("ref-defs-t"), "{g}");
}

#[test]
fn what_the_grammar_would_not_enforce_is_refused() {
    for (s, what) in [
        (json!({"type": "string", "pattern": "^a+$"}), "pattern"),
        (json!({"allOf": [{"type": "object"}]}), "allOf"),
        (json!({"type": "integer", "minimum": 1}), "minimum"),
        (json!({"type": "banana"}), "unrecognized type"),
        (json!({"$ref": "http://x/y"}), "unsupported $ref"),
        (json!("x"), "schema must be an object"),
    ] {
        let e = schema_to_grammar(&s).unwrap_err();
        assert!(e.contains(what), "{s}: {e}");
    }
}

/// libllama itself (its vocabulary only: no weights, no GPU) builds the grammars and walks valid texts through them: every
/// token of a valid text is taken, and an invalid text is refused somewhere. Skipped when the small GGUF is not on this machine.
#[test]
fn libllama_takes_what_the_schemas_allow_and_refuses_the_rest() {
    let path = std::path::Path::new("/Users/benreich/Projects/crazyidea/research/inference_runtime/converted/qwen3-0.6b-Q4_0.gguf");
    let Ok(vocab) = llama_ref::Vocab::load(path) else {
        eprintln!("SKIP: {} not found", path.display());
        return;
    };
    let person = json!({"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2}}, "required": ["name", "age"]});
    let cases = [
        (person.clone(), r#"{"name": "Bob", "age": 31}"#, r#"{"name": "Bob", "age": "x"}"#),
        (person, r#"{"name": "A", "age": -2, "tags": ["x", "y"]}"#, r#"{"name": "A", "age": 2, "tags": ["x", "y", "z"]}"#),
        (json!({"type": "object"}), r#"{"a": [1, true, null, {"b": "c"}]}"#, "[1]"),
        (json!({"enum": ["yes", "no"]}), r#""yes""#, r#""maybe""#),
        (json!({"type": ["integer", "null"]}), "null", "true"),
        (json!({"type": "string", "format": "date"}), r#""2026-11-13""#, r#""13/11/2026""#),
        (json!({"type": "array", "items": {"type": "number"}, "minItems": 2}), "[1.5, -2e3]", "[1]"),
        (json!({"$defs": {"n": {"type": "boolean"}}, "type": "object", "properties": {"ok": {"$ref": "#/$defs/n"}}, "required": ["ok"]}), r#"{"ok": true}"#, r#"{"ok": 1}"#),
    ];
    for (schema, good, bad) in cases {
        let g = schema_to_grammar(&schema).unwrap();
        let walk = |text: &str| -> bool {
            let mut gram = crate::llama_engine::Gram::new(vocab.as_ptr(), &g).unwrap_or_else(|| panic!("libllama refused the grammar of {schema}:\n{g}"));
            vocab.tokenize(text, false, false).unwrap().iter().all(|&t| gram.takes(t))
        };
        assert!(walk(good), "the grammar of {schema} refused {good}:\n{g}");
        assert!(!walk(bad), "the grammar of {schema} took {bad}:\n{g}");
    }
}
