//! JSON schema to GBNF, for `json_schema` and `response_format` (the grammar goes to the engine like a `grammar` field).
//!
//! A port of the part of llama.cpp's `common/json-schema-to-grammar.cpp` (b11669) that the usual schemas need, with the same
//! rule names and the same text, so the grammar is the one llama-server builds for these schemas. libllama does not export the
//! converter (it is in llama.cpp's `common`), so this is Rust.
//!
//! Handled: `type` (object, array, string, integer, number, boolean, null, and a list of types), `properties` / `required` /
//! `additionalProperties`, `items` / `prefixItems` / `minItems` / `maxItems`, `minLength` / `maxLength`, `format` (date, time,
//! date-time, uuid), `enum`, `const`, `anyOf` / `oneOf`, local `$ref` (`#/...`, also recursive) and a schema without a type.
//!
//! Refused with an error, where llama.cpp converts them: `pattern` (llama.cpp falls back to any string for the patterns it cannot
//! translate; here the constraint would be dropped without a word), `allOf`, and `minimum` / `maximum` on an integer. A schema
//! that asks for something the grammar would not enforce is refused rather than half applied.

use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

const SPACE_RULE: &str = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

/// (name, body, rules it needs): llama.cpp's PRIMITIVE_RULES and STRING_FORMAT_RULES.
const PRIMITIVES: &[(&str, &str, &[&str])] = &[
    ("boolean", "(\"true\" | \"false\")", &[]),
    ("decimal-part", "[0-9]{1,16}", &[]),
    ("integral-part", "[0] | [1-9] [0-9]{0,15}", &[]),
    ("number", "(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?", &["integral-part", "decimal-part"]),
    ("integer", "(\"-\"? integral-part)", &["integral-part"]),
    ("value", "object | array | string | number | boolean | null", &["object", "array", "string", "number", "boolean", "null"]),
    ("object", "\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"", &["string", "value"]),
    ("array", "\"[\" space ( value (\",\" space value)* )? space \"]\"", &["value"]),
    ("uuid", "\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\"", &[]),
    ("char", "[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})", &[]),
    ("string", "\"\\\"\" char* \"\\\"\"", &["char"]),
    ("null", "\"null\"", &[]),
    ("date", "[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )", &[]),
    ("time", "([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )", &[]),
    ("date-time", "date \"T\" time", &["date", "time"]),
    ("date-string", "\"\\\"\" date \"\\\"\"", &["date"]),
    ("time-string", "\"\\\"\" time \"\\\"\"", &["time"]),
    ("date-time-string", "\"\\\"\" date-time \"\\\"\"", &["date-time"]),
];

/// The GBNF grammar (root rule `root`) for `schema`; `Err` says what the schema asks for that is not supported.
pub fn schema_to_grammar(schema: &Value) -> Result<String, String> {
    // an absent or empty schema means any object (llama-server)
    let schema = if schema.as_object().is_some_and(|o| o.is_empty()) { serde_json::json!({"type": "object"}) } else { schema.clone() };
    let mut c = Converter { root: &schema, rules: BTreeMap::new(), resolving: HashSet::new() };
    c.rules.insert("space".into(), SPACE_RULE.into());
    c.visit(&schema, "", "#")?;
    Ok(c.rules.iter().map(|(k, v)| format!("{k} ::= {v}\n")).collect())
}

fn is_reserved(name: &str) -> bool {
    name == "root" || PRIMITIVES.iter().any(|(n, _, _)| *n == name)
}

/// A rule name made of letters, digits and dashes.
fn sanitize(name: &str) -> String {
    let mut out = String::new();
    let mut in_bad = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' {
            out.push(ch);
            in_bad = false;
        } else if !in_bad {
            out.push('-');
            in_bad = true;
        }
    }
    out
}

/// A string as a GBNF literal.
fn literal(s: &str) -> String {
    let mut out = String::from("\"");
    for ch in s.chars() {
        match ch {
            '\r' => out.push_str("\\r"),
            '\n' => out.push_str("\\n"),
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn constant(v: &Value) -> String {
    literal(&v.to_string())
}

fn repetition(item: &str, min: u64, max: Option<u64>, separator: &str) -> String {
    if max == Some(0) {
        return String::new();
    }
    if min == 0 && max == Some(1) {
        return format!("{item}?");
    }
    if separator.is_empty() {
        return match (min, max) {
            (1, None) => format!("{item}+"),
            (0, None) => format!("{item}*"),
            (_, Some(m)) => format!("{item}{{{min},{m}}}"),
            (_, None) => format!("{item}{{{min},}}"),
        };
    }
    let inner = repetition(&format!("({separator} {item})"), min.saturating_sub(1), max.map(|m| m - 1), "");
    let result = format!("{item} {inner}");
    if min == 0 {
        format!("({result})?")
    } else {
        result
    }
}

/// A trie of the names a string must not be (`additionalProperties` next to `properties`).
#[derive(Default)]
struct Trie {
    children: BTreeMap<char, Trie>,
    end: bool,
}

impl Trie {
    fn of(names: &[String]) -> Trie {
        let mut root = Trie::default();
        for n in names {
            let mut node = &mut root;
            for ch in n.chars() {
                node = node.children.entry(ch).or_default();
            }
            node.end = true;
        }
        root
    }

    fn write(&self, char_rule: &str, out: &mut String) {
        let mut rejects = String::new();
        for (i, (ch, child)) in self.children.iter().enumerate() {
            rejects.push(*ch);
            if i > 0 {
                out.push_str(" | ");
            }
            out.push_str(&format!("[{ch}]"));
            if child.children.is_empty() {
                out.push_str(&format!(" {char_rule}+"));
            } else {
                out.push_str(" (");
                child.write(char_rule, out);
                out.push(')');
            }
        }
        if !self.children.is_empty() {
            out.push_str(&format!(" | [^\"{rejects}] {char_rule}*"));
        }
    }
}

struct Converter<'a> {
    root: &'a Value,
    rules: BTreeMap<String, String>,
    resolving: HashSet<String>,
}

fn fail<T>(path: &str, msg: impl std::fmt::Display) -> Result<T, String> {
    Err(format!("JSON schema error at {path}: {msg}"))
}

/// The schema of the keys beside `properties`: `Some({})` any value, `Some(schema)` that schema, `None` no other keys.
fn additional_of(schema: &Value, path: &str) -> Result<Option<Value>, String> {
    match schema.get("additionalProperties") {
        Some(Value::Bool(true)) => Ok(Some(Value::Object(Default::default()))),
        Some(Value::Bool(false)) => Ok(None),
        Some(v @ Value::Object(_)) => Ok(Some(v.clone())),
        Some(_) => fail(path, "additionalProperties must be a boolean or a schema"),
        // {"type": "object"} on its own accepts any object
        None if schema.get("properties").is_none() => Ok(Some(Value::Object(Default::default()))),
        None => Ok(None),
    }
}

fn count(schema: &Value, key: &str, path: &str) -> Result<Option<u64>, String> {
    match schema.get(key) {
        None => Ok(None),
        Some(v) => v.as_u64().map(Some).map_or_else(|| fail(path, format!("{key} must be a non-negative integer")), Ok),
    }
}

impl Converter<'_> {
    fn add_rule(&mut self, name: &str, rule: &str) -> String {
        let esc = sanitize(name);
        if self.rules.get(&esc).is_none_or(|r| r == rule) {
            self.rules.insert(esc.clone(), rule.to_string());
            return esc;
        }
        let mut i = 0;
        while self.rules.get(&format!("{esc}{i}")).is_some_and(|r| r != rule) {
            i += 1;
        }
        let key = format!("{esc}{i}");
        self.rules.insert(key.clone(), rule.to_string());
        key
    }

    fn add_primitive(&mut self, ty: &str) -> String {
        self.add_primitive_as(ty, ty)
    }

    /// The primitive `ty` stored as rule `name` (`root` when the whole schema is that primitive), and the rules it needs.
    fn add_primitive_as(&mut self, name: &str, ty: &str) -> String {
        let (_, body, deps) = PRIMITIVES.iter().find(|(n, _, _)| *n == ty).expect("a known primitive");
        let n = self.add_rule(name, body);
        for dep in *deps {
            if !self.rules.contains_key(*dep) {
                self.add_primitive(dep);
            }
        }
        n
    }

    fn primitive(&mut self, rule_name: &str, ty: &str) -> String {
        self.add_primitive_as(if rule_name == "root" { "root" } else { ty }, ty)
    }

    fn visit(&mut self, schema: &Value, name: &str, path: &str) -> Result<String, String> {
        let rule_name = if is_reserved(name) { format!("{name}-") } else if name.is_empty() { "root".to_string() } else { name.to_string() };
        if !schema.is_object() {
            return fail(path, "schema must be an object");
        }
        if let Some(r) = schema.get("$ref") {
            let target = self.resolve_ref(r, path)?;
            return Ok(self.add_rule(&rule_name, &target));
        }
        for key in ["oneOf", "anyOf"] {
            if let Some(alts) = schema.get(key) {
                let union = self.union_of(alts.as_array(), name, &format!("{path}/{key}"))?;
                return Ok(self.add_rule(&rule_name, &union));
            }
        }
        if let Some(Value::Array(types)) = schema.get("type") {
            return self.visit_type_list(schema, types, name, &rule_name, path);
        }
        if let Some(v) = schema.get("const") {
            return Ok(self.add_rule(&rule_name, &constant(v)));
        }
        if let Some(values) = schema.get("enum") {
            return self.visit_enum(values, &rule_name, path);
        }
        self.visit_typed(schema, name, &rule_name, path)
    }

    fn union_of(&mut self, alts: Option<&Vec<Value>>, name: &str, path: &str) -> Result<String, String> {
        let alts = match alts {
            Some(a) if !a.is_empty() => a,
            _ => return fail(path, "must be a non-empty array of schemas"),
        };
        let mut rules = Vec::new();
        for (i, alt) in alts.iter().enumerate() {
            let sub = format!("{name}{}{i}", if name.is_empty() { "alternative-" } else { "-" });
            rules.push(self.visit(alt, &sub, &format!("{path}/{i}"))?);
        }
        Ok(rules.join(" | "))
    }

    fn visit_type_list(&mut self, schema: &Value, types: &[Value], name: &str, rule_name: &str, path: &str) -> Result<String, String> {
        if types.is_empty() {
            return fail(path, "type must not be empty");
        }
        let alts: Vec<Value> = types
            .iter()
            .map(|t| {
                let mut alt = schema.clone();
                alt["type"] = t.clone();
                alt
            })
            .collect();
        let union = self.union_of(Some(&alts), name, &format!("{path}/type"))?;
        Ok(self.add_rule(rule_name, &union))
    }

    fn visit_enum(&mut self, values: &Value, rule_name: &str, path: &str) -> Result<String, String> {
        match values.as_array() {
            Some(a) if !a.is_empty() => {
                let lits: Vec<String> = a.iter().map(constant).collect();
                Ok(self.add_rule(rule_name, &format!("({})", lits.join(" | "))))
            }
            _ => fail(path, "enum must be a non-empty array"),
        }
    }

    fn visit_typed(&mut self, schema: &Value, name: &str, rule_name: &str, path: &str) -> Result<String, String> {
        if schema.get("allOf").is_some() {
            return fail(path, "allOf is not supported");
        }
        let ty = match schema.get("type") {
            None | Some(Value::Null) => self.implied_type(schema),
            Some(Value::String(s)) => s.as_str(),
            Some(_) => return fail(path, "type must be a string or an array of strings"),
        };
        match ty {
            "object" => self.visit_object(schema, name, rule_name, path),
            "array" => self.visit_array(schema, name, rule_name, path),
            "string" => self.visit_string(schema, rule_name, path),
            "integer" => self.visit_integer(schema, rule_name, path),
            "number" | "boolean" | "null" => Ok(self.primitive(rule_name, ty)),
            "" => {
                let v = self.add_primitive("value");
                Ok(self.add_rule(rule_name, &v))
            }
            other => fail(path, format!("unrecognized type {other}")),
        }
    }

    /// The type the structural keywords of a schema without `type` stand for ("" when it says nothing: any value).
    fn implied_type(&self, schema: &Value) -> &'static str {
        let has = |k: &str| schema.get(k).is_some();
        if has("properties") || schema.get("additionalProperties").is_some_and(|v| *v != Value::Bool(true)) {
            "object"
        } else if has("items") || has("prefixItems") {
            "array"
        } else if has("pattern") || has("minLength") || has("maxLength") || has("format") {
            "string"
        } else {
            ""
        }
    }

    fn resolve_ref(&mut self, r: &Value, path: &str) -> Result<String, String> {
        let Some(reference) = r.as_str() else { return fail(path, "$ref must be a string") };
        let Some(pointer) = reference.strip_prefix('#').filter(|p| p.starts_with('/')) else {
            return fail(path, format!("unsupported $ref {reference}, only references into the same document are supported"));
        };
        let ref_name = format!("ref{}", sanitize(pointer));
        if !self.rules.contains_key(&ref_name) && !self.resolving.contains(reference) {
            let Some(target) = self.root.pointer(pointer) else { return fail(path, format!("cannot resolve $ref {reference}")) };
            self.resolving.insert(reference.to_string());
            let r = self.visit(target, &ref_name, reference);
            self.resolving.remove(reference);
            return r;
        }
        Ok(ref_name)
    }

    fn visit_array(&mut self, schema: &Value, name: &str, rule_name: &str, path: &str) -> Result<String, String> {
        let sub = format!("{name}{}", if name.is_empty() { "" } else { "-" });
        let items = schema.get("items").or_else(|| schema.get("prefixItems"));
        if let Some(Value::Array(tuple)) = items {
            let mut rule = String::from("\"[\" space ");
            for (i, item) in tuple.iter().enumerate() {
                if i > 0 {
                    rule.push_str(" \",\" space ");
                }
                rule.push_str(&self.visit(item, &format!("{sub}tuple-{i}"), &format!("{path}/items/{i}"))?);
            }
            rule.push_str(" space \"]\"");
            return Ok(self.add_rule(rule_name, &rule));
        }
        let (min, max) = (count(schema, "minItems", path)?.unwrap_or(0), count(schema, "maxItems", path)?);
        let Some(items) = items else {
            return Ok(if min == 0 && max.is_none() { self.primitive(rule_name, "array") } else { self.array_of_any(rule_name, &sub, min, max, path)? });
        };
        let item_rule = self.visit(items, &format!("{sub}item"), &format!("{path}/items"))?;
        let body = repetition(&item_rule, min, max, "\",\" space");
        Ok(self.add_rule(rule_name, &format!("\"[\" space {body} space \"]\"")))
    }

    fn array_of_any(&mut self, rule_name: &str, sub: &str, min: u64, max: Option<u64>, path: &str) -> Result<String, String> {
        let any = self.visit(&serde_json::json!({}), &format!("{sub}item"), &format!("{path}/items"))?;
        let body = repetition(&any, min, max, "\",\" space");
        Ok(self.add_rule(rule_name, &format!("\"[\" space {body} space \"]\"")))
    }

    fn visit_string(&mut self, schema: &Value, rule_name: &str, path: &str) -> Result<String, String> {
        if schema.get("pattern").is_some() {
            return fail(path, "pattern is not supported (the grammar would not enforce it)");
        }
        if let Some(rule) = self.format_rule(schema, rule_name) {
            return Ok(rule);
        }
        let (min, max) = (count(schema, "minLength", path)?.unwrap_or(0), count(schema, "maxLength", path)?);
        if min == 0 && max.is_none() {
            return Ok(self.primitive(rule_name, "string"));
        }
        let ch = self.add_primitive("char");
        let body = repetition(&ch, min, max, "");
        Ok(self.add_rule(rule_name, &format!("\"\\\"\" {body} \"\\\"\"")))
    }

    /// The rule of a string `format` this converter knows (date, time, date-time, uuid, uuid1 to uuid5).
    fn format_rule(&mut self, schema: &Value, rule_name: &str) -> Option<String> {
        let f = schema.get("format").and_then(Value::as_str)?;
        if f == "uuid" || (f.len() == 5 && f.starts_with("uuid") && matches!(f.as_bytes()[4], b'1'..=b'5')) {
            return Some(self.primitive(rule_name, "uuid"));
        }
        if !matches!(f, "date" | "time" | "date-time") {
            return None;
        }
        let p = self.add_primitive(&format!("{f}-string"));
        Some(self.add_rule(rule_name, &p))
    }

    fn visit_integer(&mut self, schema: &Value, rule_name: &str, path: &str) -> Result<String, String> {
        for k in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] {
            if schema.get(k).is_some() {
                return fail(path, format!("{k} on an integer is not supported (the grammar would not enforce it)"));
            }
        }
        Ok(self.primitive(rule_name, "integer"))
    }

    fn visit_object(&mut self, schema: &Value, name: &str, rule_name: &str, path: &str) -> Result<String, String> {
        let props = match schema.get("properties") {
            None => Vec::new(),
            Some(Value::Object(o)) => o.iter().collect::<Vec<_>>(),
            Some(_) => return fail(path, "properties must be an object"),
        };
        let additional = additional_of(schema, path)?;
        let any_extra = additional.as_ref().is_some_and(|a| a.as_object().is_some_and(|o| o.is_empty()));
        if props.is_empty() && any_extra {
            let obj = self.add_primitive("object");
            return Ok(self.add_rule(rule_name, &obj));
        }
        let required: HashSet<&str> = schema.get("required").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).collect();
        let rule = self.object_rule(&props, &required, name, additional.as_ref(), path)?;
        Ok(self.add_rule(rule_name, &rule))
    }

    fn object_rule(&mut self, props: &[(&String, &Value)], required: &HashSet<&str>, name: &str, additional: Option<&Value>, path: &str) -> Result<String, String> {
        let sub = format!("{name}{}", if name.is_empty() { "" } else { "-" });
        let (mut req, mut opt) = (Vec::new(), Vec::new());
        let mut kv: BTreeMap<String, String> = BTreeMap::new();
        for (pname, pschema) in props {
            let prop_rule = self.visit(pschema, &format!("{sub}{pname}"), &format!("{path}/properties/{pname}"))?;
            let kv_rule = self.add_rule(&format!("{sub}{pname}-kv"), &format!("{} space \":\" space {prop_rule}", literal(&Value::String((*pname).clone()).to_string())));
            kv.insert((*pname).clone(), kv_rule);
            if required.contains(pname.as_str()) { req.push((*pname).clone()) } else { opt.push((*pname).clone()) }
        }
        if let Some(add) = additional {
            let k = self.additional_kv(props, add, &sub, path)?;
            kv.insert("*".into(), k);
            opt.push("*".into());
        }
        if req.is_empty() && opt.is_empty() {
            return Ok("\"{\" space \"}\"".into());
        }
        let mut rule = String::from("\"{\" space ");
        rule.push_str(&req.iter().map(|k| kv[k].as_str()).collect::<Vec<_>>().join(" \",\" space "));
        if !opt.is_empty() {
            rule.push_str(&self.optional_part(&opt, &kv, &sub, !req.is_empty()));
        }
        rule.push_str(" space \"}\"");
        Ok(rule)
    }

    /// The kv rule for extra keys: any string not among the property names, with the schema's value.
    fn additional_kv(&mut self, props: &[(&String, &Value)], add: &Value, sub: &str, path: &str) -> Result<String, String> {
        let sub_name = format!("{sub}additional");
        let value_rule = if add.as_object().is_some_and(|o| o.is_empty()) { self.add_primitive("value") } else { self.visit(add, &format!("{sub_name}-value"), &format!("{path}/additionalProperties"))? };
        let key_rule = if props.is_empty() {
            self.add_primitive("string")
        } else {
            let names: Vec<String> = props.iter().map(|(n, _)| (*n).clone()).collect();
            let rule = self.not_strings(&names);
            self.add_rule(&format!("{sub_name}-k"), &rule)
        };
        Ok(self.add_rule(&format!("{sub_name}-kv"), &format!("{key_rule} \":\" space {value_rule}")))
    }

    fn not_strings(&mut self, names: &[String]) -> String {
        let char_rule = self.add_primitive("char");
        let trie = Trie::of(names);
        let mut out = String::from("[\"] ( ");
        trie.write(&char_rule, &mut out);
        out.push_str(" )");
        if !trie.end {
            out.push('?');
        }
        out.push_str(" [\"]");
        out
    }

    /// The optional properties, in llama.cpp's shape: each may be the first, the rest follow in order.
    fn optional_part(&mut self, opt: &[String], kv: &BTreeMap<String, String>, sub: &str, has_required: bool) -> String {
        let mut rule = String::from(" (");
        if has_required {
            rule.push_str(" \",\" space ( ");
        }
        for i in 0..opt.len() {
            if i > 0 {
                rule.push_str(" | ");
            }
            rule.push_str(&self.recursive_refs(&opt[i..], false, kv, sub));
        }
        if has_required {
            rule.push_str(" )");
        }
        rule.push_str(" )?");
        rule
    }

    fn recursive_refs(&mut self, ks: &[String], first_is_optional: bool, kv: &BTreeMap<String, String>, sub: &str) -> String {
        let Some(k) = ks.first() else { return String::new() };
        let kv_rule = &kv[k];
        let comma_ref = format!("( \",\" space {kv_rule} )");
        let mut res = match (first_is_optional, k == "*") {
            (true, true) => format!("{comma_ref}*"),
            (true, false) => format!("{comma_ref}?"),
            (false, true) => format!("{kv_rule} {comma_ref}*"),
            (false, false) => kv_rule.clone(),
        };
        if ks.len() > 1 {
            let rest = self.recursive_refs(&ks[1..], true, kv, sub);
            res.push(' ');
            res.push_str(&self.add_rule(&format!("{sub}{k}-rest"), &rest));
        }
        res
    }
}

#[cfg(test)]
mod tests;
