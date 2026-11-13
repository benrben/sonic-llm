use super::*;
use serde_json::json;

const CALL: &str = "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call>";

fn call(args: &str) -> Part {
    Part::Call(ToolCall { name: "f".into(), arguments: args.into() })
}

/// All parts for `pieces`, content runs merged (how the text was cut may split a run).
fn parse_with(p: ToolParser, pieces: &[&str]) -> Vec<Part> {
    let mut p = p;
    let mut out: Vec<Part> = Vec::new();
    let mut all: Vec<Part> = pieces.iter().flat_map(|s| p.push(s)).collect();
    all.extend(p.finish());
    for part in all {
        match (out.last_mut(), part) {
            (Some(Part::Content(a)), Part::Content(b)) => a.push_str(&b),
            (_, part) => out.push(part),
        }
    }
    out
}

fn parse_as(f: ToolFormat, pieces: &[&str]) -> Vec<Part> {
    parse_with(ToolParser::new(f), pieces)
}

fn parse(pieces: &[&str]) -> Vec<Part> {
    parse_as(ToolFormat::QWEN_JSON, pieces)
}

/// The text cut at every position, and one character per piece, gives the parts of the whole text.
fn same_however_cut_as(f: ToolFormat, text: &str) -> Vec<Part> {
    let whole = parse_as(f, &[text]);
    let c: Vec<String> = text.chars().map(|c| c.to_string()).collect();
    assert_eq!(parse_as(f, &c.iter().map(String::as_str).collect::<Vec<_>>()), whole, "{f:?} one char per piece: {text:?}");
    for i in (0..=text.len()).filter(|&i| text.is_char_boundary(i)) {
        assert_eq!(parse_as(f, &[&text[..i], &text[i..]]), whole, "{f:?} cut at {i}: {text:?}");
    }
    whole
}

fn same_however_cut(text: &str) -> Vec<Part> {
    same_however_cut_as(ToolFormat::QWEN_JSON, text)
}

#[test]
fn a_call_whole_split_and_with_text_around_it() {
    assert_eq!(same_however_cut(CALL), vec![call("{\"a\":1}")]);
    assert_eq!(same_however_cut(&format!("Sure.\n{CALL}")), vec![Part::Content("Sure.\n".into()), call("{\"a\":1}")]);
    assert_eq!(same_however_cut(&format!("{CALL}\nDone.")), vec![call("{\"a\":1}"), Part::Content("Done.".into())]);
    assert_eq!(same_however_cut(&format!("{CALL}\n{CALL}\n")), vec![call("{\"a\":1}"), call("{\"a\":1}")]);
}

#[test]
fn arguments_forms() {
    let body = |b: &str| format!("<tool_call>{b}</tool_call>");
    assert_eq!(parse(&[&body("{\"name\":\"f\"}")]), vec![call("{}")]);
    assert_eq!(parse(&[&body("{\"name\":\"f\",\"arguments\":null}")]), vec![call("{}")]);
    assert_eq!(parse(&[&body("{\"name\":\"f\",\"parameters\":{\"b\":2}}")]), vec![call("{\"b\":2}")]);
    assert_eq!(parse(&[&body("{\"name\":\"f\",\"arguments\":\"{\\\"c\\\": 3}\"}")]), vec![call("{\"c\": 3}")]);
    // the model's key order is kept
    assert_eq!(parse(&[&body("{\"name\":\"f\",\"arguments\":{\"z\":1,\"a\":2}}")]), vec![call("{\"z\":1,\"a\":2}")]);
}

#[test]
fn text_that_may_start_a_tag_is_held_back_and_released_when_it_is_not_one() {
    let mut p = ToolParser::new(ToolFormat::QWEN_JSON);
    assert_eq!(p.push("ok <tool"), vec![Part::Content("ok ".into())]);
    assert_eq!(p.push("kit"), vec![Part::Content("<toolkit".into())]);
    assert_eq!(p.push("<"), vec![]);
    assert_eq!(p.finish(), vec![Part::Content("<".into())]);
    assert_eq!(same_however_cut("a < b <tool_x> <tool_call"), vec![Part::Content("a < b <tool_x> <tool_call".into())]);
}

#[test]
fn a_tag_that_is_not_a_call_comes_back_as_content_verbatim() {
    for text in [
        "<tool_call>\nnot json\n</tool_call>",
        "<tool_call>{\"name\": \"f\", \"arguments\": {oops}}</tool_call>",
        "<tool_call>[1, 2]</tool_call>",
        "<tool_call>{\"arguments\": {}}</tool_call>",
        "<tool_call>{\"name\": \"\"}</tool_call>",
        "<tool_call>{\"name\": \"f\", \"arguments\": [1]}</tool_call>",
        "<tool_call>{\"name\": \"f\", \"arg", // never closed
        "<tool_call>",
    ] {
        assert_eq!(same_however_cut(text), vec![Part::Content(text.into())], "{text}");
    }
    // a bad tag does not spoil the next call
    let text = format!("<tool_call>x</tool_call>{CALL}");
    assert_eq!(same_however_cut(&text), vec![Part::Content("<tool_call>x</tool_call>".into()), call("{\"a\":1}")]);
}

// -- every format, from one table --------------------------------------------------------------------------------

/// Text in one format's style, with what the parser must make of it.
struct Sample {
    fmt: ToolFormat,
    /// One call: f({"a": 1}).
    one: &'static str,
    /// Two calls: f({"a":1}) then g({"b":2}); `None`: the format holds one call per reply.
    two: Option<&'static str>,
    /// Text that is not a call, whose start looks like one.
    lookalike: &'static str,
    /// Calls that are broken or never closed: the whole text comes back as content.
    malformed: &'static [&'static str],
}

fn samples() -> Vec<Sample> {
    vec![
        Sample {
            fmt: ToolFormat::QWEN_JSON,
            one: CALL,
            two: Some("<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call>\n<tool_call>\n{\"name\": \"g\", \"arguments\": {\"b\": 2}}\n</tool_call>"),
            lookalike: "use <tool_x> and <tool_call b> then <tool_cal",
            malformed: &["<tool_call>nope</tool_call>", "<tool_call>{\"name\": \"f\"", "<tool_call>[1]</tool_call>"],
        },
        Sample {
            fmt: ToolFormat::QWEN_XML,
            one: "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>",
            two: Some("<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=g>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>"),
            lookalike: "use <function=x> and <tool_call b> then <parameter=",
            malformed: &[
                "<tool_call>\n<function=f>\n<parameter=a>\n1\n</function>\n</tool_call>",
                "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n",
                "<tool_call>\nfunction f\n</tool_call>",
                "<tool_call>\n<function=f>\n</tool_call>",
            ],
        },
        Sample {
            fmt: ToolFormat::GEMMA4,
            one: "<|tool_call>call:f{a:1}<tool_call|>",
            two: Some("<|tool_call>call:f{a:1}<tool_call|><|tool_call>call:g{b:2}<tool_call|>"),
            lookalike: "use <|tool_x> and <|tool_call b> then <|tool_cal",
            malformed: &["<|tool_call>call:f{a:}<tool_call|>", "<|tool_call>call:f{a:1}", "<|tool_call>f{a:1}<tool_call|>", "<|tool_call>call:f{a:1<tool_call|>"],
        },
        Sample {
            fmt: ToolFormat::GPT_OSS,
            one: "<|channel|>commentary to=functions.f <|constrain|>json<|message|>{\"a\":1}",
            two: Some("<|channel|>commentary to=functions.f <|constrain|>json<|message|>{\"a\":1}<|call|><|start|>assistant<|channel|>commentary to=functions.g <|constrain|>json<|message|>{\"b\":2}"),
            lookalike: "use <|start|>assist and <|channel|>commentary to=function then <|channel|>comm",
            malformed: &["<|channel|>commentary to=functions.f <|constrain|>json<|message|>[1]", "<|channel|>commentary to=functions.f <|constrain|>json{\"a\":1}", "<|channel|>commentary to=functions.f <|constrain|>json<|message|>{\"a\": tru}<|call|>"],
        },
        Sample {
            fmt: ToolFormat::MISTRAL,
            one: "[TOOL_CALLS] [{\"name\": \"f\", \"arguments\": {\"a\": 1}, \"id\": \"abc123xyz\"}]",
            two: Some("[TOOL_CALLS] [{\"name\": \"f\", \"arguments\": {\"a\": 1}}, {\"name\": \"g\", \"arguments\": {\"b\": 2}}]"),
            lookalike: "use [TOOL_CALL] and [TOOLS] then [TOOL_CAL",
            malformed: &["[TOOL_CALLS] not json", "[TOOL_CALLS] [{\"arguments\": {}}]", "[TOOL_CALLS] {\"name\": \"f\"}", "[TOOL_CALLS] []"],
        },
        Sample {
            fmt: ToolFormat::LLAMA31,
            one: "{\"name\": \"f\", \"parameters\": {\"a\": 1}}",
            two: None,
            lookalike: "Hello {\"name\": \"f\", \"parameters\": {}} and {",
            malformed: &["{oops", "{\"parameters\": {}}", "{\"name\": \"f\", \"parameters\": [1]}", "{\"name\": \"f\", \"parameters\": {\"a\": tru}}"],
        },
    ]
}

fn g(args: &str) -> Part {
    Part::Call(ToolCall { name: "g".into(), arguments: args.into() })
}

#[test]
fn every_format_whole_split_char_by_char_two_calls_text_before_lookalike_and_malformed() {
    for s in samples() {
        let f = s.fmt;
        // one call: however the text is cut
        assert_eq!(same_however_cut_as(f, s.one), vec![call("{\"a\":1}")], "{f:?} one");
        // text before the call (a whole-reply format has none: text in front makes it all content)
        let before = format!("Sure.\n{}", s.one);
        let want = if matches!(f.spec().start, Start::Reply) { vec![Part::Content(before.clone())] } else { vec![Part::Content("Sure.\n".into()), call("{\"a\":1}")] };
        assert_eq!(same_however_cut_as(f, &before), want, "{f:?} text before");
        // two calls
        if let Some(two) = s.two {
            assert_eq!(same_however_cut_as(f, two), vec![call("{\"a\":1}"), g("{\"b\":2}")], "{f:?} two");
        }
        // a look-alike is content, verbatim
        assert_eq!(same_however_cut_as(f, s.lookalike), vec![Part::Content(s.lookalike.into())], "{f:?} look-alike");
        // malformed or unclosed: content, verbatim, and no call
        for m in s.malformed {
            assert_eq!(same_however_cut_as(f, m), vec![Part::Content((*m).into())], "{f:?} malformed {m:?}");
        }
        // a bad call does not spoil the next one (formats with a start marker)
        if let (Start::Markers(_), Some(bad)) = (f.spec().start, s.malformed.first()) {
            if !matches!(f.spec().close, Close::End) {
                let two = s.two.unwrap_or(s.one);
                let text = format!("{bad}{}", if matches!(f.spec().close, Close::MarkerOrEnd(_)) { "<|call|><|start|>assistant" } else { "" });
                let text = format!("{text}{two}");
                let parts = parse_as(f, &[&text]);
                assert!(matches!(parts.last(), Some(Part::Call(_))), "{f:?} a call after a bad one: {parts:?}");
            }
        }
    }
}

/// Negative control: format A's text read with format B's description is content, never a call.
#[test]
fn one_formats_text_read_as_another_format_is_content_and_no_call() {
    let all = samples();
    for a in &all {
        for b in &all {
            if a.fmt == b.fmt {
                continue;
            }
            for text in std::iter::once(a.one).chain(a.two) {
                let parts = parse_as(b.fmt, &[text]);
                assert!(!parts.iter().any(|p| matches!(p, Part::Call(_))), "{:?} text read as {:?}: {parts:?}", a.fmt, b.fmt);
                assert_eq!(parts, vec![Part::Content(text.into())], "{:?} text read as {:?}", a.fmt, b.fmt);
            }
        }
    }
    // and the positive control: the same texts do parse in their own format
    for a in &all {
        assert!(parse_as(a.fmt, &[a.one]).iter().any(|p| matches!(p, Part::Call(_))), "{:?}", a.fmt);
    }
}

#[test]
fn xml_values_are_typed_by_the_tools_schemas() {
    let text = "<tool_call>\n<function=f>\n<parameter=name>\n007\n</parameter>\n<parameter=n>\n3\n</parameter>\n<parameter=ok>\ntrue\n</parameter>\n<parameter=obj>\n{\"k\": [1, 2]}\n</parameter>\n<parameter=any>\nhello\n</parameter>\n<parameter=multi>\nline1\nline2\n</parameter>\n</function>\n</tool_call>";
    let tools = [json!({"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "name": {"type": "string"}, "n": {"type": "integer"}, "ok": {"type": "boolean"}, "obj": {"type": "object"}, "any": {"enum": ["hello"]}, "multi": {"type": ["string"]}}}}})];
    let args = |p: ToolParser| match parse_with(p, &[text]).remove(0) {
        Part::Call(c) => serde_json::from_str::<Value>(&c.arguments).unwrap(),
        other => panic!("{other:?}"),
    };
    assert_eq!(
        args(ToolParser::new(ToolFormat::QWEN_XML).with_tools(&tools)),
        json!({"name": "007", "n": 3, "ok": true, "obj": {"k": [1, 2]}, "any": "hello", "multi": "line1\nline2"})
    );
    // without the schema the string "007" is read as JSON where it can be (007 is not JSON: still a string), "3" is a number
    assert_eq!(args(ToolParser::new(ToolFormat::QWEN_XML))["name"], json!("007"));
    assert_eq!(args(ToolParser::new(ToolFormat::QWEN_XML))["n"], json!(3));
    // a schema that says string keeps a value that looks like a number a string (negative control)
    let s = [json!({"type": "function", "function": {"name": "f", "parameters": {"properties": {"n": {"type": "string"}}}}})];
    assert_eq!(args(ToolParser::new(ToolFormat::QWEN_XML).with_tools(&s))["n"], json!("3"));
}

#[test]
fn gemma_values_nest_and_keep_their_types() {
    let text = "<|tool_call>call:f{s:<|\"|>a, b: {c}<|\"|>,n:-1.5e2,b:true,z:null,arr:[1,<|\"|>x<|\"|>,[2],{k:<|\"|>v<|\"|>}],o:{},e:[]}<tool_call|>";
    let want = json!({"s": "a, b: {c}", "n": -150.0, "b": true, "z": null, "arr": [1, "x", [2], {"k": "v"}], "o": {}, "e": []});
    match same_however_cut_as(ToolFormat::GEMMA4, text).remove(0) {
        Part::Call(c) => assert_eq!(serde_json::from_str::<Value>(&c.arguments).unwrap(), want),
        other => panic!("{other:?}"),
    }
    // no arguments, and the model's key order is kept
    assert_eq!(parse_as(ToolFormat::GEMMA4, &["<|tool_call>call:f{}<tool_call|>"]), vec![call("{}")]);
    assert_eq!(parse_as(ToolFormat::GEMMA4, &["<|tool_call>call:f{z:1,a:2}<tool_call|>"]), vec![call("{\"z\":1,\"a\":2}")]);
}

#[test]
fn reasoning_channels_are_not_content_and_channel_headers_are_stripped() {
    let oss = "<|channel|>analysis<|message|>We need f.<|end|><|start|>assistant<|channel|>commentary to=functions.f <|constrain|>json<|message|>{\"a\":1}";
    assert_eq!(same_however_cut_as(ToolFormat::GPT_OSS, oss), vec![call("{\"a\":1}")]);
    let plain = "<|channel|>analysis<|message|>No tool.<|end|><|start|>assistant<|channel|>final<|message|>Hello!";
    assert_eq!(same_however_cut_as(ToolFormat::GPT_OSS, plain), vec![Part::Content("Hello!".into())]);
    // a preamble on the commentary channel is content; reasoning that never ends is dropped; the role-header call form works
    let pre = "<|channel|>commentary<|message|>Let me look.<|end|><|start|>assistant to=functions.f<|channel|>commentary json<|message|>{\"a\":1}<|call|>";
    assert_eq!(same_however_cut_as(ToolFormat::GPT_OSS, pre), vec![Part::Content("Let me look.".into()), call("{\"a\":1}")]);
    assert_eq!(same_however_cut_as(ToolFormat::GPT_OSS, "<|channel|>analysis<|message|>thinking and thinking"), vec![]);
    let gem = "<|channel>thought\nhmm<channel|>Sure.<|tool_call>call:f{a:1}<tool_call|>";
    assert_eq!(same_however_cut_as(ToolFormat::GEMMA4, gem), vec![Part::Content("Sure.".into()), call("{\"a\":1}")]);
}

#[test]
fn a_whole_reply_call_cut_short_by_the_model_is_completed_but_other_damage_is_not() {
    let f = ToolFormat::LLAMA31;
    assert_eq!(parse_as(f, &["{\"name\": \"f\", \"parameters\": {\"a\": [1, 2"]), vec![call("{\"a\":[1,2]}")]);
    assert_eq!(parse_as(f, &["{\"name\": \"f\", \"parameters\": {\"a\": \"x"]), vec![call("{\"a\":\"x\"}")]);
    assert_eq!(parse_as(f, &["{\"name\": \"f\", \"parameters\": {\"a\": 1,"]), vec![Part::Content("{\"name\": \"f\", \"parameters\": {\"a\": 1,".into())]);
    // a marker-closed format never completes an unclosed call
    assert_eq!(parse(&["<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": 1}"]), vec![Part::Content("<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": 1}".into())]);
    // a plain reply is content, also when it starts with white space or a brace later
    assert_eq!(same_however_cut_as(f, "  Hello {x}"), vec![Part::Content("  Hello {x}".into())]);
}

#[test]
fn white_space_after_a_call_is_not_content_in_any_format() {
    for s in samples() {
        let text = format!("{}\n\n  tail", s.one);
        let want = if matches!(s.fmt.spec().start, Start::Reply) || matches!(s.fmt.spec().close, Close::End | Close::MarkerOrEnd(_)) {
            continue; // the call runs to the end of the reply: there is no tail
        } else {
            vec![call("{\"a\":1}"), Part::Content("tail".into())]
        };
        assert_eq!(same_however_cut_as(s.fmt, &text), want, "{:?}", s.fmt);
    }
}

// -- detection ---------------------------------------------------------------------------------------------------

fn detect(src: &str, toks: &[&str]) -> Option<&'static str> {
    let set: HashSet<String> = toks.iter().map(|t| t.to_string()).collect();
    ToolFormat::detect(src, &set).map(ToolFormat::name)
}

#[test]
fn the_format_is_found_from_the_template_and_its_tokens_most_specific_first() {
    assert_eq!(ToolFormat::detect(crate::chat::ChatTemplate::chatml().source.as_str(), &HashSet::new()), Some(ToolFormat::QWEN_JSON));
    assert_eq!(crate::chat::ChatTemplate::chatml().tool_format(), Some(ToolFormat::QWEN_JSON));
    assert_eq!(detect("<start_of_turn>{{ x }}", &[]), None);
    // a template that also mentions Qwen's tags is still Gemma's: Gemma is tested first
    assert_eq!(detect("'<|tool_call>call:' x <tool_call>", &[]), Some("gemma4"));
    // Qwen 3.5's template holds Qwen's JSON markers too: the XML tags decide
    assert_eq!(detect("<tool_call> <function=x> <parameter=y>", &[]), Some("qwen-xml"));
    assert_eq!(detect("<tool_call> {\"name\"}", &[]), Some("qwen-json"));
    assert_eq!(detect("<|channel|> functions.", &[]), Some("gpt-oss"));
    // Mistral needs the token as well as the text
    assert_eq!(detect("[TOOL_CALLS] x", &["[TOOL_CALLS]"]), Some("mistral"));
    assert_eq!(detect("[TOOL_CALLS] x", &[]), None);
    assert_eq!(detect("x", &["[TOOL_CALLS]"]), None);
    assert_eq!(detect("<|python_tag|> {\"name\": f, \"parameters\": d}", &[]), Some("llama31-json"));
    // every format has a distinct name
    let names: HashSet<_> = FORMATS.iter().map(|f| f.name()).collect();
    assert_eq!(names.len(), FORMATS.len());
}

#[test]
fn only_a_formats_own_tokens_are_preserved() {
    assert!(ToolFormat::GEMMA4.preserves("<|tool_call>") && ToolFormat::GPT_OSS.preserves("<|channel|>") && ToolFormat::MISTRAL.preserves("[TOOL_CALLS]"));
    assert!(!ToolFormat::GEMMA4.preserves("<|turn>x") && !ToolFormat::QWEN_JSON.preserves("<|im_end|>") && !ToolFormat::LLAMA31.preserves("<|eot_id|>"));
}
