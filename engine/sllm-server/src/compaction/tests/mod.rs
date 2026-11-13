use super::*;

#[test]
fn cut_points_keep_the_prefix_and_the_tail() {
    assert_eq!(cut_points(5000, 600, Some(1024), None), Ok((600, 5000 - 1024)));
    // a short context keeps everything after the prefix as tail: nothing to compact
    assert!(cut_points(1500, 600, Some(1024), None).is_err());
    assert_eq!(cut_points(1700, 600, Some(1024), None), Ok((600, 676)));
    assert_eq!(cut_points(5000, 600, None, Some(4000)), Ok((600, 4000)));
    assert_eq!(cut_points(5000, 600, None, Some(5000)), Ok((600, 5000)), "no tail: summary only");
    assert!(cut_points(5000, 600, None, Some(500)).is_err(), "tail inside the prefix");
    assert!(cut_points(5000, 600, None, Some(5001)).is_err());
    assert!(cut_points(500, 600, None, None).is_err());
    assert_eq!(cut_points(3000, 0, Some(1000), None), Ok((0, 2000)), "an agent without a system prompt");
    let ctx: Vec<i32> = (0..10).collect();
    assert_eq!(feed(&ctx, 7, &[100, 101]), vec![100, 101, 7, 8, 9]);
}

#[test]
fn rollover_sizes_shrink_with_the_room_and_stop_at_the_floor() {
    let r = Rollover::default();
    assert_eq!(r.sizes(40_000), Some((128, 1024)));
    assert_eq!(r.sizes(4096), Some((128, 512)));
    assert_eq!(r.sizes(768), Some((48, 96)), "the windows of 256 agents");
    assert_eq!(r.sizes(ROLL_MIN_ROOM - 1), None);
}

fn bytes_text() -> RolloverText {
    let e: Encode = Arc::new(|t: &str| t.bytes().map(i32::from).collect());
    RolloverText::new(&TurnFrames::plain(), "", e.clone(), e, Arc::new(|ids: &[i32]| ids.iter().map(|&t| t as u8 as char).collect()), None)
}

#[test]
fn the_archive_finds_a_line_by_its_rare_word() {
    let mut a = Archive::default();
    let filler: String = (0..60).map(|i| format!("the project keeps moving along in step {i}\n")).collect();
    a.add(&filler);
    a.add("The code of project Wren is 2613.\nThe code of project Tern is 4410.\n");
    a.add("What did Wren say?\n");
    a.add(&(0..12).map(|i| format!("The code of project P{i} is {}.\n", 7000 + i)).collect::<String>());
    a.add("The code of project Wren is 2613.\n");
    assert_eq!(a.recall("What is the code of project Wren? Reply with the number only.").as_deref(), Some("The code of project Wren is 2613."));
    assert_eq!(a.recall("Which number did Tern get?").as_deref(), Some("The code of project Tern is 4410."));
    assert_eq!(a.recall("What is the code of project Zebu?"), None, "only words every line has: nothing is made up");
    assert_eq!(a.recall("project"), None);
    // an earlier question has every plain word of this one, and no answer
    a.add("What is the code of project Tern? Reply with the number only.\nassistant\n<think>\n");
    assert_eq!(a.recall("What is the code of project Wren? Reply with the number only.").as_deref(), Some("The code of project Wren is 2613."));
    assert_eq!(a.recall("What is the code of project Tern? Reply with the number only.").as_deref(), Some("The code of project Tern is 4410."));
    assert_eq!(a.recall("Say OK."), None);
}

#[test]
fn names_are_what_identifies_something() {
    assert_eq!(names("What is the code of project Wren? Reply with the number only."), ["wren"]);
    assert_eq!(names("List every IMPORTANT line."), ["important"]);
    assert_eq!(names("Open src/llama_engine.rs at line 420"), ["420", "engine", "llama", "src"]);
    assert!(names("what did we decide about this").is_empty());
}

#[test]
fn the_archive_stays_small_and_keeps_the_newest() {
    let mut a = Archive::default();
    for k in 0..40 {
        a.add(&(0..4000).map(|i| format!("entry id{k}x{i} long {}\n", "x".repeat(200))).collect::<String>());
    }
    assert!(a.bytes <= ARCHIVE_BYTES);
    assert!(a.recall("id39x77 please").is_some());
    assert!(a.recall("id0x77 please").is_none(), "the oldest went");
    assert_eq!(a.lines.len(), a.seen.len());
}

#[test]
fn recalled_lines_end_the_user_turn() {
    let text = bytes_text();
    let e = |t: &str| t.bytes().map(i32::from).collect::<Vec<i32>>();
    let p: Vec<i32> = e("a long text. question").into_iter().chain(text.to_reply.iter().copied()).collect();
    let got: String = text.with_recall(&p, "line", 1000).iter().map(|&t| t as u8 as char).collect();
    assert!(got.starts_with("a long text. question\n\nLines from earlier in this work that may help:\nline\n\nAssistant: "), "{got}");
    let got: String = text.with_recall(&e("no frame"), "line", 1000).iter().map(|&t| t as u8 as char).collect();
    assert!(got.ends_with("line\n\nno frame"), "{got}");
}

#[test]
fn a_request_loses_its_frames() {
    let t = bytes_text();
    let p: Vec<i32> = t.user_open.iter().copied().chain("count to ten".bytes().map(i32::from)).chain(t.to_reply.iter().copied()).collect();
    let got: String = request_of(&t, &p, NOTES_REQUEST_TOKENS).iter().map(|&b| b as u8 as char).collect();
    assert_eq!(got, "count to ten");
}

#[test]
fn a_tail_starts_at_a_turn_when_it_has_one() {
    assert_eq!(turn_start(&[1, 2, 7, 8, 3, 7, 8, 9], 1, &[7, 8]), Some(2));
    assert_eq!(turn_start(&[1, 2, 7, 8, 3, 7, 8, 9], 3, &[7, 8]), Some(5));
    assert_eq!(turn_start(&[1, 2, 3, 4], 0, &[7, 8]), None, "the middle of one long reply");
    assert_eq!(turn_start(&[1, 2, 3, 4], 0, &[]), None);
    let t = bytes_text();
    let f: String = t.foot_turns.iter().map(|&b| b as u8 as char).collect();
    assert_eq!(f, "\n\nAssistant: Understood.\n\n", "the notes end as an exchange of their own");
}

#[test]
fn the_ask_quotes_the_end_of_the_prompt_without_its_frame() {
    let text = bytes_text();
    let long: Vec<i32> = (0..500).chain(text.to_reply.iter().copied()).collect();
    let ask = ask_with_request(&text, &long, NOTES_REQUEST_TOKENS);
    let n = text.request.len();
    assert_eq!(ask[..n], text.request[..]);
    assert_eq!(ask[n..n + NOTES_REQUEST_TOKENS], (500 - NOTES_REQUEST_TOKENS as i32..500).collect::<Vec<_>>()[..]);
    assert_eq!(ask[n + NOTES_REQUEST_TOKENS..], text.ask_reading[..]);
}

#[test]
fn notes_lose_a_thinking_block() {
    assert_eq!(notes_text("<think>\nhm\n</think>\n\n a note \n"), "a note");
    assert_eq!(notes_text("a note"), "a note");
}

#[test]
fn policies_trigger_on_window_or_tokens_since() {
    let p = Policy { window_tokens: Some(8192), every_tokens: None, tail_tokens: 1024, summary_tokens: 300 };
    p.validate(600).unwrap();
    assert_eq!(p.due(8191, 8000), None);
    assert_eq!(p.due(8192, 10), Some(Trigger::Window));
    let e = Policy { window_tokens: None, every_tokens: Some(2000), ..p };
    assert_eq!(e.due(30_000, 1999), None, "every_tokens counts from the last compaction, not the window");
    assert_eq!(e.due(3000, 2000), Some(Trigger::Every));
    // a window trigger below what a compaction leaves would fire after every turn
    let low = Policy { window_tokens: Some(2000), ..p };
    assert!(low.validate(600).is_err());
    assert!(low.validate(0).is_ok());
    assert!(Policy { window_tokens: None, every_tokens: None, ..p }.validate(0).is_err());
    assert!(Policy { every_tokens: Some(0), window_tokens: None, ..p }.validate(0).is_err());
    // JSON defaults
    let j: Policy = serde_json::from_value(json!({"window_tokens": 16384})).unwrap();
    assert_eq!((j.tail_tokens, j.summary_tokens, j.every_tokens), (1024, 300, None));
    assert!(serde_json::from_value::<Policy>(json!({"window": 1})).is_err(), "unknown fields are refused");
}

#[test]
fn the_stub_summary_is_the_first_tokens_of_the_replaced_range() {
    let s = FirstTokensStub;
    assert_eq!(s.summarize("a", &[5, 6, 7, 8], 2).unwrap(), vec![5, 6]);
    assert_eq!(s.summarize("a", &[5], 300).unwrap(), vec![5]);
    assert_eq!(SummarizerHandle::default().0.name(), "stub-first-n");
}

#[test]
fn fact_lines_keep_digits_and_pairs_and_drop_prose() {
    let words = |t: &str| t.split_whitespace().count();
    let text = "TOOL RESULT for READ doc_03.txt:\nThe Dunay lighthouse recorded 2,615 ships guided.\nThe old gate is made of oak.\n\
                status: ok\nretries=3\nUnread files: doc_04.txt, doc_05.txt.\nLocal children helped label cupboards.\n\
                see /etc/hosts for names\nThe Dunay lighthouse recorded 2,615 ships guided.\nname: Cora Varick\nstill in the tail 77";
    let big = |t: &str| words(t) * 100;
    let sel = select_fact_lines(text, "still in the tail 77", 100_000, &big);
    assert_eq!(
        sel,
        ["TOOL RESULT for READ doc_03.txt:", "status: ok", "retries=3", "see /etc/hosts for names", "The Dunay lighthouse recorded 2,615 ships guided.", "name: Cora Varick"],
        "facts kept in order, duplicates keep the last copy, prose, unread-files and tail lines dropped"
    );
    // the cap drops the oldest lines first and is respected
    let lines: String = (0..20).map(|i| format!("item {i} has value {i}\n")).collect();
    let sel = select_fact_lines(&lines, "", 30, &words);
    assert!(!sel.is_empty() && sel.len() < 20);
    assert!(words(&sel.join("\n")) <= 30);
    assert_eq!(sel.last().unwrap(), "item 19 has value 19", "the newest line survives");
    let first = sel[0].clone();
    let n = sel.len();
    assert_eq!(first, format!("item {} has value {}", 20 - n, 20 - n), "contiguous newest lines");
    assert!(select_fact_lines("only prose here\nand more prose.", "", 256, &words).is_empty());
}

#[test]
fn fields_are_a_small_scalar_record() {
    let mut f = Fields::default();
    let m = |v: Value| v.as_object().unwrap().clone();
    f.apply(FieldsOp::Merge(m(json!({"order_id": "A-1001", "email": "a@example.com", "step": 3})))).unwrap();
    assert_eq!(f.to_json(), json!({"email": "a@example.com", "order_id": "A-1001", "step": 3}));
    f.apply(FieldsOp::Merge(m(json!({"step": 4, "email": null})))).unwrap();
    assert_eq!(f.to_json(), json!({"order_id": "A-1001", "step": 4}));
    assert_eq!(f.apply(FieldsOp::Remove("nope".into())), Ok(false));
    assert_eq!(f.apply(FieldsOp::Remove("step".into())), Ok(true));
    f.apply(FieldsOp::Replace(m(json!({"state": "waiting_for_refund"})))).unwrap();
    assert_eq!(f.to_json(), json!({"state": "waiting_for_refund"}));
    // documents are refused, and a refused op changes nothing
    assert!(f.apply(FieldsOp::Merge(m(json!({"ok": true, "doc": {"a": 1}})))).is_err());
    assert!(f.apply(FieldsOp::Merge(m(json!({"big": "x".repeat(MAX_VALUE_BYTES + 1)})))).is_err());
    assert!(f.apply(FieldsOp::Merge(m(json!({"": 1})))).is_err());
    assert_eq!(f.to_json(), json!({"state": "waiting_for_refund"}));
    let many: Map<String, Value> = (0..=MAX_FIELDS).map(|i| (format!("k{i}"), json!(i))).collect();
    assert!(f.apply(FieldsOp::Replace(many)).is_err());
    assert_eq!(f.len(), 1);
}
