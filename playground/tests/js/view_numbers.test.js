/* The headline numbers: how one engine compares with the other, which tiles a test shows, and the words around them. */
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { makeContext, load } = require('./harness');

const { ctx } = makeContext({});
load(ctx, ['charts.js', 'state.js', 'numbers.js', 'liveview.js', 'testsview.js']);
const V = ctx.LiveView;
const plain = (x) => JSON.parse(JSON.stringify(x));  // objects made in the page's context are another realm's

test('a ratio says which engine is ahead, in words, and never a bare percentage', () => {
  assert.deepEqual(plain(V.compare(360, 100, false)), { text: '3.6x faster', tone: 'win', times: '3.6x' });
  assert.equal(V.compare(10, 36, true).text, '3.6x faster');  // less time is faster
  assert.equal(V.compare(100, 150, false).text, '1.5x slower');
  assert.equal(V.compare(12, 10, true).text, '1.2x slower');
  assert.equal(V.compare(100, 120, false).tone, 'lose');
  assert.equal(V.compare(1500, 100, false).text, '15x faster');
});

test('within five percent is about the same, either way', () => {
  assert.equal(V.compare(100, 104, false).text, 'about the same');
  assert.equal(V.compare(104, 100, false).text, 'about the same');
  assert.equal(V.compare(100, 106, false).tone, 'lose');
});

test('no ratio without two real numbers', () => {
  assert.equal(V.compare(null, 5, false), null);
  assert.equal(V.compare(5, 0, false), null);
  assert.equal(V.compare(undefined, undefined, true), null);
});

test('values come with their unit: milliseconds turn into seconds, memory into gigabytes', () => {
  assert.equal(V.valueText(34, 'ms'), '34 ms');
  assert.equal(V.valueText(1234, 'ms'), '1.2 s');
  assert.equal(V.valueText(4.7, 's'), '4.7 s');
  assert.equal(V.valueText(5120, 'MB'), '5.0 GB');
  assert.equal(V.valueText(412.3, 'tok/s'), '412 tok/s');
  assert.equal(V.valueText(null, 's'), '–');
});

test('the tiles of a test: chat has no read speed, a document test has four', () => {
  assert.deepEqual(plain(V.tileSlots('chat', {}).map((s) => s.label)), ['Total time', 'First word', 'Write speed']);
  assert.deepEqual(plain(V.tileSlots('swarm', {}).map((s) => s.label)), ['Total time', 'First word', 'Read speed', 'Write speed']);
  assert.deepEqual(plain(V.tileSlots('load', {}).map((s) => s.label)), ['Total time', 'First word under load', 'Write speed under load']);
});

test('once numbers are in, a tile nobody has a number for goes away', () => {
  const final = { sllm: { wall_s: 2, ttft_ms: 40, write_tok_s: 300 } };
  assert.deepEqual(plain(V.tileSlots('swarm', final).map((s) => s.label)), ['Total time', 'First word', 'Write speed']);
});

test('a comparison tile shows both engines and the ratio; one engine shows no ratio', () => {
  const both = V.tilesHtml({ scenario: 'chat', rts: ['sllm', 'llama'], now: 0, live: {},
    final: { sllm: { wall_s: 1, ttft_ms: 50, decode_tok_s: 360 }, llama: { wall_s: 3.6, ttft_ms: 500, decode_tok_s: 100 } } });
  assert.match(both, /3\.6x faster/);
  assert.match(both, /10x faster/);
  assert.match(both, /class="t-row sllm"/);
  assert.match(both, /class="t-row llama"/);
  assert.match(both, /Sonic LLM: /);  // the engine's name is there for a screen reader
  const solo = V.tilesHtml({ scenario: 'chat', rts: ['sllm'], now: 0, live: {}, final: { sllm: { wall_s: 1, ttft_ms: 50, decode_tok_s: 360 } } });
  assert.doesNotMatch(solo, /faster|slower|about the same/);
});

test('before its own numbers arrive an engine shows what it does now, marked live', () => {
  const html = V.tilesHtml({ scenario: 'chat', rts: ['sllm'], now: 5000, final: {}, live: { sllm: { running: true, start: 2000, tok_s: 120 } } });
  assert.match(html, /3\.0<\/span><span class="t-unit">s/);
  assert.match(html, /120<\/span><span class="t-unit">tok\/s<\/span><span class="t-live">live/);
  const idle = V.tilesHtml({ scenario: 'chat', rts: ['sllm'], now: 5000, final: {}, live: { sllm: { running: false, tok_s: 120 } } });
  assert.doesNotMatch(idle, /live/);
});

test('the verdict of a finished comparison is one sentence naming the winner', () => {
  assert.equal(V.verdictText({ sllm: { wall_s: 1 }, llama: { wall_s: 2.9 } }), 'Sonic LLM finished 2.9x faster than llama.cpp.');
  assert.equal(V.verdictText({ sllm: { wall_s: 3 }, llama: { wall_s: 2 } }), 'llama.cpp finished 1.5x faster than Sonic LLM.');
  assert.equal(V.verdictText({ sllm: { wall_s: 2 }, llama: { wall_s: 2.02 } }), 'Both engines finished in about the same time.');
  assert.equal(V.verdictText({ sllm: { wall_s: 2 } }), '');
});

test('the table of every number has a column per engine and the difference in words', () => {
  const html = V.numbersTable({ sllm: { wall_s: 1, peak_mem_mb: 2048 }, llama: { wall_s: 4, peak_mem_mb: 3072 } }, ['sllm', 'llama']);
  assert.match(html, /Total time/);
  assert.match(html, /4\.0x faster/);
  assert.match(html, /2\.0 GB/);
  assert.doesNotMatch(html, /KV|prefill|ttft|decode/i);
});

test('a saved run gets short headlines: the comparison when both ran, the number when one did', () => {
  assert.deepEqual(plain(V.headlines({ metrics: { sllm: { write_tok_s: 300, wall_s: 2 }, llama: { write_tok_s: 100, wall_s: 6 } } })), ['Write speed 3.0x faster', 'Total time 3.0x faster']);
  assert.deepEqual(plain(V.headlines({ metrics: { sllm: { write_tok_s: 300, wall_s: 2 } } })), ['Write speed 300 tok/s', 'Total time 2.0 s']);
});

test('notes under an answer say identical or different in words and with an icon', () => {
  assert.match(V.statsHtml({ exact: true }), /Identical words/);
  assert.match(V.statsHtml({ exact: false, first_diff: 7 }), /differ at token 7/);
  assert.match(V.statsHtml({ agents: 9, agents_asked: 256, agents_note: 'one per document' }), /9 of 256 asked \(one per document\)/);
});

test('a tile abbreviates a very big speed, the table keeps every digit', () => {
  const m = { sllm: { wall_s: 1, read_only_tok_s: 18917 } };
  assert.match(V.tilesHtml({ scenario: 'swarm', rts: ['sllm'], now: 0, live: {}, final: m }), /18\.9k/);
  assert.match(V.numbersTable(m, ['sllm']), /18,917 tok\/s/);
});

test('a run that was cut keeps the time it had reached, marked so far, and stops counting', () => {
  const live = { sllm: { running: false, start: 1000, end: 4000 } };
  const html = V.tilesHtml({ scenario: 'chat', rts: ['sllm'], now: 99000, final: {}, live });
  assert.match(html, /3\.0<\/span><span class="t-unit">s<\/span><span class="t-live">so far/);
});

const GRADED = { graded: true, correct: 61, wrong: 2, empty: 1 };

test('a graded run gets a Correct tile with the score, the percentage and the wrong and unanswered counts in words', () => {
  const html = V.tilesHtml({ scenario: 'swarm', rts: ['sllm', 'llama'], now: 0, live: {}, final: { sllm: { wall_s: 1, ...GRADED }, llama: { wall_s: 2, graded: true, correct: 60, wrong: 3, empty: 1 } } });
  assert.match(html, /<h4><svg[^>]*><use href="#i-check"\/><\/svg>Correct<\/h4>/);
  assert.match(html, /61 of 64<\/span><span class="t-unit">95%/);
  assert.match(html, /60 of 64<\/span><span class="t-unit">94%/);
  assert.match(html, /2 wrong/);
  assert.match(html, /1 no answer/);
});

test('an ungraded run has no Correct tile, and says plainly that it was not graded', () => {
  const final = { sllm: { wall_s: 1, graded: false, graded_note: 'Not graded: this test has no single right answer.' } };
  assert.doesNotMatch(V.tilesHtml({ scenario: 'chat', rts: ['sllm'], now: 0, live: {}, final }), /Correct/);
  assert.equal(V.gradeText(final), 'Not graded: this test has no single right answer.');
  assert.equal(V.gradeText({ sllm: { wall_s: 1 } }), '');  // a run saved before grading existed
});

test('the line under the verdict names the score of each engine', () => {
  assert.equal(V.gradeText({ sllm: GRADED }), 'Sonic LLM got 61 of 64 right.');
  assert.equal(V.gradeText({ sllm: GRADED, llama: { ...GRADED } }), 'Both engines got 61 of 64 right.');
  assert.equal(V.gradeText({ sllm: GRADED, llama: { ...GRADED, correct: 60, wrong: 3 } }), 'Sonic LLM 61, llama.cpp 60 of 64 right.');
});

test('every verdict has an icon and a word, and the history shows the score of a graded run', () => {
  assert.match(V.verdictHtml('correct'), /i-check.*Correct/);
  assert.match(V.verdictHtml('wrong'), /i-no.*Wrong/);
  assert.match(V.verdictHtml('no answer'), /i-dash.*No answer/);
  assert.ok(plain(V.headlines({ metrics: { sllm: { wall_s: 2, ...GRADED } } })).includes('Sonic LLM got 61 of 64 right.'));
});

test('a graded agent is a tile with a mark: a check, a cross or a dash, named in its title', () => {
  const cell = (verdict) => V.swarmRowHtml('sllm', [{ state: 'done', label: 'samples/a.md', ttft_ms: 5, tok_s: 9, verdict }], 1);
  assert.match(cell('correct'), /cell done v-ok" title="[^"]*Correct/);
  assert.match(cell('wrong'), /cell done v-bad" title="[^"]*Wrong/);
  assert.match(cell('no answer'), /cell done v-none" title="[^"]*No answer/);
});

const BY = { by_level: { medium: [18, 24], hard: [9, 20] }, by_skill: { arithmetic: [3, 9], update: [8, 10] } };

test('the Correct tile breaks the score down by the levels that were asked, and by fields for an extract run', () => {
  const html = V.tilesHtml({ scenario: 'swarm', rts: ['sllm'], now: 0, live: {}, final: { sllm: { wall_s: 1, ...GRADED, ...BY } } });
  assert.match(html, /medium 18\/24 · hard 9\/20/);
  assert.doesNotMatch(html, /easy/);  // a level nobody was asked is not shown
  assert.equal(V.levelsText({ by_level: { hard: [2, 5] } }), 'hard 2/5');
  assert.equal(V.levelsText(GRADED), '');  // a run saved before levels existed
  const extract = V.tilesHtml({ scenario: 'extract', rts: ['sllm'], now: 0, live: {}, final: { sllm: { wall_s: 1, ...GRADED, fields_right: 31, fields_asked: 40 } } });
  assert.match(extract, /31 of 40 fields right/);
});

test('All numbers shows the score by skill for each engine, and the history pill shows the hard score', () => {
  const html = V.skillsTable({ sllm: { ...GRADED, ...BY } }, ['sllm']);
  assert.match(html, /Right by skill/);
  assert.match(html, /<th scope="row">arithmetic<\/th><td class="sllm">3 of 9<\/td>/);
  assert.match(html, /<th scope="row">update<\/th><td class="sllm">8 of 10<\/td>/);
  assert.equal(V.skillsTable({ sllm: { wall_s: 1 } }, ['sllm']), '');
  const pills = plain(V.headlines({ metrics: { sllm: { wall_s: 2, ...GRADED, ...BY } } }));
  assert.ok(pills.includes('Sonic LLM got 61 of 64 right.') && pills.includes('hard 9/20'));
});

test('the difficulty buttons are medium, hard and mixed, and the plan names the difficulty of a graded run', () => {
  assert.deepEqual(plain(V.DIFFICULTIES), ['medium', 'hard', 'mixed']);
  assert.match(V.difficultyHtml('mixed'), /value="mixed" checked/);
  assert.doesNotMatch(V.difficultyHtml('mixed'), /easy/i);
  const plan = V.planText('swarm', { n: 8, a: 8, perFile: true, first: 'a.md', files: 256, max: 256, level: 'off', grade: 'yes', difficulty: 'hard' });
  assert.match(plan.t, /Graded: each answer is checked against an answer key\. Difficulty: hard\./);
});
