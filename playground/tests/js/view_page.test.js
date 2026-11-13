/* The test cards, the first screen, the plan line, the document list and the models rows, as plain HTML. */
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { makeContext, load } = require('./harness');

const { ctx } = makeContext({});
load(ctx, ['charts.js', 'state.js', 'numbers.js', 'liveview.js', 'testsview.js', 'modelsview.js']);
const V = ctx.LiveView;
const plain = (x) => JSON.parse(JSON.stringify(x));  // objects made in the page's context are another realm's
const IDS = ['chat', 'long', 'swarm', 'team', 'followup', 'relay', 'turns', 'shared', 'classify', 'extract', 'summarize', 'compress', 'load', 'burst', 'exact'];
const GROUP = { chat: 'Talk', long: 'Talk', swarm: 'Many agents', team: 'Many agents', followup: 'Many agents', relay: 'Many agents', turns: 'Many agents',
  shared: 'Documents', classify: 'Documents', extract: 'Documents', summarize: 'Documents', compress: 'Documents', load: 'Checks', burst: 'Checks', exact: 'Checks' };
const scenarios = IDS.map((id) => ({ id, name: 'Name ' + id, group: GROUP[id], what: 'What ' + id }));

test('all fifteen tests are cards in four groups, each a radio button with its own promise', () => {
  const html = V.testCardsHtml(scenarios, 'swarm');
  assert.equal((html.match(/<fieldset class="group">/g) || []).length, 4);
  assert.deepEqual([...html.matchAll(/<legend>([^<]+)<\/legend>/g)].map((m) => m[1]), ['Talk', 'Many agents', 'Documents', 'Checks']);
  assert.equal((html.match(/type="radio" name="test"/g) || []).length, 15);
  assert.equal((html.match(/ checked>/g) || []).length, 1);
  assert.match(html, /data-t="swarm"[^>]*>/);
  IDS.forEach((id) => assert.match(html, new RegExp('What ' + id)));
});

test('every test says what it needs, and burst needs only agents', () => {
  IDS.forEach((id) => assert.ok(V.NEEDS[id], id));
  assert.deepEqual(plain(V.NEEDS.burst), { agents: true });
  assert.ok(V.NEEDS.chat.level && !V.NEEDS.relay.level);
  assert.ok(!V.NEEDS.shared.ask && V.NEEDS.team.ask);
});

test('the first screen explains itself in three steps and follows the setup', () => {
  const none = V.introHtml({ models: 0, model: '', llama: false });
  assert.match(none, /Watch many AI agents run on your Mac/);
  assert.equal((none.match(/<li class="step/g) || []).length, 3);
  assert.match(none, /id="introModels"/);
  assert.match(none, /id="introWhy"/);
  const ready = V.introHtml({ models: 2, model: 'qwen3', llama: true });
  assert.match(ready, /qwen3 is loaded and ready/);
  assert.match(ready, /step done/);
  assert.doesNotMatch(ready, /introModels|introWhy/);
});

test('a missing engine gets the build command with a copy button', () => {
  const html = V.engineMissingHtml({ build: 'cargo build --release -p sllm-server', path: '/x/sllm-server' });
  assert.match(html, /cd engine &amp;&amp; cargo build --release -p sllm-server/);
  assert.match(html, /data-copy=/);
  assert.match(html, /SONIC_SERVER_BIN/);
});

test('the plan line says what Run will do, in documents not files', () => {
  const k = { n: 0, a: 9, perFile: true, first: '', files: 9, max: 256, par: null, maxThink: false, level: 'off' };
  assert.equal(V.planText('swarm', k).t, 'No documents picked: will run 9 agents on the 9 biggest documents.');
  assert.equal(V.planText('swarm', { ...k, n: 3, a: 3 }).t, 'Will run 3 agents, one per document.');
  assert.match(V.planText('burst', { ...k, perFile: false, a: 8 }).t, /8 agents write 256 tokens/);
  assert.match(V.planText('long', { ...k, perFile: false, level: 'fast' }).t, /do not apply/);
});

test('a document of your own can be removed after one more question; a sample cannot', () => {
  const mine = { path: 'uploads/mine.md', bytes: 400, est_tokens: 100 }, sample = { path: 'samples/a.md', bytes: 400, est_tokens: 100 };
  const asked = V.fileGroupHtml('uploads', [mine], [], true, null), sure = V.fileGroupHtml('uploads', [mine], [], true, 'uploads/mine.md');
  assert.match(asked, /data-ask="uploads\/mine.md"/);
  assert.doesNotMatch(asked, /data-del/);
  assert.match(sure, /data-del="uploads\/mine.md"/);
  assert.match(sure, /data-keep/);
  assert.doesNotMatch(V.fileGroupHtml('samples', [sample], [], true, null), /data-ask|data-del/);
});

test('fit and engine hints are labelled pills with an icon, never colour alone', () => {
  const fits = V.fitPill('fits easily', 9), big = V.fitPill('too large', 120), tight = V.fitPill('tight', 30);
  assert.match(fits, /Memory: fits easily/);
  assert.match(tight, /Memory: tight fit/);
  assert.match(big, /Memory: too large for this Mac/);
  [fits, tight, big].forEach((h) => assert.match(h, /<svg class="ic"/));
  assert.equal(V.fitPill('unknown'), '');
  assert.match(V.enginePill({ engine: 'own', text: 'x' }), /Engine: Sonic LLM/);
  assert.match(V.enginePill({ engine: 'fallback', text: 'x' }), /Engine: fallback, slower/);
});

test('a model on this Mac can be loaded or removed; the one loaded is marked', () => {
  const ms = [{ id: 'a', quant: 'Q4_K_M', size_bytes: 2.5e9, arch: 'qwen3', engine: { engine: 'own', text: '' } }];
  const html = V.installedHtml(ms, 'a', null);
  assert.match(html, /data-load="a"/);
  assert.match(html, /data-ask="a"/);
  assert.match(html, /Loaded now/);
  assert.match(V.installedHtml(ms, null, 'a'), /data-remove="a"/);
  assert.match(V.installedHtml([], null, null), /No models on this Mac yet/);
});

test('downloads show progress with a label and a cancel button; finished ones only when fresh', () => {
  const ds = [{ id: 'd1', name: 'x.gguf', state: 'downloading', bytes_done: 1e9, bytes_total: 4e9, percent: 25, speed_bps: 5e7 },
    { id: 'd2', name: 'old.gguf', state: 'done' }, { id: 'd3', name: 'new.gguf', state: 'done' }];
  const html = V.downloadsHtml(ds, { d3: true });
  assert.match(html, /<progress max="4000000000" value="1000000000" aria-label="Downloading x.gguf">/);
  assert.match(html, /data-cancel="d1"/);
  assert.match(html, /1\.0 GB of 4\.0 GB · 25% · 50\.0 MB\/s/);
  assert.doesNotMatch(html, /old.gguf/);
  assert.match(html, /data-load="new"/);
  assert.equal(V.downloadsActive(ds), true);
});

test('the plan says whether the run is graded, when too many agents do not fit and when a comparison will be slow', () => {
  const base = { n: 0, a: 8, perFile: true, files: 256, max: 256, level: 'off', room: null, compare: false };
  assert.match(V.planText('swarm', { ...base, grade: 'yes' }).t, /Graded: each answer is checked against an answer key\./);
  assert.match(V.planText('swarm', { ...base, grade: 'typed' }).t, /Not graded: you wrote your own question\./);
  assert.match(V.planText('swarm', { ...base, grade: 'unkeyed' }).t, /Not graded: some documents have no answer key\./);
  const over = V.planText('swarm', { ...base, a: 100, room: 64 });
  assert.match(over.t, /More than 64 agents do not all fit/);
  assert.equal(over.warn, true);
  const slow = V.planText('swarm', { ...base, a: 100, compare: true });
  assert.match(slow.t, /With llama\.cpp compared, a run of this many agents takes much longer\./);
  assert.doesNotMatch(V.planText('swarm', { ...base, a: 32, compare: true }).t, /llama\.cpp compared/);
});
