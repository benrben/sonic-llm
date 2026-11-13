/* The live page's run and session state against the server's (findings 4 and 21 of the state-machine report). */
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { openPage, session } = require('./livepage');
const { settle } = require('./harness');

const START = { type: 'run_start', run: 'r1', scenario: 'swarm', engines: ['sllm'] };
const agent = (i, state, extra) => Object.assign({ type: 'agent', run: 'r1', rt: 'sllm', i, label: 'docs/f' + i + '.md', state }, extra || {});
const LIVE = ['reading', 'writing'];

test('a missed run_done: the next refresh clears Stopping and lets the person send again', async () => {
  const p = await openPage();
  await p.emit(START);
  p.server.session = session({ busy: true });
  p.$('send').onclick();  // Stop
  await settle();
  assert.equal(p.$('send').textContent, 'Stopping…');
  p.server.session = session({ busy: false });  // the run ended; its run_done never came
  await p.up();
  assert.equal(p.$('send').textContent, 'Send');
  assert.equal(p.$('send').disabled, false);
});

test('a run the server says is over is closed: no row stays reading or writing', async () => {
  const p = await openPage();
  await p.emit(START);
  await p.emit(agent(0, 'reading'));
  await p.emit(agent(1, 'writing', { tokens: 3 }));
  await p.up();  // server: not busy
  const rows = p.rows(p.runBoxes()[0]);
  assert.equal(rows.length, 2);
  rows.forEach((r) => assert.ok(LIVE.indexOf(r.dataset.state) < 0, 'row still ' + r.dataset.state));
});

test('run_done with an error moves every row and the answer out of reading and writing', async () => {
  const p = await openPage();
  await p.emit(START);
  await p.emit({ type: 'phase', run: 'r1', rt: 'sllm', msg: 'Sonic LLM: reading 3 files' });
  await p.emit(agent(0, 'reading'));
  await p.emit(agent(1, 'writing', { tokens: 3 }));
  await p.emit({ type: 'run_done', run: 'r1', error: 'RuntimeError: boom' });
  const b = p.runBoxes()[0];
  p.rows(b).forEach((r) => assert.ok(LIVE.indexOf(r.dataset.state) < 0, 'row still ' + r.dataset.state));
  assert.notEqual(p.answer(b).querySelector('.state').textContent, 'reading 3 files');
  assert.equal(p.answer(b).querySelector('.text').classList.contains('cursor'), false);
  assert.ok(p.$('messages').children.some((c) => c.html.indexOf('boom') >= 0));
});

test('a page opened mid-run shows the run and keeps its events', async () => {
  const p = await openPage({ session: session({ busy: true }) });
  const boxes = p.runBoxes();
  assert.equal(boxes.length, 1);
  assert.ok(boxes[0].html.indexOf('started elsewhere') >= 0);
  await p.emit({ type: 'token', run: 'r9', rt: 'sllm', text: 'hello' });
  assert.equal(p.answer(p.runBoxes()[0]).querySelector('.text').textContent, 'hello');
  assert.equal(p.$('send').textContent, 'Stop');
  await p.emit({ type: 'run_done', run: 'r9', record: null });
  assert.equal(p.runBoxes().length, 1);
  assert.equal(p.$('send').textContent, 'Send');
});

test('a run started elsewhere while the page is open gets a box when the server says busy', async () => {
  const p = await openPage();
  p.server.session = session({ busy: true });
  await p.up();
  assert.equal(p.runBoxes().length, 1);
  await p.emit({ type: 'run_start', run: 'r2', scenario: 'chat', engines: ['sllm'] });  // its start comes late: one box, not two
  assert.equal(p.runBoxes().length, 1);
  assert.ok(p.runBoxes()[0].html.indexOf('Chat') >= 0);
});

test('the snapshot event is applied like a refresh', async () => {
  const p = await openPage();
  await p.emit(START);
  assert.equal(p.$('send').textContent, 'Stop');
  await p.emit({ type: 'snapshot', state: { session: session({ busy: false }) } });
  assert.equal(p.$('send').textContent, 'Send');
});

test('a resync event asks the server for its state', async () => {
  const p = await openPage();
  const before = p.stateCalls();
  await p.emit({ type: 'resync' });
  assert.equal(p.stateCalls(), before + 1);
});

test('an engine that is restarting shows so in the status chip', async () => {
  const p = await openPage();
  const s = session({ busy: true });
  s.engines.sllm.restarting = true;
  await p.emit({ type: 'session', session: s });
  assert.match(p.$('status').textContent, /restarting/i);
});

test('a session that says it is restarting shows so in the status chip', async () => {
  const p = await openPage();
  await p.emit({ type: 'session', session: session({ busy: true, restarting: true }) });
  assert.match(p.$('status').textContent, /restarting/i);
});

test('a load that failed in another tab is said in this one', async () => {
  const p = await openPage();
  await p.emit({ type: 'session', session: null, error: 'out of memory' });
  assert.ok(p.$('messages').children.some((c) => c.html.indexOf('out of memory') >= 0));
});

test('a /state answer older than a run start does not end that run', async () => {
  const p = await openPage();
  // hold the next /state answer: ask, start a run while it is on its way, then answer "not busy" from before the start
  const held = [];
  p.ctx.fetch = ((orig) => (url, init) => (url === '/api/live/state' ? new Promise((res) => held.push(() => res(orig(url, init)))) : orig(url, init)))(p.ctx.fetch);
  await p.up();
  await p.emit(START);
  held.forEach((go) => go());
  await settle();
  assert.equal(p.$('send').textContent, 'Stop');
  await p.emit(agent(0, 'writing', { tokens: 1 }));
  assert.equal(p.rows(p.runBoxes()[0])[0].dataset.state, 'writing');
});

test('the POST /run answer starts the run even when its run_start is lost', async () => {
  const p = await openPage();
  p.$('input').value = 'hi';
  p.$('send').onclick();
  await settle();
  assert.equal(p.$('send').textContent, 'Stop');
});

test('a relay lap that reuses a row shows the new file name, not the first lap\'s', async () => {
  const p = await openPage();
  await p.emit({ type: 'run_start', run: 'r1', scenario: 'relay', engines: ['sllm'] });
  await p.emit({ type: 'agent', run: 'r1', rt: 'sllm', i: 0, label: 'docs/a.md', state: 'done', tokens: 5, reply: 'FACT: a' });
  await p.emit({ type: 'agent', run: 'r1', rt: 'sllm', i: 0, label: 'docs/b.md', state: 'reading' });
  const row = p.rows(p.runBoxes()[0])[0];
  assert.equal(row.querySelector('.nm').textContent, 'b.md');
  assert.equal(row.title, 'docs/b.md');
  assert.notEqual(row.querySelector('pre').textContent, 'FACT: a');
});
