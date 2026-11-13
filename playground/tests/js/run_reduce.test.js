/* The page's run and session state on its own: liveview.js's reduceRun, a pure function (state, event) -> state. */
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { makeContext, load } = require('./harness');

function liveView() {
  const { ctx } = makeContext({});
  load(ctx, ['charts.js', 'state.js']);
  return ctx.LiveView;
}
const V = liveView();
const run = (...evs) => evs.reduce((st, ev) => V.reduceRun(st, ev), V.RUN0);
const busy = (b) => ({ type: 'server', fresh: true, state: { session: { model: 'm', busy: b } } });

test('a run is set exactly while running or stopping, whatever the events', () => {
  const ids = ['a', 'b', 'c'];
  const make = [
    () => ({ type: 'send' }), () => ({ type: 'send_failed' }), () => ({ type: 'stop' }), () => ({ type: 'stop_failed' }),
    (r) => ({ type: 'sent', run: r }), (r) => ({ type: 'run_start', run: r, scenario: 'chat', engines: ['sllm'] }),
    (r) => ({ type: 'run_done', run: r }), (r) => ({ type: 'token', run: r }), () => busy(true), () => busy(false),
    () => ({ type: 'server', fresh: false, state: { session: null } }), () => ({ type: 'session', session: null }),
    () => ({ type: 'load' }), () => ({ type: 'loaded', session: { busy: false } }), () => ({ type: 'unloaded' })
  ];
  let seed = 7;
  const rnd = (n) => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed % n; };
  for (let k = 0; k < 400; k++) {
    let st = V.RUN0;
    for (let j = 0; j < 30; j++) {
      st = V.reduceRun(st, make[rnd(make.length)](ids[rnd(ids.length)]));
      assert.equal(st.run !== null, V.isBusy(st), JSON.stringify(st));
      assert.ok(['idle', 'sending', 'running', 'stopping'].indexOf(st.phase) >= 0);
    }
  }
});

test('Stop stays asked when the run is named, not for the next run', () => {
  let st = run(busy(true), { type: 'stop' }, { type: 'token', run: 'r1' });
  assert.equal(st.phase, 'stopping');
  assert.equal(st.run.id, 'r1');
  st = V.reduceRun(st, { type: 'run_start', run: 'r2', scenario: 'chat' });
  assert.equal(st.phase, 'running');
});

test('a late event of a run that ended does not name the run going elsewhere', () => {
  const st = run({ type: 'run_start', run: 'r1' }, { type: 'run_done', run: 'r1' }, busy(true), { type: 'agent', run: 'r1' });
  assert.equal(st.run.id, null);
  assert.equal(V.reduceRun(st, { type: 'agent', run: 'r2' }).run.id, 'r2');
});

test('an answer that is not fresh updates the session only', () => {
  const st = run({ type: 'run_start', run: 'r1' }, { type: 'server', fresh: false, state: { session: { busy: false } } });
  assert.equal(st.phase, 'running');
});

test('POST /run answered after its run ended leaves the page idle', () => {
  const st = run({ type: 'send' }, { type: 'run_start', run: 'r1' }, { type: 'run_done', run: 'r1' }, { type: 'sent', run: 'r1' });
  assert.equal(st.phase, 'idle');
  const lost = run({ type: 'send' }, { type: 'run_done', run: 'r1' }, { type: 'sent', run: 'r1' });  // its run_start was lost
  assert.equal(lost.phase, 'idle');
});

