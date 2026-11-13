/* The shared event stream (events.js, a SharedWorker) with fake ports and a fake EventSource (finding 6). */
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { STATIC, tick } = require('./harness');

function worker() {
  const streams = [];
  const ctx = { setTimeout, clearTimeout, console };
  ctx.self = ctx;
  ctx.EventSource = function (url) { this.url = url; this.readyState = 0; this.closed = false; streams.push(this); };
  ctx.EventSource.prototype.close = function () { this.closed = true; this.readyState = 2; };
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(path.join(STATIC, 'events.js'), 'utf8'), ctx, { filename: 'events.js' });
  const join = () => {
    const port = { got: [], onmessage: null, start() {}, postMessage(m) { this.got.push(JSON.parse(JSON.stringify(m))); } };
    ctx.onconnect({ ports: [port] });
    return port;
  };
  const say = (port, msg) => port.onmessage({ data: msg });
  const live = () => streams.filter((s) => !s.closed);
  return { ctx, streams, join, say, live };
}

test('a second bye from a tab does not cut off another tab', () => {
  const w = worker();
  const a = w.join(), b = w.join();
  w.say(a, 'bye');
  w.say(a, 'bye');  // pagehide again (a page back from the back/forward cache)
  w.live()[0].onmessage({ data: 'x' });
  assert.deepEqual(b.got.filter((m) => m.data), [{ data: 'x' }]);
});

test('a tab back from the back/forward cache says hi and gets events again', () => {
  const w = worker();
  const a = w.join();
  w.join();
  w.say(a, 'bye');
  w.say(a, 'hi');
  w.live()[0].onmessage({ data: 'y' });
  assert.deepEqual(a.got.filter((m) => m.data), [{ data: 'y' }]);
});

test('the stream closes when the last tab leaves and opens again for the next', () => {
  const w = worker();
  const a = w.join();
  w.say(a, 'bye');
  assert.equal(w.live().length, 0);
  w.join();
  assert.equal(w.live().length, 1);
});

test('a tab that joins while the stream is down is told so', () => {
  const w = worker();
  w.join();
  w.live()[0].onerror();
  const late = w.join();
  assert.deepEqual(late.got, [{ down: true }]);
});

test('a stream the browser gave up on is opened again', async () => {
  const w = worker();
  const a = w.join();
  const s = w.live()[0];
  s.readyState = 2;  // CLOSED: the browser will not retry (an HTTP error answer)
  s.onerror();
  await tick(2100);
  assert.equal(w.live().length, 1);
  assert.notEqual(w.live()[0], s);
  w.live()[0].onopen();
  assert.deepEqual(a.got[a.got.length - 1], { up: true });
});
