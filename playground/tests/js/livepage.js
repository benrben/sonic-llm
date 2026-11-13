/* The live page (charts.js, liveview.js, live.js) loaded in the harness: a fake server answers /api/live/state from
   `page.server`, the event stream is a fake SharedWorker port the test feeds with emit(). */
'use strict';
const { makeContext, load, response, settle } = require('./harness');

const SCENARIOS = [
  { id: 'chat', name: 'Chat', group: 'Talk', what: 'talk' },
  { id: 'swarm', name: 'Swarm', group: 'Documents', what: 'one agent per file', agents: 8 },
  { id: 'relay', name: 'Relay', group: 'Agents', what: 'laps of fresh agents', agents: 2 }
];

function session(extra) {
  return Object.assign({ model: 'm', state: 'ready', busy: false, engine: 'next', notes: 'auto', compare: false, exact: null,
    engines: { sllm: { label: 'Sonic LLM', ctx: 4096, max_parallel: 8, cells: 65536, ready_s: 1.5 } } }, extra || {});
}

async function openPage(opts) {
  opts = opts || {};
  const page = { server: { session: opts.session === undefined ? session() : opts.session, scenarios: SCENARIOS }, port: null, posted: [] };
  const routes = Object.assign({
    'GET /api/live/state': () => response(JSON.parse(JSON.stringify(page.server))),
    'GET /api/models': { models: [] },
    'GET /api/setup': { engine: { found: true, path: '/e', build: 'build' }, llama: { found: false }, models_dir: '/m', offline: false },
    'GET /api/live/files': { files: [], root: '/lib' },
    'POST /api/live/run': { run: 'r-posted' },
    'POST /api/live/run/stop': { ok: true }
  }, opts.routes || {});
  const box = makeContext({ routes, hold: opts.hold });
  const ctx = box.ctx;
  ctx.SharedWorker = function () {
    this.port = { onmessage: null, start() {}, postMessage: (m) => page.posted.push(m) };
    page.port = this.port;
  };
  load(ctx, ['charts.js', 'liveview.js']);
  ctx.Charts.line = () => ({ update() {} });  // the SVG charts need a real layout
  load(ctx, ['live.js']);
  await settle();
  Object.assign(page, box, {
    $: (id) => box.ctx.document.getElementById(id),
    emit: async (ev) => { page.port.onmessage({ data: { data: JSON.stringify(ev) } }); await settle(); },
    up: async () => { page.port.onmessage({ data: { up: true } }); await settle(); },
    stateCalls: () => box.calls.filter((c) => c.key === 'GET /api/live/state').length,
    // the message boxes of runs (not the user's own lines)
    runBoxes: () => page.$('messages').children.filter((c) => c.className === 'msg bot' && c.html.indexOf('class="scen"') >= 0),
    answer: (b, rt) => b.querySelector('.answer[data-rt="' + (rt || 'sllm') + '"]'),
    rows: (b, rt) => page.answer(b, rt).querySelector('.agentlist').children
  });
  return page;
}

module.exports = { openPage, session, SCENARIOS };
