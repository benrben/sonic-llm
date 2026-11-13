/* The stage while a test runs and after: the header, the headline numbers, the agent grid, the charts, the answers and the
   table of every number. Events arrive from live.js (App.on*), the run state is App.S.view (live.js dispatches). */
(function () {
  'use strict';
  var App = window.App, S = App.S, $ = App.$, V = App.V, C = App.C, esc = App.esc, many = V.many, isIn = App.isIn;
  var HAS_ROWS = ['swarm', 'shared', 'team', 'followup', 'relay', 'classify', 'extract', 'compress', 'summarize'];  // agents whose replies the page shows
  var HAS_GRID = HAS_ROWS.concat(['load', 'exact', 'burst', 'turns']);
  var STATE_WORDS = { running: 'Running', done: 'Done', stopped: 'Stopped', failed: 'Failed', lost: 'Ended', earlier: 'Saved run' };
  function scenarioOf(id) { return S.scenarios.filter(function (x) { return x.id === id; })[0] || { id: id, name: id }; }
  function boxKey(r) { return r.id === null ? '' : r.id; }  // a run started elsewhere, not named yet: ''
  App.boxKey = boxKey;
  function rtsNow() { return S.session ? Object.keys(S.session.engines) : ['sllm']; }

  // -- the header -----------------------------------------------------------------------------------------------------
  function setRunState(how) {
    var busy = how === 'running', ic = how === 'done' ? 'i-check' : how === 'failed' ? 'i-no' : how === 'stopped' || how === 'lost' ? 'i-stop' : 'i-clock';
    $('runState').className = 'pill' + (how === 'failed' ? ' bad' : '');
    $('runState').innerHTML = (busy ? '<span class="dot busy"></span>' : '<svg class="ic" aria-hidden="true"><use href="#' + ic + '"/></svg>') + STATE_WORDS[how];
  }
  function showStage(title, sub, ask) {
    S.hasRun = true; App.renderIntro(); $('run').hidden = false;
    $('runTitle').textContent = title; $('runSub').textContent = sub; $('elapsed').textContent = '';
    $('runAsk').hidden = !ask; $('runAsk').textContent = ask || '';
    $('messages').innerHTML = ''; S.runs = {}; S.swarm = {}; S.final = {}; S.live = {}; S.tilesHtml = ''; $('last').innerHTML = '';
    $('swarm').innerHTML = ''; $('swarmNote').textContent = ''; $('runVerdict').hidden = true; $('runGrade').hidden = true;
  }
  function subtitle(r, rts) {
    var bits = [S.session ? S.session.model : ''];
    if (S.askFiles && S.askFiles.length && !r.elsewhere) bits.push(many(S.askFiles.length, 'document'));
    bits.push(rts.length > 1 ? 'Sonic LLM, then llama.cpp' : 'Sonic LLM');
    return bits.filter(Boolean).join(' · ');
  }
  function startRun(r) {
    var sc = scenarioOf(r.scenario), key = boxKey(r), rts = r.engines || rtsNow();
    showStage(r.elsewhere ? 'A run is going (started elsewhere)' : sc.name, subtitle(r, rts), r.elsewhere ? '' : S.ask);
    var b = document.createElement('div'); b.className = 'runbox'; b.innerHTML = V.runHtml(rts);
    $('messages').appendChild(b);
    S.runs[key] = { el: b, scenario: r.scenario, rows: {}, t0: Date.now(), rts: rts }; S.gridRun = key; S.stage = { scenario: r.scenario, rts: rts };
    $('swarmCard').hidden = !isIn(HAS_GRID, r.scenario); $('charts').hidden = false;
    setRunState('running'); App.buildCharts(); renderTiles();
  }
  App.startRun = startRun;
  // An earlier run from the history: its numbers, not its answers (those are not kept)
  App.showRecord = function (rec) {
    var sc = scenarioOf(rec.scenario), rts = Object.keys(rec.metrics || {});
    showStage(sc.name, [rec.model, new Date(rec.t).toLocaleString(), rts.length > 1 ? 'Sonic LLM and llama.cpp' : 'Sonic LLM'].join(' · '), rec.message);
    S.stage = { scenario: rec.scenario, rts: rts }; $('swarmCard').hidden = true; $('charts').hidden = true;
    setRunState('earlier'); renderLast(rec);
  };
  function ans(run, rt) { var r = S.runs[run]; return r && r.el.querySelector('.answer[data-rt="' + rt + '"]'); }
  App.ans = ans;

  // -- the run box moves with the run state (live.js calls these) -----------------------------------------------------------
  App.moveBox = function (from, to) { S.runs[to] = S.runs[from]; delete S.runs[from]; if (S.gridRun === from) S.gridRun = to; };
  App.dropBox = function (key) { var r = S.runs[key]; if (r) { r.el.remove(); delete S.runs[key]; } };
  // A run that is over (how: done, stopped, failed, or lost: the server says it is not going): what did not finish (rows,
  // answers without their numbers, the grid's cells) leaves reading and writing.
  App.closeRun = function (key, how) {
    var run = S.runs[key], cut = how !== 'done';
    if (run) {
      Object.keys(run.rows).forEach(function (k) {
        var row = run.rows[k];
        if (cut && row.dataset.state !== 'done') { row.dataset.state = 'stopped'; row.querySelector('.vd').textContent = V.ENDED[how]; }
      });
      if (Object.keys(run.rows).length <= 3) Object.keys(run.rows).forEach(function (k) { run.rows[k].open = run.rows[k].dataset.state === 'done'; });
      run.rts.forEach(function (rt) {
        var a = ans(key, rt); if (!a) return;
        a.querySelector('.text').classList.remove('cursor');
        if (cut && !a.querySelector('.stats').innerHTML) a.querySelector('.state').textContent = V.ENDED[how];
      });
    }
    if (key === S.gridRun) {
      setRunState(how);
      Object.keys(S.live).forEach(function (rt) { if (S.live[rt].running) S.live[rt].end = Date.now(); S.live[rt].running = false; });
      if (cut) { V.stopCells(S.swarm); renderSwarm(); }
      applyLists();
      renderTiles();
    }
  };

  // -- agents: one row per agent under the answer, and the grid above -----------------------------------------------------
  function agentRow(ev) {
    var run = S.runs[ev.run]; if (!run || !isIn(HAS_ROWS, run.scenario)) return;
    var row = run.rows[ev.rt + ':' + ev.i] || newAgentRow(run, ev); if (!row) return;
    if (row.title !== ev.label) nameRow(row, ev.label);  // a relay's next lap reuses the row for another document
    row.dataset.state = ev.state;
    if (ev.state === 'done') showAgentDone(row, ev);
    else row.querySelector('.vd').textContent = ev.state === 'writing' ? 'writing · ' + many(ev.tokens, 'token') : ev.state;
  }
  // a new row under the engine's answer (none when the answer is not on the page)
  function newAgentRow(run, ev) {
    var box = ans(ev.run, ev.rt); if (!box) return null;
    var row = document.createElement('details'); row.className = 'arow';
    row.innerHTML = '<summary><span class="st"></span><span class="nm"></span><span class="vd"></span></summary><div class="body"><button class="btn quiet copy" type="button">Copy</button>' +
      '<dl class="facts" hidden></dl><div class="alabel" hidden>Answer</div><pre>Nothing yet.</pre></div>';
    nameRow(row, ev.label);
    box.querySelector('.agentlist').appendChild(row); run.rows[ev.rt + ':' + ev.i] = row;
    return row;
  }
  function nameRow(row, label) {
    row.title = label; row.querySelector('.nm').textContent = label.split('/').pop();
    row.querySelector('pre').textContent = 'Nothing yet.'; row.classList.remove('bad'); delete row.dataset.grade;
    row.querySelector('.facts').hidden = row.querySelector('.alabel').hidden = true;
  }
  function showAgentDone(row, ev) {
    var tok = C.num(ev.tokens) ? many(ev.tokens, 'token') : '', said = agentVerdict(ev), isGraded = ev.expected !== undefined;
    if (isGraded) { row.dataset.grade = ev.verdict; showFacts(row, ev); }
    row.querySelector('.vd').innerHTML = isGraded ? V.verdictHtml(ev.verdict) + (tok ? ' · ' + esc(tok) : '') : esc(said ? said + ' · ' + tok : tok);
    row.classList.toggle('bad', isGraded ? ev.verdict === 'wrong' : ev.ok === false || ev.stop === 'limit' || ev.found === 0);
    row.querySelector('pre').textContent = ev.reply || '(an empty reply)';
  }
  // what a graded agent was asked, what it should have said and how it came out, above its whole answer
  function showFacts(row, ev) {
    var checks = ev.checks ? ' (' + Object.keys(ev.checks).map(function (k) { return k + ' ' + (ev.checks[k] ? 'right' : 'wrong'); }).join(', ') + ')' : '';
    var rows = [['Document', ev.document.split('/').pop()], ['Question', ev.question], ['Expected', ev.expected]];
    row.querySelector('.facts').innerHTML = rows.map(function (r) { return '<dt>' + r[0] + '</dt><dd>' + esc(r[1]) + '</dd>'; }).join('') +
      '<dt>Verdict</dt><dd>' + V.verdictHtml(ev.verdict) + esc(checks) + '</dd>';
    row.querySelector('.facts').hidden = row.querySelector('.alabel').hidden = false;
  }
  // what an agent's job came to: the run's verdict, the questions it answered, or a reply cut at the limit
  function agentVerdict(ev) {
    return ev.verdict || (C.num(ev.found) ? 'answered ' + ev.found + ' of ' + ev.of : '') || (ev.stop === 'limit' ? 'cut at the length limit' : '');
  }
  // The answers of one engine: all of them, or the wrong / unanswered ones, PAGE rows at a time (256 agents stay light to scroll).
  var PAGE = 50;
  function applyList(box) {
    var rows = Array.prototype.slice.call(box.querySelectorAll('.arow')), st = box.listState = box.listState || { filter: 'all', page: 0 };
    var graded = rows.some(function (r) { return r.dataset.grade; });
    var match = rows.filter(function (r) { return st.filter === 'all' || r.dataset.grade === st.filter; });
    var pages = Math.max(1, Math.ceil(match.length / PAGE)); st.page = Math.min(st.page, pages - 1);
    var shown = match.slice(st.page * PAGE, (st.page + 1) * PAGE);
    rows.forEach(function (r) { r.hidden = shown.indexOf(r) < 0; });
    var head = box.querySelector('.alist-h'); head.hidden = !(graded || rows.length > PAGE);
    head.querySelector('.filter').hidden = !graded;
    head.querySelector('.pager').hidden = match.length <= PAGE;
    head.querySelector('.pg').textContent = match.length ? (st.page * PAGE + 1) + '–' + (st.page * PAGE + shown.length) + ' of ' + match.length : 'none';
    head.querySelector('[data-page="-1"]').disabled = st.page === 0; head.querySelector('[data-page="1"]').disabled = st.page >= pages - 1;
    box.querySelector('.agentlist').dataset.empty = match.length ? '' : 'No answer matches.';
  }
  $('messages').addEventListener('change', function (e) {
    var box = e.target.closest && e.target.closest('.answer'); if (!box || e.target.name.indexOf('afilter-') !== 0) return;
    box.listState = { filter: e.target.value, page: 0 }; applyList(box);
  });
  $('messages').addEventListener('click', function (e) {
    var b = e.target.closest && e.target.closest('[data-page]'), box = b && b.closest('.answer'); if (!box) return;
    box.listState.page += +b.dataset.page; applyList(box);
  });
  function applyLists() { Array.prototype.forEach.call($('messages').querySelectorAll('.answer'), applyList); }

  var swarmTimer = null;
  App.onAgent = function (ev) {
    S.swarm[ev.rt] = S.swarm[ev.rt] || {};
    if (S.view.run && S.view.run.elsewhere) $('swarmCard').hidden = false;  // its test is not known: its agents say it
    S.swarm[ev.rt][ev.i] = ev;
    agentRow(ev);
    if (!swarmTimer) swarmTimer = setTimeout(function () { swarmTimer = null; renderSwarm(); applyLists(); }, 150);
  };
  function renderSwarm() {
    var all = [], html = Object.keys(S.swarm).map(function (rt) {
      var agents = S.swarm[rt], ids = Object.keys(agents).map(Number).sort(function (a, b) { return a - b; });
      var done = ids.filter(function (i) { return agents[i].state === 'done'; }).length;
      var box = S.view.run && done < ids.length && ans(boxKey(S.view.run), rt); if (box) box.querySelector('.state').textContent = done + ' of ' + ids.length + ' done';
      all = all.concat(ids.map(function (i) { return agents[i].state; }));
      return V.swarmRowHtml(rt, ids.map(function (i) { return agents[i]; }), done);
    }).join('');
    $('swarm').innerHTML = html;
    $('legendGrade').hidden = !Object.keys(S.swarm).some(function (rt) { return Object.keys(S.swarm[rt]).some(function (i) { return S.swarm[rt][i].verdict; }); });
    $('swarmNote').textContent = Object.keys(S.swarm).length > 1 ? '' : V.countText(all);
  }
  $('messages').addEventListener('click', function (e) {
    var b = e.target.closest && e.target.closest('.copy'); if (!b) return;
    navigator.clipboard.writeText(b.parentNode.querySelector('pre').textContent).then(function () { b.textContent = 'Copied'; setTimeout(function () { b.textContent = 'Copy'; }, 1200); });
  });

  // -- the headline numbers -------------------------------------------------------------------------------------------
  function renderTiles() {
    if (!S.hasRun || !S.stage) return;
    var html = V.tilesHtml({ scenario: S.stage.scenario, rts: S.stage.rts, final: S.final, live: S.live, now: Date.now() });
    if (html !== S.tilesHtml) { S.tilesHtml = html; $('tiles').innerHTML = html; }
    $('tileKey').hidden = S.stage.rts.length < 2;
    var said = V.verdictText(S.final); $('runVerdict').hidden = !said; $('runVerdict').textContent = said;
    var score = V.gradeText(S.final); $('runGrade').hidden = !score; $('runGrade').textContent = score;
  }
  App.renderTiles = renderTiles;
  // the numbers of one engine when it is done: they replace its live readings
  function finalNumbers(rt, m) {
    S.final[rt] = m; S.live[rt] = Object.assign(S.live[rt] || {}, { running: false, end: Date.now() });
    renderTiles();
  }
  function renderLast(rec) {
    var rts = Object.keys(rec.metrics || {}).filter(function (rt) { return Object.keys(rec.metrics[rt]).length; });
    Object.keys(rec.metrics || {}).forEach(function (rt) { S.final[rt] = rec.metrics[rt]; });
    S.stage = S.stage || { scenario: rec.scenario, rts: rts };
    App.showMemNote(rec.metrics);
    $('last').innerHTML = V.numbersTable(rec.metrics, rts) + '<p class="small muted note">' + esc(rec.note) + '</p>';
    renderTiles();
  }

  // -- charts ---------------------------------------------------------------------------------------------------------
  var tokChart, readChart, gpuChart, memChart;
  var rz; window.addEventListener('resize', function () { clearTimeout(rz); rz = setTimeout(function () { App.buildCharts(); if (S.lastTick) App.onTick(S.lastTick); }, 200); });
  App.buildCharts = function () {
    if (!S.hasRun || $('charts').hidden) return;
    var rts = rtsNow();
    $('legendTok').innerHTML = rts.length > 1 ? rts.map(function (rt) { return '<span><i class="swatch ' + rt + '"></i>' + C.SERIES[rt].label + '</span>'; }).join('') : '';
    ['chartTok', 'chartRead', 'chartGpu', 'chartMem'].forEach(function (id) { $(id).innerHTML = ''; });
    var ser = rts.map(function (rt) { return { key: rt, label: C.SERIES[rt].label, color: C.SERIES[rt].color, soft: C.SERIES[rt].soft }; });
    App.showMemNote();
    tokChart = C.line($('chartTok'), { series: ser, unit: 'tok/s', windowS: App.WINDOW_S, label: 'Tokens written per second, last minute' });
    readChart = C.line($('chartRead'), { series: ser, unit: 'tok/s', windowS: App.WINDOW_S, label: 'Tokens read per second, last minute' });
    gpuChart = C.line($('chartGpu'), { series: [{ key: 'gpu', label: 'GPU busy', color: 'var(--blue)', soft: 'var(--blue-soft)' }], unit: '%', windowS: App.WINDOW_S, yMax: 100, label: 'GPU busy, percent' });
    memChart = C.line($('chartMem'), { series: ser, unit: 'MB', windowS: App.WINDOW_S, label: 'Memory of the engine, model file not counted',
      fmt: function (v) { return C.num(v) ? (v >= 1024 ? (v / 1024).toFixed(1) + ' GB' : (v >= 10 ? Math.round(v) : v.toFixed(1)) + ' MB') : '–'; } });
    if (S.lastTick) drawCharts(S.lastTick.t);
  };
  function drawCharts(now) {
    tokChart.update(V.tickPoints(S.ticks, 'tok_s'), now); readChart.update(V.tickPoints(S.ticks, 'read_tok_s'), now);
    gpuChart.update(S.ticks.map(function (p) { return { t: p.t, values: { gpu: p.ev.gpu_util } }; }), now); memChart.update(V.tickPoints(S.ticks, 'mem_mb'), now);
  }
  // the memory chart's note; the whole story is its tooltip
  App.showMemNote = function (per) {
    var el = $('memNote'), note = V.memNote(S.session, per);
    el.textContent = note[0]; el.title = note[1];
  };
  App.onTick = function (t) {
    S.lastTick = t;
    var run = S.view.run && S.runs[boxKey(S.view.run)];
    if (run) $('elapsed').textContent = Math.round((Date.now() - run.t0) / 1000) + ' s';
    Object.keys(t.engines || {}).forEach(function (rt) {
      S.live[rt] = Object.assign(S.live[rt] || {}, { tok_s: t.engines[rt].tok_s, read_tok_s: t.engines[rt].read_tok_s });
    });
    var now = t.t; S.ticks.push({ t: now, ev: t });
    S.ticks = S.ticks.filter(function (p) { return p.t >= now - App.WINDOW_S - 2; });
    if (tokChart) drawCharts(now);
    renderTiles();
  };

  // -- events of a run ------------------------------------------------------------------------------------------------
  App.onPhase = function (ev) {
    var a = ans(ev.run, ev.rt); if (!a) return;
    a.querySelector('.state').textContent = ev.msg.split(': ').pop();
    a.querySelector('.text').classList.toggle('cursor', !ev.metrics);
    if (/: running$/.test(ev.msg)) S.live[ev.rt] = Object.assign(S.live[ev.rt] || {}, { running: true, start: Date.now() });
    if (ev.metrics) { a.querySelector('.stats').innerHTML = V.statsHtml(ev.metrics); finalNumbers(ev.rt, ev.metrics); }
  };
  // the answer so far, its thinking apart
  App.onToken = function (ev) {
    var a = ans(ev.run, ev.rt); if (!a) return;
    var tx = a.querySelector('.text'), th = a.querySelector('.thought');
    a.dataset.raw = (a.dataset.raw || '') + ev.text;
    var parts = V.splitThinking(a.dataset.raw);
    th.textContent = parts[0]; tx.textContent = parts[1];
    th.scrollTop = th.scrollHeight; tx.scrollTop = tx.scrollHeight;
  };
  App.onRunDone = function (ev) {
    App.closeRun(ev.run, V.doneHow(ev));  // the run on the page is closed already (dispatch); this closes an earlier one
    if (ev.error) App.notify(ev.error, 'error', 'The run failed');
    else if (ev.record && ev.run === S.gridRun) renderLast(ev.record);
  };
})();
