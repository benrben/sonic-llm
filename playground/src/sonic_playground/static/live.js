/* The live playground page: model bar, chat with scenarios and files, live metrics. Server: /api/live/* (live.py). */
(function () {
  'use strict';
  var C = window.Charts, esc = C.esc, $ = function (id) { return document.getElementById(id); };
  var V = window.LiveView, many = V.many, splitThinking = V.splitThinking, statsHtml = V.statsHtml;  // liveview.js
  // S.view: the run and session state, changed only by dispatch (V.reduceRun); S.session mirrors S.view.session
  var S = { view: V.RUN0, session: null, scenarios: [], scen: 'chat', files: [], picked: [], picks: { chat: [], jobs: [] }, open: {}, shown: {}, agents: null, think: 0, level: 'off',
    runs: {}, ticks: [], lastTick: null, swarm: {}, gridRun: null, setup: null, models: 0 };
  var WINDOW_S = 60, MAX_AGENTS = 256;
  var PER_FILE = ['swarm', 'team', 'relay', 'classify', 'extract', 'compress', 'summarize'];  // one agent per picked file
  var HAS_ROWS = ['swarm', 'shared', 'team', 'followup', 'relay', 'classify', 'extract', 'compress', 'summarize'];  // agents whose replies the page shows
  var HAS_GRID = HAS_ROWS.concat(['load', 'exact', 'burst', 'turns']);
  var GROUPS = ['Talk', 'Documents', 'Agents', 'Engine tests'];
  function isIn(list, x) { return list.indexOf(x) >= 0; }
  // Chat attaches files to one message; every other task works on one set, kept while you move between them.
  function fileSet(id) { return id === 'chat' ? 'chat' : 'jobs'; }

  function api(method, path, body) {
    return fetch('/api/live' + path, { method: method, headers: body ? { 'Content-Type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined })
      .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.detail || j.error || r.statusText); return j; }); });
  }
  function logLine(msg) { var d = document.createElement('div'); d.textContent = new Date().toLocaleTimeString() + '  ' + msg; $('log').prepend(d); }

  // -- menus that open from a button --------------------------------------------------------------------------------------
  var POPS = [['optBtn', 'optPanel'], ['taskBtn', 'taskMenu'], ['setBtn', 'setPanel']];
  function closePops() {
    POPS.forEach(function (p) { $(p[1]).hidden = true; $(p[0]).setAttribute('aria-expanded', 'false'); });
  }
  POPS.forEach(function (p) {
    $(p[0]).onclick = function (e) {
      e.stopPropagation();
      var open = $(p[1]).hidden; closePops(); $(p[1]).hidden = !open; $(p[0]).setAttribute('aria-expanded', String(open));
    };
    $(p[1]).addEventListener('click', function (e) { e.stopPropagation(); });
  });
  document.addEventListener('click', closePops);
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closePops(); });

  // -- model bar -----------------------------------------------------------------------------------------------------
  function loadModels() {
    return fetch('/api/models').then(function (r) { return r.json(); }).then(function (d) {
      var ms = (d.models || []).filter(function (m) { return !m.error; });
      S.models = ms.length;
      $('model').innerHTML = V.modelOptionsHtml(ms);
      var saved = localStore('model'); if (saved && ms.some(function (m) { return m.id === saved; })) $('model').value = saved;
      S.syncedKey = null; renderSession();  // the list exists now: show the model that is really loaded
    });
  }
  // What the page needs to say about the setup: is the engine built, are there models, is llama.cpp there for the comparison.
  function loadSetup() {
    return fetch('/api/setup').then(function (r) { return r.json(); }).then(function (d) {
      S.setup = d;
      $('compare').disabled = !d.llama.found;
      $('compareNote').textContent = d.llama.found ? '' : 'llama.cpp was not found. To compare, put llama-server on your PATH (or set LLAMA_SERVER_BIN) and restart the playground.';
      renderBanner();
    });
  }
  function localStore(k, v) {
    try { if (v === undefined) return localStorage.getItem('live.' + k); localStorage.setItem('live.' + k, v); } catch (e) { storageFailed(e); return null; }
  }
  var storageWarned = false;  // a browser with no local storage (a private window): the settings last this visit only, said once
  function storageFailed(e) {
    if (!storageWarned) console.warn('live: settings are not kept (local storage failed: ' + (e && e.message || e) + ')');
    storageWarned = true;
  }
  function setStatus(text, cls, title) { var s = $('status'); s.textContent = text; s.className = 'chip ' + (cls || ''); s.title = title || text; }
  // The pickers follow the session when it changes, and are the person's own after that.
  function syncPickers() {
    var s = S.session, key = s ? [s.model, s.compare].join('|') : '';
    if (key === S.syncedKey) return;
    S.syncedKey = key;
    if (s) { $('model').value = s.model; $('compare').checked = s.compare; }
  }
  // the compare switch differs from the loaded session's
  function optionsChanged(s) { return $('compare').checked !== s.compare; }
  // a session is loading, or one is loaded and starting, or a run is asked for or going
  function loadBlocked(s) { return S.view.loading || S.view.phase !== 'idle' || !!(s && s.state === 'starting'); }
  function renderLoad() {
    var s = S.session, other = !!s && $('model').value !== s.model, opts = !!s && optionsChanged(s);
    $('load').hidden = !!s && !other && !opts;
    $('load').textContent = other ? 'Switch' : opts ? 'Reload' : 'Load';
    $('load').disabled = loadBlocked(s);
    $('stop').hidden = !s || S.view.phase !== 'idle';
  }
  function renderSend() {
    var v = S.view, s = v.session, word = cur().id === 'chat' ? 'Send' : 'Run', busy = V.isBusy(v), b = $('send');
    b.classList.toggle('danger', busy); b.classList.toggle('primary', !busy);
    b.disabled = v.loading || v.phase === 'stopping' || v.phase === 'sending' || !!(s && s.state === 'starting');
    b.textContent = v.phase === 'stopping' ? 'Stopping…' : busy ? 'Stop' : v.loading ? 'Loading…' : s ? word : 'Load and ' + word.toLowerCase();
  }
  function renderBanner() {
    var d = S.setup, msg = '';
    if (d && !d.engine.found) msg = 'The Sonic LLM engine is not built yet (expected at ' + d.engine.path + '). Build it with `' + d.engine.build + '` in the engine folder, or set SONIC_SERVER_BIN.';
    else if (d && !S.models) msg = 'No models found in ' + d.models_dir + '. Click Get models to download one, or put a .gguf file there (or set SONIC_MODELS).';
    $('banner').textContent = msg; $('banner').hidden = !msg;
  }
  function renderSession() {
    var s = S.session;
    syncPickers(); renderLoad(); renderSend(); renderBanner(); showThink();
    var ready = !!s && s.state !== 'starting';
    $('vizEmpty').hidden = !!s; $('tiles').hidden = !s; $('charts').hidden = !s;
    $('exactChip').hidden = !(ready && s.exact);
    if (!s) { setStatus(S.view.loading ? 'Loading ' + $('model').value + '…' : 'No model loaded', S.view.loading ? 'busy' : ''); $('limits').innerHTML = ''; return; }
    if (!ready) { setStatus('Loading ' + s.model + '…', 'busy'); return; }
    renderStatus(s);
    $('limits').innerHTML = V.limitsHtml(s);
    if (s.exact) renderExactChip(s.exact);
  }
  // a loaded session: running or ready, or an engine restarting (for a run that needs another context)
  function renderStatus(s) {
    var e = s.engines, x = e.sllm, busy = V.isBusy(S.view);
    var back = Object.keys(e).filter(function (rt) { return e[rt].restarting === true; }).map(function (rt) { return C.SERIES[rt].label; });
    if (!back.length && s.restarting === true) back = ['the engine'];  // the session says so, not which engine
    var said = back.length ? 'Restarting ' + back.join(' and ') + '…' : busy ? 'Running' : 'Ready';
    setStatus(said + ' · ' + C.fmt(x.ctx) + ' tokens · room for ' + many(x.max_parallel, 'agent'), busy || back.length ? 'busy' : 'ok',
      s.model + ' · ' + Object.keys(e).map(function (rt) { return C.SERIES[rt].label + ' loaded in ' + (C.num(e[rt].ready_s) ? e[rt].ready_s.toFixed(1) + ' s' : '…'); }).join(' · '));
  }
  function renderExactChip(exact) {
    $('exactChip').textContent = { exact: '✓ Exact', writing: '⚠ Writing exact' }[exact.level] || '⚠ Not exact'; $('exactChip').className = 'chip' + (exact.level === 'exact' ? '' : ' warn'); $('exactChip').title = exact.text;
  }
  function load() {
    var id = $('model').value; if (!id) return Promise.resolve(false);
    localStore('model', id);
    dispatch({ type: 'load' });
    var kv = lim('limKv');
    return api('POST', '/session', Object.assign({ model_id: id, compare: $('compare').checked }, kv ? { kv_gib: kv } : {})).then(function (s) {
      S.ticks = []; dispatch({ type: 'loaded', session: s }); buildTiles(); return true;
    }).catch(function (e) {
      dispatch({ type: 'loaded' }); refresh(); botError('Could not load ' + id + ': ' + e.message); return false;
    });
  }
  $('load').onclick = load;
  ['model', 'compare'].forEach(function (id) { $(id).onchange = renderLoad; });
  $('stop').onclick = function () { api('DELETE', '/session').then(function () { dispatch({ type: 'unloaded' }); buildTiles(); }); };
  $('exactChip').onclick = function (e) { e.stopPropagation(); $('optBtn').click(); };
  $('reset').onclick = function () { api('POST', '/reset'); $('messages').innerHTML = ''; S.runs = {}; };

  // -- tasks ---------------------------------------------------------------------------------------------------------
  function cur() { return S.scenarios.filter(function (x) { return x.id === S.scen; })[0] || S.scenarios[0] || {}; }
  function renderTools() {
    var groups = GROUPS.map(function (name) { return { name: name, items: S.scenarios.filter(function (sc) { return sc.group === name; }) }; });
    $('taskMenu').innerHTML = V.taskMenuHtml(groups, S.scen);
    var sc = cur();
    $('taskName').textContent = sc.name || 'Chat';
    var builtIn = (sc.suggest || []).length && sc.suggest[0].charAt(0) === '(';
    $('input').placeholder = builtIn ? 'Nothing to type for this task: press Run.' : sc.ask || (sc.id === 'team' ? 'One question per line (Shift+Enter for a new line)…' : 'Ask something…');
    renderSuggest(); renderAgents(); renderPlan(); renderSend();
  }
  function renderSuggest() {
    var sc = cur(), sugg = (sc.suggest || []).filter(function (x) { return x.charAt(0) !== '('; });
    if (S.files.length && (sc.id === 'chat' || sc.id === 'long')) {
      var big = S.files.slice().sort(function (a, b) { return b.bytes - a.bytes; })[0];
      sugg = sugg.concat(['📎 ' + big.path.split('/').pop() + ': what are the key decisions?']);
    }
    $('suggest').innerHTML = $('input').value ? '' : sugg.map(function (t) { return '<button class="sugg" title="' + esc(t) + '">' + esc(t.split('\n')[0]) + '</button>'; }).join('');
  }
  function agentCount() {
    var n = S.picked.length;
    if (isIn(PER_FILE, cur().id) && n) return Math.min(n, MAX_AGENTS);
    return Math.max(1, Math.min(MAX_AGENTS, Math.round(+(S.agents || cur().agents)) || 1));
  }
  function renderAgents() {
    var sc = cur(), byFiles = isIn(PER_FILE, sc.id) && S.picked.length > 0;
    $('agentsRow').hidden = !sc.agents;
    $('agents').disabled = byFiles; $('agents').title = byFiles ? 'One agent per picked file' : '';
    if (document.activeElement !== $('agents')) $('agents').value = agentCount();
  }
  // What pressing Run will really do, in one line (liveview.js writes it).
  function renderPlan() {
    var id = cur().id, legacy = thinkLegacy(), n = S.picked.length;
    var p = V.planText(id, { n: n, a: agentCount(), perFile: isIn(PER_FILE, id), first: n ? S.picked[0].split('/').pop() : '', files: S.files.length,
      max: MAX_AGENTS, par: lim('limPar'), maxThink: legacy && THINK[S.think] === 'max', level: legacy ? 'off' : S.level });
    $('plan').textContent = p.t; $('plan').className = 'plan small' + (p.warn ? ' warn' : '');
  }
  $('taskMenu').onclick = function (e) {
    var b = e.target.closest('[data-s]'); if (!b) return;
    S.picks[fileSet(S.scen)] = S.picked;
    S.scen = b.dataset.s; localStore('scen', S.scen);
    S.picked = S.picks[fileSet(S.scen)];
    closePops(); renderTools(); renderAttached(); $('input').focus();
  };
  $('suggest').onclick = function (e) {
    var b = e.target.closest('.sugg'); if (!b) return;
    var t = b.title;
    if (t.indexOf('📎 ') === 0) {
      var name = t.slice(3).split(':')[0];
      var f = S.files.filter(function (x) { return x.path.split('/').pop() === name; })[0];
      if (f && S.picked.indexOf(f.path) < 0) S.picked.push(f.path);
      renderAttached(); t = t.split(': ').slice(1).join(': ');
    }
    $('input').value = t; $('input').focus(); renderSuggest();
  };
  // read ahead: the chat prompt is read by the engine while it is typed (the newest text wins), so a sent message only
  // has its last token left to read
  var preT = null;
  function preread() {
    clearTimeout(preT);
    preT = setTimeout(function () {
      var s = S.session, text = $('input').value;
      if (!s || s.state !== 'ready' || S.view.phase !== 'idle' || s.engine !== 'next' || cur().id !== 'chat' || S.picked.length || !text.trim()) return;
      var body = { message: text }, v = lim('limAnswer');
      if (thinkLegacy()) body.thinking = THINK[S.think]; else { body.thinking = 'off'; body.level = S.level; }
      if (v) body.answer_tokens = v;
      api('POST', '/preread', body).catch(function () { /* only a head start: the run reads what is missing */ });
    }, 60);
  }
  $('input').oninput = function () { renderSuggest(); preread(); };
  $('agents').oninput = function () { S.agents = $('agents').value; renderPlan(); };
  $('agents').onblur = renderAgents;

  // -- answer level / thinking ---------------------------------------------------------------------------------------
  // The answer levels (levels.py, thinking off in all of them); gpt-oss keeps its own low / medium / high / max.
  var LEVELS = V.LEVELS, LEVEL_WHAT = V.LEVEL_WHAT, LEVEL_NOTE = V.LEVEL_NOTE, NO_TEAM = V.NO_TEAM;
  var THINK = V.THINK, THINK_WHAT = V.THINK_WHAT;
  function thinkLegacy() { return !!(S.session && /gpt-oss/i.test(S.session.model || '')); }
  function thinkNext() { return !S.session || S.session.engine === 'next'; }
  function thinkName() { return thinkLegacy() ? THINK[S.think] : S.level; }
  function showThink() {
    var legacy = thinkLegacy();
    $('think').innerHTML = V.thinkHtml(legacy, S.think, S.level);
    $('thinkWhat').textContent = legacy ? THINK_WHAT[S.think] : LEVEL_WHAT[S.level] + (S.level !== 'off' && !thinkNext() ? NO_TEAM : '') + LEVEL_NOTE;
    $('thinkHead').textContent = legacy ? 'Thinking' : 'Answer level';
    $('think').setAttribute('aria-label', legacy ? 'Thinking' : 'Answer level');
    renderSettingsNote();
  }
  try { var tv = localStorage.getItem('sonic.think'); if (tv !== null) S.think = Math.max(0, Math.min(4, +tv || 0)); } catch (e) { storageFailed(e); }
  var savedLevel = localStore('level'); if (savedLevel && LEVELS.indexOf(savedLevel) >= 0) S.level = savedLevel;
  $('think').onclick = function (e) {
    var b = e.target.closest('[data-i],[data-l]'); if (!b) return;
    if (b.dataset.l) { S.level = b.dataset.l; localStore('level', S.level); }
    else { S.think = +b.dataset.i; try { localStorage.setItem('sonic.think', String(S.think)); } catch (x) { storageFailed(x); } }
    showThink(); renderPlan();
  };

  // -- optional limits (empty = the maximum) -------------------------------------------------------------------------
  var LIMS = ['limPar', 'limCtx', 'limAnswer', 'limAgent', 'limDoc'];
  function lim(id) { var v = parseFloat($(id).value); return v > 0 ? v : null; }
  // Settings stay from one visit to the next, so anything not at its default shows on the button.
  function renderSettingsNote() {
    var set = LIMS.filter(lim), bits = thinkName() === 'off' ? [] : [(thinkLegacy() ? 'thinking ' : 'answer level ') + thinkName()];
    if (set.length) bits.push(many(set.length, 'limit'));
    $('setNote').textContent = bits.join(', '); $('setBtn').classList.toggle('on', bits.length > 0);
    $('limitsSet').textContent = set.length ? '' : 'none: everything at the maximum';
  }
  LIMS.concat('limKv').forEach(function (id) {
    var v = localStore(id); if (v) $(id).value = v;
    $(id).oninput = function () { localStore(id, $(id).value); renderSettingsNote(); renderPlan(); };
  });
  $('limClear').onclick = function () { LIMS.forEach(function (id) { $(id).value = ''; localStore(id, ''); }); renderSettingsNote(); renderPlan(); };

  // -- files ---------------------------------------------------------------------------------------------------------
  function loadFiles() {
    return api('GET', '/files').then(function (d) {
      S.files = d.files;
      var have = d.files.map(function (f) { return f.path; });
      var real = function (p) { return have.indexOf(p) >= 0; };
      S.picks.chat = S.picks.chat.filter(real); S.picks.jobs = S.picks.jobs.filter(real);
      S.picked = S.picks[fileSet(S.scen)];
      renderAttached(); renderTools();
    });
  }
  function renderFiles() {
    var q = $('filesSearch').value.toLowerCase(), groups = S.shown = V.fileGroups(S.files, q);
    $('filesList').innerHTML = Object.keys(groups).sort().map(function (g) { return V.fileGroupHtml(g, groups[g], S.picked, q || S.open[g]); }).join('') || '<p class="muted" style="padding:10px">' + (q ? 'No file matches.' : 'The library is empty: add a text file below.') + '</p>';
    $('filesList').querySelectorAll('[data-partial]').forEach(function (x) { x.indeterminate = true; });
    var tok = S.picked.reduce(function (a, p) { var f = S.files.filter(function (x) { return x.path === p; })[0]; return a + (f ? f.est_tokens : 0); }, 0);
    $('filesSel').textContent = S.picked.length + ' selected · ~' + C.fmt(tok) + ' tokens';
    $('nfiles').textContent = S.picked.length > 2 ? many(S.picked.length, 'file') : 'Files';
    $('attach').title = S.picked.length > 2 ? S.picked.map(function (p) { return p.split('/').pop(); }).join(', ') : 'Pick files from the library';
  }
  function renderAttached() {
    S.picks[fileSet(S.scen)] = S.picked;
    localStore('picks', JSON.stringify(S.picks));
    $('attached').innerHTML = V.attachedHtml(S.picked);
    renderFiles(); renderAgents(); renderPlan();
  }
  $('attached').onclick = function (e) {
    var b = e.target.closest('[data-x]'); if (!b) return;
    S.picked = b.dataset.x === '*' ? [] : S.picked.filter(function (p) { return p !== b.dataset.x; }); renderAttached();
  };
  $('attach').onclick = function () { loadFiles(); $('files').showModal(); };
  $('filesClose').onclick = $('filesDone').onclick = function () { $('files').close(); renderAttached(); };
  $('files').onclose = renderAttached;
  $('filesNone').onclick = function () { S.picked = []; renderAttached(); };
  $('filesSearch').oninput = renderFiles;
  $('filesList').onclick = function (e) {
    var b = e.target.closest('[data-open]'); if (!b) return;
    S.open[b.dataset.open] = !S.open[b.dataset.open]; renderFiles();
  };
  $('filesList').onchange = function (e) {
    var p = e.target.dataset.p, g = e.target.dataset.g;
    if (g) {
      var paths = S.shown[g].map(function (f) { return f.path; });
      S.picked = S.picked.filter(function (x) { return paths.indexOf(x) < 0; });
      if (e.target.checked) S.picked = S.picked.concat(paths);
    } else if (p) {
      if (e.target.checked) { if (S.picked.indexOf(p) < 0) S.picked.push(p); } else S.picked = S.picked.filter(function (x) { return x !== p; });
    }
    renderAttached();
  };
  // Add the user's own text files: each one is sent as the body of PUT /files/<name>, then the library is read again.
  function upload(list) {
    var sent = Array.prototype.map.call(list, function (f) {
      return fetch('/api/live/files/' + encodeURIComponent(f.name), { method: 'PUT', body: f })
        .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(f.name + ': ' + (j.detail || r.statusText)); logLine('added ' + j.path); }); })
        .catch(function (e) { alert(e.message); });
    });
    return Promise.all(sent).then(loadFiles).then(function () { $('upload').value = ''; });
  }
  $('upload').onchange = function () { upload($('upload').files); };
  $('drop').addEventListener('dragover', function (e) { e.preventDefault(); $('drop').classList.add('over'); });
  $('drop').addEventListener('dragleave', function () { $('drop').classList.remove('over'); });
  $('drop').addEventListener('drop', function (e) { e.preventDefault(); $('drop').classList.remove('over'); upload(e.dataTransfer.files); });

  // -- sending -------------------------------------------------------------------------------------------------------
  function go() {
    var s = S.session, sc = cur(), text = $('input').value;
    if (V.isBusy(S.view)) return stopRun();
    if (sc.id === 'chat' && !text.trim()) { $('input').focus(); return; }
    if (s) return send(text);
    load().then(function (ok) { if (ok) send(text); });
  }
  function stopRun() {
    dispatch({ type: 'stop' });
    api('POST', '/run/stop').catch(function (e) { dispatch({ type: 'stop_failed' }); refresh(); botError(e.message); });
  }
  function send(text) {
    var sc = cur();
    var body = { scenario: sc.id, message: text, files: S.picked.slice(), agents: agentCount() };
    if (thinkLegacy()) body.thinking = THINK[S.think];
    else { body.thinking = 'off'; body.level = S.level; }
    [['limPar', 'parallel'], ['limCtx', 'ctx'], ['limAnswer', 'answer_tokens'], ['limAgent', 'agent_tokens'], ['limDoc', 'doc_chars']].forEach(function (p) {
      var v = lim(p[0]); if (v) body[p[1]] = v;
    });
    $('welcome') && $('welcome').remove();
    var u = document.createElement('div'); u.className = 'msg user';
    u.innerHTML = esc(text || sc.name) + (body.files.length ? '<div class="files">📎 ' + (body.files.length > 3 ? many(body.files.length, 'file') : body.files.map(function (p) { return esc(p.split('/').pop()); }).join(', ')) + '</div>' : '');
    $('messages').appendChild(u);
    dispatch({ type: 'send' });
    api('POST', '/run', body).then(function (r) { dispatch({ type: 'sent', run: r && r.run, scenario: sc.id }); })
      .catch(function (e) { dispatch({ type: 'send_failed' }); botError(e.message); });
    $('input').value = ''; renderSuggest(); scrollDown(true);
    if (sc.id === 'chat') { S.picked = []; renderAttached(); }
  }
  function botError(msg) {
    $('welcome') && $('welcome').remove();
    var b = document.createElement('div'); b.className = 'msg bot'; b.innerHTML = '<div class="answer"><b>Error:</b> ' + esc(msg) + '</div>';
    $('messages').appendChild(b); scrollDown(true);
  }
  $('send').onclick = go;
  $('input').onkeydown = function (e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!$('send').disabled && !V.isBusy(S.view)) go(); } };

  // -- the run state on the page ---------------------------------------------------------------------------------------
  // Every change goes through V.reduceRun; then the boxes follow: a run that ended is closed, a new one gets its box, a
  // run started elsewhere keeps its box when its first event names it.
  function dispatch(ev) {
    var was = S.view;
    S.view = V.reduceRun(was, ev); S.session = S.view.session;
    if (S.view === was) return;
    followRun(was, S.view, ev);
    renderSession();
  }
  function boxKey(r) { return r.id === null ? '' : r.id; }  // a run started elsewhere, not named yet: ''
  function followRun(was, now, ev) {
    var a = was.run, b = now.run;
    if (a === b) return;
    if (a) leaveBox(a, b, endedHow(ev, a, was.phase));
    if (b && !S.runs[boxKey(b)]) startRun(b);
  }
  // the box of run a, now that run b is going (or none): an ended run is closed; a stand-in (a run started elsewhere,
  // not named yet) is named by its first event, or replaced when its run_start comes
  function leaveBox(a, b, how) {
    if (a.id !== null) { if (!b || b.id !== a.id) closeRun(a.id, how); }
    else if (!b) closeRun('', how);
    else if (b.elsewhere) moveBox('', b.id);
    else dropBox('');
  }
  // how a run ended: its run_done says, else the server says no run is going (its run_done was missed)
  function endedHow(ev, run, phase) { return ev.type === 'run_done' && ev.run === run.id ? V.doneHow(ev) : phase === 'stopping' ? 'stopped' : 'lost'; }
  function moveBox(from, to) { S.runs[to] = S.runs[from]; delete S.runs[from]; if (S.gridRun === from) S.gridRun = to; }
  function dropBox(key) { var r = S.runs[key]; if (r) { r.el.remove(); delete S.runs[key]; } }
  function startRun(r) {
    var sc = S.scenarios.filter(function (x) { return x.id === r.scenario; })[0] || {}, key = boxKey(r), rts = r.engines || rtsNow();
    var b = document.createElement('div'); b.className = 'msg bot';
    b.innerHTML = V.runHtml(r.elsewhere ? 'A run is going (started elsewhere)' : sc.name || r.scenario, rts);
    $('messages').appendChild(b);
    S.runs[key] = { el: b, scenario: r.scenario, rows: {}, t0: Date.now(), rts: rts }; S.gridRun = key;
    S.swarm = {}; $('swarm').innerHTML = ''; $('swarmNote').textContent = ''; $('swarmCard').hidden = !isIn(HAS_GRID, r.scenario);
    scrollDown(true);
  }
  function ans(run, rt) { var r = S.runs[run]; return r && r.el.querySelector('.answer[data-rt="' + rt + '"]'); }
  // Follows the newest words only for someone at the bottom: reading further up stays put.
  function scrollDown(force) {
    var m = $('messages');
    if (force || m.scrollHeight - m.scrollTop - m.clientHeight < 140) m.scrollTop = m.scrollHeight;
  }

  // -- agents: one row per agent under the answer, and the grid on the right -----------------------------------------
  function agentRow(ev) {
    var run = S.runs[ev.run]; if (!run || !isIn(HAS_ROWS, run.scenario)) return;
    var row = run.rows[ev.rt + ':' + ev.i] || newAgentRow(run, ev); if (!row) return;
    if (row.title !== ev.label) nameRow(row, ev.label);  // a relay's next lap reuses the row for another file
    row.dataset.state = ev.state;
    if (ev.state === 'done') showAgentDone(row, ev);
    else row.querySelector('.vd').textContent = ev.state === 'writing' ? 'writing · ' + many(ev.tokens, 'token') : ev.state;
  }
  // a new row under the engine's answer (none when the answer is not on the page)
  function newAgentRow(run, ev) {
    var box = ans(ev.run, ev.rt); if (!box) return null;
    var row = document.createElement('details'); row.className = 'arow';
    row.innerHTML = '<summary><span class="st"></span><span class="nm"></span><span class="vd"></span></summary><div class="body"><button class="link copy">Copy</button><pre>Nothing yet.</pre></div>';
    nameRow(row, ev.label);
    box.querySelector('.agentlist').appendChild(row); run.rows[ev.rt + ':' + ev.i] = row; scrollDown();
    return row;
  }
  function nameRow(row, label) {
    row.title = label; row.querySelector('.nm').textContent = label.split('/').pop();
    row.querySelector('pre').textContent = 'Nothing yet.'; row.classList.remove('bad');
  }
  function showAgentDone(row, ev) {
    var tok = C.num(ev.tokens) ? many(ev.tokens, 'token') : '', said = agentVerdict(ev);
    row.querySelector('.vd').textContent = said ? said + ' · ' + tok : tok;
    row.classList.toggle('bad', ev.ok === false || ev.stop === 'limit' || ev.found === 0);
    row.querySelector('pre').textContent = ev.reply || '(an empty reply)';
  }
  // what an agent's job came to: the run's verdict, the questions it answered, or a reply cut at the limit
  function agentVerdict(ev) {
    return ev.verdict || (C.num(ev.found) ? 'answered ' + ev.found + ' of ' + ev.of : '') || (ev.stop === 'limit' ? 'cut at the length limit' : '');
  }
  var swarmTimer = null;
  function swarmEvent(ev) {
    var key = ev.rt; S.swarm[key] = S.swarm[key] || {};
    if (S.view.run && S.view.run.elsewhere) $('swarmCard').hidden = false;  // its task is not known: its agents say it
    S.swarm[key][ev.i] = ev;
    agentRow(ev);
    if (!swarmTimer) swarmTimer = setTimeout(function () { swarmTimer = null; renderSwarm(); }, 150);
  }
  function renderSwarm() {
    var all = [], html = Object.keys(S.swarm).map(function (rt) {
      var agents = S.swarm[rt], ids = Object.keys(agents).map(Number).sort(function (a, b) { return a - b; });
      var done = ids.filter(function (i) { return agents[i].state === 'done'; }).length;
      var box = S.view.run && done < ids.length && ans(boxKey(S.view.run), rt); if (box) box.querySelector('.state').textContent = done + ' of ' + ids.length + ' done';
      all = all.concat(ids.map(function (i) { return agents[i].state; }));
      return V.swarmRowHtml(rt, ids.map(function (i) { return agents[i]; }), done);
    }).join('');
    $('swarm').innerHTML = html;
    $('swarmNote').textContent = V.countText(all);
  }
  // A run that is over (how: done, stopped, failed, or lost: the server says it is not going): what did not finish (rows,
  // answers without their numbers, the grid's cells) leaves reading and writing.
  function closeRun(key, how) {
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
      scrollDown();
    }
    if (cut && key === S.gridRun) { V.stopCells(S.swarm); renderSwarm(); }
  }

  // -- tiles and charts ----------------------------------------------------------------------------------------------
  var tokChart, readChart, gpuChart, memChart;
  var rz; window.addEventListener('resize', function () { clearTimeout(rz); rz = setTimeout(function () { buildTiles(); if (S.lastTick) onTick(S.lastTick); }, 200); });
  function rtsNow() { return S.session ? Object.keys(S.session.engines) : ['sllm']; }
  function buildTiles() {
    var rts = rtsNow();
    $('legendTok').innerHTML = rts.length > 1 ? rts.map(function (rt) { return '<span><i style="background:' + C.SERIES[rt].color + '"></i>' + C.SERIES[rt].label + '</span>'; }).join('') : '';
    ['chartTok', 'chartRead', 'chartGpu', 'chartMem'].forEach(function (id) { $(id).innerHTML = ''; });
    var ser = rts.map(function (rt) { return { key: rt, label: C.SERIES[rt].label, color: C.SERIES[rt].color, soft: C.SERIES[rt].soft }; });
    showMemNote();
    tokChart = C.line($('chartTok'), { series: ser, unit: 'tok/s', windowS: WINDOW_S, label: 'Tokens per second, last minute' });
    readChart = C.line($('chartRead'), { series: ser, unit: 'tok/s', windowS: WINDOW_S, label: 'Prompt tokens read per second' });
    gpuChart = C.line($('chartGpu'), { series: [{ key: 'gpu', label: 'GPU busy', color: 'var(--series-1)', soft: 'var(--series-1-soft)' }], unit: '%', windowS: WINDOW_S, yMax: 100, label: 'GPU busy %' });
    memChart = C.line($('chartMem'), { series: ser, unit: 'MB', windowS: WINDOW_S, label: 'Process footprint, model file not counted',
      fmt: function (v) { return C.num(v) ? (v >= 1024 ? (v / 1024).toFixed(1) + 'G' : Math.round(v) + 'M') : '–'; } });
    renderTiles(S.lastTick);
  }
  // the memory chart's note: the KV cache reserved and (after a run) the most of it in use; the whole story is its tooltip
  function showMemNote(per) {
    var el = $('memNote'), note = V.memNote(S.session, per); if (!el) return;
    el.textContent = note[0]; el.title = note[1];
  }
  function renderTiles(t) { $('tiles').innerHTML = V.tilesHtml(rtsNow(), t); }
  function onTick(t) {
    S.lastTick = t;
    var run = S.view.run && S.runs[boxKey(S.view.run)];
    if (run) run.el.querySelector('.elapsed').textContent = Math.round((Date.now() - run.t0) / 1000) + ' s';
    var now = t.t; S.ticks.push({ t: now, ev: t });
    S.ticks = S.ticks.filter(function (p) { return p.t >= now - WINDOW_S - 2; });
    tokChart && tokChart.update(V.tickPoints(S.ticks, 'tok_s'), now);
    readChart && readChart.update(V.tickPoints(S.ticks, 'read_tok_s'), now);
    gpuChart && gpuChart.update(S.ticks.map(function (p) { return { t: p.t, values: { gpu: p.ev.gpu_util } }; }), now);
    memChart && memChart.update(V.tickPoints(S.ticks, 'mem_mb'), now);
    renderTiles(t);
  }

  function renderLast(rec) {
    $('lastCard').hidden = false;
    var sc = S.scenarios.filter(function (x) { return x.id === rec.scenario; })[0] || {};
    $('lastTitle').textContent = 'Last run · ' + (sc.name || rec.scenario);
    showMemNote(rec.metrics);
    $('last').innerHTML = C.pairedBars(C.METRICS, rec.metrics) + '<p class="small muted">' + esc(rec.note) + '</p>';
  }

  // -- events --------------------------------------------------------------------------------------------------------
  // One stream for all tabs (events.js), or one of its own where a browser has no shared workers. The server ends the stream
  // every 20 s and it reconnects at once, replaying what it missed (Last-Event-ID): only a stream that stays down is shown,
  // and tiles are rebuilt only when the loaded model changed.
  function onPhase(ev) {
    var a = ans(ev.run, ev.rt); if (!a) return;
    a.querySelector('.state').textContent = ev.msg.split(': ').pop();
    a.querySelector('.text').classList.toggle('cursor', !ev.metrics);
    if (ev.metrics) a.querySelector('.stats').innerHTML = statsHtml(ev.metrics);
  }
  // the answer so far, its thinking apart
  function onToken(ev) {
    var a = ans(ev.run, ev.rt); if (!a) return;
    var tx = a.querySelector('.text'), th = a.querySelector('.thought');
    if (!th) { th = document.createElement('div'); th.className = 'thought'; tx.parentNode.insertBefore(th, tx); }
    a.dataset.raw = (a.dataset.raw || '') + ev.text;
    var parts = splitThinking(a.dataset.raw);
    th.textContent = parts[0]; tx.textContent = parts[1];
    th.scrollTop = th.scrollHeight; tx.scrollTop = tx.scrollHeight; scrollDown();
  }
  // the server's events by type; other types are ignored
  var ON = {
    tick: onTick,
    log: function (ev) { logLine(ev.msg); },
    session: function (ev) {
      if (ev.error && !S.view.loading) botError('The model could not load: ' + ev.error);  // a load from another tab
      if (!ev.session || ev.session.state === 'ready') buildTiles();
    },
    snapshot: function (ev) { applyState(ev.state, true); },  // the first event after every connect
    resync: refresh,
    phase: onPhase,
    token: onToken,
    agent: swarmEvent,
    run_done: function (ev) {
      closeRun(ev.run, V.doneHow(ev));  // the run on the page is closed already (dispatch); this closes an earlier one
      if (ev.error) botError(ev.error);
      else if (ev.record) renderLast(ev.record);
    }
  };
  // every event changes the run state first (run_start, run_done, session, a run's first event), then shows itself
  function onEvent(ev) {
    dispatch(ev);
    if (Object.prototype.hasOwnProperty.call(ON, ev.type)) ON[ev.type](ev);
  }
  function connect() {
    var downTimer = null;
    function onData(data) { onEvent(JSON.parse(data)); }
    function onDown() { if (!downTimer) downTimer = setTimeout(function () { setStatus('reconnecting…', 'busy'); }, 1500); }
    function onUp() { clearTimeout(downTimer); downTimer = null; refresh(); }
    if (window.SharedWorker) {
      var w = new SharedWorker('/static/events.js');
      w.port.onmessage = function (m) { if (m.data.up) onUp(); else if (m.data.down) onDown(); else onData(m.data.data); };
      w.port.start();
      window.addEventListener('pagehide', function () { w.port.postMessage('bye'); });
      window.addEventListener('pageshow', function (e) { if (e.persisted) w.port.postMessage('hi'); });  // back from the back/forward cache
      return;
    }
    var es = new EventSource('/api/live/events');
    es.onmessage = function (m) { onData(m.data); };
    es.onerror = function () { onDown(); if (es.readyState === 2) setTimeout(connect, 2000); };  // 2: the browser gave up (an error answer)
    es.onopen = onUp;
  }
  // The server's state (GET /state, or the snapshot event); fresh: no run started or ended here while it was asked for
  function applyState(st, fresh) {
    var before = S.session && S.session.model;
    dispatch({ type: 'server', state: st, fresh: fresh });
    if ((S.session && S.session.model) !== before) buildTiles();
  }
  function refresh() {
    var asked = V.runKey(S.view); return api('GET', '/state').then(function (st) { applyState(st, V.runKey(S.view) === asked); });
  }

  $('messages').addEventListener('click', function (e) {
    var b = e.target.closest('.copy'); if (!b) return;
    navigator.clipboard.writeText(b.parentNode.querySelector('pre').textContent).then(function () { b.textContent = 'Copied'; setTimeout(function () { b.textContent = 'Copy'; }, 1200); });
  });

  showThink();
  api('GET', '/state').then(function (st) {
    S.scenarios = st.scenarios;
    var saved = localStore('scen'); if (saved && st.scenarios.some(function (x) { return x.id === saved; })) S.scen = saved;
    try { var kept = JSON.parse(localStore('picks') || '{}'); S.picks = { chat: kept.chat || [], jobs: kept.jobs || [] }; } catch (e) { console.warn('live: the saved file picks are not readable: ' + e.message); }
    S.picked = S.picks[fileSet(S.scen)];
    renderTools(); dispatch({ type: 'server', state: st, fresh: true }); buildTiles();
  }).then(loadSetup).then(loadModels).then(loadFiles).then(connect);
  document.addEventListener('models-changed', function (e) { loadModels().then(function () { if (e.detail.id) $('model').value = e.detail.id; renderLoad(); renderBanner(); }); });
})();
