/* The test picker: the cards, the controls the picked test needs (question, documents, agents, answer level, limits), the
   one-line plan, and the Run button (run, stop, load first when no model is loaded). */
(function () {
  'use strict';
  var App = window.App, S = App.S, $ = App.$, V = App.V, esc = App.esc, isIn = App.isIn;
  var PER_FILE = ['swarm', 'team', 'relay', 'classify', 'extract', 'compress', 'summarize'];  // one agent per picked document
  // Chat attaches documents to one message; every other test works on one set, kept while you move between them.
  function fileSet(id) { return id === 'chat' ? 'chat' : 'jobs'; }
  App.fileSet = fileSet;
  App.cur = function () { return S.scenarios.filter(function (x) { return x.id === S.scen; })[0] || S.scenarios[0] || {}; };

  // -- the cards ------------------------------------------------------------------------------------------------------
  function placeControls() {
    var el = $('tests').querySelector('[data-t="' + S.scen + '"]');
    if (el) el.appendChild($('controls'));
  }
  App.renderTools = function () {
    $('controlsPark').appendChild($('controls'));
    $('tests').innerHTML = V.testCardsHtml(S.scenarios, S.scen);
    placeControls(); applyNeeds();
  };
  // show only the controls the picked test needs, labelled for it
  function applyNeeds() {
    var sc = App.cur(), n = V.NEEDS[sc.id] || {};
    $('askBox').hidden = !n.ask; $('askLabel').textContent = n.ask || '';
    $('input').placeholder = sc.ask || (sc.id === 'team' ? 'One question per line (Shift+Enter for a new line)' : 'Type here');
    $('docsBox').hidden = !n.docs; $('docsLabel').textContent = n.docs || '';
    $('levelBox').hidden = !(n.level || thinkLegacy());
    renderSuggest(); renderAgents(); renderPlan(); App.renderSend();
  }
  $('tests').addEventListener('change', function (e) {
    if (e.target.name !== 'test') return;
    S.picks[fileSet(S.scen)] = S.picked;
    S.scen = e.target.value; App.store('scen', S.scen);
    S.picked = S.picks[fileSet(S.scen)];
    Array.prototype.forEach.call($('tests').querySelectorAll('.test'), function (el) { el.classList.toggle('on', el.dataset.t === S.scen); });
    placeControls(); applyNeeds(); App.renderAttached();
  });
  $('tests').addEventListener('click', function (e) {  // a mouse click on a card goes straight to its question (a key press does not)
    if (e.detail > 0 && e.target.closest && e.target.closest('.pick') && !$('askBox').hidden) $('input').focus();
  });

  // -- the question ---------------------------------------------------------------------------------------------------
  function renderSuggest() {
    var sc = App.cur(), n = V.NEEDS[sc.id] || {}, chips = n.ask ? (sc.suggest || []).filter(function (x) { return x.charAt(0) !== '('; }).map(function (t) { return { t: t }; }) : [];
    if (S.files.length && !S.picked.length && n.ask && isIn(['chat', 'long'], sc.id)) {
      var big = S.files.slice().sort(function (a, b) { return b.bytes - a.bytes; })[0], what = big.path.split('/').pop();
      chips.push({ t: 'What are the key decisions in ' + what + '?', file: big.path, label: 'Ask about ' + what });
    }
    $('suggest').innerHTML = $('input').value ? '' : chips.map(function (c) {
      return '<button class="sugg" type="button" data-t="' + esc(c.t) + '"' + (c.file ? ' data-file="' + esc(c.file) + '"' : '') + ' title="' + esc(c.t) + '">' + esc((c.label || c.t).split('\n')[0]) + '</button>';
    }).join('');
  }
  App.renderSuggest = renderSuggest;
  $('suggest').addEventListener('click', function (e) {
    var b = e.target.closest && e.target.closest('.sugg'); if (!b) return;
    if (b.dataset.file && S.picked.indexOf(b.dataset.file) < 0) { S.picked.push(b.dataset.file); App.renderAttached(); }
    $('input').value = b.dataset.t; $('input').focus(); renderSuggest();
  });
  // read ahead: the chat prompt is read by the engine while it is typed (the newest text wins), so a sent message only
  // has its last token left to read
  var preT = null;
  function preread() {
    clearTimeout(preT);
    preT = setTimeout(function () {
      var s = S.session, text = $('input').value;
      if (!s || s.state !== 'ready' || S.view.phase !== 'idle' || s.engine !== 'next' || App.cur().id !== 'chat' || S.picked.length || !text.trim()) return;
      var body = { message: text }, v = App.lim('limAnswer');
      if (thinkLegacy()) body.thinking = V.THINK[S.think]; else { body.thinking = 'off'; body.level = S.level; }
      if (v) body.answer_tokens = v;
      App.api('POST', '/preread', body).catch(function () { /* only a head start: the run reads what is missing */ });
    }, 60);
  }
  $('input').oninput = function () { renderSuggest(); preread(); };

  // -- agents and the plan --------------------------------------------------------------------------------------------
  function agentCount() {
    var n = S.picked.length;
    if (isIn(PER_FILE, App.cur().id) && n) return Math.min(n, App.MAX_AGENTS);
    return Math.max(1, Math.min(App.MAX_AGENTS, Math.round(+(S.agents || App.cur().agents)) || 1));
  }
  function renderAgents() {
    var sc = App.cur(), byFiles = isIn(PER_FILE, sc.id) && S.picked.length > 0;
    $('agentsRow').hidden = !(V.NEEDS[sc.id] || {}).agents;
    $('agents').disabled = byFiles; $('agents').title = byFiles ? 'One agent per picked document' : '';
    if (document.activeElement !== $('agents')) $('agents').value = agentCount();
  }
  // What pressing Run will really do, in one line (liveview.js writes it).
  function renderPlan() {
    var id = App.cur().id, legacy = thinkLegacy(), n = S.picked.length;
    var p = V.planText(id, { n: n, a: agentCount(), perFile: isIn(PER_FILE, id), first: n ? S.picked[0].split('/').pop() : '', files: S.files.length,
      max: App.MAX_AGENTS, par: App.lim('limPar'), maxThink: legacy && V.THINK[S.think] === 'max', level: legacy ? 'off' : S.level });
    $('plan').textContent = p.t; $('plan').className = 'plan' + (p.warn ? ' warn' : '');
  }
  App.renderPlan = renderPlan; App.renderAgents = renderAgents;
  $('agents').oninput = function () { S.agents = $('agents').value; renderPlan(); };
  $('agents').onblur = renderAgents;

  // -- answer level / thinking ----------------------------------------------------------------------------------------
  function thinkLegacy() { return !!(S.session && /gpt-oss/i.test(S.session.model || '')); }
  function thinkName() { return thinkLegacy() ? V.THINK[S.think] : S.level; }
  App.showThink = function () {
    var legacy = thinkLegacy(), next = !S.session || S.session.engine === 'next';
    $('think').innerHTML = V.thinkHtml(legacy, S.think, S.level);
    $('thinkWhat').textContent = legacy ? V.THINK_WHAT[S.think] : V.LEVEL_WHAT[S.level] + (S.level !== 'off' && !next ? V.NO_TEAM : '') + V.LEVEL_NOTE;
    $('thinkHead').textContent = legacy ? 'Thinking' : 'Answer level';
    $('levelBox').hidden = !((V.NEEDS[App.cur().id] || {}).level || legacy);
  };
  try { var tv = localStorage.getItem('sonic.think'); if (tv !== null) S.think = Math.max(0, Math.min(4, +tv || 0)); } catch (e) { App.storageFailed(e); }
  var savedLevel = App.store('level'); if (savedLevel && V.LEVELS.indexOf(savedLevel) >= 0) S.level = savedLevel;
  $('think').addEventListener('change', function (e) {
    if (thinkLegacy()) { S.think = +e.target.value; try { localStorage.setItem('sonic.think', String(S.think)); } catch (x) { App.storageFailed(x); } }
    else { S.level = e.target.value; App.store('level', S.level); }
    App.showThink(); renderPlan();
  });

  // -- optional limits (empty = the most the model and the memory allow) ----------------------------------------------
  var LIMS = ['limPar', 'limCtx', 'limAnswer', 'limAgent', 'limDoc'];
  App.lim = function (id) { var v = parseFloat($(id).value); return v > 0 ? v : null; };
  function renderLimitsNote() {
    var set = LIMS.filter(App.lim);
    $('limitsSet').textContent = set.length ? '(' + set.length + ' set)' : '';
  }
  LIMS.concat('limKv').forEach(function (id) {
    var v = App.store(id); if (v) $(id).value = v;
    $(id).oninput = function () { App.store(id, $(id).value); renderLimitsNote(); renderPlan(); };
  });
  renderLimitsNote();
  $('limClear').onclick = function () { LIMS.forEach(function (id) { $(id).value = ''; App.store(id, ''); }); renderLimitsNote(); renderPlan(); };

  // -- the Run button -------------------------------------------------------------------------------------------------
  App.renderSend = function () {
    var v = S.view, s = v.session, busy = V.isBusy(v), b = $('send');
    b.classList.toggle('danger', busy); b.classList.toggle('primary', !busy);
    b.disabled = v.loading || v.phase === 'stopping' || v.phase === 'sending' || !!(s && s.state === 'starting') || (!s && (!$('model').value || (!!S.setup && !S.setup.engine.found)));
    b.textContent = v.phase === 'stopping' ? 'Stopping…' : busy ? 'Stop' : v.phase === 'sending' ? 'Starting…' : v.loading ? 'Loading…' : s ? 'Run' : 'Load and run';
  };
  function go() {
    var s = S.session, sc = App.cur(), text = $('input').value;
    if (V.isBusy(S.view)) return stopRun();
    if (sc.id === 'chat' && !text.trim()) { $('input').focus(); return; }
    if (s) return send(text);
    App.load().then(function (ok) { if (ok) send(text); });
  }
  function stopRun() {
    App.dispatch({ type: 'stop' });
    App.api('POST', '/run/stop').catch(function (e) { App.dispatch({ type: 'stop_failed' }); App.refresh(); App.notify(e.message, 'error'); });
  }
  function send(text) {
    var sc = App.cur();
    var body = { scenario: sc.id, message: text, files: S.picked.slice(), agents: agentCount() };
    if (thinkLegacy()) body.thinking = V.THINK[S.think]; else { body.thinking = 'off'; body.level = S.level; }
    [['limPar', 'parallel'], ['limCtx', 'ctx'], ['limAnswer', 'answer_tokens'], ['limAgent', 'agent_tokens'], ['limDoc', 'doc_chars']].forEach(function (p) {
      var v = App.lim(p[0]); if (v) body[p[1]] = v;
    });
    S.ask = (V.NEEDS[sc.id] || {}).ask ? text : ''; S.askFiles = body.files;
    $('notice').innerHTML = '';
    App.dispatch({ type: 'send' });
    App.api('POST', '/run', body).then(function (r) { App.dispatch({ type: 'sent', run: r && r.run, scenario: sc.id }); })
      .catch(function (e) { App.dispatch({ type: 'send_failed' }); App.notify(e.message, 'error'); });
    if (window.matchMedia && window.matchMedia('(max-width: 1000px)').matches) $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
    if (sc.id === 'chat') { S.picked = []; App.renderAttached(); }
  }
  $('send').onclick = go;
  $('input').onkeydown = function (e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!$('send').disabled && !V.isBusy(S.view)) go(); } };
})();
