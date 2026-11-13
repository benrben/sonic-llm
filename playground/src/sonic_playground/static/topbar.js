/* The top bar and what hangs on it: the model picker and its load state, the compare switch, the engine details, and the
   guide a newcomer sees (it depends on the same setup facts). */
(function () {
  'use strict';
  var App = window.App, S = App.S, $ = App.$, V = App.V, C = App.C;
  App.popover('optBtn', 'optPanel');
  App.popover('compareWhy', 'comparePanel');

  // -- what the page knows of the setup: engine built, models found, llama.cpp there ------------------------------------
  App.loadModels = function () {
    return fetch('/api/models').then(function (r) { return r.json(); }).then(function (d) {
      var ms = (d.models || []).filter(function (m) { return !m.error; });
      S.models = ms.length;
      $('model').innerHTML = ms.length ? V.modelOptionsHtml(ms) : '<option value="">No models yet</option>';
      var saved = App.store('model'); if (saved && ms.some(function (m) { return m.id === saved; })) $('model').value = saved;
      S.syncedKey = null; App.renderSession();  // the list exists now: show the model that is really loaded
    });
  };
  App.loadSetup = function () {
    return fetch('/api/setup').then(function (r) { return r.json(); }).then(function (d) {
      S.setup = d;
      $('compare').disabled = !d.llama.found; $('compareWhy').hidden = d.llama.found;
      $('compareNote').innerHTML = 'To compare, install llama.cpp so that <code>llama-server</code> is on your PATH (for example <code>brew install llama.cpp</code>), or set <code>LLAMA_SERVER_BIN</code>, then restart the playground.';
      App.renderSession();
    });
  };

  // -- the status pill ------------------------------------------------------------------------------------------------
  function setStatus(text, dot, title) { $('statusText').textContent = text; $('status').querySelector('.dot').className = 'dot ' + (dot || ''); $('status').title = title || text; }
  App.setStatus = setStatus;
  // The pickers follow the session when it changes, and are the person's own after that.
  function syncPickers() {
    var s = S.session, key = s ? [s.model, s.compare].join('|') : '';
    if (key === S.syncedKey) return;
    S.syncedKey = key;
    if (s) { $('model').value = s.model; $('compare').checked = s.compare; }
  }
  // a session is loading, or one is loaded and starting, or a run is asked for or going
  function loadBlocked(s) { return S.view.loading || S.view.phase !== 'idle' || !!(s && s.state === 'starting'); }
  function engineMissing() { return !!S.setup && !S.setup.engine.found; }
  function renderLoad() {
    var s = S.session, other = !!s && $('model').value !== s.model, opts = !!s && $('compare').checked !== s.compare;
    $('load').hidden = !!s && !other && !opts;
    $('load').textContent = other ? 'Switch' : opts ? 'Reload' : 'Load';
    $('load').disabled = loadBlocked(s) || engineMissing() || !$('model').value;
    $('stop').hidden = !s || S.view.phase !== 'idle';
  }
  App.renderLoad = renderLoad;
  // a loaded session: running or ready, or an engine restarting (for a run that needs another context)
  function renderStatus(s) {
    var e = s.engines, busy = V.isBusy(S.view);
    var back = Object.keys(e).filter(function (rt) { return e[rt].restarting === true; }).map(function (rt) { return C.SERIES[rt].label; });
    if (!back.length && s.restarting === true) back = ['the engine'];  // the session says so, not which engine
    var said = back.length ? 'Restarting ' + back.join(' and ') + '…' : busy ? 'Running' : 'Ready';
    setStatus(said, busy || back.length ? 'busy' : 'on',
      s.model + ' · ' + Object.keys(e).map(function (rt) { return C.SERIES[rt].label + ' loaded in ' + (C.num(e[rt].ready_s) ? e[rt].ready_s.toFixed(1) + ' s' : '…'); }).join(' · '));
  }
  // The engine says its answers are not word for word the same as llama.cpp's: said once per loaded model
  function sayExact(s) {
    if (!s.exact || s.exact.level === 'exact' || S.exactSaid === s.model) return;
    S.exactSaid = s.model; App.notify(s.exact.text, 'info', 'A note about this model');
  }
  App.renderSession = function () {
    var s = S.session;
    syncPickers(); renderLoad(); App.renderSend(); App.showThink(); App.renderIntro(); App.renderSetup(); App.renderRoom(); App.renderPlan();
    if (!s) { setStatus(S.view.loading ? 'Loading ' + $('model').value + '…' : 'No model loaded', S.view.loading ? 'busy' : ''); $('limits').innerHTML = ''; return; }
    if (s.state === 'starting') { setStatus('Loading ' + s.model + '…', 'busy'); return; }
    renderStatus(s); sayExact(s);
    $('limits').innerHTML = V.limitsHtml(s);
  };

  // -- the guide and the setup problems -------------------------------------------------------------------------------
  App.renderIntro = function () {
    $('intro').hidden = S.hasRun;
    if (S.hasRun) return;
    var s = S.session, ready = !!s && s.state !== 'starting';
    var html = V.introHtml({ models: S.models, model: ready ? s.model : '', loading: S.view.loading || !!(s && s.state === 'starting'), llama: !!(S.setup && S.setup.llama.found), noEngine: engineMissing() });
    if (html !== S.introHtml) { S.introHtml = html; $('intro').innerHTML = html; }
  };
  App.renderSetup = function () {
    var html = engineMissing() ? V.engineMissingHtml(S.setup.engine) : '';
    if (html !== S.setupHtml) { S.setupHtml = html; $('setup').innerHTML = html; }
  };
  $('intro').addEventListener('click', function (e) {
    var t = e.target.closest && e.target.closest('button'); if (!t) return;
    if (t.id === 'introModels') $('modelsBtn').click();
    else if (t.id === 'introWhy') $('compareWhy').click();
  });

  // -- load, unload ---------------------------------------------------------------------------------------------------
  App.load = function () {
    var id = $('model').value; if (!id) return Promise.resolve(false);
    App.store('model', id);
    App.dispatch({ type: 'load' });
    var kv = App.lim('limKv');
    return App.api('POST', '/session', Object.assign({ model_id: id, compare: $('compare').checked }, kv ? { kv_gib: kv } : {})).then(function (s) {
      S.ticks = []; App.dispatch({ type: 'loaded', session: s }); App.buildCharts(); return true;
    }).catch(function (e) {
      App.dispatch({ type: 'loaded' }); App.refresh(); App.notify(e.message, 'error', 'Could not load ' + id); return false;
    });
  };
  $('load').onclick = App.load;
  ['model', 'compare'].forEach(function (id) { $(id).onchange = renderLoad; });
  $('stop').onclick = function () { App.api('DELETE', '/session').then(function () { App.dispatch({ type: 'unloaded' }); App.buildCharts(); }); };
  document.addEventListener('models-changed', function (e) {
    App.loadModels().then(function () { if (e.detail.id) $('model').value = e.detail.id; renderLoad(); });
  });
  document.addEventListener('models-load', function (e) { $('model').value = e.detail.id; renderLoad(); App.load(); });
})();
