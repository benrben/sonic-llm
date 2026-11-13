/* The playground page's shared ground: its state (App.S), the calls to the server, the small things every part uses
   (settings kept in the browser, notices, popovers, dialogs). The parts that follow (topbar, tests, files, result,
   history, live, models) add their functions to window.App. */
(function () {
  'use strict';
  var C = window.Charts, V = window.LiveView;
  var App = window.App = { C: C, V: V, esc: C.esc, WINDOW_S: 60, MAX_AGENTS: 256 };
  App.$ = function (id) { return document.getElementById(id); };
  // S.view: the run and session state, changed only by dispatch (V.reduceRun); S.session mirrors S.view.session
  App.S = { view: V.RUN0, session: null, scenarios: [], scen: 'chat', files: [], picked: [], picks: { chat: [], jobs: [] }, open: {}, shown: {}, sure: null,
    agents: null, think: 0, level: 'off', runs: {}, ticks: [], lastTick: null, swarm: {}, gridRun: null, setup: null, models: 0,
    final: {}, live: {}, tilesHtml: '', hasRun: false, started: null, room: null, roomAsked: null };
  App.isIn = function (list, x) { return list.indexOf(x) >= 0; };

  // JSON over fetch; an error answer throws with the server's own words
  App.request = function (method, url, body) {
    return fetch(url, { method: method, headers: body ? { 'Content-Type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined })
      .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.detail || j.error || r.statusText); return j; }); });
  };
  App.api = function (method, path, body) { return App.request(method, '/api/live' + path, body); };

  // Settings kept in this browser (a private window may refuse): said once, the page works without them
  var warned = false;
  App.storageFailed = function (e) {
    if (!warned) console.warn('live: settings are not kept (local storage failed: ' + (e && e.message || e) + ')');
    warned = true;
  };
  App.store = function (k, v) {
    try { if (v === undefined) return localStorage.getItem('live.' + k); localStorage.setItem('live.' + k, v); } catch (e) { App.storageFailed(e); }
    return null;
  };
  App.logLine = function (msg) { var d = document.createElement('div'); d.textContent = new Date().toLocaleTimeString() + '  ' + msg; App.$('log').prepend(d); };

  // A message about what just happened (a failed load or run), above the results
  App.notify = function (msg, kind, title) { App.$('notice').innerHTML = V.noticeHtml(msg, kind, title); };
  App.$('notice').addEventListener('click', function (e) { if (e.target.closest && e.target.closest('[data-dismiss]')) App.$('notice').innerHTML = ''; });

  // -- popovers: open from a button, close on a click outside, on Escape (focus goes back to the button) -------------------
  var pops = [];
  function closePops(except) {
    pops.forEach(function (p) { if (p[0] === except) return; App.$(p[1]).hidden = true; App.$(p[0]).setAttribute('aria-expanded', 'false'); });
  }
  App.popover = function (btn, panel) {
    pops.push([btn, panel]);
    App.$(btn).addEventListener('click', function (e) {
      e.stopPropagation();
      var open = App.$(panel).hidden; closePops(btn); App.$(panel).hidden = !open; App.$(btn).setAttribute('aria-expanded', String(open));
    });
    App.$(panel).addEventListener('click', function (e) { e.stopPropagation(); });
  };
  document.addEventListener('click', function () { closePops(); });
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Escape') return;
    pops.forEach(function (p) { if (!App.$(p[1]).hidden) { closePops(); App.$(p[0]).focus(); } });
  });

  // -- dialogs: modal (focus stays inside, Escape closes); focus returns to what opened them; a click outside closes ---------
  App.openDialog = function (dlg) {
    var opener = document.activeElement;
    dlg.addEventListener('close', function () { if (opener && opener.focus) opener.focus(); }, { once: true });
    dlg.showModal();
  };
  App.closeOnBackdrop = function (dlg) {
    dlg.addEventListener('click', function (e) {
      var r = dlg.getBoundingClientRect();
      if (e.target === dlg && (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom)) dlg.close();
    });
  };

  // the sticky parts of the page sit below the top bar, whatever height it wraps to
  var bar = document.querySelector('.top');
  function fitTop() { document.documentElement.style.setProperty('--top-h', bar.offsetHeight + 'px'); }
  if (bar) { fitTop(); window.addEventListener('resize', fitTop); }
  // text that a person can copy with one button (the build command)
  document.addEventListener('click', function (e) {
    var b = e.target.closest && e.target.closest('[data-copy]'); if (!b) return;
    navigator.clipboard.writeText(b.dataset.copy).then(function () { b.lastChild.textContent = 'Copied'; setTimeout(function () { b.lastChild.textContent = 'Copy'; }, 1500); });
  });
})();
