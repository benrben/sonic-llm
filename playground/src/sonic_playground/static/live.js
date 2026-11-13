/* The live page's engine room: every server event goes through dispatch (the run state, then the stage), the event stream
   with its reconnects, and the start of the page. Server: /api/live/* (live.py). */
(function () {
  'use strict';
  var App = window.App, S = App.S, V = App.V;

  // -- the run state on the page ------------------------------------------------------------------------------------------
  // Every change goes through V.reduceRun; then the stage follows: a run that ended is closed, a new one gets its box, a
  // run started elsewhere keeps its box when its first event names it.
  App.dispatch = function (ev) {
    var was = S.view;
    S.view = V.reduceRun(was, ev); S.session = S.view.session;
    if (S.view === was) return;
    followRun(was, S.view, ev);
    App.renderSession();
  };
  function followRun(was, now, ev) {
    var a = was.run, b = now.run;
    if (a === b) return;
    if (a) leaveBox(a, b, endedHow(ev, a, was.phase));
    if (b && !S.runs[App.boxKey(b)]) App.startRun(b);
  }
  // the box of run a, now that run b is going (or none): an ended run is closed; a stand-in (a run started elsewhere,
  // not named yet) is named by its first event, or replaced when its run_start comes
  function leaveBox(a, b, how) {
    if (a.id !== null) { if (!b || b.id !== a.id) App.closeRun(a.id, how); }
    else if (!b) App.closeRun('', how);
    else if (b.elsewhere) App.moveBox('', b.id);
    else App.dropBox('');
  }
  // how a run ended: its run_done says, else the server says no run is going (its run_done was missed)
  function endedHow(ev, run, phase) { return ev.type === 'run_done' && ev.run === run.id ? V.doneHow(ev) : phase === 'stopping' ? 'stopped' : 'lost'; }

  // -- events ---------------------------------------------------------------------------------------------------------
  // One stream for all tabs (events.js), or one of its own where a browser has no shared workers. The server ends the stream
  // every 20 s and it reconnects at once, replaying what it missed (Last-Event-ID): only a stream that stays down is shown.
  var ON = {  // the server's events by type; other types are ignored
    tick: App.onTick,
    log: function (ev) { App.logLine(ev.msg); },
    session: function (ev) {
      if (ev.error && !S.view.loading) App.notify('The model could not load: ' + ev.error, 'error');  // a load from another tab
      if (!ev.session || ev.session.state === 'ready') App.buildCharts();
    },
    snapshot: function (ev) { applyState(ev.state, true); },  // the first event after every connect
    resync: function () { App.refresh(); },
    phase: App.onPhase, token: App.onToken, agent: App.onAgent, run_done: App.onRunDone
  };
  // every event changes the run state first (run_start, run_done, session, a run's first event), then shows itself
  function onEvent(ev) {
    App.dispatch(ev);
    if (Object.prototype.hasOwnProperty.call(ON, ev.type)) ON[ev.type](ev);
  }
  function connect() {
    var downTimer = null;
    function onData(data) { onEvent(JSON.parse(data)); }
    function onDown() { if (!downTimer) downTimer = setTimeout(function () { App.setStatus('Reconnecting…', 'busy'); }, 1500); }
    function onUp() { clearTimeout(downTimer); downTimer = null; App.refresh(); }
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
    App.dispatch({ type: 'server', state: st, fresh: fresh });
    if ((S.session && S.session.model) !== before) App.buildCharts();
  }
  App.refresh = function () {
    var asked = V.runKey(S.view); return App.api('GET', '/state').then(function (st) { applyState(st, V.runKey(S.view) === asked); });
  };

  // -- the start --------------------------------------------------------------------------------------------------------
  App.showThink();
  App.api('GET', '/state').then(function (st) {
    S.scenarios = st.scenarios;
    var saved = App.store('scen'); if (saved && st.scenarios.some(function (x) { return x.id === saved; })) S.scen = saved;
    try { var kept = JSON.parse(App.store('picks') || '{}'); S.picks = { chat: kept.chat || [], jobs: kept.jobs || [] }; } catch (e) { console.warn('live: the saved document picks are not readable: ' + e.message); }
    S.picked = S.picks[App.fileSet(S.scen)];
    App.renderTools(); App.dispatch({ type: 'server', state: st, fresh: true });
  }).then(App.loadSetup).then(App.loadModels).then(App.loadFiles).then(connect).catch(function (e) { App.notify('The playground could not start: ' + e.message, 'error'); });
})();
