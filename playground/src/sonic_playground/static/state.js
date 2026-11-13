/* The page's run and session state: all it knows of the run and the session, changed only by LiveView.reduceRun. A pure
   function (state, event) -> state, no DOM and no network. Loaded after charts.js, before the other LiveView files. */
(function () {
  'use strict';
  var C = window.Charts;
  var V = window.LiveView = {};
  V.many = function (n, one) { return C.fmt(n) + ' ' + one + (n === 1 ? '' : 's'); };

  // -- the run and session state -------------------------------------------------------------------------------------
  // All the page knows of the run and the session, changed only by V.reduceRun(state, event) -> new state:
  //   session  what the server said last (GET /state, its snapshot and session events)
  //   loading  this tab's POST /session is on its way
  //   phase    idle: no run · sending: this tab asked for a run, no answer yet · running · stopping: Stop was asked
  //   run      {id, elsewhere, scenario, engines}, set exactly while running or stopping; id is null for a run started
  //            elsewhere that has sent no event yet (the page was opened mid-run, or its run_start was lost)
  //   gone     the last runs that ended: a late event of one does not bring it back
  // The server's word wins over the page's: a fresh "not busy" ends the run the page shows, a fresh "busy" with no run
  // shows one started elsewhere. An answer to /state asked before a run started or ended on the page is not fresh (it
  // only updates the session).
  V.RUN0 = { session: null, loading: false, phase: 'idle', run: null, gone: [] };
  V.runKey = function (st) { return st.phase + ':' + (st.run ? st.run.id : ''); };
  V.isBusy = function (st) { return st.phase === 'running' || st.phase === 'stopping'; };
  function put(st, o) { return Object.assign({}, st, o); }
  function seen(st, id) { return st.gone.indexOf(id) >= 0; }
  function remember(st, id) { return id && !seen(st, id) ? put(st, { gone: st.gone.concat(id).slice(-16) }) : st; }
  function ended(st, id) { return put(remember(st, id), { phase: 'idle', run: null }); }
  // a run going: `ev` gives its id, scenario and engines (run_start has all three); a Stop asked for it stays asked
  function going(st, ev, elsewhere) {
    var same = !!st.run && (st.run.id === null || st.run.id === ev.run);
    return put(st, { phase: same && st.phase === 'stopping' ? 'stopping' : 'running',
      run: { id: ev.run, elsewhere: elsewhere, scenario: ev.scenario || '', engines: ev.engines || null } });
  }
  function fromServer(st, state, fresh) {
    var s = state.session, busy = !!(s && s.busy);
    var out = put(st, { session: s });
    if (!fresh) return out;
    if (!busy && V.isBusy(st)) return ended(out, st.run.id);
    if (busy && st.phase === 'idle') return going(out, { run: null }, true);
    return out;
  }
  // an event of a run: the first one names a run started elsewhere
  function heard(st, ev) { return st.run && st.run.id === null && !seen(st, ev.run) ? going(st, ev, true) : st; }
  function isRun(st, id) { return !!st.run && (st.run.id === id || (st.run.id === null && !seen(st, id))); }
  var STEPS = {
    load: function (st) { return put(st, { loading: true, session: null }); },
    loaded: function (st, ev) { return put(st, { loading: false, session: 'session' in ev ? ev.session : st.session }); },
    unloaded: function (st) { return ended(put(st, { session: null }), st.run && st.run.id); },
    server: function (st, ev) { return fromServer(st, ev.state, ev.fresh); },
    session: function (st, ev) { return fromServer(st, { session: ev.session }, true); },
    send: function (st) { return st.phase === 'idle' ? put(st, { phase: 'sending' }) : st; },
    // POST /run's answer: the run is going even if its run_start is lost (or it is over already)
    sent: function (st, ev) { return st.phase !== 'sending' || !ev.run ? st : seen(st, ev.run) ? put(st, { phase: 'idle' }) : going(st, ev, false); },
    send_failed: function (st) { return st.phase === 'sending' ? put(st, { phase: 'idle' }) : st; },
    run_start: function (st, ev) { return seen(st, ev.run) || (st.run && st.run.id === ev.run) ? st : going(st, ev, false); },
    stop: function (st) { return st.phase === 'running' ? put(st, { phase: 'stopping' }) : st; },
    stop_failed: function (st) { return st.phase === 'stopping' ? put(st, { phase: 'running' }) : st; },
    run_done: function (st, ev) { return isRun(st, ev.run) ? ended(st, ev.run) : remember(st, ev.run); },
    phase: heard, token: heard, agent: heard
  };
  V.reduceRun = function (st, ev) { return Object.prototype.hasOwnProperty.call(STEPS, ev.type) ? STEPS[ev.type](st, ev) : st; };
  // how a run_done says its run ended; what a run that did not finish says on its rows and answers
  V.doneHow = function (ev) { return ev.stopped ? 'stopped' : ev.error ? 'failed' : 'done'; };
  V.ENDED = { stopped: 'stopped', failed: 'failed', lost: 'ended' };
})();
