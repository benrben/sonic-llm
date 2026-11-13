/* The history drawer: the runs the playground saved (GET /api/live/history, newest first). "Show numbers" puts one on the
   stage. Only the numbers are kept, not the answers. */
(function () {
  'use strict';
  var App = window.App, S = App.S, $ = App.$, V = App.V, esc = App.esc, many = V.many;
  var SHOWN = 50, runs = [];

  function when(t) { return new Date(t).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }); }
  function itemHtml(r, i) {
    var sc = S.scenarios.filter(function (x) { return x.id === r.scenario; })[0] || { name: r.scenario };
    var meta = [r.model, r.files && r.files.length ? many(r.files.length, 'document') : '', Object.keys(r.metrics || {}).length > 1 ? 'compared with llama.cpp' : 'Sonic LLM only'].filter(Boolean);
    return '<li class="hitem"><div class="top2"><b>' + esc(sc.name) + '</b><span class="when">' + esc(when(r.t)) + '</span></div>' +
      '<div class="meta">' + esc(meta.join(' · ')) + '</div>' +
      '<div class="heads">' + V.headlines(r).map(function (h) { return '<span class="pill">' + esc(h) + '</span>'; }).join('') + '</div>' +
      '<button class="btn" type="button" data-i="' + i + '"' + (V.isBusy(S.view) ? ' disabled title="A run is going: wait for it to end"' : '') + '>Show numbers</button></li>';
  }
  function render(list) {
    runs = list.slice(0, SHOWN);
    $('historyList').innerHTML = runs.length ? runs.map(itemHtml).join('') :
      '<li class="empty"><b>No runs yet</b>Every run you make is saved here, so you can look at its numbers again.</li>';
  }
  function open() {
    $('historyList').innerHTML = '<li class="empty">Loading…</li>';
    App.openDialog($('history'));
    App.api('GET', '/history').then(function (d) { render(d.runs); })
      .catch(function (e) { $('historyList').innerHTML = '<li class="empty"><b>The history could not be read</b>' + esc(e.message) + '</li>'; });
  }
  $('historyBtn').onclick = open;
  $('historyClose').onclick = function () { $('history').close(); };
  App.closeOnBackdrop($('history'));
  $('historyList').onclick = function (e) {
    var b = e.target.closest && e.target.closest('[data-i]'); if (!b || b.disabled) return;
    $('history').close();
    App.showRecord(runs[+b.dataset.i]);
    if (window.matchMedia && window.matchMedia('(max-width: 1000px)').matches) $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  };
})();
