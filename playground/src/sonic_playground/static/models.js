/* The models dialog: models on this Mac (load, remove), recommended models, a search on Hugging Face, a link of your own,
   and the downloads going on. Server: /api/catalog, /api/hub/*, /api/downloads, /api/models (model_api.py, app.py). It
   tells the page through the "models-changed" event when a model arrived or went, and "models-load" when one is to load. */
(function () {
  'use strict';
  var App = window.App, $ = App.$, V = App.V, esc = App.esc;
  var TABS = ['Installed', 'Rec', 'Hub'];
  var M = { seen: {}, fresh: {}, timer: null, searchTimer: null, repo: null, rec: [], installed: [], sure: null, hubLoaded: false, recLoaded: false };
  function api(method, path, body) { return App.request(method, path, body); }
  function say(el, msg) { el.textContent = msg || ''; }
  function fail(msg) { $('modelsErr').innerHTML = '<div class="callout error"><svg class="ic" aria-hidden="true"><use href="#i-alert"/></svg><div>' + esc(msg) + '</div></div>'; }

  // -- tabs (arrow keys move between them) ----------------------------------------------------------------------------------
  function selectTab(name, focus) {
    TABS.forEach(function (t) {
      var on = t === name;
      $('tab' + t).setAttribute('aria-selected', String(on)); $('tab' + t).tabIndex = on ? 0 : -1; $('pan' + t).hidden = !on;
      if (on && focus) $('tab' + t).focus();
    });
    if (name === 'Rec' && !M.recLoaded) loadRecommended();
    if (name === 'Hub' && !M.hubLoaded) search();
  }
  TABS.forEach(function (t, i) {
    $('tab' + t).onclick = function () { selectTab(t); };
    $('tab' + t).onkeydown = function (e) {
      var next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: TABS.length - 1 }[e.key];
      if (next === undefined) return;
      e.preventDefault(); selectTab(TABS[(next + TABS.length) % TABS.length], true);
    };
  });

  // -- on this Mac ----------------------------------------------------------------------------------------------------------
  function loadInstalled() {
    return api('GET', '/api/models').then(function (d) {
      M.installed = d.models; $('modelsDir').textContent = 'Models folder: ' + d.dir; renderInstalled();
    }).catch(function (e) { $('installedList').innerHTML = '<li class="empty"><b>The list could not be read</b>' + esc(e.message) + '</li>'; });
  }
  function renderInstalled() {
    var s = App.S.session; $('installedList').innerHTML = V.installedHtml(M.installed, s && s.model, M.sure);
  }
  $('installedList').onclick = function (e) {
    var b = e.target.closest && e.target.closest('button'); if (!b) return;
    if (b.dataset.ask) { M.sure = b.dataset.ask; renderInstalled(); }
    else if (b.dataset.keep) { M.sure = null; renderInstalled(); }
    else if (b.dataset.load) loadModel(b.dataset.load);
    else if (b.dataset.remove) {
      M.sure = null;
      api('DELETE', '/api/models/' + encodeURIComponent(b.dataset.remove)).then(function () { changed(null); }).catch(function (err) { fail(err.message); loadInstalled(); });
    }
  };
  function loadModel(id) { $('models').close(); document.dispatchEvent(new CustomEvent('models-load', { detail: { id: id } })); }
  function changed(id) {
    document.dispatchEvent(new CustomEvent('models-changed', { detail: { id: id } }));
    loadInstalled(); if (M.recLoaded) loadRecommended();
  }

  // -- recommended ----------------------------------------------------------------------------------------------------------
  function loadRecommended() {
    return api('GET', '/api/catalog').then(function (d) {
      M.rec = d.models; M.recLoaded = true; $('recList').innerHTML = V.recommendedHtml(d.models);
    }).catch(function (e) { $('recList').innerHTML = '<li class="empty"><b>The list could not be read</b>' + esc(e.message) + '</li>'; });
  }
  $('recList').onclick = function (e) {
    var b = e.target.closest && e.target.closest('[data-rec]'); if (!b) return;
    var m = M.rec[+b.dataset.rec];
    startDownload({ source: m.repo, file: m.file, sha256: m.sha256 });
  };

  // -- search ---------------------------------------------------------------------------------------------------------------
  function search() {
    say($('hubStatus'), 'Searching…');
    var q = $('hubSearch').value;
    api('GET', '/api/hub/search?q=' + encodeURIComponent(q)).then(function (d) {
      M.hubLoaded = true;
      say($('hubStatus'), d.results.length ? (q.trim() ? '' : 'The most downloaded GGUF models:') : 'Nothing found.');
      $('hubResults').innerHTML = V.hubResultsHtml(d.results);
    }).catch(function (e) { say($('hubStatus'), e.message); $('hubResults').innerHTML = ''; });
  }
  $('hubSearch').oninput = function () { clearTimeout(M.searchTimer); M.searchTimer = setTimeout(search, 350); };
  $('hubResults').onclick = function (e) {
    var b = e.target.closest && e.target.closest('[data-repo]'); if (!b) return;
    say($('hubStatus'), 'Reading ' + b.dataset.repo + '…');
    api('GET', '/api/hub/repo?id=' + encodeURIComponent(b.dataset.repo)).then(function (d) {
      M.repo = d; say($('hubStatus'), ''); $('hubFiles').innerHTML = V.repoHtml(d);
      $('hubFiles').scrollIntoView({ block: 'nearest' });
    }).catch(function (err) { say($('hubStatus'), err.message); $('hubFiles').innerHTML = ''; });
  };
  $('hubFiles').onclick = function (e) {
    var b = e.target.closest && e.target.closest('[data-file]'); if (!b) return;
    var f = M.repo.files[+b.dataset.file];
    startDownload({ source: M.repo.id, file: f.path, sha256: f.sha256 });
  };
  $('customGo').onclick = function () { startDownload({ source: $('customSrc').value, file: $('customFile').value }); };

  // -- downloads ------------------------------------------------------------------------------------------------------------
  function startDownload(body) {
    $('modelsErr').innerHTML = '';
    api('POST', '/api/downloads', body).then(function () { poll(); }).catch(function (e) { fail(e.message); });
  }
  function poll() {
    return api('GET', '/api/downloads').then(function (r) {
      r.downloads.forEach(function (d) {
        if (d.state !== 'done' || M.seen[d.id]) return;
        M.seen[d.id] = true;
        if (M.opened) M.fresh[d.id] = true;
        changed(d.name.replace(/\.gguf$/, ''));
      });
      $('dlStrip').innerHTML = V.downloadsHtml(r.downloads, M.fresh);
      clearTimeout(M.timer); if (V.downloadsActive(r.downloads)) M.timer = setTimeout(poll, 700);
    }).catch(function (e) { fail(e.message); });
  }
  $('dlStrip').onclick = function (e) {
    var b = e.target.closest && e.target.closest('button'); if (!b) return;
    if (b.dataset.cancel) api('POST', '/api/downloads/' + b.dataset.cancel + '/cancel').then(poll).catch(function (err) { fail(err.message); });
    else if (b.dataset.load) loadModel(b.dataset.load);
  };

  // -- the dialog -----------------------------------------------------------------------------------------------------------
  $('modelsBtn').onclick = function () {
    M.opened = false; M.fresh = {}; M.sure = null;
    App.openDialog($('models'));
    var ready = App.S.setup ? Promise.resolve(App.S.setup) : api('GET', '/api/setup');
    ready.then(function (s) {
      $('modelsOffline').hidden = !s.offline; $('tabRec').hidden = $('tabHub').hidden = s.offline;
      return loadInstalled().then(function () { selectTab(M.installed.length || s.offline ? 'Installed' : 'Rec'); return poll(); });
    }).then(function () { M.opened = true; });
  };
  $('modelsClose').onclick = function () { $('models').close(); };
  App.closeOnBackdrop($('models'));
})();
