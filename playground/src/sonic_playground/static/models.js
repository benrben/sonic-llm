/* The models dialog of the live page: recommended models, a search on Hugging Face, a link of your own, the downloads going on and
   the models on this computer. Server: /api/catalog, /api/hub/*, /api/downloads, /api/models (model_api.py, app.py). It tells
   live.js through the "models-changed" event when a model arrived or was removed. */
(function () {
  'use strict';
  var C = window.Charts, esc = C.esc, $ = function (id) { return document.getElementById(id); };
  var M = { seen: {}, timer: null, searchTimer: null, files: null, rec: [] };

  function api(method, path, body) {
    return fetch(path, { method: method, headers: body ? { 'Content-Type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined })
      .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.detail || r.statusText); return j; }); });
  }
  function gb(bytes) { return (bytes / 1e9).toFixed(1) + ' GB'; }
  function say(el, msg) { el.textContent = msg || ''; }
  var FIT = { 'fits easily': 'fits easily', tight: 'tight on this Mac', 'too large': 'too large for this Mac', unknown: '' };
  function row(main, sub, action) {
    return '<div class="mrow2"><div><b>' + main + '</b><div class="small muted">' + sub + '</div></div><div>' + action + '</div></div>';
  }
  function button(label, attrs, cls) { return '<button class="' + (cls || 'ghost') + '" ' + attrs + '>' + label + '</button>'; }

  // -- recommended -----------------------------------------------------------------------------------------------------
  function loadRecommended() {
    return api('GET', '/api/catalog').then(function (d) {
      $('recList').innerHTML = d.models.map(function (m, i) {
        var sub = gb(m.size_bytes) + ' on disk · about ' + m.ram_gb + ' GB of memory · ' + esc(FIT[m.fit]) + ' · ' + esc(m.engine.text);
        return row(esc(m.name), sub, m.installed ? '<span class="small muted">Installed</span>' : button('Download', 'data-rec="' + i + '"'));
      }).join('');
      M.rec = d.models;
    }).catch(function (e) { $('recList').textContent = e.message; });
  }
  $('recList').onclick = function (e) {
    var b = e.target.closest('[data-rec]'); if (!b) return;
    var m = M.rec[+b.dataset.rec];
    startDownload({ source: m.repo, file: m.file, sha256: m.sha256 });
  };

  // -- search ----------------------------------------------------------------------------------------------------------
  function search() {
    say($('hubStatus'), 'Searching…');
    api('GET', '/api/hub/search?q=' + encodeURIComponent($('hubSearch').value)).then(function (d) {
      say($('hubStatus'), d.results.length ? ($('hubSearch').value.trim() ? '' : 'The most downloaded GGUF models:') : 'Nothing found.');
      $('hubResults').innerHTML = d.results.map(function (r) {
        return row(esc(r.id), C.fmt(r.downloads) + ' downloads · ' + C.fmt(r.likes) + ' likes', button('Show files', 'data-repo="' + esc(r.id) + '"'));
      }).join('');
    }).catch(function (e) { say($('hubStatus'), e.message); $('hubResults').innerHTML = ''; });
  }
  $('hubSearch').oninput = function () { clearTimeout(M.searchTimer); M.searchTimer = setTimeout(search, 350); };
  $('hubResults').onclick = function (e) {
    var b = e.target.closest('[data-repo]'); if (!b) return;
    say($('hubStatus'), 'Reading ' + b.dataset.repo + '…');
    api('GET', '/api/hub/repo?id=' + encodeURIComponent(b.dataset.repo)).then(function (d) {
      M.files = d; say($('hubStatus'), '');
      var hidden = d.hidden.split + d.hidden.projector ? '<p class="small muted">Not shown: ' + d.hidden.split + ' files in several parts and ' + d.hidden.projector + ' vision add-ons.</p>' : '';
      $('hubFiles').innerHTML = '<h3>' + esc(d.id) + '</h3><p class="small muted">' + esc(d.engine.text) + (d.gated ? ' This repository is gated: it needs HF_TOKEN.' : '') + '</p>' +
        (d.files.length ? d.files.map(function (f, i) {
          return row(esc(f.name), esc(f.quant) + ' · ' + gb(f.size) + ' · about ' + f.ram_gb + ' GB of memory · ' + esc(FIT[f.fit]), button('Download', 'data-file="' + i + '"'));
        }).join('') : '<p class="small muted">This repository has no single-file .gguf models.</p>') + hidden;
    }).catch(function (err) { say($('hubStatus'), err.message); $('hubFiles').innerHTML = ''; });
  };
  $('hubFiles').onclick = function (e) {
    var b = e.target.closest('[data-file]'); if (!b) return;
    var f = M.files.files[+b.dataset.file];
    startDownload({ source: M.files.id, file: f.path, sha256: f.sha256 });
  };
  $('customGo').onclick = function () { startDownload({ source: $('customSrc').value, file: $('customFile').value }); };

  // -- downloads -------------------------------------------------------------------------------------------------------
  function startDownload(body) {
    api('POST', '/api/downloads', body).then(function () { poll(); }).catch(function (e) { alert(e.message); });
  }
  function dlRow(d) {
    var bar = d.bytes_total ? '<progress max="' + d.bytes_total + '" value="' + d.bytes_done + '"></progress> ' : '';
    var sub = d.state === 'downloading' ? gb(d.bytes_done) + ' of ' + gb(d.bytes_total) + ' · ' + (d.percent || 0) + '% · ' + (d.speed_bps / 1e6).toFixed(1) + ' MB/s' + (d.resumed ? ' · resumed' : '')
      : d.state === 'failed' ? esc(d.error) : d.state === 'cancelled' ? 'Cancelled. Start it again to continue where it stopped.' : d.state === 'done' ? 'Done: it is in the model list.' : d.state;
    var act = d.state === 'downloading' || d.state === 'starting' ? button('Cancel', 'data-cancel="' + d.id + '"') : '';
    return row(esc(d.name), bar + sub, act);
  }
  function poll() {
    api('GET', '/api/downloads').then(function (r) {
      $('dlList').innerHTML = r.downloads.length ? r.downloads.map(dlRow).join('') : '<span class="small muted">None yet.</span>';
      var active = r.downloads.some(function (d) { return d.state === 'starting' || d.state === 'downloading' || d.state === 'verifying'; });
      r.downloads.forEach(function (d) {
        if (d.state === 'done' && !M.seen[d.id]) { M.seen[d.id] = true; changed(d.name.replace(/\.gguf$/, '')); }
      });
      clearTimeout(M.timer); if (active) M.timer = setTimeout(poll, 700);
    });
  }
  $('dlList').onclick = function (e) {
    var b = e.target.closest('[data-cancel]'); if (b) api('POST', '/api/downloads/' + b.dataset.cancel + '/cancel').then(poll);
  };

  // -- on this computer ------------------------------------------------------------------------------------------------
  function changed(id) {
    document.dispatchEvent(new CustomEvent('models-changed', { detail: { id: id } }));
    loadInstalled(); loadRecommended();
  }
  function loadInstalled() {
    return api('GET', '/api/models').then(function (d) {
      $('modelsDir').textContent = d.dir;
      $('installedList').innerHTML = d.models.length ? d.models.map(function (m) {
        var sub = esc(m.quant) + ' · ' + gb(m.size_bytes) + (m.arch ? ' · ' + esc(m.arch) : '') + (m.error ? ' · ' + esc(m.error) : ' · ' + esc(m.engine.text));
        return row(esc(m.id), sub, button('Remove', 'data-remove="' + esc(m.id) + '"'));
      }).join('') : '<span class="small muted">No models yet.</span>';
    });
  }
  $('installedList').onclick = function (e) {
    var b = e.target.closest('[data-remove]'); if (!b) return;
    if (!window.confirm('Remove ' + b.dataset.remove + ' from your computer? Its file is deleted.')) return;
    api('DELETE', '/api/models/' + b.dataset.remove).then(function () { changed(null); }).catch(function (err) { alert(err.message); });
  };

  // -- the dialog ------------------------------------------------------------------------------------------------------
  $('modelsBtn').onclick = function () {
    $('models').showModal();
    api('GET', '/api/setup').then(function (s) {
      $('modelsOffline').hidden = !s.offline; $('modelsOnline').hidden = s.offline;
      loadInstalled(); poll();
      if (!s.offline) { loadRecommended(); search(); }
    });
  };
  $('modelsClose').onclick = function () { $('models').close(); };
})();
