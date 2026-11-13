/* The documents the tests read: the dialog that lists the samples and your own files (search, pick, add by button or
   drop, remove your own), and the chips of the picked ones in the picker. Server: /api/live/files. */
(function () {
  'use strict';
  var App = window.App, S = App.S, $ = App.$, V = App.V, C = App.C, many = V.many;
  function say(msg) { $('filesMsg').textContent = msg || ''; }

  App.loadFiles = function () {
    return App.api('GET', '/files').then(function (d) {
      S.files = d.files;
      var have = d.files.map(function (f) { return f.path; });
      var real = function (p) { return have.indexOf(p) >= 0; };
      S.picks.chat = S.picks.chat.filter(real); S.picks.jobs = S.picks.jobs.filter(real);
      S.picked = S.picks[App.fileSet(S.scen)];
      App.renderAttached();
    });
  };
  function renderFiles() {
    var q = $('filesSearch').value.toLowerCase(), groups = S.shown = V.fileGroups(S.files, q);
    $('filesList').innerHTML = Object.keys(groups).sort().map(function (g) { return V.fileGroupHtml(g, groups[g], S.picked, q || S.open[g] !== false, S.sure); }).join('') ||
      '<p class="empty">' + (q ? 'No document matches.' : 'There are no documents yet. Add a text file below.') + '</p>';
    Array.prototype.forEach.call($('filesList').querySelectorAll('[data-partial]'), function (x) { x.indeterminate = true; });
    var tok = S.picked.reduce(function (a, p) { var f = S.files.filter(function (x) { return x.path === p; })[0]; return a + (f ? f.est_tokens : 0); }, 0);
    $('filesSel').textContent = S.picked.length ? many(S.picked.length, 'document') + ' picked · about ' + C.fmt(tok) + ' tokens' : 'None picked: the tests use the biggest documents';
    $('nfiles').textContent = S.picked.length ? many(S.picked.length, 'document') + ' picked' : 'Choose documents';
    $('attach').title = S.picked.map(function (p) { return p.split('/').pop(); }).join(', ');
  }
  App.renderAttached = function () {
    S.picks[App.fileSet(S.scen)] = S.picked;
    App.store('picks', JSON.stringify(S.picks));
    $('attached').innerHTML = V.attachedHtml(S.picked);
    renderFiles(); App.renderAgents(); App.renderPlan(); App.renderSuggest();
  };
  $('attached').onclick = function (e) {
    var b = e.target.closest && e.target.closest('[data-x]'); if (!b) return;
    S.picked = b.dataset.x === '*' ? [] : S.picked.filter(function (p) { return p !== b.dataset.x; }); App.renderAttached();
  };
  $('attach').onclick = function () { say(''); App.openDialog($('files')); App.loadFiles(); };
  App.closeOnBackdrop($('files'));
  $('filesClose').onclick = $('filesDone').onclick = function () { $('files').close(); };
  $('files').onclose = App.renderAttached;
  $('filesNone').onclick = function () { S.picked = []; App.renderAttached(); };
  $('filesSearch').oninput = renderFiles;
  $('filesList').onclick = function (e) {
    var t = e.target.closest && e.target.closest('button'); if (!t) return;
    if (t.dataset.open) S.open[t.dataset.open] = S.open[t.dataset.open] === false;
    else if (t.dataset.ask) S.sure = t.dataset.ask;
    else if (t.dataset.keep) S.sure = null;
    else if (t.dataset.del) return remove(t.dataset.del);
    renderFiles();
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
    App.renderAttached();
  };

  // Your own documents: each is sent as the body of PUT /files/<name>, then the library is read again.
  function upload(list) {
    var sent = Array.prototype.map.call(list, function (f) {
      return fetch('/api/live/files/' + encodeURIComponent(f.name), { method: 'PUT', body: f })
        .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(f.name + ': ' + (j.detail || r.statusText)); App.logLine('added ' + j.path); say('Added ' + f.name + '.'); }); })
        .catch(function (e) { say(e.message); });
    });
    return Promise.all(sent).then(App.loadFiles).then(function () { $('upload').value = ''; });
  }
  function remove(path) {
    S.sure = null;
    return fetch('/api/live/files/' + encodeURIComponent(path.split('/').pop()), { method: 'DELETE' })
      .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.detail || r.statusText); App.logLine('removed ' + j.path); say('Removed ' + path.split('/').pop() + '.'); }); })
      .catch(function (e) { say(e.message); })
      .then(function () { S.picked = S.picked.filter(function (p) { return p !== path; }); return App.loadFiles(); });
  }
  $('upload').onchange = function () { upload($('upload').files); };
  $('drop').addEventListener('dragover', function (e) { e.preventDefault(); $('drop').classList.add('over'); });
  $('drop').addEventListener('dragleave', function () { $('drop').classList.remove('over'); });
  $('drop').addEventListener('drop', function (e) { e.preventDefault(); $('drop').classList.remove('over'); upload(e.dataTransfer.files); });
})();
