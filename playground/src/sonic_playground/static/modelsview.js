/* The models dialog's rows, built from plain data: models on this Mac, the recommended ones, Hugging Face search results
   and a repository's files, and the downloads going on. Fit and engine hints are small labelled pills (an icon and words,
   never colour alone). Extends window.LiveView. */
(function () {
  'use strict';
  var C = window.Charts, esc = C.esc, V = window.LiveView;
  function gb(bytes) { return bytes < 1e9 ? Math.round(bytes / 1e6) + ' MB' : (bytes / 1e9).toFixed(1) + ' GB'; }
  function icon(id) { return '<svg class="ic" aria-hidden="true"><use href="#i-' + id + '"/></svg>'; }
  function pill(ic, text, title, cls) { return '<span class="pill ' + (cls || '') + '" title="' + esc(title || '') + '">' + icon(ic) + esc(text) + '</span>'; }

  // Will it fit this Mac: the word says it, the icon repeats it
  var FIT = { 'fits easily': ['check', 'Memory: fits easily', 'good'], tight: ['alert', 'Memory: tight fit', ''], 'too large': ['no', 'Memory: too large for this Mac', 'bad'] };
  V.fitPill = function (fit, ramGb) {
    var f = FIT[fit]; return f ? pill(f[0], f[1], ramGb ? 'Needs about ' + ramGb + ' GB of memory' : f[1], f[2]) : '';
  };
  // Which engine runs it: Sonic LLM's own, or the fallback inside the server (it works, without the speed-ups)
  V.enginePill = function (engine) {
    if (!engine || engine.engine === 'unknown') return engine ? pill('info', 'Engine: not known yet', engine.text) : '';
    return engine.engine === 'own' ? pill('chip', 'Engine: Sonic LLM', engine.text, 'blue') : pill('alert', 'Engine: fallback, slower', engine.text);
  };
  function row(name, meta, pills, action) {
    return '<li class="mrow2"><div><div class="name">' + name + '</div><div class="meta">' + meta + '</div>' + (pills ? '<div class="pills">' + pills + '</div>' : '') + '</div><div class="act">' + action + '</div></li>';
  }
  function btn(label, attrs, cls) { return '<button class="btn ' + (cls || '') + '" type="button" ' + attrs + '>' + label + '</button>'; }

  // On this Mac: Load, Remove (asks again first: `sure` is the id asked about)
  V.installedHtml = function (ms, loaded, sure) {
    if (!ms.length) return '<li class="empty"><b>No models on this Mac yet</b>Download one from Recommended or search Hugging Face.</li>';
    return ms.map(function (m) {
      var meta = esc([m.quant, m.size_bytes ? gb(m.size_bytes) : '', m.arch].filter(Boolean).join(' · '));
      var pills = (m.id === loaded ? pill('check', 'Loaded now', '', 'good') : '') + (m.error ? pill('no', m.error, '', 'bad') : V.enginePill(m.engine));
      var act = sure === m.id ? '<span class="small">Delete the file?</span>' + btn('Yes, remove', 'data-remove="' + esc(m.id) + '"', 'danger') + btn('Keep', 'data-keep="1"') :
        (m.error ? '' : btn('Load', 'data-load="' + esc(m.id) + '"', 'primary')) + btn(icon('trash') + 'Remove', 'data-ask="' + esc(m.id) + '"');
      return row(esc(m.id), meta, pills, act);
    }).join('');
  };
  V.recommendedHtml = function (ms) {
    return ms.map(function (m, i) {
      return row(esc(m.name), gb(m.size_bytes) + ' on disk · needs about ' + m.ram_gb + ' GB of memory', V.fitPill(m.fit, m.ram_gb) + V.enginePill(m.engine),
        m.installed ? pill('check', 'Installed', '', 'good') : btn(icon('download') + 'Download', 'data-rec="' + i + '"'));
    }).join('');
  };
  V.hubResultsHtml = function (results) {
    return results.map(function (r) {
      return row(esc(r.id), C.fmt(r.downloads) + ' downloads · ' + C.fmt(r.likes) + ' likes', '', btn('Show files', 'data-repo="' + esc(r.id) + '"'));
    }).join('');
  };
  // One repository: its engine hint, and each single-file model with its quantization, size and fit
  V.repoHtml = function (d) {
    var hidden = d.hidden.split + d.hidden.projector ? '<p class="small muted">Not shown: ' + d.hidden.split + ' files in several parts and ' + d.hidden.projector + ' vision add-ons. The playground loads one file.</p>' : '';
    return '<div class="repo-h"><h3>' + esc(d.id) + '</h3>' + V.enginePill(d.engine) + (d.gated ? pill('alert', 'Gated: needs HF_TOKEN', 'This repository needs a Hugging Face account') : '') + '</div>' +
      (d.files.length ? '<ul class="mlist">' + d.files.map(function (f, i) {
        return row(esc(f.name), esc(f.quant) + ' · ' + gb(f.size) + ' · needs about ' + f.ram_gb + ' GB of memory', V.fitPill(f.fit, f.ram_gb), btn(icon('download') + 'Download', 'data-file="' + i + '"'));
      }).join('') + '</ul>' : '<p class="small muted">This repository has no single-file .gguf models.</p>') + hidden;
  };

  // -- downloads ------------------------------------------------------------------------------------------------------
  function dlMeta(d) {
    if (d.state === 'downloading') return gb(d.bytes_done) + ' of ' + gb(d.bytes_total) + ' · ' + (d.percent || 0) + '% · ' + (d.speed_bps / 1e6).toFixed(1) + ' MB/s' + (d.resumed ? ' · resumed' : '');
    return { starting: 'Starting…', verifying: 'Checking the file…', failed: d.error, cancelled: 'Cancelled. Start it again to continue where it stopped.', done: 'Done. It is in your model list.' }[d.state] || d.state;
  }
  // The downloads worth showing: going on, stopped, or finished while this dialog was open (`fresh`: their ids)
  V.downloadsHtml = function (ds, fresh) {
    return ds.filter(function (d) { return d.state !== 'done' || fresh[d.id]; }).map(function (d) {
      var live = isLive(d), name = d.name.replace(/\.gguf$/, '');
      return '<div class="dl' + (d.state === 'failed' ? ' bad' : '') + '"><span class="name">' + esc(d.name) + '</span>' +
        (live ? btn('Cancel', 'data-cancel="' + esc(d.id) + '"') : d.state === 'done' ? btn('Load', 'data-load="' + esc(name) + '"', 'primary') : '<span></span>') +
        (d.bytes_total && live ? '<progress max="' + d.bytes_total + '" value="' + d.bytes_done + '" aria-label="Downloading ' + esc(d.name) + '"></progress>' : '') +
        '<span class="meta">' + esc(dlMeta(d)) + '</span></div>';
    }).join('');
  };
  function isLive(d) { return d.state === 'starting' || d.state === 'downloading' || d.state === 'verifying'; }
  V.downloadsActive = function (ds) { return ds.some(isLive); };
})();
