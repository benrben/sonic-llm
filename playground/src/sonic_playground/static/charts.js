/* Small SVG charts for the live playground: a line chart with a crosshair tooltip, a ring gauge and paired bars.
   Colors come from CSS tokens (--series-1 Sonic LLM, --series-2 llama.cpp); text always wears text tokens. */
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  var C = window.Charts = {};
  C.SERIES = { sllm: { label: 'Sonic LLM', color: 'var(--series-1)', soft: 'var(--series-1-soft)' },
               llama: { label: 'llama.cpp', color: 'var(--series-2)', soft: 'var(--series-2-soft)' } };
  C.esc = function (s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };
  C.num = function (v) { return typeof v === 'number' && isFinite(v); };
  C.fmt = function (v, d) {
    if (!C.num(v)) return '–';
    if (d == null && Number.isInteger(v)) return v.toLocaleString('en-US');
    if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString('en-US');
    return v.toFixed(d == null ? (Math.abs(v) >= 100 ? 0 : 1) : d);
  };
  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function niceMax(v) {
    if (!(v > 0)) return 1;
    var p = Math.pow(10, Math.floor(Math.log10(v))), m = v / p;
    return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
  }
  var tip = function () { return document.getElementById('tip'); };
  C.showTip = function (html, x, y) {
    var t = tip(); if (!t) return;
    t.innerHTML = html; t.hidden = false;
    var w = t.offsetWidth, h = t.offsetHeight;
    t.style.left = Math.min(window.innerWidth - w - 8, x + 14) + 'px';
    t.style.top = Math.max(8, y - h - 10) + 'px';
  };
  C.hideTip = function () { var t = tip(); if (t) t.hidden = true; };

  /* Line chart. opts: {series: [{key, label, color}], unit, windowS, yMax, fmt}. update(points) where
     points = [{t (s), values: {key: number}}]; the x axis is the last windowS seconds. */
  C.line = function (host, opts) {
    // drawn at the host's own pixel size, so text and strokes are never stretched (rebuild the chart on resize)
    var W = Math.max(240, host.clientWidth || 600), H = host.clientHeight || 160, pad = { l: 44, r: 12, t: 8, b: 20 };
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': opts.label || '' }, host);
    var grid = el('g', {}, svg), lines = el('g', {}, svg), hover = el('g', {}, svg);
    var cross = el('line', { y1: pad.t, y2: H - pad.b, stroke: 'var(--axis)', 'stroke-width': 1, visibility: 'hidden' }, hover);
    var dots = {};
    var paths = {}, areas = {};
    opts.series.forEach(function (s) {
      areas[s.key] = el('path', { fill: s.soft || 'none', stroke: 'none' }, lines);
      paths[s.key] = el('path', { fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'vector-effect': 'non-scaling-stroke' }, lines);
      dots[s.key] = el('circle', { r: 4, fill: s.color, stroke: 'var(--surface-1)', 'stroke-width': 2, visibility: 'hidden' }, hover);
    });
    var hit = el('rect', { x: pad.l, y: pad.t, width: W - pad.l - pad.r, height: H - pad.t - pad.b, fill: 'transparent' }, svg);
    var last = [], xs = null, ys = null, fmt = opts.fmt || function (v) { return C.fmt(v); };
    function draw(points, now) {
      last = points;
      var t1 = now, t0 = now - opts.windowS;
      var vis = points.filter(function (p) { return p.t >= t0; });
      var mx = opts.yMax || 0;
      vis.forEach(function (p) { opts.series.forEach(function (s) { var v = p.values[s.key]; if (C.num(v) && v > mx) mx = v; }); });
      mx = opts.yMax || niceMax(mx * 1.1);
      xs = function (t) { return pad.l + (t - t0) / (t1 - t0) * (W - pad.l - pad.r); };
      ys = function (v) { return H - pad.b - v / mx * (H - pad.t - pad.b); };
      grid.innerHTML = '';
      for (var i = 0; i <= 4; i++) {
        var v = mx * i / 4, y = ys(v);
        el('line', { x1: pad.l, x2: W - pad.r, y1: y, y2: y, stroke: i ? 'var(--grid)' : 'var(--axis)', 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke' }, grid);
        var tx = el('text', { x: pad.l - 6, y: y + 4, 'text-anchor': 'end', 'font-size': 10, fill: 'var(--text-muted)' }, grid);
        tx.textContent = fmt(v);
      }
      var xl = opts.xLabels || ['-' + opts.windowS + ' s', '-' + opts.windowS / 2 + ' s', 'now'];
      [opts.windowS, opts.windowS / 2, 0].forEach(function (s, i) {
        if (!xl[i]) return;
        var tx = el('text', { x: xs(t1 - s), y: H - 5, 'text-anchor': i === 0 ? 'start' : i === 2 ? 'end' : 'middle', 'font-size': 10, fill: 'var(--text-muted)' }, grid);
        tx.textContent = xl[i];
      });
      opts.series.forEach(function (s) {
        var pts = vis.filter(function (p) { return C.num(p.values[s.key]); });
        if (!pts.length) { paths[s.key].setAttribute('d', ''); areas[s.key].setAttribute('d', ''); return; }
        var d = pts.map(function (p, i) { return (i ? 'L' : 'M') + xs(p.t).toFixed(1) + ',' + ys(p.values[s.key]).toFixed(1); }).join('');
        paths[s.key].setAttribute('d', d);
        areas[s.key].setAttribute('d', d + 'L' + xs(pts[pts.length - 1].t).toFixed(1) + ',' + ys(0) + 'L' + xs(pts[0].t).toFixed(1) + ',' + ys(0) + 'Z');
      });
    }
    function onMove(ev) {
      if (!xs || !last.length) return;
      var r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * W;
      var best = null, bd = 1e9;
      last.forEach(function (p) { var d = Math.abs(xs(p.t) - x); if (d < bd) { bd = d; best = p; } });
      if (!best) return;
      var bx = xs(best.t);
      cross.setAttribute('x1', bx); cross.setAttribute('x2', bx); cross.setAttribute('visibility', 'visible');
      var rows = opts.series.map(function (s) {
        var v = best.values[s.key];
        dots[s.key].setAttribute('visibility', C.num(v) ? 'visible' : 'hidden');
        if (C.num(v)) { dots[s.key].setAttribute('cx', bx); dots[s.key].setAttribute('cy', ys(v)); }
        return '<div><i style="background:' + s.color + '"></i>' + C.esc(s.label) + ' <b>' + fmt(v) + '</b> ' + C.esc(opts.unit || '') + '</div>';
      }).join('');
      C.showTip(rows, ev.clientX, ev.clientY);
    }
    hit.addEventListener('mousemove', onMove);
    hit.addEventListener('mouseleave', function () {
      cross.setAttribute('visibility', 'hidden'); C.hideTip();
      for (var k in dots) dots[k].setAttribute('visibility', 'hidden');
    });
    return { update: draw };
  };

  /* Ring gauge 0..100, one value. */
  C.ring = function (value, color, size) {
    size = size || 44;
    var r = size / 2 - 4, c = 2 * Math.PI * r, v = C.num(value) ? Math.max(0, Math.min(100, value)) : 0;
    return '<svg class="ring" width="' + size + '" height="' + size + '" viewBox="0 0 ' + size + ' ' + size + '" aria-hidden="true">' +
      '<circle cx="' + size / 2 + '" cy="' + size / 2 + '" r="' + r + '" fill="none" stroke="var(--surface-2)" stroke-width="5"/>' +
      '<circle cx="' + size / 2 + '" cy="' + size / 2 + '" r="' + r + '" fill="none" stroke="' + color + '" stroke-width="5" stroke-linecap="round" ' +
      'stroke-dasharray="' + (c * v / 100).toFixed(1) + ' ' + c.toFixed(1) + '" transform="rotate(-90 ' + size / 2 + ' ' + size / 2 + ')"/></svg>';
  };

  /* How many times better Sonic LLM or llama.cpp is on metric m, when both engines have a value above 0 ('' otherwise). */
  function ratioHtml(m, vals) {
    if (!(vals.length === 2 && C.num(vals[0]) && C.num(vals[1]) && vals[0] > 0 && vals[1] > 0)) return '';
    var r = m.lower ? vals[1] / vals[0] : vals[0] / vals[1];
    return '<b>' + (r >= 1 ? 'Sonic LLM ' + r.toFixed(2) + '×' : 'llama.cpp ' + (1 / r).toFixed(2) + '×') + '</b>';
  }

  /* Paired horizontal bars for one run: metrics = [{key, label, unit, lower}], per = {sllm: {...}, llama: {...}}. */
  C.pairedBars = function (metrics, per) {
    var rts = Object.keys(per);
    return metrics.filter(function (m) { return rts.some(function (rt) { return C.num(per[rt][m.key]); }); }).map(function (m) {
      var vals = rts.map(function (rt) { return per[rt][m.key]; });
      var mx = Math.max.apply(null, vals.filter(C.num).concat([0])) || 1;
      var bars = rts.map(function (rt) {
        var v = per[rt][m.key];
        return '<div class="bar" title="' + C.esc(C.SERIES[rt].label + ': ' + C.fmt(v) + ' ' + (m.unit || '')) + '" style="width:' +
          (C.num(v) ? Math.max(1, v / mx * 100) : 0) + '%;background:' + C.SERIES[rt].color + '"></div>';
      }).join('');
      var ratio = ratioHtml(m, vals);
      var txt = rts.map(function (rt) { return C.fmt(per[rt][m.key]); }).join(' · ');
      return '<div class="mrow"><span class="name">' + C.esc(m.label) + (m.lower ? ' <span class="muted small">(lower is better)</span>' : '') + '</span>' +
        '<div class="bars">' + bars + '</div><span class="vals">' + (ratio || '') + '<br>' + C.esc(txt) + ' ' + C.esc(m.unit || '') + '</span></div>';
    }).join('');
  };

  /* Metric labels of the run summary. */
  C.METRICS = [
    { key: 'decode_tok_s', label: 'Decode speed', unit: 'tok/s' },
    { key: 'ttft_ms', label: 'Time to first token', unit: 'ms', lower: true },
    { key: 'itl_p95_ms', label: 'Inter-token p95', unit: 'ms', lower: true },
    { key: 'swarm_tok_s', label: 'Swarm output, whole swarm', unit: 'tok/s' },
    { key: 'write_tok_s', label: 'Writing only (first to last token)', unit: 'tok/s' },
    { key: 'read_only_tok_s', label: 'Reading only (evaluated prompt tokens, until the last first token)', unit: 'tok/s' },
    { key: 'read_wall_tok_s', label: 'Reading over the whole swarm (writing included)', unit: 'tok/s' },
    { key: 'swarm_wall_s', label: 'Swarm time', unit: 's', lower: true },
    { key: 'agg_tok_s', label: 'Whole run tok/s (reading included)', unit: 'tok/s' },
    { key: 'max_running', label: 'Most running at once', unit: '' },
    { key: 'per_agent_tok_s', label: 'Each agent', unit: 'tok/s' },
    { key: 'read_tokens', label: 'Tokens read by the agents', unit: '' },
    { key: 'found_pct', label: 'Questions answered per document', unit: '%' },
    { key: 'ttft_ms_p50', label: 'Agent TTFT p50', unit: 'ms', lower: true },
    { key: 'cache_hit_pct', label: 'Prompt reused from cache', unit: '%' },
    { key: 'labeled', label: 'Files with a label', unit: '' },
    { key: 'json_ok', label: 'Replies that are valid JSON', unit: '' },
    { key: 'kept_pct', label: 'Kept of what was read', unit: '%', lower: true },
    { key: 'final_ttft_ms', label: 'Merge / manager TTFT', unit: 'ms', lower: true },
    { key: 'ttft_alone_ms', label: 'Chat TTFT alone', unit: 'ms', lower: true },
    { key: 'ttft_loaded_ms', label: 'Chat TTFT under load', unit: 'ms', lower: true },
    { key: 'decode_alone_tok_s', label: 'Chat speed alone', unit: 'tok/s' },
    { key: 'decode_loaded_tok_s', label: 'Chat speed under load', unit: 'tok/s' },
    { key: 'chat_slowdown', label: 'Chat slowdown under load', unit: '×', lower: true },
    { key: 'prefill_tok_s', label: 'Prefill (reading)', unit: 'tok/s' },
    { key: 'cached_ttft_ms', label: 'TTFT, asked again', unit: 'ms', lower: true },
    { key: 'prompt_tokens', label: 'Prompt tokens', unit: '' },
    { key: 'tokens_generated', label: 'Tokens generated', unit: '' },
    { key: 'wall_s', label: 'Total time', unit: 's', lower: true },
    { key: 'peak_mem_mb', label: 'Footprint peak, model file not counted', unit: 'MB', lower: true },
    { key: 'session_peak_mem_mb', label: 'Footprint peak, whole session', unit: 'MB', lower: true },
    { key: 'kv_reserved_mb', label: 'KV cache reserved', unit: 'MB', lower: true },
    { key: 'kv_used_mb', label: 'KV cache in use (peak)', unit: 'MB', lower: true },
    { key: 'weights_mb', label: 'Model file (mapped, not in the footprint)', unit: 'MB', lower: true },
    { key: 'spec_accept_rate_total', label: 'Speculation acceptance', unit: '' }
  ];
})();
