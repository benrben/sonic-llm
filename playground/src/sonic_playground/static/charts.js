/* Small SVG charts for the live playground: a line chart with a crosshair tooltip, and the two engines' names and colours.
   Colors come from CSS tokens (blue Sonic LLM, orange llama.cpp); text always wears text tokens. */
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  var C = window.Charts = {};
  C.SERIES = { sllm: { label: 'Sonic LLM', color: 'var(--blue)', soft: 'var(--blue-soft)' },
               llama: { label: 'llama.cpp', color: 'var(--orange)', soft: 'var(--orange-soft)' } };
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
    var W = Math.max(240, host.clientWidth || 600), H = host.clientHeight || 160, pad = { l: 56, r: 12, t: 8, b: 20 };
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': opts.label || '' }, host);
    var grid = el('g', {}, svg), lines = el('g', {}, svg), hover = el('g', {}, svg);
    var cross = el('line', { y1: pad.t, y2: H - pad.b, stroke: 'var(--axis)', 'stroke-width': 1, visibility: 'hidden' }, hover);
    var dots = {};
    var paths = {}, areas = {};
    opts.series.forEach(function (s) {
      areas[s.key] = el('path', { fill: s.soft || 'none', stroke: 'none' }, lines);
      paths[s.key] = el('path', { fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'vector-effect': 'non-scaling-stroke' }, lines);
      dots[s.key] = el('circle', { r: 4, fill: s.color, stroke: 'var(--surface)', 'stroke-width': 2, visibility: 'hidden' }, hover);
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
        var tx = el('text', { x: pad.l - 6, y: y + 4, 'text-anchor': 'end', 'font-size': 12, fill: 'var(--muted)' }, grid);
        tx.textContent = fmt(v);
      }
      var xl = opts.xLabels || ['-' + opts.windowS + ' s', '-' + opts.windowS / 2 + ' s', 'now'];
      [opts.windowS, opts.windowS / 2, 0].forEach(function (s, i) {
        if (!xl[i]) return;
        var tx = el('text', { x: xs(t1 - s), y: H - 5, 'text-anchor': i === 0 ? 'start' : i === 2 ? 'end' : 'middle', 'font-size': 12, fill: 'var(--muted)' }, grid);
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
})();
