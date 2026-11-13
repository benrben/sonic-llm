/* What the live page shows, built from plain data: the loaded engine's details, a run's answer boxes, the one-line plan of
   a run, the answer levels, the agent grid, the document library, chart notes. No state of its own, no network. Extends
   window.LiveView (state.js, numbers.js come first). */
(function () {
  'use strict';
  var C = window.Charts, esc = C.esc, V = window.LiveView, many = V.many;

  // Engine details: what is loaded now, each engine's room for agents, the trained context, the exact status
  V.limitsHtml = function (s) {
    var e = s.engines;
    var gib = function (cells) { return C.num(s.kv_bytes_per_token) ? (cells * s.kv_bytes_per_token / 1073741824).toFixed(1) + ' GiB' : C.fmt(cells) + ' cells'; };
    return '<p><b>Loaded now:</b> ' + esc(s.model) + '</p>' + Object.keys(e).map(function (rt) {
      return '<p>' + C.SERIES[rt].label + ': ' + (rt === 'sllm' ? C.fmt(e[rt].ctx) + ' tokens per agent, room for ' + many(e[rt].max_parallel, 'agent') : 'one shared room, ' + e[rt].max_parallel + ' slots') +
        ', memory for agents ' + gib(e[rt].cells) + '</p>';
    }).join('') + (C.num(s.train_ctx) ? '<p>The model was trained on up to ' + C.fmt(s.train_ctx) + ' tokens.</p>' : '') + (s.exact ? '<p>' + esc(s.exact.text) + '</p>' : '');
  };

  // A run's answers: one box per engine (rts), each with its state, text, agents and notes.
  V.runHtml = function (rts) {
    return '<div class="answers ' + (rts.length > 1 ? 'two' : '') + '">' + rts.map(function (rt) {
      return '<div class="answer" data-rt="' + rt + '"><div class="who"><i class="swatch ' + rt + '"></i>' + C.SERIES[rt].label +
        '<span class="state">waiting</span></div><div class="thought"></div><div class="text"></div><div class="agentlist"></div><div class="stats"></div></div>';
    }).join('') + '</div>';
  };

  // -- what pressing Run will really do, in one line ------------------------------------------------------------------
  // k: n documents picked, a agents, perFile (one agent per document), first (the first picked document's name), files (in the
  // library), max (the most agents), par (the agents-at-once limit, or null), maxThink (gpt-oss thinking max), level (the
  // answer level; 'off' for gpt-oss, which has its thinking instead). Gives {t: the text, warn}.
  V.planText = function (id, k) {
    var p = k.perFile ? perFilePlan(id, k) : { t: PLAN[id] ? PLAN[id](k.n, k.a, k) : k.n ? 'Your ' + many(k.n, 'document') + ' go with this message.' : '', warn: false };
    if (k.maxThink && k.perFile) { p.t += ' Thinking max: replies have no length limit.'; p.warn = true; }
    if (k.level !== 'off') p.t += V.LEVEL_FREE.indexOf(id) >= 0 ? ' Answer levels do not apply to this test.' : ' Answer level: ' + k.level + '.';
    return p;
  };
  function perFilePlan(id, k) {
    var n = k.n, a = k.a;
    if (n) return { t: 'Will run ' + many(a, 'agent') + ', one per document.' + (n > k.max ? ' Only the first ' + k.max + ' documents.' : ''), warn: false };
    var m = Math.min(a, k.files);
    return { t: 'No documents picked: will run ' + many(m, 'agent') + ' on the ' + (m === 1 ? 'biggest document' : m + ' biggest documents') + '.', warn: m > 8 };
  }
  var PLAN = {
    shared: function (n, a, k) { return 'Will run ' + many(a, 'agent') + ' on ' + (n ? k.first : 'the biggest document') + ', each with its own question.'; },
    load: function (n, a) { return 'The chat runs alone, then again beside ' + many(a, 'agent') + '.'; },
    exact: function (n, a) { return a > 1 ? 'The question runs alone, then beside ' + many(a - 1, 'other agent') + '.' : 'The question runs alone, then alone again.'; },
    burst: function (n, a) { return many(a, 'agent') + (a === 1 ? ' writes' : ' write') + ' 256 tokens.'; },
    turns: function (n, a, k) { return many(a, 'agent') + (a === 1 ? ' takes' : ' take') + ' turns, ' + Math.min(k.par || 8, a) + ' writing at once.'; },
    long: function (n) { return (n ? many(n, 'picked document') : 'All the documents') + ' joined into one long text, asked twice.'; }
  };

  // -- answer level / thinking ----------------------------------------------------------------------------------------
  V.LEVELS = ['off', 'sonic', 'fast', 'normal', 'deep'];
  V.LEVEL_FREE = ['exact', 'burst', 'turns', 'long'];  // tests of the raw engine: they always run plain
  V.LEVEL_WHAT = {
    off: 'The model answers as it is.',
    sonic: 'Instant: one short line.',
    fast: 'A regular chat answer, written right away at full speed.',
    normal: 'Smarter: answers twice, checks the two agree, and works step by step when they do not.',
    deep: 'Long work (code, research): being built. For now the same as normal.'
  };
  V.LEVEL_NOTE = ' The recipes were measured on Qwen3.6-35B-A3B.';
  V.NO_TEAM = ' The levels were measured on the Sonic LLM engine itself; on this engine they run the same way, unmeasured.';
  V.THINK = ['off', 'low', 'medium', 'high', 'max'];
  V.THINK_WHAT = ['The model answers straight away.', 'Short thinking first (gpt-oss: low; Qwen3 / Gemma 4: thinking on).',
    'Medium thinking (gpt-oss: medium; Qwen3 / Gemma 4: thinking on).', 'Long thinking (gpt-oss: high; Qwen3 / Gemma 4: thinking on).',
    'The most thinking the model knows, with no limit on the reply.'];
  function cap(t) { return t.charAt(0).toUpperCase() + t.slice(1); }
  function radio(name, value, on, text, title) {
    return '<label' + (title ? ' title="' + esc(title) + '"' : '') + '><input type="radio" name="' + name + '" value="' + value + '"' + (on ? ' checked' : '') + '><span>' + cap(text) + '</span></label>';
  }
  // The level buttons: gpt-oss's thinking (legacy, think: its index) or the answer levels (level).
  V.thinkHtml = function (legacy, think, level) {
    if (legacy) return V.THINK.map(function (t, i) { return radio('lvl', i, i === think, t); }).join('');
    return V.LEVELS.map(function (t) { return radio('lvl', t, t === level, t, V.LEVEL_WHAT[t]); }).join('');
  };

  // Same rules as chat_template.split_thinking (chat_template.py): Qwen3 </think>, harmony's switch to the final channel.
  var THOUGHT_END = /<\/think>|(?:<\|end\|>)?(?:<\|start\|>)?assistant(?:<\|channel\|>)?final(?:<\|message\|>)?/;
  var THOUGHT_START = /^\s*(?:<think>|<\|channel\|>analysis<\|message\|>|analysis(?=\S))/;
  V.splitThinking = function (raw) {
    var started = THOUGHT_START.test(raw), m = raw.match(THOUGHT_END);
    if (m && (started || m[0] === '</think>')) return [raw.slice(0, m.index).replace(THOUGHT_START, '').trim(), raw.slice(m.index + m[0].length).replace(/^\s+/, '')];
    if (started) return [raw.replace(THOUGHT_START, ''), ''];  // still thinking
    return ['', raw];
  };

  // -- the agent grid -------------------------------------------------------------------------------------------------
  // One engine's row: a cell per agent (in order), how many are done, and the bar.
  V.swarmRowHtml = function (rt, agents, done) {
    return '<div class="' + rt + '"><div class="swarmrow"><span class="lbl"><i class="swatch ' + rt + '"></i>' + C.SERIES[rt].label + '</span><div class="cells">' +
      agents.map(function (a) { return '<span class="cell ' + a.state + cellKind(a) + '" title="' + esc(cellTitle(a)) + '"></span>'; }).join('') +
      '</div><span class="count">' + done + ' of ' + agents.length + ' done</span></div><div class="barbox ' + rt + '"><i style="width:' + (agents.length ? 100 * done / agents.length : 0) + '%"></i></div></div>';
  };
  // a run that did not finish: its agents not done are stopped (swarm: {rt: {i: agent event}})
  V.stopCells = function (swarm) {
    Object.keys(swarm).forEach(function (rt) { Object.keys(swarm[rt]).forEach(function (i) { if (swarm[rt][i].state !== 'done') swarm[rt][i].state = 'stopped'; }); });
  };
  function cellTitle(a) {
    if (a.state !== 'done') return a.label + ' · ' + a.state + (C.num(a.tokens) ? ' · ' + many(a.tokens, 'token') : '');
    return a.label + ' · first word ' + V.valueText(a.ttft_ms, 'ms') + ' · ' + C.fmt(a.tok_s) + ' tokens per second' + (C.num(a.cached) ? ' · ' + a.cached + ' tokens from memory' : '') + (C.num(a.found) ? ' · answered ' + a.found + ' of ' + a.of : '');
  }
  // a research agent that found nothing is hollow, one that found some of its questions is faded
  function cellKind(a) { return a.state === 'done' && C.num(a.found) ? (a.found === 0 ? ' none' : a.found < a.of ? ' part' : '') : ''; }

  // The model picker: the models found (ms), each with its size.
  V.modelOptionsHtml = function (ms) {
    return ms.map(function (m) { return '<option value="' + esc(m.id) + '">' + esc(m.id) + (m.size_bytes ? '  ·  ' + (m.size_bytes / 1e9).toFixed(1) + ' GB' : '') + '</option>'; }).join('');
  };
  // The documents going with the next message: a chip each with its remove button, or one clear button when there are many.
  function name(p) { return p.split('/').pop(); }
  V.attachedHtml = function (picked) {
    var x = '<svg class="ic s" aria-hidden="true"><use href="#i-x"/></svg>';
    return picked.length > 2 ? '<button class="link" data-x="*" type="button">Clear selection</button>' : picked.map(function (p) {
      return '<span class="chip" title="' + esc(p) + '"><span>' + esc(name(p)) + '</span><button data-x="' + esc(p) + '" type="button" aria-label="Remove ' + esc(name(p)) + '">' + x + '</button></span>';
    }).join('');
  };

  // -- the document library -------------------------------------------------------------------------------------------
  // The library's documents matching q (lower case, '' for all) by folder.
  V.fileGroups = function (files, q) {
    var groups = {};
    files.filter(function (f) { return !q || f.path.toLowerCase().indexOf(q) >= 0; }).forEach(function (f) {
      var g = f.path.split('/').slice(0, -1).join(' / ') || '(root)'; (groups[g] = groups[g] || []).push(f);
    });
    return groups;
  };
  V.FOLDER = { samples: 'Samples that come with the playground', uploads: 'Your own documents' };
  // One folder of the library: its box (all, some or none picked), its name and size, and when it is open a row per document.
  // Your own documents (uploads) have a remove button; `sure` is the path asked "really remove?" right now.
  V.fileGroupHtml = function (g, fs, picked, open, sure) {
    var on = fs.filter(function (f) { return picked.indexOf(f.path) >= 0; }).length, tok = fs.reduce(function (a, f) { return a + f.est_tokens; }, 0);
    return '<div class="fgroup"><input type="checkbox" aria-label="All of ' + esc(V.FOLDER[g] || g) + '" data-g="' + esc(g) + '"' + (on === fs.length ? ' checked' : '') + (on && on < fs.length ? ' data-partial="1"' : '') + '>' +
      '<button class="fopen" data-open="' + esc(g) + '" aria-expanded="' + !!open + '" type="button"><svg class="ic s" aria-hidden="true"><use href="#i-chev"/></svg>' + esc(V.FOLDER[g] || g) + '</button>' +
      '<span class="n">' + (on ? on + ' of ' : '') + many(fs.length, 'document') + ' · about ' + C.fmt(tok) + ' tokens</span></div>' +
      (open ? fs.map(function (f, i) { return fileRowHtml(f, picked.indexOf(f.path) >= 0, g === 'uploads', sure === f.path, 'f-' + g.replace(/\W/g, '') + i); }).join('') : '');
  };
  function fileRowHtml(f, on, mine, sure, id) {
    var act = !mine ? '<span></span>' : sure ? '<span class="sure"><button class="btn danger" data-del="' + esc(f.path) + '" type="button">Remove</button><button class="link" data-keep="1" type="button">Keep</button></span>' :
      '<button class="btn icon quiet gone" data-ask="' + esc(f.path) + '" type="button" aria-label="Remove ' + esc(name(f.path)) + '"><svg class="ic s" aria-hidden="true"><use href="#i-trash"/></svg></button>';
    return '<div class="frow"><input type="checkbox" data-p="' + esc(f.path) + '" id="' + id + '"' + (on ? ' checked' : '') + '><label class="p" for="' + id + '">' + esc(name(f.path)) +
      '</label><span class="n">about ' + C.fmt(f.est_tokens) + ' tokens</span>' + act + '</div>';
  }

  // The memory chart's note: [short, its tooltip]: the memory set aside for agents and (after a run, per) the most of it used.
  V.memNote = function (s, per) {
    var parts = ['memory of the engine, model file not counted'], short = 'MB';
    if (s && C.num(s.kv_bytes_per_token) && s.engines.sllm) { short = (s.engines.sllm.cells * s.kv_bytes_per_token / 1073741824).toFixed(1) + ' GiB for agents'; parts.push(short + ' set aside'); }
    var u = per && per.sllm && per.sllm.kv_used_mb; if (C.num(u)) parts.push('agents used ' + (u / 1024).toFixed(2) + ' GiB in the last run');
    return [short, parts.join(' · ')];
  };
  // a chart's points from the ticks kept: each engine's number `key`
  V.tickPoints = function (ticks, key) {
    return ticks.map(function (p) { var v = {}; for (var rt in p.ev.engines) v[rt] = p.ev.engines[rt][key]; return { t: p.t, values: v }; });
  };
  // "done 3 · writing 2": how many agents are in each state
  V.countText = function (states) {
    var counts = {};
    states.forEach(function (x) { counts[x] = (counts[x] || 0) + 1; });
    return Object.keys(counts).sort().map(function (k) { return k + ' ' + counts[k]; }).join(' · ');
  };
})();
