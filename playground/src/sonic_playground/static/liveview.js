/* What the live page (live.js) shows, built from plain data: the numbers under an answer, the agent grid, the tiles, the
   loaded session's limits, a run's answer boxes, the texts of the answer levels, and the one function that changes the
   page's run and session state (reduceRun). No state of its own, no network; loaded before live.js, which reads it as
   window.LiveView. */
(function () {
  'use strict';
  var C = window.Charts, esc = C.esc;
  var V = window.LiveView = {};
  function many(n, one) { return C.fmt(n) + ' ' + one + (n === 1 ? '' : 's'); }
  V.many = many;

  // Options › what is loaded now: each engine's context, agents and context memory, the trained context, the exact status
  V.limitsHtml = function (s) {
    var e = s.engines;
    var gib = function (cells) { return C.num(s.kv_bytes_per_token) ? (cells * s.kv_bytes_per_token / 1073741824).toFixed(1) + ' GiB' : C.fmt(cells) + ' cells'; };
    return '<p><b>Loaded now:</b> ' + esc(s.model) + '</p>' + Object.keys(e).map(function (rt) {
      return '<p>' + C.SERIES[rt].label + ': ' + (rt === 'sllm' ? C.fmt(e[rt].ctx) + ' tokens per agent, room for ' + many(e[rt].max_parallel, 'agent') : 'shared context, ' + e[rt].max_parallel + ' slots') + ', context memory ' + gib(e[rt].cells) + '</p>';
    }).join('') + (C.num(s.train_ctx) ? '<p>Model trained to ' + C.fmt(s.train_ctx) + ' tokens.</p>' : '') +
      (s.exact ? '<p class="' + (s.exact.level === 'exact' ? '' : 'warnline') + '">' + esc(s.exact.text) + '</p>' : '');
  };
  // A run's message: its scenario and one answer box per engine (rts), each with its state, text, agents and numbers.
  V.runHtml = function (name, rts) {
    return '<div class="scen"><span>' + esc(name) + (rts.length > 1 ? ' · Sonic LLM first, then llama.cpp' : '') + '</span><span class="elapsed"></span></div>' +
      '<div class="answers ' + (rts.length > 1 ? 'two' : '') + '">' + rts.map(function (rt) {
        return '<div class="answer" data-rt="' + rt + '"><div class="who"><i style="background:' + C.SERIES[rt].color + '"></i>' + C.SERIES[rt].label +
          '<span class="state">waiting</span></div><div class="text"></div><div class="agentlist"></div><div class="stats"></div></div>';
      }).join('') + '</div>';
  };

  // The task menu: a group of tasks under each heading, the picked one (scen) checked.
  V.taskMenuHtml = function (groups, scen) {
    return groups.map(function (g) {
      return '<div class="mgroup"><h3>' + esc(g.name) + '</h3>' + g.items.map(function (sc) {
        return '<button class="mitem" role="menuitemradio" aria-checked="' + (sc.id === scen) + '" data-s="' + sc.id + '" title="' + esc(sc.what) + '"><b>' + esc(sc.name) + '</b><span>' + esc(sc.what) + '</span></button>';
      }).join('') + '</div>';
    }).join('');
  };

  // -- what pressing Run will really do, in one line -----------------------------------------------------------------
  // k: n files picked, a agents, perFile (one agent per file), first (the first picked file's name), files (in the
  // library), max (the most agents), par (the agents-at-once limit, or null), maxThink (gpt-oss thinking max), level
  // (the answer level; 'off' for gpt-oss, which has its thinking instead). Gives {t: the text, warn}.
  V.planText = function (id, k) {
    var p = k.perFile ? perFilePlan(id, k) : { t: PLAN[id] ? PLAN[id](k.n, k.a, k) : k.n ? 'Your ' + many(k.n, 'file') + ' go with this message.' : '', warn: false };
    if (k.maxThink && k.perFile) { p.t += ' Thinking max: replies have no length limit.'; p.warn = true; }
    if (k.level !== 'off') p.t += V.LEVEL_FREE.indexOf(id) >= 0 ? ' Answer levels do not apply to this task.' : ' Answer level ' + k.level + '.';
    return p;
  };
  // The tasks with one agent per file: the picked files, else the biggest files.
  function perFilePlan(id, k) {
    var n = k.n, a = k.a;
    if (n) return { t: 'Will run ' + many(a, 'agent') + ', one per file.' + (n > k.max ? ' Only the first ' + k.max + ' files.' : ''), warn: false };
    var m = Math.min(a, k.files);
    return { t: 'No files picked: will run ' + many(m, 'agent') + ' on the ' + (m === 1 ? 'biggest file' : m + ' biggest files') + '.', warn: m > 8 };
  }
  // The other tasks' plans from the files picked (n) and the agents (a); a task not here only says which files go along.
  var PLAN = {
    shared: function (n, a, k) { return 'Will run ' + many(a, 'agent') + ' on ' + (n ? k.first : 'the biggest file') + ', each with its own question.'; },
    load: function (n, a) { return 'The chat runs alone, then again beside ' + many(a, 'agent') + '.'; },
    exact: function (n, a) { return a > 1 ? 'The question runs alone, then beside ' + many(a - 1, 'other agent') + '.' : 'The question runs alone, then alone again.'; },
    burst: function (n, a) { return many(a, 'agent') + (a === 1 ? ' writes' : ' write') + ' 256 tokens.'; },
    turns: function (n, a, k) { return many(a, 'agent') + (a === 1 ? ' takes' : ' take') + ' turns, ' + Math.min(k.par || 8, a) + ' writing at once.'; },
    long: function (n) { return (n ? many(n, 'picked file') : 'The library') + ' joined into one long prompt, asked twice.'; }
  };

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

  // -- answer level / thinking ---------------------------------------------------------------------------------------
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

  // The level buttons: gpt-oss's thinking (legacy, think: its index) or the answer levels (level).
  V.thinkHtml = function (legacy, think, level) {
    if (legacy) return V.THINK.map(function (t, i) { return '<button role="radio" aria-checked="' + (i === think) + '" data-i="' + i + '">' + t + '</button>'; }).join('');
    return V.LEVELS.map(function (t) { return '<button role="radio" aria-checked="' + (t === level) + '" data-l="' + t + '" title="' + esc(V.LEVEL_WHAT[t]) + '">' + t + '</button>'; }).join('');
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

  // The first few show under an answer, the rest open on request.
  var STAT_KEYS = [['labeled', 'with a label', 'files'], ['json_ok', 'valid JSON', 'files'], ['kept_pct', 'kept', '% of what was read'], ['found_pct', 'answered', '%'],
    ['wall_s', 'total', 's'], ['swarm_wall_s', 'swarm', 's'], ['ttft_ms', 'first word', 'ms'], ['ttft_ms_p50', 'first word, median', 'ms'], ['decode_tok_s', '', 'tok/s'], ['write_tok_s', 'writing', 'tok/s'], ['peak_mem_mb', 'memory peak', 'MB'],
    ['itl_p95_ms', 'token gap p95', 'ms'], ['max_running', 'most running at once', ''], ['agg_tok_s', 'whole run', 'tok/s'], ['read_only_tok_s', 'reading only', 'tok/s'], ['per_agent_tok_s', 'each', 'tok/s'], ['questions', '', 'questions'], ['read_tokens', 'read', 'tokens'], ['swarm_tok_s', '', 'tok/s swarm'], ['cache_hit_pct', 'cache', '%'], ['reused_pct', 'read from memory', '%'], ['laps', '', 'laps'], ['stories_done', 'stories done', ''], ['retries', '', 'retries'], ['ttft_loaded_ms', 'first word, loaded', 'ms'],
    ['kv_reserved_mb', 'KV reserved', 'MB'], ['kv_used_mb', 'KV in use (peak)', 'MB'], ['weights_mb', 'model file (not in memory peak)', 'MB'],
    ['chat_slowdown', 'slowdown', '×'], ['prefill_tok_s', 'reading', 'tok/s'], ['cached_ttft_ms', 'again', 'ms'], 
    ['rounds', '', 'rounds'], ['at_once', 'write at once', ''], ['suspends', 'parked', 'times'], ['restores', 'woken', 'times'], ['restore_ms_p50', 'wake p50', 'ms'], ['restore_ms_p95', 'wake p95', 'ms'], ['ssd_mb', 'SSD peak', 'MB'], ['notes_tokens', 'all notes', 'tokens']];
  var STATS_SHOWN = 6;
  // "59 of 256 asked (one per document)": agents that really ran next to the number asked for
  function agentsText(m) {
    if (!C.num(m.agents)) return '';
    return m.agents + (C.num(m.agents_asked) && m.agents_asked !== m.agents ? ' of ' + m.agents_asked + ' asked' : ' agents') + (m.agents_note ? ' (' + m.agents_note + ')' : '');
  }
  V.statsHtml = function (m) {
    var out = C.num(m.agents) ? ['<span><b>' + esc(agentsText(m)) + '</b></span>'] : [];
    if (m.cut_off) out.push('<span class="warn"><b>' + C.fmt(m.cut_off) + '</b> hit the length limit</span>');
    if (m.exact === true) out.push('<span class="verdict good">✓ bit-identical</span>');
    if (m.exact === false) out.push('<span class="verdict bad">✗ tokens differ at ' + m.first_diff + '</span>');
    if (m.same_as_ram === true) out.push('<span class="verdict good">✓ same as never leaving RAM</span>');
    if (m.same_as_ram === false) out.push('<span class="verdict bad">✗ differs from the RAM run</span>');
    if (m.same_answer_under_load === true) out.push('<span>same answer under load ✓</span>');
    out = out.concat(STAT_KEYS.filter(function (k) { return C.num(m[k[0]]); }).map(function (k) {
      return '<span>' + (k[1] ? esc(k[1]) + ' ' : '') + '<b>' + C.fmt(m[k[0]]) + '</b> ' + esc(k[2]) + '</span>';
    }));
    var more = out.slice(STATS_SHOWN);
    return out.slice(0, STATS_SHOWN).join('') + (more.length ? '<details class="more"><summary>' + more.length + ' more numbers</summary><div>' + more.join('') + '</div></details>' : '');
  };

  // -- the agent grid ------------------------------------------------------------------------------------------------
  // One engine's row: a cell per agent (in order), how many are done, and the bar.
  V.swarmRowHtml = function (rt, agents, done) {
    return '<div class="swarmrow"><span class="lbl"><span class="dot" style="background:' + C.SERIES[rt].color + '"></span>' + C.SERIES[rt].label + '</span><div class="cells">' +
      agents.map(function (a) {
        return '<span class="cell ' + a.state + '" title="' + esc(cellTitle(a)) + '"' + cellStyle(a, C.SERIES[rt].color) + '></span>';
      }).join('') + '</div><span class="count">' + done + ' / ' + agents.length + '</span></div><div class="barbox"><i style="width:' + (agents.length ? 100 * done / agents.length : 0) + '%"></i></div>';
  };
  // a run that did not finish: its agents not done are stopped (swarm: {rt: {i: agent event}})
  V.stopCells = function (swarm) {
    Object.keys(swarm).forEach(function (rt) { Object.keys(swarm[rt]).forEach(function (i) { if (swarm[rt][i].state !== 'done') swarm[rt][i].state = 'stopped'; }); });
  };
  // one agent's cell in the grid: its numbers when done, else its state
  function cellTitle(a) {
    if (a.state !== 'done') return a.label + ' · ' + a.state + (C.num(a.tokens) ? ' · ' + many(a.tokens, 'token') : '');
    return a.label + ' · first word ' + C.fmt(a.ttft_ms) + ' ms · ' + C.fmt(a.tok_s) + ' tok/s' + (C.num(a.cached) ? ' · cached ' + a.cached : '') + (C.num(a.found) ? ' · answered ' + a.found + '/' + a.of : '');
  }
  // writing and done cells in the engine's colour; a team agent that found nothing hollow, one that found some faded
  function cellStyle(a, color) {
    if (a.state === 'writing') return ' style="background:' + color + '"';
    if (a.state !== 'done') return '';
    if (C.num(a.found) && a.found === 0) return ' style="background:var(--surface-1);border-color:' + color + '"';
    return ' style="background:' + color + (C.num(a.found) && a.found < a.of ? ';opacity:.55' : '') + '"';
  }

  // The model picker: the models found (ms), each with its size.
  V.modelOptionsHtml = function (ms) {
    return ms.map(function (m) {
      return '<option value="' + esc(m.id) + '">' + esc(m.id) + '  ·  ' + (m.size_bytes ? (m.size_bytes / 1e9).toFixed(1) + ' GB' : '') + '</option>';
    }).join('');
  };
  // The files going with the next message: a chip each with its ✕, or one ✕ for all when there are more than two.
  V.attachedHtml = function (picked) {
    var name = function (p) { return p.split('/').pop(); };
    return picked.length > 2 ? '<button class="link" data-x="*" aria-label="Clear all files" title="Clear all files">✕</button>' : picked.map(function (p) {
      return '<span class="chip" title="' + esc(p) + '">' + esc(name(p)) + ' <button data-x="' + esc(p) + '" aria-label="Remove ' + esc(name(p)) + '">✕</button></span>';
    }).join('');
  };
  // -- the file library ----------------------------------------------------------------------------------------------
  // The library's files matching q (lower case, '' for all) by folder.
  V.fileGroups = function (files, q) {
    var groups = {};
    files.filter(function (f) { return !q || f.path.toLowerCase().indexOf(q) >= 0; }).forEach(function (f) {
      var g = f.path.split('/').slice(0, -1).join(' / ') || '(root)'; (groups[g] = groups[g] || []).push(f);
    });
    return groups;
  };
  // One folder of the library: its box (all, some or none of its files picked), its name and size, and when it is open
  // a row per file.
  V.fileGroupHtml = function (g, fs, picked, open) {
    var on = fs.filter(function (f) { return picked.indexOf(f.path) >= 0; }).length;
    var tok = fs.reduce(function (a, f) { return a + f.est_tokens; }, 0);
    return '<div class="fgroup"><input type="checkbox" aria-label="All of ' + esc(g) + '" data-g="' + esc(g) + '"' + (on === fs.length ? ' checked' : '') + (on && on < fs.length ? ' data-partial="1"' : '') + '>' +
      '<button class="fopen" data-open="' + esc(g) + '" aria-expanded="' + !!open + '">' + (open ? '▾ ' : '▸ ') + esc(g) + '</button><span class="n">' + (on ? on + ' of ' : '') + many(fs.length, 'file') + ' · ~' + C.fmt(tok) + ' tok</span></div>' +
      (open ? fs.map(function (f) {
        return '<label class="frow"><input type="checkbox" data-p="' + esc(f.path) + '"' + (picked.indexOf(f.path) >= 0 ? ' checked' : '') + '>' +
          '<span class="p">' + esc(f.path.split('/').pop()) + '</span><span class="n">~' + C.fmt(f.est_tokens) + ' tok</span></label>';
      }).join('') : '');
  };

  // The memory chart's note: [short, its tooltip]: the KV cache reserved and (after a run, per) the most of it in use.
  V.memNote = function (s, per) {
    var parts = ['process footprint, model file not counted'], short = 'MB';
    if (s && C.num(s.kv_bytes_per_token) && s.engines.sllm) { short = 'KV ' + (s.engines.sllm.cells * s.kv_bytes_per_token / 1073741824).toFixed(1) + ' GiB'; parts.push(short + ' reserved'); }
    var u = per && per.sllm && per.sllm.kv_used_mb; if (C.num(u)) parts.push('KV in use, last run ' + (u / 1024).toFixed(2) + ' GiB');
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

  // -- tiles ---------------------------------------------------------------------------------------------------------
  // One tile per engine (rts) from the last tick t (speed, reading, requests, memory), and the GPU tile.
  V.tilesHtml = function (rts, t) {
    var e = (t && t.engines) || {};
    var tiles = rts.map(function (rt) {
      var x = e[rt] || {};
      return '<div class="tile"><div class="k"><i style="background:' + C.SERIES[rt].color + '"></i>' + C.SERIES[rt].label + ' now</div>' +
        '<div class="v">' + C.fmt(x.tok_s, 0) + '<small>tok/s written</small></div><div class="sub">reading ' + C.fmt(x.read_tok_s, 0) + ' tok/s · ' + (x.active || 0) + ' requests · ' +
        (C.num(x.mem_mb) ? (x.mem_mb / 1024).toFixed(2) + ' GB' : '–') + '</div></div>';
    });
    tiles.push('<div class="tile"><div class="k">GPU busy</div>' + C.ring(t && t.gpu_util, 'var(--series-1)') +
      '<div class="v">' + C.fmt(t && t.gpu_util, 0) + '<small>%</small></div></div>');
    return tiles.join('');
  };
})();
