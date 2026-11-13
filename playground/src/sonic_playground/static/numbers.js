/* The numbers of a run, in plain words: the headline tiles (one per question a newcomer asks), how much better one engine
   is than the other, the table of every number, and the short notes under an answer. Builds HTML from data, nothing else. */
(function () {
  'use strict';
  var C = window.Charts, esc = C.esc, V = window.LiveView;
  var SAME = 1.05;  // within 5% is "about the same"

  // -- how one number reads -------------------------------------------------------------------------------------------
  // [number, unit] for value v of a unit: tok/s, ms, s, MB, %, x, or none
  function parts(v, unit, exact) {
    if (!C.num(v)) return ['–', ''];
    if (!exact && unit === 'tok/s' && v >= 10000) return [(v / 1000).toFixed(1) + 'k', unit];  // a tile is small: 18.9k, the table has all digits
    if (unit === 'ms') return v >= 1000 ? [(v / 1000).toFixed(1), 's'] : [String(Math.round(v)), 'ms'];
    if (unit === 's') return [v >= 100 ? String(Math.round(v)) : v.toFixed(1), 's'];
    if (unit === 'MB') return v >= 1024 ? [(v / 1024).toFixed(1), 'GB'] : [String(Math.round(v)), 'MB'];
    return [C.fmt(v), unit || ''];
  }
  V.valueText = function (v, unit) { var p = parts(v, unit, true); return p[1] ? p[0] + ' ' + p[1] : p[0]; };
  function times(r) { return (r >= 10 ? Math.round(r) : r.toFixed(1)) + 'x'; }

  // How Sonic LLM compares with llama.cpp on one number: {text, tone} with tone win, lose or same; null when either is missing.
  V.compare = function (sllm, llama, lower) {
    if (!(C.num(sllm) && C.num(llama) && sllm > 0 && llama > 0)) return null;
    var r = lower ? llama / sllm : sllm / llama;
    if (r >= SAME) return { text: times(r) + ' faster', tone: 'win', times: times(r) };
    if (r <= 1 / SAME) return { text: times(1 / r) + ' slower', tone: 'lose', times: times(1 / r) };
    return { text: 'about the same', tone: 'same' };
  };
  // One sentence for a finished comparison, from the total time of both engines ('' while either is missing)
  V.verdictText = function (final) {
    var s = final.sllm && final.sllm.wall_s, l = final.llama && final.llama.wall_s, c = V.compare(s, l, true);
    if (!c) return '';
    return c.tone === 'win' ? 'Sonic LLM finished ' + c.times + ' faster than llama.cpp.' : c.tone === 'lose' ? 'llama.cpp finished ' + c.times + ' faster than Sonic LLM.' : 'Both engines finished in about the same time.';
  };

  // -- the headline slots ---------------------------------------------------------------------------------------------
  // Each slot is one tile: the keys of a run's numbers it can come from (the first one a run has), and the live reading that
  // stands in until the run's own number arrives (a tick's tok_s or read_tok_s, or the time since the engine started).
  var SLOTS = {
    total: { label: 'Total time', unit: 's', keys: ['wall_s'], lower: true, live: 'elapsed' },
    first: { label: 'First word', unit: 'ms', keys: ['ttft_ms', 'ttft_ms_p50'], lower: true },
    firstLoaded: { label: 'First word under load', unit: 'ms', keys: ['ttft_loaded_ms'], lower: true },
    read: { label: 'Read speed', unit: 'tok/s', keys: ['read_only_tok_s', 'prefill_tok_s', 'read_wall_tok_s'], live: 'read_tok_s' },
    write: { label: 'Write speed', unit: 'tok/s', keys: ['write_tok_s', 'decode_tok_s', 'agg_tok_s'], live: 'tok_s' },
    writeLoaded: { label: 'Write speed under load', unit: 'tok/s', keys: ['decode_loaded_tok_s'], live: 'tok_s' },
    same: { label: 'Same words?', verdict: 'exact', yes: 'Identical', no: 'Different' },
    sameParked: { label: 'Same words after parking?', verdict: 'same_as_ram', yes: 'Identical', no: 'Different' },
    correct: { label: 'Correct', icon: 'i-check', score: true }
  };
  var TILES = {
    chat: ['total', 'first', 'write'], burst: ['total', 'first', 'write'], long: ['total', 'first', 'read', 'write'],
    load: ['total', 'firstLoaded', 'writeLoaded'], exact: ['total', 'same'], turns: ['total', 'write', 'sameParked']
  };
  var DEFAULT_TILES = ['total', 'first', 'read', 'write'];
  function slotValue(slot, m) {
    for (var i = 0; m && i < slot.keys.length; i++) if (C.num(m[slot.keys[i]])) return m[slot.keys[i]];
    return null;
  }
  // The tiles a test shows (and the score of a graded run, last). Once numbers are in (final), a tile no engine has a number for is left out.
  V.tileSlots = function (scenario, final) {
    var done = Object.keys(final || {}).filter(function (rt) { return Object.keys(final[rt]).length; });
    var ids = (TILES[scenario] || DEFAULT_TILES).concat(done.some(function (rt) { return graded(final[rt]); }) ? ['correct'] : []);
    return ids.map(function (id) { return SLOTS[id]; }).filter(function (slot) {
      if (!done.length) return true;
      return done.some(function (rt) {
        return slot.score ? graded(final[rt]) : slot.verdict ? typeof final[rt][slot.verdict] === 'boolean' : slotValue(slot, final[rt]) !== null;
      });
    });
  };

  // -- a graded run's score: how many answers were right, and in words and icons, never colour alone -----------------------------
  var GRADE = { 'correct': ['i-check', 'Correct', 'ok'], 'wrong': ['i-no', 'Wrong', 'bad'], 'no answer': ['i-dash', 'No answer', 'none'] };
  function graded(m) { return !!m && m.graded === true; }
  function asked(m) { return m.correct + m.wrong + m.empty; }
  V.verdictWord = function (v) { return GRADE[v] ? GRADE[v][1] : ''; };
  // the icon and the word of a verdict (`text` instead of the word, for a count: "3 wrong")
  V.verdictHtml = function (v, text) { var g = GRADE[v]; return g ? '<svg class="ic s g-' + g[2] + '" aria-hidden="true"><use href="#' + g[0] + '"/></svg>' + esc(text || g[1]) : ''; };
  function scoreDetail(m) { return V.verdictHtml('wrong', m.wrong + ' wrong') + ' ' + V.verdictHtml('no answer', m.empty + ' no answer'); }
  // One plain line under the verdict: the score of each engine, or why the run is not graded ('' for a run from before grading)
  V.gradeText = function (final) {
    var rts = Object.keys(final || {}).filter(function (rt) { return graded(final[rt]); }), a = final.sllm, b = final.llama;
    if (!rts.length) return Object.keys(final || {}).map(function (rt) { return final[rt] && final[rt].graded_note; }).filter(Boolean)[0] || '';
    var note = (final[rts[0]].graded_note ? ' ' + final[rts[0]].graded_note : '');
    if (rts.length === 1) return C.SERIES[rts[0]].label + ' got ' + final[rts[0]].correct + ' of ' + asked(final[rts[0]]) + ' right.' + note;
    return (a.correct === b.correct && asked(a) === asked(b) ? 'Both engines got ' + a.correct + ' of ' + asked(a) + ' right.' :
      'Sonic LLM ' + a.correct + ', llama.cpp ' + b.correct + ' of ' + asked(a) + ' right.') + note;
  };

  // One engine's reading for a slot: {text, unit, raw, live}: its own number, else the live one, else a dash.
  function reading(slot, rt, o) {
    var m = (o.final || {})[rt], now = (o.live || {})[rt];
    if (slot.score) {
      return graded(m) ? { text: m.correct + ' of ' + asked(m), unit: Math.round(100 * m.correct / Math.max(1, asked(m))) + '%', raw: null, live: scoreDetail(m) } :
        { text: '–', unit: '', raw: null, live: '' };
    }
    if (slot.verdict) {
      var b = m && m[slot.verdict];
      return { text: typeof b === 'boolean' ? (b ? slot.yes : slot.no) : '–', unit: '', raw: null, live: '' };
    }
    var own = slotValue(slot, m);
    if (own !== null) { var p = parts(own, slot.unit); return { text: p[0], unit: p[1], raw: own, live: '' }; }
    var lv = slot.live === 'elapsed' ? (now && now.start ? ((now.end || o.now) - now.start) / 1000 : null) : now && now.running ? now[slot.live] : null;
    if (C.num(lv) && lv > 0) { var q = parts(lv, slot.unit); return { text: q[0], unit: q[1], raw: null, live: now.running ? 'live' : 'so far' }; }
    return { text: '–', unit: parts(1, slot.unit)[1], raw: null, live: '' };
  }
  function rowHtml(slot, rt, r, who) {
    return '<div class="t-row ' + (who ? rt : 'solo') + (r.text === '–' ? ' blank' : '') + '">' + (who ? '<i class="swatch ' + rt + '"></i><span class="sr">' + esc(C.SERIES[rt].label) + ': </span>' : '') +
      '<span class="t-num">' + esc(r.text) + '</span>' + (r.unit ? '<span class="t-unit">' + esc(r.unit) + '</span>' : '') + (r.live ? '<span class="t-live">' + r.live + '</span>' : '') + '</div>';
  }
  // The headline tiles. o: {scenario, rts, final: {rt: numbers}, live: {rt: {tok_s, read_tok_s, running, start}}, now (ms)}.
  V.tilesHtml = function (o) {
    var many2 = o.rts.length > 1;
    return V.tileSlots(o.scenario, o.final).map(function (slot) {
      var rs = o.rts.map(function (rt) { return reading(slot, rt, o); });
      var cmp = many2 && !slot.verdict && !slot.score && rs[0].raw !== null && rs[1].raw !== null ? V.compare(rs[0].raw, rs[1].raw, slot.lower) : null;
      return '<div class="tile"><h4>' + (slot.icon ? '<svg class="ic s" aria-hidden="true"><use href="#' + slot.icon + '"/></svg>' : '') + esc(slot.label) + '</h4>' + rs.map(function (r, i) { return rowHtml(slot, o.rts[i], r, many2); }).join('') +
        '<div class="t-ratio ' + (cmp ? cmp.tone : '') + '">' + (cmp ? esc(cmp.text) : '') + '</div></div>';
    }).join('');
  };
  // The same tiles with nothing in them yet: what the first screen shows of what is coming.
  V.ghostTilesHtml = function () {
    return ['total', 'first', 'read', 'write'].map(function (id) {
      return '<div class="tile"><h4>' + SLOTS[id].label + '</h4><div class="t-row solo blank"><span class="t-num">–</span><span class="t-unit">' + parts(1, SLOTS[id].unit)[1] + '</span></div><div class="t-ratio"></div></div>';
    }).join('');
  };

  // -- every number, in a table ---------------------------------------------------------------------------------------
  var NUMBERS = [
    ['wall_s', 'Total time', 's', true], ['swarm_wall_s', 'Time for all agents', 's', true], ['ttft_ms', 'First word', 'ms', true],
    ['ttft_ms_p50', 'First word, typical agent', 'ms', true], ['final_ttft_ms', 'First word of the merged answer', 'ms', true],
    ['ttft_alone_ms', 'First word, alone', 'ms', true], ['ttft_loaded_ms', 'First word, under load', 'ms', true], ['cached_ttft_ms', 'First word, asked again', 'ms', true],
    ['decode_tok_s', 'Write speed', 'tok/s'], ['write_tok_s', 'Write speed, while writing', 'tok/s'], ['agg_tok_s', 'Write speed, whole run', 'tok/s'], ['swarm_tok_s', 'Write speed, all agents', 'tok/s'],
    ['per_agent_tok_s', 'Write speed of each agent', 'tok/s'], ['decode_alone_tok_s', 'Write speed, alone', 'tok/s'], ['decode_loaded_tok_s', 'Write speed, under load', 'tok/s'],
    ['read_only_tok_s', 'Read speed', 'tok/s'], ['read_wall_tok_s', 'Read speed, whole swarm', 'tok/s'], ['prefill_tok_s', 'Read speed, long prompt', 'tok/s'],
    ['itl_p95_ms', 'Slowest gaps between words', 'ms', true], ['chat_slowdown', 'Chat slowdown under load', 'x', true],
    ['max_running', 'Most agents at once', ''], ['read_tokens', 'Tokens read by the agents', ''], ['prompt_tokens', 'Tokens read', ''], ['tokens_generated', 'Tokens written', ''],
    ['found_pct', 'Questions answered', '%'], ['labeled', 'Documents labelled', ''], ['json_ok', 'Replies that are valid JSON', ''], ['kept_pct', 'Length kept', '%', true],
    ['cache_hit_pct', 'Read from memory, not again', '%'], ['reused_pct', 'Read from memory, not again', '%'], ['stories_done', 'Stories done', ''], ['laps', 'Laps', ''], ['retries', 'Retries', '', true],
    ['rounds', 'Rounds', ''], ['at_once', 'Writing at once', ''], ['suspends', 'Times parked on disk', ''], ['restores', 'Times woken', ''], ['restore_ms_p50', 'Wake-up time, typical', 'ms', true],
    ['restore_ms_p95', 'Wake-up time, slowest', 'ms', true], ['ssd_mb', 'Disk used, peak', 'MB', true],
    ['peak_mem_mb', 'Memory used, peak', 'MB', true], ['session_peak_mem_mb', 'Memory used, whole session', 'MB', true], ['kv_reserved_mb', 'Memory set aside for agents', 'MB', true],
    ['kv_used_mb', 'Memory the agents used, peak', 'MB', true], ['weights_mb', 'Model file, not counted above', 'MB', true]
  ];
  V.numbersTable = function (per, rts) {
    var two = rts.length > 1;
    var rows = NUMBERS.filter(function (n) { return rts.some(function (rt) { return C.num((per[rt] || {})[n[0]]); }); }).map(function (n) {
      var vals = rts.map(function (rt) { return (per[rt] || {})[n[0]]; });
      var cmp = two && n[2] !== '' && n[2] !== 'MB' ? V.compare(vals[0], vals[1], !!n[3]) : null;
      return '<tr><th scope="row">' + esc(n[1]) + '</th>' + rts.map(function (rt, i) { return '<td class="' + rt + '">' + esc(V.valueText(vals[i], n[2])) + '</td>'; }).join('') +
        (two ? '<td class="diff">' + (cmp ? esc(cmp.text) : '') + '</td>' : '') + '</tr>';
    });
    if (!rows.length) return '<p class="muted small">No numbers yet.</p>';
    return '<table class="numtable"><thead><tr><th scope="col">Measure</th>' + rts.map(function (rt) { return '<th scope="col">' + C.SERIES[rt].label + '</th>'; }).join('') +
      (two ? '<th scope="col">Sonic LLM vs llama.cpp</th>' : '') + '</tr></thead><tbody>' + rows.join('') + '</tbody></table>';
  };
  // Two short pills for the history list: the headline of a saved run.
  V.headlines = function (rec) {
    var rts = Object.keys(rec.metrics || {}).filter(function (rt) { return Object.keys(rec.metrics[rt]).length; });
    var score = V.gradeText(rec.metrics || {});
    return ['write', 'total'].map(function (id) {
      var slot = SLOTS[id], vals = rts.map(function (rt) { return slotValue(slot, rec.metrics[rt]); });
      if (rts.length > 1 && rts[0] === 'sllm') { var c = V.compare(vals[0], vals[1], slot.lower); if (c) return slot.label + ' ' + c.text; }
      return vals[0] !== null && vals[0] !== undefined ? slot.label + ' ' + V.valueText(vals[0], slot.unit) : '';
    }).concat(rts.some(function (rt) { return graded(rec.metrics[rt]); }) ? [score] : []).filter(Boolean);
  };

  // -- notes under an answer ------------------------------------------------------------------------------------------
  // "59 of 256 asked (one per document)": agents that really ran next to the number asked for
  function agentsText(m) {
    if (!C.num(m.agents)) return '';
    return m.agents + (C.num(m.agents_asked) && m.agents_asked !== m.agents ? ' of ' + m.agents_asked + ' asked' : ' agents') + (m.agents_note ? ' (' + m.agents_note + ')' : '');
  }
  function note(ok, text) { return '<span class="verdict' + (ok ? '' : ' bad') + '"><svg class="ic s" aria-hidden="true"><use href="#' + (ok ? 'i-check' : 'i-no') + '"/></svg>' + esc(text) + '</span>'; }
  V.statsHtml = function (m) {
    var out = C.num(m.agents) ? ['<span class="pill">' + esc(agentsText(m)) + '</span>'] : [];
    if (m.cut_off) out.push('<span class="pill">' + C.fmt(m.cut_off) + ' stopped at the length limit</span>');
    if (m.exact === true) out.push(note(true, 'Identical words'));
    if (m.exact === false) out.push(note(false, 'Words differ at token ' + m.first_diff));
    if (m.same_as_ram === true) out.push(note(true, 'Same words as without parking'));
    if (m.same_as_ram === false) out.push(note(false, 'Words differ from the run without parking'));
    if (m.same_answer_under_load === true) out.push(note(true, 'Same answer under load'));
    if (m.same_answer_under_load === false) out.push(note(false, 'The answer changed under load'));
    return out.join('');
  };
})();
