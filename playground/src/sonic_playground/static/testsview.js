/* The test cards and the first screen, built from plain data: which controls each test needs, the grouped cards, the
   guide a newcomer sees before any run, and the callout for an engine that is not built. Extends window.LiveView. */
(function () {
  'use strict';
  var C = window.Charts, esc = C.esc, V = window.LiveView;

  V.GROUPS = ['Talk', 'Many agents', 'Documents', 'Checks'];
  var ICON = { chat: 'chat', long: 'doc', swarm: 'swarm', team: 'team', followup: 'reply', relay: 'relay', turns: 'turns', shared: 'layers',
    classify: 'tag', extract: 'code', summarize: 'lines', compress: 'shrink', load: 'pulse', burst: 'bolt', exact: 'target' };
  // What each test asks of you: ask (the label of the text box), docs (the label of the documents control), agents, level.
  V.NEEDS = {
    chat: { ask: 'Your question', docs: 'Documents to ask about (optional)', level: true },
    long: { ask: 'Your question about the documents', docs: 'Documents to join (all of them by default)' },
    swarm: { ask: 'Question for every agent', docs: 'Documents, one agent each', agents: true, level: true },
    team: { ask: 'Questions, one per line', docs: 'Documents, one agent each', agents: true, level: true },
    followup: { ask: 'New question for the same agents', level: true },
    relay: { ask: 'What should each agent look for?', docs: 'Documents, one story each', agents: true },
    turns: { docs: 'Documents, one agent each', agents: true },
    shared: { docs: 'Document to ask about', agents: true },
    classify: { ask: 'Labels, separated by commas', docs: 'Documents, one agent each', agents: true },
    extract: { ask: 'Fields to pull out, separated by commas', docs: 'Documents, one agent each', agents: true },
    summarize: { ask: 'How should it be summarized?', docs: 'Documents, one agent each', agents: true },
    compress: { ask: 'How short?', docs: 'Documents, one agent each', agents: true },
    load: { ask: 'Chat question', docs: 'Documents the busy agents read', agents: true, level: true },
    burst: { agents: true },
    exact: { ask: 'Question to compare', agents: true }
  };

  // The tests as cards in groups (fieldsets of radio buttons: arrow keys move between them); the picked one (scen) is on.
  V.testCardsHtml = function (scenarios, scen) {
    var names = V.GROUPS.concat(scenarios.map(function (s) { return s.group; }).filter(function (g, i, a) { return V.GROUPS.indexOf(g) < 0 && a.indexOf(g) === i; }));
    return names.map(function (g) {
      var items = scenarios.filter(function (s) { return s.group === g; });
      return items.length ? '<fieldset class="group"><legend>' + esc(g) + '</legend>' + items.map(function (s) { return card(s, s.id === scen); }).join('') + '</fieldset>' : '';
    }).join('');
  };
  function card(s, on) {
    return '<div class="test' + (on ? ' on' : '') + '" data-t="' + esc(s.id) + '"><label class="pick"><input type="radio" name="test" value="' + esc(s.id) + '"' + (on ? ' checked' : '') + '>' +
      '<span class="ico"><svg class="ic" aria-hidden="true"><use href="#i-' + (ICON[s.id] || 'chat') + '"/></svg></span>' +
      '<span><b>' + esc(s.name) + '</b><span class="what">' + esc(s.what) + '</span></span>' +
      '<svg class="ic tick" aria-hidden="true"><use href="#i-check"/></svg></label></div>';
  }

  // -- the first screen -----------------------------------------------------------------------------------------------
  // o: {models (how many on this Mac), model (the loaded one's name or ''), loading, llama (found), noEngine}
  V.introHtml = function (o) {
    var s1 = o.model ? { done: true, text: esc(o.model) + ' is loaded and ready.' } :
      o.noEngine ? { text: 'Build the engine first: the box above says how. Then pick a model.' } :
      !o.models ? { text: 'There is no model on this Mac yet. Download one: it takes a few minutes.', button: '<button class="btn primary" id="introModels" type="button"><svg class="ic s" aria-hidden="true"><use href="#i-download"/></svg>Get models</button>' } :
      { text: o.loading ? 'Loading the model…' : 'Pick a model in the top bar and press Load.' };
    var compare = o.llama ? 'Turn on <b>Compare with llama.cpp</b> in the top bar to run every test on both engines, side by side.' :
      'Optional: with llama.cpp installed, every test also runs on llama.cpp, side by side. <button class="link" id="introWhy" type="button">How to turn it on</button>';
    return '<div class="intro"><div><h2>Watch many AI agents run on your Mac</h2></div>' +
      '<p class="lead">Sonic LLM is a local LLM server for Apple Silicon that stays fast when many agents run at once. Pick a test, press Run, and the speed, the agents and the memory show up here as they happen.</p>' +
      '<ol class="steps">' + step(1, 'Get a model', s1.text, s1.done, !s1.done, s1.button) +
      step(2, 'Pick a test', 'Browse the tests. Each one says in a line what you will see.', false, !!o.model) +
      step(3, 'Run it', 'Press Run. Numbers, agents and charts fill in here, live.', false, false) + '</ol>' +
      '<div class="preview"><p>Your numbers will appear here, like this:</p><div class="tiles">' + V.ghostTilesHtml() + '</div></div>' +
      '<p class="tail">' + compare + '</p></div>';
  };
  function step(n, title, text, done, now, button) {
    return '<li class="step' + (done ? ' done' : now ? ' now' : '') + '"><span class="n">' + (done ? '<svg class="ic s" aria-hidden="true"><use href="#i-check"/></svg>' : n) + '</span><h3>' + esc(title) + '</h3><p>' + text + '</p>' + (button || '') + '</li>';
  }

  // The engine is not built: what to run, with a copy button.
  V.engineMissingHtml = function (engine) {
    var cmd = 'cd engine && ' + engine.build;
    return '<div class="callout error"><svg class="ic" aria-hidden="true"><use href="#i-alert"/></svg><div><h3>The Sonic LLM engine is not built yet</h3>' +
      '<p>The playground starts the engine for you, but it has to exist first. Build it once, from the folder of this repository:</p>' +
      '<div class="cmd"><code>' + esc(cmd) + '</code><button class="btn" type="button" data-copy="' + esc(cmd) + '"><svg class="ic s" aria-hidden="true"><use href="#i-copy"/></svg>Copy</button></div>' +
      '<p>Then reload this page. It looks for the engine at <code>' + esc(engine.path) + '</code>; to use another build, set <code>SONIC_SERVER_BIN</code>.</p></div></div>';
  };
  // A message about what just happened (a load or a run that failed), with a dismiss button.
  V.noticeHtml = function (msg, kind, title) {
    return '<div class="callout ' + (kind || '') + '"><svg class="ic" aria-hidden="true"><use href="#i-' + (kind === 'error' ? 'alert' : 'info') + '"/></svg><div>' + (title ? '<h3>' + esc(title) + '</h3>' : '') + '<p>' + esc(msg) + '</p></div>' +
      '<button class="btn icon quiet x" type="button" data-dismiss="1" aria-label="Dismiss"><svg class="ic" aria-hidden="true"><use href="#i-x"/></svg></button></div>';
  };
})();
