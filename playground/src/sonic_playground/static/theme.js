/* Light or dark: the system's choice until the person picks one. Loaded in the head so the page never flashes the wrong theme. */
(function () {
  'use strict';
  var KEY = 'live.theme', root = document.documentElement;
  function stored() {
    try { return localStorage.getItem(KEY); } catch (e) { console.warn('live: the theme is not kept (' + e.message + ')'); return null; }
  }
  function effective() {
    return root.getAttribute('data-theme') || (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  }
  function show(button) {
    var dark = effective() === 'dark';
    button.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
    button.title = dark ? 'Switch to light theme' : 'Switch to dark theme';
    button.querySelector('use').setAttribute('href', dark ? '#i-sun' : '#i-moon');
  }
  var saved = stored();
  if (saved === 'light' || saved === 'dark') root.setAttribute('data-theme', saved);
  document.addEventListener('DOMContentLoaded', function () {
    var b = document.getElementById('theme');
    show(b);
    b.addEventListener('click', function () {
      var next = effective() === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem(KEY, next); } catch (e) { console.warn('live: the theme is not kept (' + e.message + ')'); }
      show(b);
    });
  });
})();
