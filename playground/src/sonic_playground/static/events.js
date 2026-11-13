/* One event stream for every playground tab of a browser (a SharedWorker). A browser gives a site 6 connections: with a
   stream per tab, six tabs left none for requests, so Stop waited until a stream ended, up to 20 s.
   A tab says 'bye' when it is hidden and 'hi' when it comes back from the back/forward cache; the stream is open only
   while a tab listens, and one the browser gave up on (an error answer: no retry of its own) is opened again. */
var ports = [], es = null, up = false, again = null;
var REOPEN_MS = 2000;
function all(msg) { ports.forEach(function (p) { p.postMessage(msg); }); }
function open() {
  es = new EventSource('/api/live/events');
  es.onopen = function () { up = true; all({ up: true }); };
  es.onerror = function () {
    up = false; all({ down: true });
    if (es && es.readyState === 2 && !again) again = setTimeout(reopen, REOPEN_MS);
  };
  es.onmessage = function (m) { all({ data: m.data }); };
}
function reopen() { again = null; if (es) es.close(); es = null; if (ports.length) open(); }
function shut() { if (es) es.close(); es = null; up = false; clearTimeout(again); again = null; }
function join(p) {
  if (ports.indexOf(p) < 0) ports.push(p);
  if (!es) open(); else p.postMessage(up ? { up: true } : { down: true });
}
function leave(p) {
  var i = ports.indexOf(p);
  if (i >= 0) ports.splice(i, 1);
  if (!ports.length) shut();
}
self.onconnect = function (e) {
  var p = e.ports[0];
  p.onmessage = function (m) { if (m.data === 'bye') leave(p); else if (m.data === 'hi') join(p); };
  p.start();
  join(p);
};
