/* A tiny stand-in for the browser, enough to load the playground's static scripts in node's vm and drive them: elements
   are made on first use (getElementById, querySelector), fetch answers from a table of routes, and the event stream is a
   fake SharedWorker port or EventSource that the test feeds. No dependencies. */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const STATIC = path.resolve(__dirname, '../../src/sonic_playground/static');

class ClassList {
  constructor(el) { this.el = el; }
  set() { return new Set(String(this.el.className || '').split(/\s+/).filter(Boolean)); }
  write(s) { this.el.className = Array.from(s).join(' '); }
  add(c) { const s = this.set(); s.add(c); this.write(s); }
  remove(c) { const s = this.set(); s.delete(c); this.write(s); }
  contains(c) { return this.set().has(c); }
  toggle(c, on) { const want = on === undefined ? !this.contains(c) : !!on; if (want) this.add(c); else this.remove(c); return want; }
}

// An element: what a page writes to it is kept; querySelector(sel) gives the same child for the same selector.
class El {
  constructor(tag, key) {
    this.tagName = tag; this.key = key; this.children = []; this.parentNode = null; this.dataset = {}; this.style = {};
    this.attrs = {}; this.q = {}; this.html = ''; this.textContent = ''; this.className = ''; this.title = '';
    this.hidden = false; this.disabled = false; this.value = ''; this.checked = false; this.open = false;
    this.classList = new ClassList(this); this.listeners = {};
  }
  get innerHTML() { return this.html; }
  set innerHTML(v) { this.html = String(v); this.q = {}; this.children = []; }
  querySelector(sel) {
    if (!this.q[sel]) { this.q[sel] = new El('q', sel); this.q[sel].parentNode = this; }
    return this.q[sel];
  }
  // 'A B': the B inside every A asked for so far; anything else: none
  querySelectorAll(sel) {
    const parts = sel.split(' ');
    if (parts.length !== 2) return [];
    return Object.keys(this.q).filter((k) => k.startsWith(parts[0])).map((k) => this.q[k].querySelector(parts[1]));
  }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  prepend(c) { c.parentNode = this; this.children.unshift(c); }
  insertBefore(c, ref) { c.parentNode = this; const i = this.children.indexOf(ref); this.children.splice(i < 0 ? this.children.length : i, 0, c); return c; }
  remove() { if (this.parentNode) { const k = this.parentNode.children; const i = k.indexOf(this); if (i >= 0) k.splice(i, 1); } this.removed = true; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  closest() { return null; }
  focus() {} showModal() {} close() {} scrollIntoView() {}
  get scrollHeight() { return 0; } get clientHeight() { return 0; }
}

function response(body, status) {
  const st = status || 200;
  return { ok: st < 400, status: st, statusText: 'x', json: () => Promise.resolve(body), text: () => Promise.resolve(JSON.stringify(body)) };
}

// A fresh vm context with a document, window, storage, timers, fetch and a stream. opts.routes: {'GET /api/x': body or fn}.
function makeContext(opts) {
  const ids = {};
  const calls = [];
  const held = [];  // fetches a test holds back: {key, resolve}
  const ctx = {
    console, setTimeout, clearTimeout, setInterval: () => 0, clearInterval: () => {}, Promise, JSON, Object, Array,
    URLSearchParams, Date, Math, Number, String, Error, isFinite, parseInt, parseFloat
  };
  ctx.window = ctx; ctx.self = ctx;
  ctx.document = {
    readyState: 'complete', activeElement: null, listeners: {},
    getElementById: (id) => ids[id] || (ids[id] = new El('div', id)),
    createElement: (tag) => new El(tag, null),
    createElementNS: (ns, tag) => new El(tag, null),
    addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); },
    querySelector: () => null, querySelectorAll: () => []
  };
  ctx.listeners = {};
  ctx.addEventListener = (type, fn) => { (ctx.listeners[type] = ctx.listeners[type] || []).push(fn); };
  const store = {};
  ctx.localStorage = { getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); } };
  ctx.navigator = { clipboard: { writeText: () => Promise.resolve() } };
  ctx.location = { search: '', pathname: '/', hash: '' };
  ctx.fetch = (url, init) => {
    const method = (init && init.method) || 'GET', key = method + ' ' + url;
    calls.push({ key, body: init && init.body ? JSON.parse(init.body) : null });
    const route = (opts.routes || {})[key];
    const answer = () => (typeof route === 'function' ? route(init) : route === undefined ? response({ detail: 'no route ' + key }, 404) : response(route));
    if (opts.hold && opts.hold(key)) return new Promise((resolve) => held.push({ key, resolve: () => resolve(answer()) }));
    return Promise.resolve(answer());
  };
  return { ctx, ids, calls, held };
}

function load(ctx, names) {
  vm.createContext(ctx);
  names.forEach((n) => vm.runInContext(fs.readFileSync(path.join(STATIC, n), 'utf8'), ctx, { filename: n }));
}

// The inline scripts of a static page (no src), in order.
function inlineScripts(html) {
  const text = fs.readFileSync(path.join(STATIC, html), 'utf8');
  return Array.from(text.matchAll(/<script>([\s\S]*?)<\/script>/g)).map((m) => m[1]);
}
function runCode(ctx, code, name) { vm.runInContext(code, ctx, { filename: name }); }

const tick = (ms) => new Promise((r) => setTimeout(r, ms || 0));
async function settle() { for (let i = 0; i < 20; i++) await tick(0); }

module.exports = { El, STATIC, makeContext, load, inlineScripts, runCode, response, tick, settle };
