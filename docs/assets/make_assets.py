"""README images for Sonic LLM: assets.py RESULTS_DIR OUT_DIR -> logo, banners, charts, diagram (light + dark SVG)."""
import json, statistics, sys
DIR, OUT = sys.argv[1], sys.argv[2]
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
T = {"light": dict(bg="#fcfcfb", card="#f3f2ee", ink="#0b0b0b", ink2="#52514e", mute="#8a897f", grid="#e1dfd8", s1="#2a78d6", s2="#eb6834", edge="#d9d7cf"),
     "dark": dict(bg="#1a1a19", card="#242423", ink="#f0efec", ink2="#c3c2b7", mute="#8f8e85", grid="#33332f", s1="#3987e5", s2="#d95926", edge="#3a3a36")}
M = [("gemma-4-E2B-it-Q4_0", "Gemma 4 E2B"), ("Qwen3-4B-Q4_0", "Qwen3 4B"), ("gemma-3-4b-it-qat-Q4_0", "Gemma 3 4B"), ("Phi-3.5-mini-instruct-Q4_0", "Phi 3.5 mini"),
     ("Mistral-7B-Instruct-v0.3-Q4_K_M", "Mistral 7B"), ("Llama-3.1-8B-Instruct-Q4_0", "Llama 3.1 8B"), ("gemma-4-12b-it-qat-q4_0", "Gemma 4 12B"), ("gpt-oss-20b-MXFP4", "gpt-oss 20B")]
S = [("shared", "16 questions, one document", "wall_s"), ("burst", "32 requests at once", "wall_s"), ("followup", "Follow-up to 8 agents", "swarm_wall_s"),
     ("long", "One long document", "wall_s"), ("team", "Research team of 8", "swarm_wall_s"), ("swarm", "Swarm of 8 agents", "swarm_wall_s"),
     ("classify", "Classify 8 documents", "wall_s"), ("extract", "Extract from 8 documents", "wall_s"), ("summarize", "Summarize 8 documents", "wall_s"),
     ("compress", "Compress 8 documents", "wall_s"), ("relay", "Relay of 8 agents", "wall_s"), ("chat", "One chat turn", "wall_s")]
D = {}
for m, _ in M:
    for l in open(f"{DIR}/{m}.jsonl"):
        d = json.loads(l); mt = (d.get("record") or {}).get("metrics") or {}
        D[(m, d["scenario"])] = (mt.get("sllm") or {}, mt.get("llama") or {})


def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;")
def svg(w, h, body, c): return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="{FONT}"><rect width="{w}" height="{h}" rx="14" fill="{c["bg"]}" stroke="{c["edge"]}"/>{body}</svg>\n'
def text(x, y, s, size, fill, weight=400, anchor="start"): return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{esc(s)}</text>'


def mark(x, y, s, c):
    """The logo mark: a rounded tile with three speed streaks that end in a forward chevron."""
    k = s / 96
    return (f'<g transform="translate({x},{y}) scale({k})"><rect width="96" height="96" rx="22" fill="{c["s1"]}"/>'
            '<g stroke="#fff" stroke-width="9" stroke-linecap="round" fill="none">'
            '<path d="M16 30 H46"/><path d="M24 48 H50"/><path d="M16 66 H46"/>'
            '<path d="M56 24 L78 48 L56 72" stroke-linejoin="round"/></g></g>')


def logo():
    c = T["light"]
    open(f"{OUT}/logo.svg", "w").write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" width="96" height="96">{mark(0, 0, 96, c)}</svg>\n')


def banner(mode):
    c = T[mode]; w, h = 1200, 420
    b = mark(64, 64, 88, c) + text(176, 110, "Sonic LLM", 60, c["ink"], 700) + text(178, 146, "One Mac, many agents.", 24, c["ink2"])
    b += text(64, 212, "A local LLM server for Apple Silicon. Same GGUF files and same interface as llama.cpp,", 21, c["ink2"])
    b += text(64, 240, "same answers, and several times faster when many requests run together.", 21, c["ink2"])
    tiles = [("3.7x-6.3x", "faster: 16 questions", "about one document"), ("2.0x-3.9x", "faster: 32 requests", "sent at once"),
             ("0.1 s", "to a chat's first word while", "8 agents work (llama.cpp: 18-52 s)"), ("103 of 112", "model-and-mode cells", "faster than llama.cpp")]
    tw, gap, x0 = 259, 12, 64
    for i, (big, l1, l2) in enumerate(tiles):
        x = x0 + i * (tw + gap)
        b += f'<rect x="{x}" y="276" width="{tw}" height="112" rx="12" fill="{c["card"]}"/>'
        b += text(x + 18, 322, big, 34, c["s1"] if mode == "dark" else "#1f5fb0", 700) + text(x + 18, 348, l1, 14.5, c["ink"]) + text(x + 18, 368, l2, 12.5, c["ink2"])
    b += text(w - 64, 110, "vs llama-server build 11669", 14, c["mute"], anchor="end") + text(w - 64, 130, "Apple M5 Pro, 48 GB, one run per cell", 14, c["mute"], anchor="end")
    open(f"{OUT}/banner-{mode}.svg", "w").write(svg(w, h, b, c))


def ratios(sid, key):
    out = []
    for m, _ in M:
        a, b = D.get((m, sid), ({}, {}))
        if a.get(key) and b.get(key): out.append(b[key] / a[key])
    return out


def speedup(mode):
    c = T[mode]; rows = sorted(((n, ratios(s, k)) for s, n, k in S), key=lambda r: -statistics.median(r[1]))
    w, left, right, top, rh = 1200, 300, 150, 112, 40
    h = top + rh * len(rows) + 64; pw = w - left - right; mx = 6.5
    X = lambda v: left + pw * v / mx
    b = text(40, 48, "How many times faster than llama.cpp, by kind of work", 24, c["ink"], 700)
    b += text(40, 76, "Total time for the same requests on the same model. Bar: median of 8 models. Line: slowest to fastest model.", 15, c["ink2"])
    for v in range(0, 7):
        b += f'<line x1="{X(v)}" y1="{top - 8}" x2="{X(v)}" y2="{top + rh * len(rows)}" stroke="{c["grid"]}"/>' + text(X(v), top + rh * len(rows) + 22, f"{v}x", 13, c["mute"], anchor="middle")
    b += f'<line x1="{X(1)}" y1="{top - 8}" x2="{X(1)}" y2="{top + rh * len(rows)}" stroke="{c["ink2"]}" stroke-dasharray="4 4"/>' + text(X(1) + 6, top - 12, "llama.cpp = 1x", 13, c["ink2"])
    for i, (name, r) in enumerate(rows):
        y = top + i * rh; med = statistics.median(r)
        b += text(left - 14, y + 25, name, 15.5, c["ink"], anchor="end")
        b += f'<path d="M{left} {y + 8} H{X(med) - 4} a4 4 0 0 1 4 4 V{y + 28} a4 4 0 0 1 -4 4 H{left} Z" fill="{c["s1"]}"/>'
        b += f'<line x1="{X(min(r))}" y1="{y + 20}" x2="{X(max(r))}" y2="{y + 20}" stroke="{c["ink"]}" stroke-width="2"/>'
        for v in (min(r), max(r)): b += f'<line x1="{X(v)}" y1="{y + 14}" x2="{X(v)}" y2="{y + 26}" stroke="{c["ink"]}" stroke-width="2"/>'
        b += text(X(max(max(r), med)) + 12, y + 25, f"{med:.1f}x", 15.5, c["ink"], 700) + text(X(max(max(r), med)) + 58, y + 25, f"({min(r):.2f}-{max(r):.1f})", 13, c["ink2"])
    b += text(40, h - 16, "Apple M5 Pro, 48 GB. llama-server build 11669, same GGUF files. One run per cell; differences under 10% are noise.", 13, c["mute"])
    open(f"{OUT}/speedup-{mode}.svg", "w").write(svg(w, h, b, c))


def firstword(mode):
    c = T[mode]; rows = []
    for m, n in M:
        a, b = D.get((m, "load"), ({}, {}))
        if a.get("ttft_loaded_ms") and b.get("ttft_loaded_ms"): rows.append((n, a["ttft_loaded_ms"] / 1000, b["ttft_loaded_ms"] / 1000))
    w, left, right, top, rh = 1200, 190, 110, 138, 58
    h = top + rh * len(rows) + 64; pw = w - left - right; mx = 60
    X = lambda v: left + pw * v / mx
    b = text(40, 48, "Chat while 8 agents work: seconds until the first word", 24, c["ink"], 700)
    b += text(40, 76, "A chat message arrives while eight background agents are busy. Shorter is better.", 15, c["ink2"])
    b += f'<rect x="40" y="96" width="14" height="14" rx="3" fill="{c["s1"]}"/>' + text(62, 108, "Sonic LLM", 14.5, c["ink"]) + f'<rect x="160" y="96" width="14" height="14" rx="3" fill="{c["s2"]}"/>' + text(182, 108, "llama.cpp", 14.5, c["ink"])
    for v in range(0, 61, 10):
        b += f'<line x1="{X(v)}" y1="{top - 8}" x2="{X(v)}" y2="{top + rh * len(rows)}" stroke="{c["grid"]}"/>' + text(X(v), top + rh * len(rows) + 22, f"{v} s", 13, c["mute"], anchor="middle")
    for i, (name, a, l) in enumerate(rows):
        y = top + i * rh
        b += text(left - 14, y + 32, name, 15.5, c["ink"], anchor="end")
        b += f'<rect x="{left}" y="{y + 8}" width="{max(3, X(a) - left):.1f}" height="16" rx="2" fill="{c["s1"]}"/>' + text(max(X(a), left + 3) + 10, y + 21, f"{a:.2f} s", 14.5, c["ink"], 700)
        b += f'<path d="M{left} {y + 28} H{X(l) - 4} a4 4 0 0 1 4 4 V{y + 40} a4 4 0 0 1 -4 4 H{left} Z" fill="{c["s2"]}"/>' + text(X(l) + 10, y + 41, f"{l:.1f} s", 14.5, c["ink2"])
    b += text(40, h - 16, "Apple M5 Pro, 48 GB, one run per model. Mistral 7B is missing: its run did not finish in the time allowed.", 13, c["mute"])
    open(f"{OUT}/first-word-{mode}.svg", "w").write(svg(w, h, b, c))


def diagram(mode):
    c = T[mode]; w, h = 1200, 380
    b = text(40, 48, "Why many questions about one document are fast", 24, c["ink"], 700)
    def panel(x, title, sub, shared):
        s = f'<rect x="{x}" y="76" width="540" height="270" rx="12" fill="{c["card"]}"/>' + text(x + 24, 110, title, 17, c["ink"], 700) + text(x + 24, 132, sub, 13.5, c["ink2"])
        for i in range(4):
            y = 154 + i * 44
            if shared:
                if i == 0: s += f'<rect x="{x + 24}" y="{154}" width="300" height="162" rx="6" fill="{c["s1"]}"/>' + text(x + 174, 242, "document, read once", 15, "#fff", 700, "middle")
                s += f'<rect x="{x + 332}" y="{y}" width="60" height="30" rx="6" fill="{c["s2"]}"/>' + text(x + 404, y + 20, f"question {i + 1}", 13.5, c["ink2"])
            else:
                s += f'<rect x="{x + 24}" y="{y}" width="300" height="30" rx="6" fill="{c["s1"]}"/>' + text(x + 174, y + 20, "document, read again", 13.5, "#fff", 600, "middle")
                s += f'<rect x="{x + 332}" y="{y}" width="60" height="30" rx="6" fill="{c["s2"]}"/>' + text(x + 404, y + 20, f"question {i + 1}", 13.5, c["ink2"])
        return s
    b += panel(40, "Without shared reading", "every request reads the whole document", False) + panel(620, "Sonic LLM", "the shared start is read once; each request reads only its question", True)
    open(f"{OUT}/shared-reading-{mode}.svg", "w").write(svg(w, h, b, c))


logo()
for mode in T:
    banner(mode); speedup(mode); firstword(mode); diagram(mode)
