"""A plain numpy Gemma 4 forward (llama.cpp's gemma4.cpp graph, written out), to find where the Metal path goes wrong.
usage: gemma_np.py REF.json INDEX [DUMP.npz]   prints the next-token argmax after prompt INDEX; DUMP keeps the last
token's hidden state after every layer."""
import json
import struct
import sys

import numpy as np

M = "/Users/benreich/Projects/crazyidea/models/gguf/gemma-4-E2B-it-Q4_0.gguf"


def gguf(path):
    f = open(path, "rb")
    u = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    s = lambda: f.read(u("Q")).decode("utf8", "replace")
    sz = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    assert f.read(4) == b"GGUF"
    u("I"); nt = u("Q"); nk = u("Q")
    meta = {}
    for _ in range(nk):
        k = s(); t = u("I")
        if t == 8:
            meta[k] = s()
        elif t == 9:
            at, n = u("I"), u("Q")
            meta[k] = [s() for _ in range(n)] if at == 8 else [u(sz[at]) for _ in range(n)]
        else:
            meta[k] = u(sz[t])
    ts = {}
    for _ in range(nt):
        n = s(); nd = u("I"); d = [u("Q") for _ in range(nd)]; t = u("I"); o = u("Q")
        ts[n] = (d, t, o)
    align = meta.get("general.alignment", 32)
    data = (f.tell() + align - 1) // align * align
    return meta, ts, data


META, TS, DATA = gguf(M)
RAW = np.memmap(M, dtype=np.uint8, mode="r")


def scale_min_k4(sc):  # sc [n, 12] uint8 -> (scales [n, 8], mins [n, 8])
    s = np.zeros(sc.shape[:-1] + (8,), np.float32)
    m = np.zeros_like(s)
    for j in range(8):
        if j < 4:
            s[..., j] = sc[..., j] & 63
            m[..., j] = sc[..., j + 4] & 63
        else:
            s[..., j] = (sc[..., j + 4] & 0xF) | ((sc[..., j - 4] >> 6) << 4)
            m[..., j] = (sc[..., j + 4] >> 4) | ((sc[..., j] >> 6) << 4)
    return s, m


def rows(name, idx=None):
    """dequantized rows [M, K] of a tensor (idx: only those rows)"""
    d, t, o = TS[name]
    K = d[0]
    Mr = int(np.prod(d[1:])) if len(d) > 1 else 1
    bpr = {0: K * 4, 30: K * 2, 2: K // 32 * 18, 3: K // 32 * 20, 12: K // 256 * 144, 13: K // 256 * 176}[t]
    raw = RAW[DATA + o: DATA + o + bpr * Mr].reshape(Mr, bpr)
    if idx is not None:
        raw = raw[np.asarray(idx)]
    n = raw.shape[0]
    if t == 0:
        return raw.view(np.float32).reshape(n, K).copy()
    if t == 30:
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(n, K)
    if t in (2, 3):
        bs = 18 if t == 2 else 20
        b = raw.reshape(n, K // 32, bs)
        dd = b[..., 0:2].copy().view(np.float16).astype(np.float32)
        off = 2 if t == 2 else 4
        qs = b[..., off:off + 16]
        lo, hi = (qs & 15).astype(np.float32), (qs >> 4).astype(np.float32)
        if t == 2:
            v = np.concatenate([lo - 8, hi - 8], -1) * dd
        else:
            mm = b[..., 2:4].copy().view(np.float16).astype(np.float32)
            v = np.concatenate([lo, hi], -1) * dd + mm
        return v.reshape(n, K)
    if t == 12:
        b = raw.reshape(n, K // 256, 144)
        d_ = b[..., 0:2].copy().view(np.float16).astype(np.float32)
        dm = b[..., 2:4].copy().view(np.float16).astype(np.float32)
        s, m = scale_min_k4(b[..., 4:16])
        q = b[..., 16:144].reshape(n, K // 256, 4, 32)
        out = np.zeros((n, K // 256, 8, 32), np.float32)
        for g in range(4):
            out[:, :, 2 * g] = (d_ * s[..., 2 * g:2 * g + 1]) * (q[:, :, g] & 0xF) - dm * m[..., 2 * g:2 * g + 1]
            out[:, :, 2 * g + 1] = (d_ * s[..., 2 * g + 1:2 * g + 2]) * (q[:, :, g] >> 4) - dm * m[..., 2 * g + 1:2 * g + 2]
        return out.reshape(n, K)
    if t == 13:
        b = raw.reshape(n, K // 256, 176)
        d_ = b[..., 0:2].copy().view(np.float16).astype(np.float32)
        dm = b[..., 2:4].copy().view(np.float16).astype(np.float32)
        s, m = scale_min_k4(b[..., 4:16])
        qh = b[..., 16:48]
        q = b[..., 48:176].reshape(n, K // 256, 4, 32)
        out = np.zeros((n, K // 256, 8, 32), np.float32)
        for g in range(4):
            h0 = ((qh >> (2 * g)) & 1).astype(np.float32) * 16
            h1 = ((qh >> (2 * g + 1)) & 1).astype(np.float32) * 16
            out[:, :, 2 * g] = (d_ * s[..., 2 * g:2 * g + 1]) * ((q[:, :, g] & 0xF) + h0) - dm * m[..., 2 * g:2 * g + 1]
            out[:, :, 2 * g + 1] = (d_ * s[..., 2 * g + 1:2 * g + 2]) * ((q[:, :, g] >> 4) + h1) - dm * m[..., 2 * g + 1:2 * g + 2]
        return out.reshape(n, K)
    raise ValueError(t)


def rms(x, w=None, eps=1e-6):
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return y * w if w is not None else y


def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * x * (1 + 0.044715 * x * x)))


def rope(x, pos, base, ff=None):  # x [n, H, D], NEOX
    D = x.shape[-1]
    i = np.arange(D // 2)
    th = pos[:, None] * base ** (-2.0 * i / D)
    if ff is not None:
        th = th / ff
    c, s = np.cos(th)[:, None, :], np.sin(th)[:, None, :]
    a, b = x[..., :D // 2], x[..., D // 2:]
    return np.concatenate([a * c - b * s, a * s + b * c], -1)


def W(name):
    return rows(name)


ref = json.load(open(sys.argv[1]))
toks = ref["prompts"][int(sys.argv[2])]
n, E, L, PLE = len(toks), META["gemma4.embedding_length"], META["gemma4.block_count"], META["gemma4.embedding_length_per_layer_input"]
swa, ffl = META["gemma4.attention.sliding_window_pattern"], META["gemma4.feed_forward_length"]
win, KVL = META["gemma4.attention.sliding_window"], L - META["gemma4.attention.shared_kv_layers"]
pos = np.arange(n, dtype=np.float64)
x = rows("token_embd.weight", toks) * np.float32(np.sqrt(E))
raw = rows("per_layer_token_embd.weight", toks) * 16.0
proj = (x @ W("per_layer_model_proj.weight").T) / np.sqrt(E)
proj = rms(proj.reshape(n, L, PLE), W("per_layer_proj_norm.weight")[0])
ple = (proj + raw.reshape(n, L, PLE)) / np.sqrt(2.0)
ff = W("rope_freqs.weight")[0]
kv, dump = {}, [x[-1].copy()]
mask_full = np.tril(np.ones((n, n), bool))
for il in range(L):
    p = f"blk.{il}."
    D = META["gemma4.attention.key_length_swa"] if swa[il] else META["gemma4.attention.key_length"]
    h = rms(x, W(p + "attn_norm.weight")[0])
    q = (h @ W(p + "attn_q.weight").T).reshape(n, 8, D)
    q = rope(rms(q, W(p + "attn_q_norm.weight")[0]), pos, 1e4 if swa[il] else 1e6, None if swa[il] else ff)
    if il < KVL:
        k = (h @ W(p + "attn_k.weight").T).reshape(n, 1, D)
        v = (h @ W(p + "attn_v.weight").T).reshape(n, 1, D)
        k = rope(rms(k, W(p + "attn_k_norm.weight")[0]), pos, 1e4 if swa[il] else 1e6, None if swa[il] else ff)
        kv[il] = (k[:, 0], rms(v)[:, 0])
    K_, V_ = kv[il if il < KVL else KVL - (2 if swa[il] else 1)]
    mask = mask_full & ((np.arange(n)[:, None] - np.arange(n)[None, :]) < win) if swa[il] else mask_full
    sc = np.einsum("nhd,md->hnm", q, K_)
    sc = np.where(mask[None], sc, -np.inf)
    sc = np.exp(sc - sc.max(-1, keepdims=True))
    a = np.einsum("hnm,md->nhd", sc / sc.sum(-1, keepdims=True), V_).reshape(n, 8 * D)
    x = x + rms(a @ W(p + "attn_output.weight").T, W(p + "post_attention_norm.weight")[0])
    dump.append(x[-1].copy())
    h = rms(x, W(p + "ffn_norm.weight")[0])
    f = (gelu(h @ W(p + "ffn_gate.weight").T) * (h @ W(p + "ffn_up.weight").T)) @ W(p + "ffn_down.weight").T
    x = x + rms(f, W(p + "post_ffw_norm.weight")[0])
    dump.append(x[-1].copy())
    g = gelu(x @ W(p + "inp_gate.weight").T) * ple[:, il]
    x = (x + rms(g @ W(p + "proj.weight").T, W(p + "post_norm.weight")[0])) * W(p + "layer_output_scale.weight")[0, 0]
    dump.append(x[-1].copy())
hN = rms(x[-1:], W("output_norm.weight")[0])
best, arg = -np.inf, -1
for c0 in range(0, TS["token_embd.weight"][0][1], 32768):
    lg = (rows("token_embd.weight", np.arange(c0, min(c0 + 32768, TS["token_embd.weight"][0][1]))) @ hN[0])
    j = int(lg.argmax())
    if lg[j] > best:
        best, arg = lg[j], c0 + j
print(json.dumps({"next": arg, "today": ref["ref"][int(sys.argv[2])][0]}))
if len(sys.argv) > 3:
    np.savez(sys.argv[3], *dump)
