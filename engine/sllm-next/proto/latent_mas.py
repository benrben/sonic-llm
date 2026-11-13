"""LatentMAS on sllm-server --engine next: GSM8K, sequential planner -> critic -> refiner -> solver, text vs latent.

The prompts are LatentMAS's own (github.com/Gen-Verse/LatentMAS, prompts.py, sequential, gsm8k), copied verbatim. Text: each
agent a fresh completion whose prompt carries the earlier agents' text (their TextMAS). Latent: one agent whose turns are
the four prompts; the first three are silent (`latent_steps`), only the solver writes. Greedy (temperature 0) for both.

  python3 latent_mas.py [Q questions] [N at once] [M latent steps] [methods: latent,stack,text]
Env: SLLM_BIN, MODEL, CTX (8192), MAXNEW (2048 per agent), OFFSET (first question), GSM (test.jsonl path), PORT.
"""
import json, os, re, subprocess, sys, tempfile, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

Q = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
M = int(sys.argv[3]) if len(sys.argv) > 3 else 40
METHODS = (sys.argv[4] if len(sys.argv) > 4 else "latent,text").split(",")
BIN = os.environ.get("SLLM_BIN", "/Users/benreich/Projects/crazyidea/sllm-wt/shared-target/release/sllm-server")
MODEL = os.environ.get("MODEL", "/Users/benreich/Projects/crazyidea/models/gguf/Qwen3-4B-Q4_0.gguf")
CTX = int(os.environ.get("CTX", "8192"))
MAXNEW = int(os.environ.get("MAXNEW", "2048"))
OFFSET = int(os.environ.get("OFFSET", "0"))
GSM = os.environ.get("GSM", "/private/tmp/claude-501/-Users-benreich-Projects-crazyidea/19a4af43-be72-48ea-a293-2bbc175b4830/scratchpad/gsm8k/test.jsonl")
PORT = int(os.environ.get("PORT", "8791"))
OUT = os.environ.get("OUT", "")
TEXTCTX = os.environ.get("TEXTCTX", "raw")
SYS = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def http(method, path, body=None, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        t = r.read()
        return json.loads(t) if t else {}


def chat(user):  # Qwen3's template, add_generation_prompt (thinking on: no think tags added)
    return f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def ids(text):
    return http("POST", "/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]


# ---- LatentMAS prompts (sequential, gsm8k), verbatim ----------------------------------------------------------------
def latent_prompt(role, q):
    if role == "planner":
        return f"""You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

Question: {q}

Your outlined plan should be concise with a few bulletpoints for each step. Do not produce the final answer.
Now output your plan to solve the question below:
"""
    if role == "critic":
        return f"""
Question: {q}

You are a Critic Agent to evaluate the correctness of the input plan for the given question and provide helpful feedback for improving the plan.
The plan information is provided in latent KV representation format. Review the plan and question and output:
(1) original plan contents
(2) constructive feedback on the original plan.

Format your response as follows:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:
"""
    if role == "refiner":
        return f"""
Question: {q}

You are a Refiner Agent to provide a refined step-by-step plan for solving the given question.
You are provided with:
(1) latent-format information: a previous plan with feedback
(2) text-format information: the input question you need to solve.

Based on the input, write a refined and improved plan to solve the question. Make sure your output plan is correct and concise.

Now, output your refined plan below:
"""
    return f"""
Target Question: {q}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve.

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""


def text_prompt(role, q, ctx):
    ctx = ctx[:-1]  # their text_mas_context_length default -1: context[:-1]
    if role == "planner":
        return f"""
You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

## Input Question:
{q}

Your outlined plan should be concise with a few bullet points for each step. Do not produce the final answer.

## Format your response as follows:
Planner Agent's Output:
[Your detailed plan here]

Now output your plan to solve the question below:
"""
    if role == "critic":
        return f"""
You are a Critic Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan in text format.

Your job is to carefully evaluate the correctness and completeness of the plan and provide helpful feedback.

## Input Question:
{q}

## Plan from Planner Agent:
{ctx}

## Format your response as follows:
Critic Agent's Output:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:
"""
    if role == "refiner":
        return f"""
You are a Refiner Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan together with Critic Agent's feedback in text format.

Your job is to incorporate the feedback and produce an improved, refined step-by-step plan.

## Input Question:
{q}

## Original Plan and Critic Feedback:
{ctx}

## Format your response as follows:
Refiner Agent's Output:
[Your refined and improved plan here]

Make sure your output plan is logically correct, concise, and sufficient to guide final problem solving.
Now, output your refined plan below:
"""
    return f"""
Target Question: {q}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""


ROLES = ["planner", "critic", "refiner", "judger"]
NAMES = {"planner": "Planner", "critic": "Critic", "refiner": "Refiner", "judger": "Judger"}


def answer(text):  # their extract_gsm8k_answer
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        m = re.search(r"[-+]?\d+(?:\.\d+)?", boxes[-1].replace(",", ""))
        return m.group(0) if m else boxes[-1].strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def same(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def gen(prompt_ids, n):
    r = http("POST", "/completion", {"prompt": prompt_ids, "n_predict": n, "temperature": 0, "stream": False, "lane": "background", "return_tokens": True})
    return r.get("content", ""), len(r.get("tokens") or [])


def run_text(q):
    ctx, toks, t0 = "", 0, time.time()
    for role in ROLES:
        out, n = gen(ids(chat(text_prompt(role, q, ctx))), MAXNEW)
        toks += n
        if role != "judger":
            # their TextMAS passes the whole reply; with long thinking (hard tasks) that overflows the window, so TEXTCTX=clean
            # passes the answer after the thinking (the tail of the thinking when it was cut before an answer)
            if TEXTCTX == "clean":
                after = out.split("</think>")[-1].strip() if "</think>" in out else out[-2000:]
                out = after or out[-2000:]
            ctx += f"[{NAMES[role]}]:\n{out.strip()}\n\n"
    return out, toks, time.time() - t0


def run_latent(q, m=None):
    m = M if m is None else m  # m 0: the three prompts only read (no silent steps), the control
    t0 = time.time()
    a = http("POST", "/v1/agents", {"lane": "background", "system_ids": []})
    aid = a.get("id") or a.get("agent_id")
    try:
        n_in = 0
        for role in ROLES[:3]:
            p_ = ids(chat(latent_prompt(role, q)))
            n_in += len(p_)
            body = {"latent_steps": m} if m > 0 else {"n_predict": 0}
            r = http("POST", f"/v1/agents/{aid}/turn", {"prompt": p_, **body, "temperature": 0, "stream": False, "return_tokens": True})
            assert not (r.get("tokens") or []), r
        ctx = http("GET", f"/v1/agents/{aid}").get("context_tokens")
        assert ctx == n_in + 3 * m, (ctx, n_in, m)  # the silent cells are there
        r = http("POST", f"/v1/agents/{aid}/turn", {"prompt": ids(chat(latent_prompt("judger", q))), "n_predict": MAXNEW, "temperature": 0, "stream": False, "return_tokens": True})
        return r.get("content", ""), len(r.get("tokens") or []), time.time() - t0
    finally:
        http("DELETE", f"/v1/agents/{aid}")


def main():
    items = [json.loads(l) for l in open(GSM)][OFFSET:OFFSET + Q]
    for it in items:
        it["gold"] = it["answer"].split("####")[-1].strip().replace(",", "")
    tmp = tempfile.mkdtemp()
    log = open(os.path.join(tmp, "srv.log"), "w")
    p = subprocess.Popen([BIN, "--model", MODEL, "--port", str(PORT), "--host", "127.0.0.1", "--engine", "next", "--ctx", str(CTX),
                          "--kv-pages", str(N * CTX // 256 + 4), "--prompt-cache", "off", "--agent-dir", tmp + "/a", "--spill-dir", tmp + "/s"],
                         stdout=log, stderr=subprocess.STDOUT)
    res = {"model": os.path.basename(MODEL), "questions": Q, "offset": OFFSET, "at_once": N, "latent_steps": M, "max_new": MAXNEW, "log": log.name}
    try:
        for _ in range(900):
            try:
                http("GET", "/health", timeout=2)
                break
            except Exception:
                time.sleep(0.2)
        for meth in METHODS:
            f = run_latent if meth == "latent" else (lambda q: run_latent(q, 0)) if meth == "stack" else run_text
            t0 = time.time()
            with ThreadPoolExecutor(N) as ex:
                outs = list(ex.map(f, [it["question"] for it in items]))
            wall = time.time() - t0
            preds = [answer(o[0]) for o in outs]
            ok = sum(same(p_, it["gold"]) for p_, it in zip(preds, items))
            res[meth] = {"correct": ok, "acc": round(ok / Q, 3), "wall_s": round(wall, 1), "tokens_written": sum(o[1] for o in outs),
                         "mean_question_s": round(sum(o[2] for o in outs) / Q, 1), "preds": preds}
            print(meth, json.dumps({k: v for k, v in res[meth].items() if k != "preds"}), flush=True)
            if OUT:
                with open(OUT + f".{meth}.jsonl", "w") as fo:
                    for it, o, p_ in zip(items, outs, preds):
                        fo.write(json.dumps({"q": it["question"], "gold": it["gold"], "pred": p_, "tokens": o[1], "out": o[0]}) + "\n")
        res["golds"] = [it["gold"] for it in items]
        try:
            res["counters"] = http("GET", "/metrics", timeout=5)
        except Exception:
            pass
        print(json.dumps({k: v for k, v in res.items() if k != "counters"}))
    finally:
        p.terminate()
        p.wait()


if __name__ == "__main__":
    main()
