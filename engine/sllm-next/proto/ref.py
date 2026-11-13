"""Reference for the prototype: 32 real chat prompts (16 stories, 16 questions) tokenized by today's engine, and today's
engine's greedy answer to each, alone (speculation off). usage: ref.py OUT.json N_OUT [VERSIONS]
VERSIONS > 1 adds "(version k)" copies, 32 distinct prompts per version."""
import os
import sys
import tempfile

sys.path.insert(0, "/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/rollover")
from common import *  # noqa: E402,F403

OUT, N_OUT = sys.argv[1], int(sys.argv[2])
VERSIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
FR = json.load(open(os.environ.get("FRAMES", "/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/limits/frames.json")))[os.environ.get("MODELKEY", "qwen")]
FR["file"] = os.environ.get("MODELFILE", FR["file"])
TOPICS = ["a lighthouse keeper", "a lost robot", "a deep sea diver", "a mountain village", "a clockmaker", "a desert caravan", "a river pilot",
          "a space station cook", "a night train", "a beekeeper", "an old map", "a storm at sea", "a chess master", "a small bakery",
          "a volcano island", "a spaceship crew"]
QUESTIONS = ["What is a KV cache, in three sentences?", "Explain what a GPU does, briefly.", "Why is the sky blue?", "What makes a good code review?",
             "How does a hash map work?", "What is the difference between TCP and UDP?", "Why do batteries wear out?", "How do vaccines work?",
             "What is compound interest?", "Why do leaves change colour?", "How does a compiler work?", "What is entropy?",
             "Why do we sleep?", "How do airplanes fly?", "What is a black hole?", "How does public key encryption work?"]

tmp = tempfile.mkdtemp()
p = start(MODELS + FR["file"], ["--host", "127.0.0.1", "--ctx", "1024", "--kv-pages", "64", "--prompt-cache", "off", "--spec", "off",
                                "--agent-dir", tmp + "/a", "--spill-dir", tmp + "/s"], OUT + ".srv.log")
try:
    a, b = FR["task"].split("@@MSG@@")
    tk = lambda text: post("/tokenize", {"content": text, "add_special": FR["bos"], "parse_special": True})["tokens"]
    prompts = []
    for v in range(VERSIONS):  # distinct prompts, so nothing is shared between agents
        tag = f" (version {v + 1})" if v else ""
        prompts += [tk(FR["prefix"]["chat"] + a + f"Write a long, detailed story about {t}{tag}." + b) for t in TOPICS]
        prompts += [tk(FR["prefix"]["chat"] + a + q[:-1] + tag + q[-1] + b) for q in QUESTIONS]
    ref = []
    for q in prompts:
        r = post("/completion", {"prompt": q, "n_predict": N_OUT, "temperature": 0, "return_tokens": True, "cache_prompt": False, "ignore_eos": True})
        ref.append(r["tokens"])
    json.dump({"prompts": prompts, "ref": ref, "n_out": N_OUT}, open(OUT, "w"))
    print("prompts", len(prompts), "lengths", min(map(len, prompts)), max(map(len, prompts)), "ref tokens", [len(x) for x in ref][:4])
finally:
    stop(p)
