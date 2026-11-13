"""32 distinct short prompts in a model's own tokens for srv_test.py (many agents at once; "ref" left empty, so its
solo_equals_today reads 0/N). usage: mkmulti.py OUT.json [MODELFILE env]"""
import json
import os
import sys
import tempfile

sys.path.insert(0, "/Users/benreich/Projects/crazyidea/sllm-wt/rollover-20261101/results/rollover")
from common import *  # noqa: E402,F403

MODEL = MODELS + os.environ["MODELFILE"]
BIN = "/Users/benreich/Projects/crazyidea/sllm-wt/shared-target/release/sllm-server"
TOPICS = ["a lighthouse keeper", "a lost robot", "a desert caravan", "an old violin", "a city under the sea", "a talking cat",
          "the last train home", "a forgotten garden", "a mountain rescue", "a clockmaker's apprentice", "a comet over the village",
          "a library at midnight", "two rival chefs", "a ship in a bottle", "the first snow", "a map with no names",
          "a beekeeper's secret", "a radio from the future", "a bridge builder", "a night market", "a broken compass",
          "a painter who lost her colors", "an island of mirrors", "a letter never sent", "a farmer and a storm",
          "a museum guard", "a quiet astronaut", "a game of chess in the park", "a fox in the city", "a paper boat",
          "the oldest tree in the forest", "a train station piano"]
tmp = tempfile.mkdtemp()
p = subprocess.Popen([BIN, "--model", MODEL, "--port", str(PORT), "--host", "127.0.0.1", "--ctx", "2048", "--agent-dir", tmp + "/a"],
                     stdout=open("/tmp/mkmulti.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(600):
        try:
            get("/health")
            break
        except Exception:
            time.sleep(0.3)
    prompts = [post("/tokenize", {"content": f"Write a short story about {t}. Give it a title, then tell it in a few paragraphs.\n\n", "add_special": True})["tokens"] for t in TOPICS]
    json.dump({"prompts": prompts, "ref": [[] for _ in prompts]}, open(sys.argv[1], "w"))
    print(len(prompts), min(map(len, prompts)), max(map(len, prompts)))
finally:
    stop(p)
