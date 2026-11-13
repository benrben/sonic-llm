#!/usr/bin/env python3
"""A stand-in for sllm-server that the browser check starts in place of it. It takes the command line the playground builds
(--model, --port, --ctx, ...) and runs the real binary named by SONIC_REAL_SERVER_BIN in its `--fake` mode, which needs no model
and no GPU. `--print-profile` (the memory plan) is answered here."""

import json
import os
import sys

args = sys.argv[1:]
if "--print-profile" in args:
    print(json.dumps({"budget": {"kv_alloc_bytes": 4 * 100_000, "kv_cells": {"allocated": 4, "max_fitting": 262144}}}))
    sys.exit(0)


def value(flag: str) -> str:
    return args[args.index(flag) + 1]


real = os.environ["SONIC_REAL_SERVER_BIN"]
os.execv(real, [real, "--fake", "--host", value("--host"), "--port", value("--port"), "--fake-token-ms", "4", "--fake-prefill-ms", "5"])
