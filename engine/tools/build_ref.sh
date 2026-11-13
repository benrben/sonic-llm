#!/bin/sh
# Build engine/tools/ref_llama.c against the vendored llama.cpp b11669 libllama (research/inference_runtime/vendor).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
V=/Users/benreich/Projects/crazyidea/research/inference_runtime/vendor
out=${1:-$here/ref_llama}
clang -O2 -o "$out" "$here/ref_llama.c" -I"$V/llama-src/include" -I"$V/llama-src/ggml/include" \
    -L"$V/llama-bin" -lllama -lggml -lggml-base -Wl,-rpath,"$V/llama-bin"
echo "built $out"
