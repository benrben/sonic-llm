#!/bin/sh
# Put the official llama.cpp b11669 ggml and libllama dylibs into ./lib and verify them against SHA256SUMS.
#
# The dylibs are large binaries and are not in git; only the checksums and the headers are. With no argument the
# official release archive is downloaded from GitHub. With an argument, the dylibs are copied from that directory
# (an unpacked llama-b11669-bin-macos-arm64 release).
#
# Usage: fetch.sh [directory-with-the-release-dylibs]
set -eu

here=$(cd "$(dirname "$0")" && pwd)
url="https://github.com/ggml-org/llama.cpp/releases/download/b11669/llama-b11669-bin-macos-arm64.tar.gz"
tmp=""
if [ $# -ge 1 ]; then
    src=$1
else
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    echo "fetch.sh: downloading $url"
    curl -fL --retry 3 -o "$tmp/llama.tar.gz" "$url"
    tar -xzf "$tmp/llama.tar.gz" -C "$tmp"
    src=$(dirname "$(find "$tmp" -name 'libggml-base.0.27.0.dylib' | head -1)")
fi
[ -f "$src/libggml-base.0.27.0.dylib" ] || { echo "fetch.sh: no ggml dylibs in $src" >&2; exit 1; }

mkdir -p "$here/lib"
for name in ggml-base ggml ggml-cpu ggml-metal ggml-blas ggml-rpc; do
    # cp -P keeps the symlink chain lib<name>.dylib -> .0.dylib -> .0.27.0.dylib
    cp -P "$src"/lib${name}.0.27.0.dylib "$src"/lib${name}.0.dylib "$src"/lib${name}.dylib "$here/lib/"
done
cp -P "$src"/libllama.0.7.0.dylib "$src"/libllama.0.dylib "$src"/libllama.dylib "$here/lib/"

cd "$here"
shasum -a 256 -c SHA256SUMS
echo "ggml b11669 + libllama dylibs in $here/lib verified"
