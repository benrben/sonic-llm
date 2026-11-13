#!/bin/sh
# Developer helper: regenerate patches/0010-row-exact.patch from apply_edits.py against the pristine b11669 tree,
# then rebuild the dylib. Usage: mkpatch.sh [path-to-research-inference_runtime-vendor-dir]
# patches/0011-sllm-copy-rows.patch is not made here: it is kept by hand and applies on top of 0010 (build.sh applies it
# last), so the build at the end of this script checks that it still applies to the regenerated tree.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
vendor_src=${1:-/Users/benreich/Projects/crazyidea/research/inference_runtime/vendor}
tmp=$(mktemp -d)
mkdir "$tmp/a" "$tmp/b"
tar xzf "$vendor_src/llama-source.tar.gz" -C "$tmp/a" --strip-components=1 ggml-org-llama.cpp-620d222/ggml
tar xzf "$vendor_src/llama-source.tar.gz" -C "$tmp/b" --strip-components=1 ggml-org-llama.cpp-620d222/ggml
# the patches before the row-exact one (qmm, 0001-0003) go into both trees, so 0010-row-exact.patch holds only apply_edits.py
for p in "$here"/patches/000*.patch; do
    patch -s -p1 -d "$tmp/a/ggml" < "$p"
    patch -s -p1 -d "$tmp/b/ggml" < "$p"
done
python3 "$here/apply_edits.py" "$tmp/b/ggml"
mkdir -p "$here/patches"
(cd "$tmp" && mv a/ggml a2 && mv b/ggml b2 && rmdir a b && mv a2 a && mv b2 b && diff -ruN a b) > "$here/patches/0010-row-exact.patch" || true
# no timestamps in the patch: the same bytes whenever it is made
sed -i '' -E 's/^(---|\+\+\+) ([^	]*)	.*$/\1 \2/' "$here/patches/0010-row-exact.patch"
rm -rf "$tmp"
"$here/build.sh" "$vendor_src"
