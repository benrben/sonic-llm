#!/bin/sh
# Build the row-exact libggml-metal: the OFFICIAL llama.cpp b11669 Metal backend (tarball 620d222) plus patches/ (the qmm
# patches 0001-0003, the row-exact patch 0010 and the copy-rows patch 0011; with every SLLM switch off every decision is
# the official one).
#
# No cmake on this machine, so this script mirrors what ggml/src/ggml-metal/CMakeLists.txt does for a release build
# with GGML_METAL_EMBED_LIBRARY=ON (the official macOS build):
#   1. extract ggml/ from the pinned source tarball into a fresh build/src (sha256 checked);
#   2. apply patches/*.patch in order (none of them touches anything but ggml/src/ggml-metal);
#   3. embed every kernels/*.metal the way the CMake custom command does (prepend common.h [+ dequantize.h ...],
#      drop the internal #include/#pragma once lines, inline ggml-common.h and ggml-metal-impl.h, emit an .s with
#      .incbin and the _ggml_metallib_<kind>_{start,end} symbols);
#   4. compile the ggml-metal sources with the release flags (-O3 -DNDEBUG, the GGML_* defines of a shared build);
#   5. link libggml-metal.0.27.0.dylib (install name @rpath/libggml-metal.0.dylib, rpath @loader_path) against the
#      OFFICIAL libggml-base;
#   6. lay out lib/: the patched libggml-metal plus byte-identical copies of the official ggml-base, ggml, ggml-cpu,
#      ggml-blas, ggml-rpc and libllama (sha256 checked against ../ggml-b11669/SHA256SUMS). Copies, not symlinks:
#      libggml loads its backends through @loader_path, so the whole set must sit in one directory for the patched
#      metal backend to be the one that libggml picks up;
#   7. write MANIFEST.json (source commit + tarball sha256, patch sha256s, flags, output sha256s).
#
# The official dylibs in ../ggml-b11669/lib are only read, never modified. llama.cpp baselines keep using them.
#
# Usage: build.sh [directory-with-llama-source.tar.gz]
set -eu

here=$(cd "$(dirname "$0")" && pwd)
official="$here/../ggml-b11669"
# With an argument: the directory that holds llama-source.tar.gz. Without: the pinned source archive is downloaded
# from GitHub into ./download (its sha256 is checked below either way).
if [ $# -ge 1 ]; then
    tarball="$1/llama-source.tar.gz"
else
    tarball="$here/download/llama-source.tar.gz"
    if [ ! -f "$tarball" ]; then
        mkdir -p "$here/download"
        echo "build.sh: downloading the llama.cpp source at 620d222"
        curl -fL --retry 3 -o "$tarball" "https://api.github.com/repos/ggml-org/llama.cpp/tarball/620d222"
    fi
fi
TARBALL_SHA256=da36b3375b80d9d237e3787ff04fa72887bce2c8eefb717023205e4e6bbdeaad
SRC_COMMIT=620d222
TOP="ggml-org-llama.cpp-$SRC_COMMIT"

[ -f "$tarball" ] || { echo "build.sh: $tarball not found" >&2; exit 1; }
[ -f "$official/lib/libggml-base.0.27.0.dylib" ] || { echo "build.sh: run $official/fetch.sh first" >&2; exit 1; }
got=$(shasum -a 256 "$tarball" | cut -d' ' -f1)
[ "$got" = "$TARBALL_SHA256" ] || { echo "build.sh: tarball sha256 $got != $TARBALL_SHA256" >&2; exit 1; }
(cd "$official" && shasum -a 256 -c SHA256SUMS >/dev/null) || { echo "build.sh: official dylibs fail SHA256SUMS" >&2; exit 1; }

B="$here/build"
rm -rf "$B"
mkdir -p "$B/src" "$B/obj" "$B/embed"
tar xzf "$tarball" -C "$B/src" "$TOP/ggml"
G="$B/src/$TOP/ggml"
M="$G/src/ggml-metal"

# 2. patches
for p in "$here"/patches/*.patch; do
    [ -f "$p" ] || continue
    echo "applying $(basename "$p")"
    patch -s -p1 -d "$G" < "$p"
done
# SLLM_METAL_PROFILE=1: also the profiling patches (patches-profile/: per-step GPU time of the command buffers,
# exported as ggml_backend_metal_sllm_gpu_times). Off by default, so the default build stays the same bytes.
if [ "${SLLM_METAL_PROFILE:-0}" = 1 ]; then
    for p in "$here"/patches-profile/*.patch; do
        echo "applying $(basename "$p") (profile)"
        patch -s -p1 -d "$G" < "$p"
    done
fi

MINOS=13.3
objs=""

# 3. embedded kernel sources: the METALLIB_KERNEL_SOURCES list of CMakeLists.txt (after patching)
kernels=$(sed -n '/^set(METALLIB_KERNEL_SOURCES/,/^)/p' "$M/CMakeLists.txt" | grep 'kernels/.*\.metal' | sed 's|.*kernels/||; s|\.metal.*||')
[ -n "$kernels" ] || { echo "build.sh: no kernel sources found in CMakeLists.txt" >&2; exit 1; }
for kind in $kernels; do
    src="$M/kernels/$kind.metal"
    hdrs="$M/kernels/common.h"
    grep -q '#include "dequantize\.h"' "$src" && hdrs="$hdrs $M/kernels/dequantize.h"
    grep -q '#include "quantize\.h"' "$src" && hdrs="$hdrs $M/kernels/quantize.h"
    grep -q '#include "fa_common\.metal"' "$src" && hdrs="$hdrs $M/kernels/fa_common.metal"
    grep -q '#include "fa_vec_common\.metal"' "$src" && hdrs="$hdrs $M/kernels/fa_vec_common.metal"
    e="$B/embed/ggml-metal-embed-$kind.metal"
    # shellcheck disable=SC2086
    cat $hdrs "$src" > "$e.tmp1"
    sed -e '/#include "common.h"/d' -e '/#include "dequantize.h"/d' -e '/#include "quantize.h"/d' \
        -e '/#include "fa_common.metal"/d' -e '/#include "fa_vec_common.metal"/d' -e '/#pragma once/d' < "$e.tmp1" > "$e.tmp2"
    sed -e "/__embed_ggml-common.h__/r $G/src/ggml-common.h" -e '/__embed_ggml-common.h__/d' < "$e.tmp2" > "$e.tmp3"
    sed -e "/#include \"ggml-metal-impl.h\"/r $M/ggml-metal-impl.h" -e '/#include "ggml-metal-impl.h"/d' < "$e.tmp3" > "$e"
    sym=$(echo "$kind" | tr '-' '_')
    {
        echo ".section __DATA,__ggml_metallib"
        echo ".globl _ggml_metallib_${sym}_start"
        echo "_ggml_metallib_${sym}_start:"
        echo ".incbin \"$e\""
        echo ".globl _ggml_metallib_${sym}_end"
        echo "_ggml_metallib_${sym}_end:"
    } > "$B/embed/ggml-metal-embed-$kind.s"
    clang -c -arch arm64 -mmacosx-version-min=$MINOS "$B/embed/ggml-metal-embed-$kind.s" -o "$B/obj/embed-$kind.o"
done

# 4. compile. The defines are those of a shared release build of ggml-metal (ggml/src/CMakeLists.txt:
#    ggml_add_backend_library, GGML_SCHED_MAX_COPIES, _DARWIN_C_SOURCE / _XOPEN_SOURCE on Darwin).
DEFS="-DNDEBUG -DGGML_METAL_EMBED_LIBRARY -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_BUILD \
-DGGML_SCHED_MAX_COPIES=4 -D_XOPEN_SOURCE=600 -D_DARWIN_C_SOURCE"
INC="-I$G/include -I$G/src -I$M"
# The checkout path must not reach the binary (ggml logs __FILE__): map it away so every checkout builds the same bytes.
PREFIX_MAP="-ffile-prefix-map=$B/src/="
COMMON="-arch arm64 -mmacosx-version-min=$MINOS -O3 -fPIC $PREFIX_MAP $DEFS $INC"
CFLAGS="$COMMON -std=gnu11"
CXXFLAGS="$COMMON -std=gnu++17"
srcs=$(sed -n '/^ggml_add_backend_library(ggml-metal/,/)/p' "$M/CMakeLists.txt" | grep -E '\.(m|cpp)' | tr -d ' ')
for s in $srcs; do
    case "$s" in
        *.cpp) clang++ $CXXFLAGS -c "$M/$s" -o "$B/obj/$s.o" ;;
        *.m)   clang $CFLAGS -c "$M/$s" -o "$B/obj/$s.o" ;;
    esac
    objs="$objs $B/obj/$s.o"
done
# the embedded sources go in the CMake list order (the section layout of the official binary)
for kind in $kernels; do objs="$objs $B/obj/embed-$kind.o"; done

# 5. link against the official libggml-base
mkdir -p "$here/lib"
out="$here/lib/libggml-metal.0.27.0.dylib"
rm -f "$out"
clang++ -arch arm64 -mmacosx-version-min=$MINOS -dynamiclib -o "$out" $objs \
    -install_name @rpath/libggml-metal.0.dylib -compatibility_version 0 -current_version 0.27.0 \
    -Wl,-rpath,@loader_path -L"$official/lib" -lggml-base \
    -framework Foundation -framework Metal -framework MetalKit
ln -sf libggml-metal.0.27.0.dylib "$here/lib/libggml-metal.0.dylib"
ln -sf libggml-metal.0.dylib "$here/lib/libggml-metal.dylib"

# 6. the official set beside it (byte-identical copies)
for name in ggml-base ggml ggml-cpu ggml-blas ggml-rpc; do
    cp -f "$official/lib/lib$name.0.27.0.dylib" "$here/lib/"
    ln -sf "lib$name.0.27.0.dylib" "$here/lib/lib$name.0.dylib"
    ln -sf "lib$name.0.dylib" "$here/lib/lib$name.dylib"
done
cp -f "$official/lib/libllama.0.7.0.dylib" "$here/lib/"
ln -sf libllama.0.7.0.dylib "$here/lib/libllama.0.dylib"
ln -sf libllama.0.dylib "$here/lib/libllama.dylib"
(cd "$here" && grep ' lib/' "$official/SHA256SUMS" | grep -v libggml-metal | shasum -a 256 -c - >/dev/null) \
    || { echo "build.sh: copied official dylibs do not match SHA256SUMS" >&2; exit 1; }

# the patch changes host code only: every embedded kernel source must be byte-identical to the official dylib's
python3 "$here/verify_embed.py" "$official/lib/libggml-metal.0.27.0.dylib" "$out"

# 7. manifest
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
{
    echo "{"
    echo "  \"what\": \"libggml-metal 0.27.0 from llama.cpp b11669 + the row-exact patch; every other dylib is the official b11669 binary\","
    echo "  \"source_commit\": \"$SRC_COMMIT\","
    echo "  \"source_tarball\": \"research/inference_runtime/vendor/llama-source.tar.gz\","
    echo "  \"source_tarball_sha256\": \"$TARBALL_SHA256\","
    echo "  \"patches\": ["
    first=1
    for p in "$here"/patches/*.patch; do
        [ -f "$p" ] || continue
        [ $first = 1 ] || echo ","
        printf '    {"file": "patches/%s", "sha256": "%s"}' "$(basename "$p")" "$(sha "$p")"
        first=0
    done
    echo ""
    echo "  ],"
    echo "  \"profile_patches\": ${SLLM_METAL_PROFILE:-0},"
    echo "  \"build_sh_sha256\": \"$(sha "$here/build.sh")\","
    echo "  \"compiler\": \"$(clang --version | head -1)\","
    echo "  \"sdk\": \"$(xcrun --show-sdk-version 2>/dev/null || echo unknown)\","
    echo "  \"flags\": \"-arch arm64 -mmacosx-version-min=$MINOS -O3 -fPIC -ffile-prefix-map=<build>/src/= $DEFS (-std=gnu11 for .m, -std=gnu++17 for .cpp; -I ggml/include ggml/src ggml/src/ggml-metal)\","
    echo "  \"embedded_kernels\": \"$(echo $kernels)\","
    echo "  \"outputs\": {"
    echo "    \"lib/libggml-metal.0.27.0.dylib\": \"$(sha "$out")\""
    echo "  },"
    echo "  \"official_copies_verified_against\": \"engine/vendor/ggml-b11669/SHA256SUMS\""
    echo "}"
} > "$here/MANIFEST.json"
echo "built $out"
echo "sha256 $(sha "$out")"
