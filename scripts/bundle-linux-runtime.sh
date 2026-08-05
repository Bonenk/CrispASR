#!/usr/bin/env bash
# bundle-linux-runtime.sh — make a staged Linux release directory self-contained.
#
# Two jobs, in this order:
#
#   1. Rewrite RUNPATH to `$ORIGIN` on every ELF in the directory.
#   2. Copy in every non-system shared library the binaries still need.
#
# Both are required, and step 1 is the one that was missing. The published
# v0.8.25 artifact carried:
#
#     crispasr           RUNPATH=$ORIGIN:/home/runner/work/CrispASR/.../c2pa/lib
#     crispasr-quantize  RUNPATH=/home/runner/work/CrispASR/.../c2pa/lib
#
# — a build-machine path that exists on no user's disk, and for
# `crispasr-quantize` no `$ORIGIN` at all, so it could not find the
# `libc2pa_c.so` sitting right beside it. Bundling a library the loader has no
# way to look for buys nothing, which is why this script does both.
#
# Supersedes the openblas-only `bundle-openblas.sh`: driving the copy from the
# actual `ldd` output picks up libgomp (OpenMP, shipped with gcc and absent on
# minimal installs) and anything a future flag introduces, instead of naming one
# library and rotting the moment a second appears.
#
# GPU runtimes are deliberately NOT bundled — libcuda/libcudart/libcublas,
# libamdhip64/librocblas and libvulkan belong to the host's driver or toolkit
# install. Pass them to check-bundled-deps.py with --allow so the contract is
# recorded rather than assumed.
#
# Usage: scripts/bundle-linux-runtime.sh <staged-dir>
set -euo pipefail

DEST="${1:-}"
[ -n "$DEST" ] && [ -d "$DEST" ] || { echo "bundle-linux-runtime: no dest dir '$DEST'" >&2; exit 1; }

command -v patchelf >/dev/null 2>&1 || {
    echo "bundle-linux-runtime: patchelf not installed — RUNPATH cannot be fixed" >&2; exit 1; }

is_elf() { head -c 4 "$1" 2>/dev/null | grep -q $'\x7fELF'; }

# ── 1. RUNPATH -> $ORIGIN ────────────────────────────────────────────────────
# Unconditional: an inherited build-tree RUNPATH is at best useless and at worst
# points somewhere that exists on the build machine only.
for f in "$DEST"/*; do
    [ -f "$f" ] || continue
    [ -L "$f" ] && continue
    is_elf "$f" || continue
    before=$(patchelf --print-rpath "$f" 2>/dev/null || echo "")
    patchelf --set-rpath '$ORIGIN' "$f" 2>/dev/null || {
        echo "bundle-linux-runtime: patchelf failed on $(basename "$f")" >&2; exit 1; }
    echo "  rpath  $(basename "$f"): '${before}' -> '\$ORIGIN'"
done

# ── 2. copy the non-system dependency closure ───────────────────────────────
# `ldd` prints the full transitive set already resolved, so one pass per binary
# is enough. Skip the C/C++ runtime (present everywhere) and the GPU runtimes.
copied=0
for f in "$DEST"/*; do
    [ -f "$f" ] || continue
    [ -L "$f" ] && continue
    is_elf "$f" || continue
    while read -r lib; do
        [ -n "$lib" ] || continue
        base=$(basename "$lib")
        case "$base" in
            libc.so.*|libm.so.*|libdl.so.*|libpthread.so.*|librt.so.*|libutil.so.*) continue ;;
            libgcc_s.so.*|libstdc++.so.*|libresolv.so.*|ld-linux*) continue ;;
            libcuda.so.*|libcudart.so.*|libcublas*.so.*|libnv*.so.*) continue ;;
            libamdhip64.so.*|librocblas.so.*|libhsa*.so.*|libvulkan.so.*) continue ;;
        esac
        [ -e "$DEST/$base" ] && continue
        cp -Lf "$lib" "$DEST/$base"
        patchelf --set-rpath '$ORIGIN' "$DEST/$base" 2>/dev/null || true
        echo "  bundle $base  (from $lib)"
        copied=$((copied + 1))
    done < <(ldd "$f" 2>/dev/null | awk '/=>/ {print $3}' | grep '^/' | sort -u)
done

echo "bundle-linux-runtime: $DEST — rpaths normalised, $copied librar(ies) bundled"
