#!/usr/bin/env bash
# verify-lib-bundle.sh — prove a packaged libcrispasr bundle actually LOADS.
#
# Usage: tools/verify-lib-bundle.sh <bundle-dir>
#
# WHY THIS EXISTS
#
# release.yml already gated on "every @rpath/DT_NEEDED dependency is present in
# the bundle". That check passed for v0.8.18 and the bundles were still broken
# for every consumer, because presence is not resolvability:
#
#   macOS: LC_RPATH was the CI runner's build tree
#          (/Users/runner/work/CrispASR/CrispASR/build-libs/ggml/src), which
#          exists nowhere else, so @rpath/libggml.0.dylib resolved to nothing.
#   Linux: DT_RUNPATH was '$ORIGIN:$ORIGIN/../../ggml/src' — one level too high,
#          pointing outside the bundle. Relative, so it leaked no CI path and
#          looked correct.
#
# Both shipped. The dependency really was in the tarball; the loader simply
# could not find it. So this script does the only check that matters: relocate
# the bundle somewhere unrelated to the build tree and dlopen it, the way a
# downstream FFI consumer does.
#
# Exits non-zero on failure, so it fails the RELEASE instead of the consumer.

set -euo pipefail

BUNDLE="${1:?usage: verify-lib-bundle.sh <bundle-dir>}"
[ -d "$BUNDLE" ] || { echo "ERROR: no such bundle dir: $BUNDLE" >&2; exit 1; }

# Relocate. Loading in place can succeed by accident — a build-tree rpath still
# resolves on the machine that built it, which is exactly how this shipped.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp -R "$BUNDLE" "$TMP/relocated"
ROOT="$TMP/relocated"

echo "verifying bundle loads after relocation: $ROOT"

# Report the resolved paths first — on failure this is what tells you WHY.
for lib in "$ROOT"/src/libcrispasr.*; do
    [ -f "$lib" ] || continue
    case "$lib" in *.dylib|*.so|*.so.*) ;; *) continue ;; esac
    if command -v otool >/dev/null 2>&1; then
        echo "  LC_RPATH of $(basename "$lib"):"
        otool -l "$lib" | awk '/LC_RPATH/{f=1} f&&/ path /{print "    "$2; f=0}'
    elif command -v patchelf >/dev/null 2>&1; then
        echo "  RUNPATH of $(basename "$lib"): $(patchelf --print-rpath "$lib" 2>/dev/null || echo '(none)')"
    fi
    break
done

python3 - "$ROOT" <<'PY'
import ctypes, glob, os, sys

root = sys.argv[1]
pats = ("libcrispasr.*.dylib", "libcrispasr.so.*")
# lib/ is the flattened layout; src/ is the historical one (and, after
# package-lib-bundle.sh, a symlink to lib/). Check both so this script works on
# a bundle from either era.
cands = []
for sub in ("lib", "src"):
    for p in pats:
        cands += [f for f in glob.glob(os.path.join(root, sub, p)) if not os.path.islink(f)]
    if cands:
        break
if not cands:
    print("  ERROR: no concrete libcrispasr library found in <bundle>/lib or <bundle>/src")
    sys.exit(1)

lib = sorted(cands)[0]
try:
    h = ctypes.CDLL(lib)
except OSError as e:
    print("  ERROR: dlopen FAILED on the packaged bundle:")
    for line in str(e).splitlines()[:4]:
        print("   ", line[:300])
    print("  Every downstream FFI consumer would hit exactly this.")
    sys.exit(1)

# A handle alone does not prove a usable ABI — lazy binding can defer the
# failure to first call. Resolve real symbols now.
required = ["crispasr_session_open_explicit", "crispasr_session_close"]
missing = [s for s in required if not hasattr(h, s)]
if missing:
    print("  ERROR: loaded, but these symbols do not resolve:", ", ".join(missing))
    sys.exit(1)

print(f"  OK: {os.path.basename(lib)} dlopens after relocation; session ABI resolves")
PY
