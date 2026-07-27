#!/usr/bin/env python3
"""Stage a prebuilt libcrispasr bundle into the `crispasr` package for wheel
building.

The release workflow (`release.yml`) already builds relocatable
`libcrispasr-<platform>[-cuda|-vulkan].tar.gz` bundles whose libraries have
their rpaths rewritten to `$ORIGIN` / `@loader_path` (see
`tools/package-lib-bundle.sh`). This script copies those libraries next to
`crispasr/_binding.py` — where `_find_lib()` probes first — and best-effort
compiles the tiny `_helpers.c` shim, so that `python -m build` produces a
self-contained platform wheel.

Usage:
    python tools/stage_libs.py --bundle <extracted-bundle-dir> [--pkg crispasr]

Symlinks are materialised as real files: the wheel (a zip) cannot be relied on
to preserve symlinks across pip versions, and soname-based inter-library
resolution needs the versioned file to exist as real bytes.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

LIB_GLOBS = ("lib*.so", "lib*.so.*", "*.so", "*.dylib", "*.dll")


def find_lib_dir(bundle: Path) -> Path:
    for sub in ("lib", "src", "."):
        d = bundle / sub if sub != "." else bundle
        if d.is_dir() and any(_is_lib(p) for p in d.iterdir()):
            return d
    raise SystemExit(f"stage_libs: no shared libraries found under {bundle}")


def _is_lib(p: Path) -> bool:
    n = p.name
    return (
        n.endswith(".dylib")
        or n.endswith(".dll")
        or n.endswith(".so")
        or ".so." in n
    )


def copy_libs(lib_dir: Path, pkg: Path) -> list[str]:
    copied = []
    for entry in sorted(lib_dir.iterdir()):
        if not _is_lib(entry):
            continue
        # Resolve symlinks to real bytes but keep the entry's own name, so both
        # `libcrispasr.so` (what _find_lib opens) and `libcrispasr.so.1` (what
        # the ggml backends' SONAME refers to) exist as loadable real files.
        src = entry.resolve()
        if not src.is_file():
            continue
        dst = pkg / entry.name
        shutil.copyfile(src, dst)
        shutil.copymode(src, dst)
        copied.append(entry.name)
    if not copied:
        raise SystemExit(f"stage_libs: nothing copied from {lib_dir}")
    return copied


def compile_helpers(pkg: Path, include_dir: Path) -> None:
    """Compile `_helpers.c` into the package dir. Best-effort: the legacy
    whisper `CrispASR` class uses it, but the modern `Session` API does not, so
    a failure only degrades that one class — never fail the wheel over it."""
    src = pkg / "_helpers.c"
    if not src.exists():
        print("stage_libs: no _helpers.c, skipping helpers", flush=True)
        return
    if not (include_dir / "crispasr.h").exists():
        print(f"stage_libs: no crispasr.h under {include_dir}, skipping helpers",
              flush=True)
        return
    system = platform.system()
    try:
        if system == "Windows":
            # cl needs the import lib; the bundle ships crispasr.lib alongside.
            implib = next(iter(pkg.glob("crispasr.lib")), None) or next(
                iter(include_dir.parent.rglob("crispasr.lib")), None)
            if implib is None:
                print("stage_libs: no crispasr.lib, skipping Windows helpers",
                      flush=True)
                return
            out = pkg / "crispasr_helpers.dll"
            subprocess.run(
                ["cl", "/nologo", "/LD", str(src), f"/I{include_dir}",
                 str(implib), f"/Fe:{out}"],
                check=True,
            )
        else:
            ext = "dylib" if system == "Darwin" else "so"
            out = pkg / f"libcrispasr_helpers.{ext}"
            rpath = "@loader_path" if system == "Darwin" else "$ORIGIN"
            cc = os.environ.get("CC", "cc")
            subprocess.run(
                [cc, "-shared", "-fPIC", str(src), f"-I{include_dir}",
                 f"-L{pkg}", "-lcrispasr", f"-Wl,-rpath,{rpath}", "-o", str(out)],
                check=True,
            )
            if system == "Darwin":
                subprocess.run(["codesign", "--force", "--sign", "-", str(out)],
                               check=False)
        print(f"stage_libs: compiled helpers -> {out.name}", flush=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"stage_libs: WARNING helpers compile failed ({exc}); the wheel "
              "will work for the Session API but not the legacy CrispASR class",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path,
                    help="extracted libcrispasr-<platform> bundle directory")
    ap.add_argument("--pkg", type=Path, default=Path(__file__).parent.parent / "crispasr",
                    help="the crispasr package directory to stage into")
    args = ap.parse_args()

    bundle = args.bundle
    if not bundle.is_dir():
        raise SystemExit(f"stage_libs: bundle not found: {bundle}")
    pkg = args.pkg
    if not (pkg / "_binding.py").exists():
        raise SystemExit(f"stage_libs: {pkg} is not the crispasr package")

    lib_dir = find_lib_dir(bundle)
    copied = copy_libs(lib_dir, pkg)
    print(f"stage_libs: staged {len(copied)} libs from {lib_dir}:", flush=True)
    for n in copied:
        print(f"  {n}", flush=True)

    include_dir = bundle / "include"
    compile_helpers(pkg, include_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
