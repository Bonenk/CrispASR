#!/usr/bin/env python3
"""Check that every chat export is named by each chat wrapper.

This is a source drift guard, not callable-API, signature, ownership or model
acceptance. In particular, a bare declaration can satisfy this census. Run the
language ABI contracts and full-library/platform/model gates separately.
WASM chat is opt-in and currently restricted to single-thread modules.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = os.path.join(REPO, "include", "crispasr_chat.h")

# Each binding, and the file(s) that must name every entry point. A binding
# split across files (Java: a JNA interface plus the session wrapper) may
# satisfy a symbol from any of them.
BINDINGS: dict[str, list[str]] = {
    "ruby": ["bindings/ruby/ext/ruby_crispasr_aux.h"],
    "wasm": ["bindings/javascript/session_chat.h"],
    "csharp": ["bindings/csharp/CrispASR/ChatSession.cs"],
    "python": ["python/crispasr/_binding.py"],
    "go": ["bindings/go/chat.go"],
    "rust": ["crispasr-sys/src/lib.rs"],
    "dart": ["flutter/crispasr/lib/src/chat.dart"],
    "java": [
        "bindings/java/src/main/java/io/github/ggerganov/whispercpp/chat/ChatNative.java",
        "bindings/java/src/main/java/io/github/ggerganov/whispercpp/chat/ChatLib.java",
    ],
}

# `CRISPASR_CHAT_API <return type> <name>(` — the exported functions. The
# trailing `(` is what separates them from the typedefs and the handle type,
# which are not entry points and are not expected in every binding.
DECL = re.compile(r"CRISPASR_CHAT_API\s+[A-Za-z_][A-Za-z0-9_ *]*?\b(crispasr_chat_[a-z0-9_]+)\s*\(")


def header_symbols(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    # Drop block comments so a symbol named only in prose does not count.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return sorted(set(DECL.findall(text)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the exported symbols and exit")
    args = ap.parse_args()

    if not os.path.exists(HEADER):
        print(f"error: {HEADER} not found", file=sys.stderr)
        return 2
    symbols = header_symbols(HEADER)
    if args.list:
        for s in symbols:
            print(s)
        return 0
    if not symbols:
        # A header that parses to nothing would make every binding vacuously
        # complete, which is the one way this check could pass while blind.
        print("error: no CRISPASR_CHAT_API declarations found — has the header or the regex changed?",
              file=sys.stderr)
        return 2

    gaps = 0
    for lang, rel_paths in sorted(BINDINGS.items()):
        blobs = []
        for rel in rel_paths:
            path = os.path.join(REPO, rel)
            if not os.path.exists(path):
                print(f"MISSING FILE  {lang}: {rel}", file=sys.stderr)
                gaps += 1
                continue
            with open(path, encoding="utf-8") as fh:
                blobs.append(fh.read())
        joined = "\n".join(blobs)
        missing = [s for s in symbols if s not in joined]
        if missing:
            gaps += len(missing)
            for s in missing:
                print(f"UNBOUND  {lang}: {s}  (add it to {' or '.join(rel_paths)})", file=sys.stderr)
        else:
            print(f"ok  {lang:6s} names all {len(symbols)} entry points")

    if gaps:
        print(f"\n{gaps} gap(s): a chat ABI entry point is not bound everywhere.", file=sys.stderr)
        return 1
    print(f"\nRESULT: PASS — {len(symbols)} entry points named by {len(BINDINGS)} wrappers (source census only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
