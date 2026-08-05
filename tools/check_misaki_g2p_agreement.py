#!/usr/bin/env python3
"""Measure CrispASR's English G2P against misaki, Kokoro's own G2P (#316).

Kokoro-82M was trained on misaki's output, so misaki IS the reference: any
phoneme string we hand the model that misaki would not have produced is a token
sequence it never saw in training. This runs both over the same running prose
and reports how often they agree.

It compiles a ~40-line dumper against `src/core/g2p_en.h` with one `c++` call —
no CMake target, no library, nothing to keep in sync — and drives misaki
through its Python package. That means it measures the SAME code path the
product uses (`g2p_en::text_to_ipa` + `core_phoneme::convert`), including
whether the misaki conventions are switched on at all, which is precisely the
question that went unanswered through 0.8.24 and 0.8.25.

    pip install misaki                     # plus its spacy model, see misaki
    python tools/check_misaki_g2p_agreement.py --corpus my-prose.txt
    python tools/check_misaki_g2p_agreement.py --corpus my-prose.txt --verbose

Sentences where misaki itself emits `❓` (a word its lexicon does not have and
no fallback was configured for) are counted separately: we produce a
pronunciation there and misaki does not, so they are not our disagreements.

Exit code 0 always — this is a measurement, not a gate. The gate is
tests/test-kokoro-misaki-wiring.cpp.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DUMPER = r"""
#include "core/g2p_en.h"
#include "core/g2p_inflect.h"
#include "core/phoneme_dialect.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>

static g2p_en::context C;

int main(int argc, char** argv) {
    const std::string dir = argv[1];
    int total = 0;
    for (const char* w : {"us_gold.json", "us_silver.json"})
        total += g2p_en::load_misaki_json(C.espeak_ipa, C.phrase_final, dir + "/" + w);
    if (total == 0) { fprintf(stderr, "no lexicon under %s\n", dir.c_str()); return 1; }
    C.espeak_ipa.loaded = true;
    C.phrase_final.loaded = !C.phrase_final.entries.empty();
    C.inflect_fallback = [](const std::string& w) -> std::string {
        core_g2p_inflect::Params p;
        p.reduced_vowel = "\xe1\xb5\xbb";
        p.flap = "T";
        return core_g2p_inflect::inflect(w, [](const std::string& stem) -> std::string {
            auto it = C.espeak_ipa.entries.find(stem);
            return it == C.espeak_ipa.entries.end() ? std::string() : it->second;
        }, p);
    };
    const bool misaki = argc > 2 && std::string(argv[2]) == "misaki";
    const g2p_en::style st = misaki ? g2p_en::misaki_style() : g2p_en::style{};
    std::string line;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && line.back() == '\r') line.pop_back();
        printf("%s\n", core_phoneme::convert(g2p_en::text_to_ipa(C, line, st),
                                             core_phoneme::Dialect::Misaki).c_str());
    }
    return 0;
}
"""


def build_dumper(workdir: Path) -> Path:
    src = workdir / "g2p_dump.cpp"
    src.write_text(DUMPER, encoding="utf-8")
    out = workdir / "g2p_dump"
    cmd = ["c++", "-std=c++17", "-O1", "-I", str(ROOT / "src"), "-o", str(out), str(src)]
    subprocess.run(cmd, check=True)
    return out


def misaki_data_dir() -> Path:
    from misaki import data  # type: ignore

    return Path(data.__file__).parent


def run_misaki(lines: list[str]) -> list[str]:
    from misaki import en  # type: ignore

    g = en.G2P(trf=False, british=False, fallback=None)
    return [g(line)[0].replace("\n", " ") for line in lines]


def run_ours(binary: Path, data_dir: Path, style: str, lines: list[str]) -> list[str]:
    p = subprocess.run(
        [str(binary), str(data_dir), style],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    return p.stdout.splitlines()


def report(name: str, ref: list[str], got: list[str], verbose: bool) -> None:
    clean = [(a, b) for a, b in zip(ref, got) if "❓" not in a]
    exact = sum(1 for a, b in zip(ref, got) if a.split() == b.split())
    exact_clean = sum(1 for a, b in clean if a.split() == b.split())
    total_w = matched_w = 0
    scorable_w = scorable_match = 0
    diffs: Counter = Counter()
    for a, b in zip(ref, got):
        A, B = a.split(), b.split()
        total_w += max(len(A), len(B))
        matched_w += sum(1 for x, y in zip(A, B) if x == y)
        for x, y in zip(A, B):
            # A ❓ is a word misaki has no pronunciation for and we do — not our
            # disagreement, and in a novel full of proper names it dominates.
            if "❓" in x:
                continue
            scorable_w += 1
            scorable_match += x == y
        if len(A) == len(B):
            for x, y in zip(A, B):
                if x != y and "❓" not in x:
                    diffs[(x, y)] += 1
    print(
        f"{name:8s}  exact sentences {exact:4d}/{len(ref)} ({100 * exact / max(1, len(ref)):5.1f}%)"
        f"   token agreement {100 * matched_w / max(1, total_w):5.2f}%"
    )
    print(
        f"{'':8s}  ...over the {len(clean)} sentences misaki phonemizes without a "
        f"❓: {exact_clean} ({100 * exact_clean / max(1, len(clean)):5.1f}%)"
    )
    print(
        f"{'':8s}  token agreement over the {scorable_w} tokens misaki DOES "
        f"phonemize: {100 * scorable_match / max(1, scorable_w):5.2f}%"
    )
    if verbose and diffs:
        print(f"{'':8s}  top disagreements (misaki -> ours):")
        for (x, y), n in diffs.most_common(20):
            print(f"{'':10s}  {n:4d}  {x}  ->  {y}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=Path, help="one sentence per line")
    ap.add_argument("--lexicon-dir", type=Path, help="misaki/data (default: the installed package)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N lines")
    ap.add_argument("--verbose", action="store_true", help="list the top disagreements")
    args = ap.parse_args()

    lines = [l.strip() for l in args.corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        lines = lines[: args.limit]
    if not lines:
        print("empty corpus", file=sys.stderr)
        return 1

    data_dir = args.lexicon_dir or misaki_data_dir()
    with tempfile.TemporaryDirectory(prefix="crispasr-g2p-") as td:
        binary = build_dumper(Path(td))
        ref = run_misaki(lines)
        print(f"{len(lines)} sentences, misaki lexicon at {data_dir}\n")
        report("misaki", ref, run_ours(binary, data_dir, "misaki", lines), args.verbose)
        print()
        report("piper", ref, run_ours(binary, data_dir, "piper", lines), args.verbose)
        print(
            "\n`piper` is the same dictionary read with the ESPEAK consumer's conventions —"
            "\nit is the control, not a target. It should score far lower; when it does not,"
            "\nthe misaki conventions have stopped being applied (which is #316 round 2)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
