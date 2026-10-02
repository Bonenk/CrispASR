#!/usr/bin/env python3
"""Run the staged CUDA archive checks using the existing Heavy CPU workflow."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli-run", type=int, required=True)
    parser.add_argument("--library-run", type=int, required=True)
    args = parser.parse_args()
    assert args.cli_run > 0 and args.library_run > 0
    subprocess.run(["uptime"], check=True)
    subprocess.run(["free", "-h"], check=True)
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "cuda-package-pair"
    output = Path(os.environ["HEAVY_OUT"])
    output.mkdir(parents=True, exist_ok=True)
    checker = Path(__file__).with_name("check_cuda_package_pair.py")
    repository = os.environ.get("GITHUB_REPOSITORY", "CrispStrobe/CrispASR")
    for flavor in ("cuda", "cuda126"):
        cli = scratch / flavor / "cli"
        libraries = scratch / flavor / "libraries"
        for run, artifact, destination in (
            (args.cli_run, f"crispasr-windows-x86_64-{flavor}-split", cli),
            (args.library_run, f"libcrispasr-windows-x86_64-{flavor}", libraries),
        ):
            subprocess.run(["gh", "run", "download", str(run), "--repo", repository,
                            "--name", artifact, "--dir", str(destination)], check=True)
        subprocess.run([sys.executable, str(checker), "--flavor", flavor,
                        "--cli", str(cli), "--libraries", str(libraries),
                        "--output", str(output / f"package-pair-{flavor}.json")], check=True)
    (output / "summary.md").write_text(
        "CUDA 12.8 and CUDA 12.6 CLI/runtime/library SHA-256 pairing PASS.\n"
        f"CLI run: {args.cli_run}; library run: {args.library_run}.\n")


if __name__ == "__main__":
    main()
