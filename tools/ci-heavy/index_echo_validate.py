#!/usr/bin/env python3
"""Build Index-Echo's shared ABI, CLI and stage diff on a hosted CPU runner."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
BUILD = Path(os.environ['HEAVY_SCRATCH']) / 'index-echo-build'
OUT.mkdir(parents=True, exist_ok=True)


def run(*args):
    print('run:', *args, flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


parser = argparse.ArgumentParser()
parser.add_argument('--build-only', action='store_true')
args = parser.parse_args()
run('cmake', '-S', ROOT, '-B', BUILD, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
    '-DBUILD_SHARED_LIBS=ON', '-DCRISPASR_BUILD_SERVER=OFF', '-DGGML_NATIVE=OFF')
run('cmake', '--build', BUILD, '--target', 'crispasr-cli', 'crispasr-lib', 'crispasr-diff',
    'test-index-echo-windows', 'test-session-autochunk', 'test-arch-backend-map', '-j', '4')
for test in ['test-index-echo-windows', 'test-session-autochunk', 'test-arch-backend-map']:
    run(BUILD / 'bin' / test)
run(sys.executable, ROOT / 'tools/gen-feature-matrix.py', '--crispasr', BUILD / 'bin/crispasr')
run(sys.executable, ROOT / 'tools/gen-backend-caps-table.py', '--crispasr', BUILD / 'bin/crispasr')
for relative in ['docs/feature-matrix.md', 'docs/feature-matrix.json', 'src/core/backend_caps_table.h']:
    source = ROOT / relative
    if source.exists(): shutil.copy2(source, OUT / source.name)
if args.build_only:
    (OUT / 'summary.md').write_text('Index-Echo shared library, CLI, diff and integration unit tests built. '
                                   'Model parity remains pending.\n')
    sys.exit(0)
raise RuntimeError('Live validation must wait for the independent reference producer')
