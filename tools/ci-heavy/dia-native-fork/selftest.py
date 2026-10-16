"""Compile the exact pinned actual sampler and exercise the scratch fork helper."""
import argparse
import importlib.util
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('native_fork_patch', HERE/'patch.py')
patcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(patcher)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--scratch', type=Path, required=True)
args = parser.parse_args(); args.scratch.mkdir(parents=True, exist_ok=False)
for name in ['dia_sampling.h', 'dia_exponential_rng.h', 'dia_tts.cpp']:
    value = subprocess.check_output(['git', 'show', patcher.SOURCE+':src/'+name], cwd=args.repo)
    (args.scratch/name).write_bytes(value)
patcher.patch(args.scratch/'dia_tts.cpp')
try: patcher.patch(args.scratch/'dia_tts.cpp')
except AssertionError: pass
else: raise AssertionError('Repeated patch accepted')
# Release native builds must not disable these assertions.
subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',str(HERE/'guard.cpp'),
                '-I'+str(args.scratch),'-I'+str(HERE),'-o',str(args.scratch/'guard')],check=True)
subprocess.run([str(args.scratch/'guard'),str(args.scratch)],check=True)
print('DIA_NATIVE_FORK_PATCH_PASS: immutable source and repeated-patch rejection')
