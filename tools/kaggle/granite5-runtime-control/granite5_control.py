#!/usr/bin/env python3
"""Run the source-matched F16 GPU control; no Q4 qualification."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

VERSION = 'granite5-runtime-control-v1'
SOURCE = '4b433e1bbe54cf3bcbf0dfee731cdbdfdbb27ad7'
RUNTIME_SOURCE = 'e2368b9e6ba94368a31798d003dca3b61e0bda10'
RUNTIME_REVISION = 'd597fd6b5cde7126f181fdce08b5d85499c54cb9'
RUNTIME_SHA256 = '132ba8f3e82abe6dcc6b66de808883d5f634997f65210a435ef88689ef7473d9'
STAGING = 'cstr/crispasr-granite5-validation-public'
RUNTIME_PATH = f'runtime/{RUNTIME_SOURCE}/granite5-cuda-validation.tar.gz'
for value in (SOURCE, RUNTIME_SOURCE, RUNTIME_REVISION):
    if not re.fullmatch('[0-9a-f]{40}', value):
        raise RuntimeError('Immutable worker/runtime source pins are required')
if not re.fullmatch('[0-9a-f]{64}', RUNTIME_SHA256):
    raise RuntimeError('Immutable runtime content hash is required')
OUT = Path('/kaggle/working')
SCRATCH = Path('/kaggle/temp/granite5-runtime-control')
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(SCRATCH), HF_HOME=str(SCRATCH / 'hf'), HF_XET_CACHE=str(SCRATCH / 'xet'),
                  OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='0')
hardware = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=name,compute_cap,driver_version',
                                    '--format=csv,noheader'], text=True).strip()
if not hardware:
    raise RuntimeError('Actual GPU0 is required')
print(VERSION, SOURCE, hardware, flush=True)
repo = SCRATCH / 'CrispASR'
for command in (['git', 'init', repo], ['git', '-C', repo, 'remote', 'add', 'origin',
                                       'https://github.com/CrispStrobe/CrispASR.git'],
                ['git', '-C', repo, 'fetch', '--depth=1', 'origin', SOURCE],
                ['git', '-C', repo, 'checkout', 'FETCH_HEAD']):
    subprocess.run(list(map(str, command)), check=True)
if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != SOURCE:
    raise RuntimeError('Worker source pin mismatch')
sys.path.insert(0, str(repo / 'tools/kaggle'))
import kaggle_harness as kh
kh.init_progress()
kh.provenance(VERSION, repo)
os.environ['HF_TOKEN'] = kh.resolve_hf_token(require=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy<3', 'gguf==0.17.1',
                'huggingface_hub>=0.36,<2'], check=True)
from huggingface_hub import hf_hub_download


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


runtime = Path(hf_hub_download(STAGING, RUNTIME_PATH, revision=RUNTIME_REVISION, local_dir=SCRATCH / 'inputs'))
if sha(runtime) != RUNTIME_SHA256:
    raise RuntimeError('Pinned runtime content mismatch')
with tarfile.open(runtime) as archive:
    archive.extractall(SCRATCH, filter='data')
bundle = SCRATCH / 'bundle'
if json.loads((bundle / 'provenance.json').read_text())['source'] != RUNTIME_SOURCE:
    raise RuntimeError('Runtime build source pin mismatch')
(OUT / 'entry-provenance.json').write_text(json.dumps(dict(version=VERSION, source=SOURCE,
    runtime_source=RUNTIME_SOURCE, runtime_revision=RUNTIME_REVISION, runtime_sha256=RUNTIME_SHA256,
    entry_sha256=sha(__file__), hardware=hardware), indent=2) + '\n')
command = [sys.executable, repo / 'tools/ci-heavy/granite5_runtime_control.py', '--bundle', bundle,
           '--scratch', SCRATCH / 'control', '--out', OUT / 'control']
with kh.build_heartbeat('granite5.runtime.control'):
    result = subprocess.run(list(map(str, command)))
report = OUT / 'control/receipt.json'
if not report.is_file():
    raise RuntimeError('Control ended before a receipt; inspect the original worker log')
summary = json.loads(report.read_text())
kh.step('granite5.runtime.control.result', returncode=result.returncode, complete=summary['complete'],
        runtime_control_pass=summary['runtime_control_pass'], q4_acceptance=summary['q4_acceptance'])
if not (result.returncode == 0 and summary['complete'] and summary['runtime_control_pass'] and not summary['q4_acceptance']):
    raise RuntimeError('Runtime control failed; preserve terminal logs before any rerun')
print('GRANITE5_RUNTIME_CONTROL_PASSED', VERSION, flush=True)
