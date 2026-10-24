#!/usr/bin/env python3
"""Run prebuilt Granite5 acceptance on Kaggle after explicit CPU-input pinning."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

VERSION = 'granite5-q4-cuda-v1'
SOURCE = 'a581c66f086b61c0f12bd5d6c8f69f06b4a40197'
CPU_SOURCE = '8fbb1794da4eea3851d8384934695d39b3518b72'
CPU_RUN = 40068653973
CPU_REVISION = 'e425e802e3a029edcf15d51cb31e863d59b5b0d2'
CPU_SHA256 = 'de60017c1e5d443539021124927a6c30d99bfa399d24ea018e274574ae9f53d7'
SELECTED_CASE = 'value-down-f16'
RUNTIME_SOURCE = 'e2368b9e6ba94368a31798d003dca3b61e0bda10'
RUNTIME_REVISION = 'd597fd6b5cde7126f181fdce08b5d85499c54cb9'
RUNTIME_SHA256 = '132ba8f3e82abe6dcc6b66de808883d5f634997f65210a435ef88689ef7473d9'
STAGING = 'cstr/crispasr-granite5-validation-public'
RUNTIME_PATH = f'runtime/{RUNTIME_SOURCE}/granite5-cuda-validation.tar.gz'
# Deliberately unlaunchable until reviewed immutable CPU/runtime pins exist.
for value in (SOURCE, CPU_SOURCE, RUNTIME_SOURCE, CPU_REVISION, RUNTIME_REVISION):
    if not re.fullmatch('[0-9a-f]{40}', value):
        raise RuntimeError('Reviewed immutable CPU and runtime revisions are required before pushing this kernel')
if not all(re.fullmatch('[0-9a-f]{64}', value) for value in (CPU_SHA256, RUNTIME_SHA256)):
    raise RuntimeError('Reviewed CPU receipt hash is required before pushing this kernel')
if not re.fullmatch('[a-z][a-z0-9-]*', SELECTED_CASE):
    raise RuntimeError('A reviewed CPU-qualified Q4 recipe must be selected before pushing this kernel')
OUT = Path('/kaggle/working')
SCRATCH = Path('/kaggle/temp/granite5-q4')
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(SCRATCH), HF_HOME=str(SCRATCH / 'hf'), HF_XET_CACHE=str(SCRATCH / 'xet'),
                  OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='0')
hardware = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=name,compute_cap,driver_version',
                                    '--format=csv,noheader'], text=True).strip()
if not hardware:
    raise RuntimeError('Actual GPU0 is required')
print(VERSION, SOURCE, SELECTED_CASE, hardware, flush=True)
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


cpu_path = f'q4-recipes/{CPU_SOURCE}/runs/{CPU_RUN}/cpu-evidence/receipt.json'
cpu = Path(hf_hub_download(STAGING, cpu_path, revision=CPU_REVISION, local_dir=SCRATCH / 'inputs'))
runtime = Path(hf_hub_download(STAGING, RUNTIME_PATH, revision=RUNTIME_REVISION, local_dir=SCRATCH / 'inputs'))
if sha(cpu) != CPU_SHA256 or sha(runtime) != RUNTIME_SHA256:
    raise RuntimeError('Pinned CPU or runtime content mismatch')
evidence = json.loads(cpu.read_text())
if not (evidence['source'] == CPU_SOURCE and evidence['run'] == CPU_RUN and evidence['complete']):
    raise RuntimeError('The complete pinned CPU diagnostic receipt is required')
sys.path.insert(0, str(repo / 'tools/ci-heavy'))
from granite5_q4_cuda import approved_case
approved_case(evidence, SELECTED_CASE)
with tarfile.open(runtime) as archive:
    archive.extractall(SCRATCH, filter='data')
bundle = SCRATCH / 'bundle'
if json.loads((bundle / 'provenance.json').read_text())['source'] != RUNTIME_SOURCE:
    raise RuntimeError('Runtime build source pin mismatch')
(OUT / 'entry-provenance.json').write_text(json.dumps(dict(version=VERSION, source=SOURCE,
    selected_recipe=SELECTED_CASE, cpu_run=CPU_RUN, cpu_source=CPU_SOURCE, cpu_revision=CPU_REVISION, cpu_sha256=CPU_SHA256,
    runtime_source=RUNTIME_SOURCE, runtime_revision=RUNTIME_REVISION, runtime_sha256=RUNTIME_SHA256,
    entry_sha256=sha(__file__), hardware=hardware), indent=2) + '\n')
command = [sys.executable, repo / 'tools/ci-heavy/granite5_q4_cuda.py', '--bundle', bundle,
           '--cpu-receipt', cpu, '--case', SELECTED_CASE, '--scratch', SCRATCH / 'models', '--out', OUT / 'cuda']
with kh.build_heartbeat('granite5.q4.cuda.acceptance'):
    result = subprocess.run(list(map(str, command)))
summary = json.loads((OUT / 'cuda/receipt.json').read_text())
kh.step('granite5.cuda.result', returncode=result.returncode, complete=summary['complete'],
        cuda_acceptance=summary['cuda_acceptance'])
if not (result.returncode == 0 and summary['complete'] and summary['cuda_acceptance']):
    raise RuntimeError('Actual CUDA acceptance failed; preserve terminal logs before any rerun')
print('GRANITE5_Q4_CUDA_ACCEPTED', VERSION, flush=True)
