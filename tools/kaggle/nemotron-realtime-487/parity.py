#!/usr/bin/env python3
"""PR #487 real CUDA window parity. Uploaded script version 1."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

REF = 'fix/487-realtime'
WORK = Path('/kaggle/working')
REPO = WORK / 'CrispASR'
subprocess.run(['git', 'clone', '--depth', '1', '--recursive', '--branch', REF,
                'https://github.com/CrispStrobe/CrispASR', REPO], check=True)
sys.path.insert(0, str(REPO / 'tools/kaggle'))
import kaggle_harness as kh
kh.init_progress()
os.environ['HF_TOKEN'] = kh.resolve_hf_token(require=True)
kh.step('hardware', script_version=1, sha=subprocess.check_output(
    ['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip())
subprocess.run(['nvidia-smi'], check=True)
subprocess.run(['uptime'], check=True)
subprocess.run(['free', '-h'], check=True)
kh.install_build_toolchain()
build = REPO / 'build'
flags = kh.cuda_build_flags(kh.detect_cuda_arch()) + kh.cache_and_link_flags()
def command(cmd):
    kh.sh_with_progress(shlex.join(list(map(str, cmd))))
with kh.build_heartbeat('configure'):
    command(['cmake', '-S', REPO, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
             '-DGGML_NATIVE=OFF', '-DCRISPASR_BUILD_TESTS=ON', '-DCRISPASR_BUILD_SERVER=ON',
             '-DCRISPASR_OPUS=OFF', '-DCRISPASR_AMR=OFF'] + flags)
with kh.build_heartbeat('build'):
    command(['cmake', '--build', build, '--target', 'test-nemotron', 'crispasr-cli',
             'crispasr-quantize', '-j' + kh.safe_build_jobs(gpu=True)])
from huggingface_hub import hf_hub_download
revision = 'bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2'
f16 = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF',
    'nemotron-3.5-asr-streaming-0.6b-f16.gguf', revision=revision)
q8 = WORK / 'nemotron-q8.gguf'
command([build / 'bin/crispasr-quantize', f16, q8, 'q8_0'])
results = []
for quant in ('q8_0', 'q4_k', 'f16'):
    model = str(q8) if quant == 'q8_0' else hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF',
        f'nemotron-3.5-asr-streaming-0.6b-{quant}.gguf', revision=revision)
    env = dict(os.environ, CRISPASR_MODEL_NEMOTRON=model, CRISPASR_TEST_N_THREADS='4')
    log = WORK / f'cuda-{quant}.log'
    with kh.build_heartbeat('cuda-' + quant), log.open('w') as output:
        proc = subprocess.run([build / 'bin/test-nemotron',
            'nemotron: realtime stream gives the same output as a full recompute'],
            cwd=REPO, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=3600)
    text = log.read_text()
    print(text[-4500:], flush=True)
    assert 'nemotron: backend = CUDA' in text, 'CPU fallback cannot prove CUDA parity'
    assert proc.returncode == 0, quant
    results.append({'quant': quant, 'passed': True})
    (WORK / 'receipt.json').write_text(json.dumps(results, indent=2))
    kh.step('cuda.parity.pass', quant=quant)
print('NEMOTRON_CUDA_REALTIME_PARITY_PASS', flush=True)
