#!/usr/bin/env python3
"""PR #487: exact full-window/token-confidence parity and real WebSocket tests."""
import json
import os
from pathlib import Path
import subprocess
import sys

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)

def run(cmd, label, env=None):
    with (OUT / (label + '.log')).open('w') as log:
        proc = subprocess.run(list(map(str, cmd)), cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT, timeout=3600)
    print(label, proc.returncode, (OUT / (label + '.log')).read_text()[-2500:], flush=True)
    assert proc.returncode == 0, label

subprocess.run(['uptime'], check=True)
subprocess.run(['free', '-h'], check=True)
build = SCRATCH / 'build'
run(['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
     '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF',
     '-DCRISPASR_BUILD_TESTS=ON', '-DCRISPASR_BUILD_SERVER=ON',
     '-DCRISPASR_OPUS=OFF', '-DCRISPASR_AMR=OFF'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-cli', 'test-nemotron',
     'test-realtime-turn-buffer', '-j4'], 'build')
# Protocol test discovers this conventional path; no local build is overwritten.
assert not (ROOT / 'build').exists()
(ROOT / 'build').symlink_to(build, target_is_directory=True)
receipt = {'sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
           'revision': 'bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2', 'cases': []}
run([build / 'bin/test-realtime-turn-buffer'], 'turn-buffer')
for quant in ('q8_0', 'q4_k', 'f16'):
    model = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF',
                           f'nemotron-3.5-asr-streaming-0.6b-{quant}.gguf',
                           revision=receipt['revision'])
    env = dict(os.environ, CRISPASR_MODEL_NEMOTRON=model, CRISPASR_TEST_CPU_ONLY='1')
    for threads in (1, 4, 8):
        label = f'window-{quant}-{threads}'
        run([build / 'bin/test-nemotron', 'nemotron: realtime stream gives the same output as a full recompute'],
            label, dict(env, CRISPASR_TEST_N_THREADS=str(threads)))
        receipt['cases'].append(label)
        (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if quant == 'q8_0':
        for vad in (False, True):
            command = [sys.executable, ROOT / 'tests/test-server-realtime-api.py',
                       '--backend', 'nemotron', '--model', model, '--language', 'en']
            if vad:
                command.append('--server-vad')
            run(command, 'server-vad' if vad else 'server', env)
print('NEMOTRON_REALTIME_PARITY_PASS', json.dumps(receipt), flush=True)
