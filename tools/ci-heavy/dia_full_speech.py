#!/usr/bin/env python3
"""Dia complete-speech acceptance, isolated C ABI generation and ASR processes."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
PHRASES = [
    ('hello', 42, 'Hello there, how are you doing today? I really hope you are having a wonderful and pleasant time. The weather outside is lovely and bright.'),
    ('fox', 123, 'The quick brown fox jumps over the lazy dog. Please listen carefully, because this is a test of speech synthesis with a chosen number of CPU threads.'),
]
p = argparse.ArgumentParser()
p.add_argument('--child', choices=('generate', 'recognize'))
p.add_argument('--model')
p.add_argument('--lib')
p.add_argument('--phrase', choices=('hello', 'fox'))
p.add_argument('--threads', type=int, default=4)
p.add_argument('--steps', type=int, default=0)
p.add_argument('--quant', choices=('f16', 'q8_0'), default='q8_0')
p.add_argument('--matrix', default='4,1,8')
a = p.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)

if a.child:
    import numpy as np
    sys.path.insert(0, str(ROOT / 'python'))
    from crispasr import Session
    key, seed, text = next(c for c in PHRASES if c[0] == a.phrase)
    tag = f'{key}-{a.threads}'
    path = OUT / (tag + '.npy')
    if a.child == 'generate':
        if a.steps:
            os.environ['CRISPASR_DIA_MAX_STEPS'] = str(a.steps)
        with Session(a.model, lib_path=a.lib, backend='dia', n_threads=a.threads) as session:
            session.set_temperature(1.2, seed=seed)
            start = time.perf_counter()
            pcm = session.synthesize('[S1] ' + text)
            elapsed = time.perf_counter() - start
            assert session.output_sample_rate() == 44100
        assert np.isfinite(pcm).all() and np.sqrt(np.mean(pcm.astype(np.float64) ** 2)) > 1e-4
        np.save(path, pcm)
        result = {'phrase': text, 'seed': seed, 'threads': a.threads, 'steps_override': a.steps,
                  'audio_seconds': len(pcm) / 44100, 'generation_seconds': elapsed}
        assert result['audio_seconds'] > 3, result
        (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2))
    else:
        with Session(a.model, lib_path=a.lib, backend='nemotron', n_threads=4) as session:
            transcript = ' '.join(s.text for s in session.transcribe(np.load(path), sample_rate=44100, language='en'))
        def words(t):
            return re.findall('[a-z]+', re.sub(r'<[^>]*>', '', t).lower())
        ref, actual = words(text), words(transcript)
        row = list(range(len(actual) + 1))
        for i, r in enumerate(ref, 1):
            new = [i]
            for j, x in enumerate(actual, 1):
                new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (r != x)))
            row = new
        result = json.loads((OUT / (tag + '.json')).read_text())
        result.update(transcript=transcript, wer=row[-1] / len(ref))
        (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        assert result['wer'] <= .2, result
    sys.exit(0)

from huggingface_hub import hf_hub_download

def run(cmd, tag):
    with (OUT / (tag + '.log')).open('w') as log:
        proc = subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=2400)
    print(tag, proc.returncode, (OUT / (tag + '.log')).read_text()[-2200:], flush=True)
    assert proc.returncode == 0, tag

subprocess.run(['uptime'], check=True)
subprocess.run(['free', '-h'], check=True)
build = SCRATCH / 'dia-build'
run(['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
     '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF',
     '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_OPUS=OFF', '-DCRISPASR_AMR=OFF'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-lib', '-j4'], 'build')
lib = next(build.rglob('libcrispasr.so'))
repo, revision = 'cstr/dia-1.6b-GGUF', '3233fbcb32be47761d2e736857b6d1a075b9ba7e'
model = hf_hub_download(repo, f'dia-1.6b-{a.quant}.gguf', revision=revision)
hf_hub_download(repo, 'dac-44khz.gguf', revision=revision)
asr = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF', 'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf', revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2')
for key, _, _ in PHRASES:
    if a.phrase and key != a.phrase:
        continue
    for threads in map(int, a.matrix.split(',')):
        for child, weights in (('generate', model), ('recognize', asr)):
            run([sys.executable, __file__, '--child', child, '--model', weights, '--lib', lib,
                 '--phrase', key, '--threads', threads, '--steps', a.steps], f'{key}-{threads}-{child}')
print('DIA_FULL_SPEECH_PASS', flush=True)
