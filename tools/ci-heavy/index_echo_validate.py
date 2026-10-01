#!/usr/bin/env python3
"""Build Index-Echo's shared ABI, CLI and stage diff on a hosted CPU runner."""
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
BUILD = Path(os.environ['HEAVY_SCRATCH']) / 'index-echo-build'
OUT.mkdir(parents=True, exist_ok=True)


def run(*args):
    print('run:', *args, flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


parser = argparse.ArgumentParser()
parser.add_argument('--build-only', action='store_true')
parser.add_argument('--clips', nargs='+', choices=['jfk', 'zh', 'jfk-tail'], default=['jfk', 'zh', 'jfk-tail'])
args = parser.parse_args()
run('cmake', '-S', ROOT, '-B', BUILD, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
    '-DBUILD_SHARED_LIBS=ON', '-DCRISPASR_BUILD_SERVER=OFF', '-DGGML_NATIVE=OFF')
run('cmake', '--build', BUILD, '--target', 'crispasr-cli', 'crispasr-lib', 'crispasr-diff',
    'test-index-echo-windows', 'test-index-echo-batch', 'test-session-autochunk', 'test-arch-backend-map', '-j', '4')
for test in ['test-index-echo-windows', 'test-index-echo-batch', 'test-session-autochunk', 'test-arch-backend-map']:
    run(BUILD / 'bin' / test)
run(sys.executable, ROOT / 'tools/gen-feature-matrix.py', '--crispasr', BUILD / 'bin/crispasr')
run(sys.executable, ROOT / 'tools/gen-backend-caps-table.py', '--crispasr', BUILD / 'bin/crispasr')
for relative in ['docs/feature-matrix.md', 'docs/feature-matrix.html', 'src/core/backend_caps_table.h']:
    source = ROOT / relative
    if source.exists(): shutil.copy2(source, OUT / source.name)
run('cmake', '--build', BUILD, '--target', 'crispasr-lib', '-j', '4')
if args.build_only:
    (OUT / 'summary.md').write_text('Index-Echo shared library, CLI, diff and integration unit tests built. '
                                   'Model parity remains pending.\n')
    sys.exit(0)
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=os.environ.get('HF_TOKEN'))
destination = 'cstr/index-echo-2b-GGUF'
required = {f'reference/{clip}-ref.gguf' for clip in args.clips} | {'conversion-receipt.json'}
for attempt in range(90):
    present = set(api.list_repo_files(destination))
    if required <= present:
        break
    print('waiting for independent references:', sorted(required - present), flush=True)
    time.sleep(30)
else:
    raise RuntimeError('Independent reference producer did not complete in 45 minutes')
models = Path(snapshot_download(destination, local_dir=Path(os.environ['HEAVY_SCRATCH']) / 'index-echo-models',
    allow_patterns=['index-echo-2b-f16.gguf', 'index-echo-2b-decoder-f16.gguf', 'reference/*']))
os.environ['TMPDIR'] = os.environ['HEAVY_SCRATCH']
failures = []
for clip in args.clips:
    audio = models / 'reference/jfk-tail.wav' if clip == 'jfk-tail' else ROOT / 'samples' / ('paraformer_zh.wav' if clip == 'zh' else 'jfk.wav')
    log_path = OUT / f'f16-{clip}-diff.log'
    command = [str(BUILD / 'bin/crispasr-diff'), 'index-echo', str(models / 'index-echo-2b-f16.gguf'),
               str(models / f'reference/{clip}-ref.gguf'), str(audio)]
    with log_path.open('w') as log:
        result = subprocess.run(command, env=dict(os.environ, CRISPASR_DIFF_NO_GPU='1'),
                                stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    print(clip, 'stage diff rc:', result.returncode, log_path.read_text()[-16000:], flush=True)
    if result.returncode: failures.append(clip)
(OUT / 'stage-results.json').write_text(json.dumps(dict(f16_failed=failures), indent=2))
if failures:
    raise RuntimeError('F16 stage parity failed: ' + ', '.join(failures))

# Open a model with an arbitrary filename through the actual Python Session,
# which tests shared metadata detection and the shipped C ABI, not CLI heuristics.
sys.path.insert(0, str(ROOT / 'python'))
import numpy as np
import wave
from gguf import GGUFReader
from crispasr import Session
library = next(BUILD.rglob('libcrispasr.so'))
assert 'index-echo' in Session.available_backends(lib_path=str(library))
renamed = models / 'model-without-backend-hint.gguf'
renamed.symlink_to(models / 'index-echo-2b-f16.gguf')
decoded = {}


def reference_cues(text):
    # Released timestamp / transcript / translation format, parsed independently
    # of the native implementation. These fixtures have no optional context.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    expected = []
    for i in range(0, len(lines), 3):
        match = re.fullmatch(r'\[(\d+):(\d+(?:\.\d+)?)-(\d+):(\d+(?:\.\d+)?)\]', lines[i])
        if not match or i + 2 >= len(lines):
            raise RuntimeError('Malformed independent decoded reference')
        expected.append(dict(start=60 * int(match[1]) + float(match[2]),
                             end=60 * int(match[3]) + float(match[4]),
                             text=lines[i + 1] + '\n' + lines[i + 2]))
    return expected


with Session(str(renamed), lib_path=str(library), n_threads=4) as session:
    assert session.backend == 'index-echo', session.backend
    for clip in args.clips:
        audio = models / 'reference/jfk-tail.wav' if clip == 'jfk-tail' else ROOT / 'samples' / ('paraformer_zh.wav' if clip == 'zh' else 'jfk.wav')
        with wave.open(str(audio), 'rb') as wav:
            assert wav.getframerate() == 16000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        segments = session.transcribe(pcm)
        reader = GGUFReader(models / f'reference/{clip}-ref.gguf')
        reference = reader.fields['crispasr.ref.generated_text'].contents()
        decoded[clip] = dict(reference=reference, segments=[dict(start=s.start, end=s.end, text=s.text) for s in segments])
        if not segments or any(not s.text or s.end < s.start for s in segments):
            failures.append(clip + ': missing/invalid decoded cues')
        expected = reference_cues(reference)
        actual = decoded[clip]['segments']
        if len(actual) != len(expected) or any(
                a['text'] != e['text'] or abs(a['start'] - e['start']) > 0.0051 or
                abs(a['end'] - e['end']) > 0.0051 for a, e in zip(actual, expected)):
            failures.append(clip + ': decoded text/timestamp mismatch')
        # Preserve complete text and timing for review rather than hiding a
        # numerically correct but behaviorally wrong output behind cosine.
        print('decoded', clip, json.dumps(decoded[clip], ensure_ascii=False), flush=True)
        del reader
(OUT / 'decoded-f16.json').write_text(json.dumps(decoded, indent=2, ensure_ascii=False))
if failures:
    raise RuntimeError('; '.join(failures))
(OUT / 'summary.md').write_text('F16 stage/magnitude/prompt/cache parity passed; Python Session '
                               'metadata autodetection and exact decoded text/timestamp parity passed.\n')
