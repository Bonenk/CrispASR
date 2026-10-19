#!/usr/bin/env python3
"""Hosted-only CPU Whisper pointer/Python metadata acceptance, no parity claim.

Build the actual library, pin the public GGML fixture at its resolved revision,
and retain model/audio/binary identities, raw metadata and phase logs on failure.
"""
import dataclasses
import hashlib
import json
import os
import resource
import shutil
from pathlib import Path
import subprocess
import sys
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise RuntimeError('Hosted-only worker; do not build or download models locally')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'whisper-wrapper-validation'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    receipt = {'scope': 'CPU pointer/Python metadata smoke; no original-model, WER, GPU or speed parity claim',
               'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'passed': False, 'phases': {}}

    def save():
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

    def run(command, label, timeout=3600):
        save()
        with (out / (label + '.log')).open('w') as log:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout)
        receipt['phases'][label] = {'returncode': result.returncode}
        save()
        if result.returncode:
            raise RuntimeError(f'{label} failed with {result.returncode}; see retained log')

    try:
        save()
        run(['cmake', '-S', ROOT, '-B', scratch / 'build', '-DCMAKE_BUILD_TYPE=Release',
             '-DBUILD_SHARED_LIBS=ON', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DGGML_NATIVE=OFF',
             '-DGGML_CUDA=OFF', '-DGGML_METAL=OFF', '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF',
             '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=OFF'], 'configure')
        run(['cmake', '--build', scratch / 'build', '--target', 'crispasr-lib', '-j2'], 'build', 5400)
        libraries = list((scratch / 'build').rglob('libcrispasr.so'))
        if len(libraries) != 1:
            raise RuntimeError('Expected one actual Linux shared library')
        library = libraries[0].resolve()
        receipt['library_sha256'] = sha(library)
        receipt['native_source_sha256'] = {name: sha(ROOT / name) for name in (
            'src/crispasr.cpp', 'src/crispasr_c_api.cpp', 'include/crispasr.h', 'include/crispasr_session.h')}

        receipt['native_dependencies'] = {str(p.relative_to(scratch / 'build')): sha(p)
                                          for p in (scratch / 'build').rglob('*.so') if p.is_file()}
        native_out = out / 'native'
        native_out.mkdir()
        for dependency in (scratch / 'build').rglob('*.so*'):
            if dependency.is_file():
                shutil.copy2(dependency, native_out / dependency.name)
        shutil.copy2(scratch / 'build/compile_commands.json', out / 'compile_commands.json')
        shutil.copy2(ROOT / 'samples/jfk.wav', out / 'fixture.wav')
        receipt['audio_sha256'] = sha(ROOT / 'samples/jfk.wav')
        with urllib.request.urlopen('https://huggingface.co/api/models/ggerganov/whisper.cpp/revision/main', timeout=60) as stream:
            revision = json.load(stream)['sha']
        if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            raise RuntimeError('Expected immutable public model revision')
        model_url = f'https://huggingface.co/ggerganov/whisper.cpp/resolve/{revision}/ggml-tiny.en.bin'
        receipt['model_revision'] = revision
        receipt['model_url'] = model_url
        model = scratch / 'ggml-tiny.en.bin'
        run(['curl', '-fL', '--retry', '3', '--output', model, model_url], 'download-model')
        if model.stat().st_size < 1_000_000:
            raise RuntimeError('Truncated GGML model')
        receipt['model_sha256'] = sha(model)
        binary = scratch / 'whisper-model-wrapper-acceptance'
        source = ROOT / 'tests/fixtures/whisper-model-wrapper-acceptance.cpp'
        receipt['probe_source_sha256'] = sha(source)
        run(['c++', '-std=c++17', '-O2', '-I' + str(ROOT / 'include'), '-I' + str(ROOT / 'ggml/include'),
             source, library, '-Wl,-rpath,' + str(library.parent), '-o', binary], 'compile-probe')
        receipt['probe_binary_sha256'] = sha(binary)
        shutil.copy2(binary, native_out / binary.name)
        run([binary, model, ROOT / 'samples/jfk.wav', out / 'native-metadata.tsv'], 'native-pointer-matrix')
        rows = (out / 'native-metadata.tsv').read_text().splitlines()
        if sum(row.startswith('run\t') for row in rows) != 12:
            raise RuntimeError('Native matrix did not complete all twelve runs')
        run([sys.executable, ROOT / 'tests/test_whisper_parallel_runtime.py', '-v'], 'runtime-regressions')
        run([sys.executable, ROOT / 'tests/test_whisper_vad_timeline.py', '-v'], 'vad-regressions')

        import numpy as np
        sys.path.insert(0, str(ROOT / 'python'))
        from crispasr import CrispASR
        with wave.open(str(ROOT / 'samples/jfk.wav')) as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                raise RuntimeError('Unexpected JFK WAV format')
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768

        def normalize(value):
            if dataclasses.is_dataclass(value):
                return normalize(dataclasses.asdict(value))
            if isinstance(value, bytes):
                return {'hex': value.hex()}
            if isinstance(value, dict):
                return {k: normalize(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [normalize(v) for v in value]
            return value

        snapshots = []
        for dtw in (False, True):
            model_handle = CrispASR(str(model), lib_path=str(library), dtw=dtw, dtw_aheads_preset=3)
            try:
                model_handle._helpers = None  # Require the exported native pointer route.
                for processors in (1, 2, 3):
                    segments = model_handle.transcribe_pcm(pcm, language='en', n_processors=processors,
                        decode_options={'token_timestamps': True, 'alt_n': 2, 'n_threads': 2,
                                        'no_context': True, 'print_progress': False, 'print_realtime': False})
                    if not segments or not any(segment.tokens for segment in segments):
                        raise RuntimeError('Actual Python result metadata missing')
                    frozen = normalize(segments)
                    snapshots.append((segments, frozen))
                    (out / 'python-metadata.json').write_text(json.dumps([f for _, f in snapshots],
                        indent=2, ensure_ascii=False, allow_nan=False) + '\n')
            finally:
                model_handle.close()
        if any(normalize(segments) != frozen for segments, frozen in snapshots):
            raise RuntimeError('Python owned metadata changed after reuse/close')
        (out / 'python-metadata.json').write_text(json.dumps([f for _, f in snapshots], indent=2, ensure_ascii=False) + '\n')
        receipt['native_runs'] = 12
        receipt['python_runs'] = 6
        receipt['passed'] = True
    finally:
        save()
        (out / 'summary.md').write_text(f"Whisper pointer/Python CPU smoke: {'PASS' if receipt['passed'] else 'FAIL'}\n\n"
            'Real model metadata/ownership validation; numerical-reference, transcription-quality and GPU parity are separate.\n')


if __name__ == '__main__':
    main()
