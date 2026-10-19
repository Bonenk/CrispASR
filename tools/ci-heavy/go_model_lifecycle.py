#!/usr/bin/env python3
"""Hosted-only real Go/Whisper lifecycle; no original-reference or GPU claim."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
MODEL_REV = '5359861c739e955e79d9a303bcbc70fb988958b1'
MODEL_SHA = '921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f'
AUDIO_SHA = '59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise RuntimeError('Hosted only: no local build or model execution')
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'go-model-lifecycle'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    receipt = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   passed=False, model_revision=MODEL_REV, phases={},
                   scope='real CPU tiny.en/JFK high-level Go lifecycle, callback reentry/close, abort recovery and owned segments; no WER/reference/GPU acceptance')

    def save():
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

    def run(argv, label, env=None, cwd=ROOT, timeout=5400):
        save()
        with (out / (label + '.log')).open('w') as log:
            result = subprocess.run(list(map(str, argv)), cwd=cwd, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout)
        receipt['phases'][label] = dict(returncode=result.returncode)
        save()
        if result.returncode:
            raise RuntimeError(label + ' failed; inspect retained log')

    try:
        audio = ROOT / 'samples/jfk.wav'
        if sha(audio) != AUDIO_SHA:
            raise ValueError('Pinned JFK fixture changed')
        shutil.copy2(audio, out / 'fixture.wav')
        build = scratch / 'build'
        run(['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
             '-DBUILD_SHARED_LIBS=ON', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
             '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_METAL=OFF',
             '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF', '-DCRISPASR_BUILD_SERVER=OFF',
             '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_OPUS_FETCH=ON'], 'configure')
        run(['cmake', '--build', build, '--target', 'crispasr-lib', '-j2'], 'build')
        libraries = list(build.rglob('libcrispasr.so'))
        if len(libraries) != 1:
            raise ValueError('Expected one real native library')
        native = out / 'native'
        native.mkdir(exist_ok=True)
        for dependency in build.rglob('*.so*'):
            if dependency.is_file():
                shutil.copy2(dependency, native / dependency.name)
        shutil.copy2(build / 'compile_commands.json', out / 'compile_commands.json')
        receipt.update(library_sha256=sha(libraries[0]), probe_sha256=sha(ROOT / 'tools/ci-heavy/go-model-lifecycle/main.go'))
        save()
        directories = sorted({p.parent for pattern in ('*.a', '*.so*') for p in build.rglob(pattern) if p.is_file()})
        env = dict(os.environ)
        env['CGO_ENABLED'] = '1'
        env['CGO_LDFLAGS'] = ' '.join(shlex.quote('-L' + str(p)) for p in directories)
        env['LD_LIBRARY_PATH'] = ':'.join(map(str, directories))
        source = ROOT / 'tools/ci-heavy/go-model-lifecycle/main.go'
        binary = scratch / 'go-model-lifecycle'
        run(['go', 'version'], 'go-version', env=env)
        # Compile the actual public packages and their ordinary generated linker
        # directives; no substitute headers, stub libraries or altered bindings.
        run(['go', 'build', '-race', '-o', binary, source], 'go-build', env=env, cwd=ROOT / 'bindings/go')
        shutil.copy2(binary, native / binary.name)
        model = scratch / 'ggml-tiny.en.bin'
        url = f'https://huggingface.co/ggerganov/whisper.cpp/resolve/{MODEL_REV}/ggml-tiny.en.bin'
        with urllib.request.urlopen(url, timeout=180) as response, model.open('wb') as target:
            shutil.copyfileobj(response, target)
        if sha(model) != MODEL_SHA:
            raise ValueError('Pinned model checksum mismatch')
        receipt.update(model_sha256=MODEL_SHA, audio_sha256=AUDIO_SHA, binary_sha256=sha(binary))
        save()
        run([binary, model, audio, out / 'go-results.json'], 'go-model-acceptance', env=env, timeout=1800)
        if 'GO_REAL_MODEL_LIFECYCLE_PASS' not in (out / 'go-model-acceptance.log').read_text():
            raise ValueError('Probe did not complete')
        receipt.update(model_sha256=MODEL_SHA, audio_sha256=AUDIO_SHA,
                       library_sha256=sha(libraries[0]), binary_sha256=sha(binary), probe_sha256=sha(source),
                       source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                                      [source, ROOT / 'tools/ci-heavy/go_model_lifecycle.py'] +
                                      sorted((ROOT / 'bindings/go').glob('*.go')) +
                                      sorted((ROOT / 'bindings/go/pkg/whisper').glob('*.go'))}, passed=True)
    except BaseException:
        receipt['failure'] = traceback.format_exc()
        raise
    finally:
        receipt['artifacts'] = {str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=sha(p))
                                for p in out.rglob('*') if p.is_file() and p.name != 'receipt.json'}
        save()
        (out / 'summary.md').write_text('Real-model Go lifecycle: ' + ('PASS' if receipt['passed'] else 'FAIL') + '\n')


if __name__ == '__main__':
    main()
