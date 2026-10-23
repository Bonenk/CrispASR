#!/usr/bin/env python3
"""Build #555 CUDA validation on GitHub; execute real CUDA only on Kaggle."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'qwen3-voice-pack-cuda-out'
BUILD = ROOT / 'qwen3-voice-pack-cuda-build'
CUDA = Path(os.environ.get('CUDA_PATH', '/usr/local/cuda'))
OUT.mkdir(exist_ok=True)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(command, label):
    print(label, flush=True)
    with (OUT / (label + '.log')).open('w') as log:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=7000)


def main():
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    ggml = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT / 'ggml', text=True).strip()
    receipt = dict(source=source, ggml=ggml, cuda='12.4.1', architectures=[60, 75],
                   issue=555,
                   build_complete=False, gpu_executed=False, model_acceptance=False)
    (OUT / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    stubs = list(CUDA.rglob('stubs/libcuda.so'))
    assert stubs, 'Toolkit driver stub required for GPU-less CI linker'
    links = BUILD / 'link-stubs'
    links.mkdir(parents=True, exist_ok=True)
    (links / 'libcuda.so.1').symlink_to(stubs[0])
    # Keep the production DL package configuration. The full test CMake file
    # contains unrelated direct ggml-cpu MODULE links; inject only these probes.
    proof = BUILD / 'proof-targets.cmake'
    proof.write_text(f"""add_executable(qwen3-voice-pack-probe "{ROOT}/tools/ci-heavy/qwen3_voice_pack_probe.cpp")
    target_include_directories(qwen3-voice-pack-probe PRIVATE "{ROOT}/src")
    target_link_libraries(qwen3-voice-pack-probe PRIVATE qwen3_tts)
    target_compile_features(qwen3-voice-pack-probe PRIVATE cxx_std_17)
    add_executable(test-qwen3tts-voice-pack "{ROOT}/tests/test-qwen3tts-voice-pack.cpp")
    target_include_directories(test-qwen3tts-voice-pack PRIVATE "{ROOT}/src")
    target_link_libraries(test-qwen3tts-voice-pack PRIVATE crispasr-core ggml ggml-base crispasr_link_ggml_cpu)
    target_compile_features(test-qwen3tts-voice-pack PRIVATE cxx_std_17)
    set_target_properties(qwen3-voice-pack-probe test-qwen3tts-voice-pack PROPERTIES RUNTIME_OUTPUT_DIRECTORY "{BUILD}/bin")
    """)
    run(['cmake', '-G', 'Ninja', '-S', ROOT, '-B', BUILD,
         f'-DCMAKE_PROJECT_crispasr_INCLUDE={proof}',
         '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
         '-DGGML_BACKEND_DL=ON', '-DGGML_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=60;75',
         '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF', '-DGGML_OPENBLAS=OFF',
         '-DGGML_VULKAN=OFF', '-DGGML_METAL=OFF',
         '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF',
         '-DCRISPASR_BUILD_EXAMPLES=ON',
         f'-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,{links}',
         '-DCMAKE_C_COMPILER_LAUNCHER=ccache', '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
         '-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache'], 'configure')
    run(['cmake', '--build', BUILD, '--target', 'crispasr-cli', 'crispasr-lib',
         'test-qwen3tts-voice-pack', 'qwen3-voice-pack-probe', '-j2'], 'build')
    bundle = OUT / 'bundle'
    bundle.mkdir()
    for name in ['crispasr']:
        shutil.copy2(BUILD / 'bin' / name, bundle / name)
    for name in ['test-qwen3tts-voice-pack', 'qwen3-voice-pack-probe']:
        shutil.copy2(BUILD / 'bin' / name, bundle / name)
    for path in BUILD.rglob('*.so*'):
        if path.is_file() and not path.name.startswith('libcuda.so'):
            soname = subprocess.check_output(['patchelf', '--print-soname', str(path)], text=True).strip()
            dest = bundle / (soname or path.name)
            if dest.exists():
                assert sha(dest) == sha(path), f'Conflicting library {dest.name}'
            else:
                shutil.copy2(path, dest)
            # Explicit C ABI paths and dlopen discovery need the original
            # aliases as well as the ELF SONAME (e.g. libcrispasr.so -> .so.1).
            alias = bundle / path.name
            if path.name != dest.name and not alias.exists():
                alias.symlink_to(dest.name)
    # Copy toolkit user-space runtime, never driver or compatibility shim.
    for name in ['libcudart.so.12', 'libcublas.so.12', 'libcublasLt.so.12']:
        paths = list(CUDA.rglob(name))
        assert paths, name
        shutil.copy2(paths[0], bundle / name)
    for path in bundle.iterdir():
        subprocess.run(['patchelf', '--set-rpath', '$ORIGIN', str(path)], check=True)
    run(['bash', ROOT / 'scripts/bundle-linux-runtime.sh', bundle], 'runtime-closure')
    run(['python3', ROOT / 'scripts/check-bundled-deps.py', bundle,
         '--allow', 'libcuda.so.*'], 'dependency-check')
    assert (bundle / 'libggml-cuda.so').is_file()
    assert (bundle / 'libcrispasr.so').is_file()
    run([bundle / 'test-qwen3tts-voice-pack'], 'unit-lifecycle')
    # The binding must be the same composition as the compiled C ABI.
    shutil.copytree(ROOT / 'python/crispasr', bundle / 'python/crispasr',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.so*'))
    shutil.copy2(ROOT / 'LICENSE', bundle)
    shutil.copy2(ROOT / 'THIRD_PARTY_NOTICES.txt', bundle)
    receipt['build_complete'] = True
    receipt['files'] = {str(p.relative_to(bundle)): dict(sha256=sha(p), bytes=p.stat().st_size)
                        for p in bundle.rglob('*') if p.is_file()}
    (bundle / 'provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    archive = OUT / 'qwen3-voice-pack-cuda-speech-validation.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(bundle, arcname='bundle')
    receipt['archive'] = dict(name=archive.name, sha256=sha(archive), bytes=archive.stat().st_size)
    (OUT / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    shutil.rmtree(bundle)
    print(json.dumps(receipt['archive']), flush=True)


if __name__ == '__main__':
    main()
