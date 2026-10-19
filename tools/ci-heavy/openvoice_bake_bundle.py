#!/usr/bin/env python3
"""Targeted original-runtime build for GPU parity; no model acceptance here."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'openvoice-cuda-out'
BUILD = ROOT / 'openvoice-cuda-build'
PROJECT = ROOT / 'openvoice-cuda-project'
CUDA = Path(os.environ.get('CUDA_PATH', '/usr/local/cuda'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run(command, label):
    print(label, flush=True)
    with (OUT / (label + '.log')).open('w') as log:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=6000)


def main():
    OUT.mkdir(exist_ok=True)
    PROJECT.mkdir(exist_ok=True)
    # CUDA's linker stub has SONAME libcuda.so.1 but ships as libcuda.so.
    # This alias is link-only, outside the runtime bundle; never package it.
    links = PROJECT / 'link-only'
    links.mkdir(exist_ok=True)
    alias = links / 'libcuda.so.1'
    if not alias.exists():
        alias.symlink_to(next(CUDA.rglob('stubs/libcuda.so')))
    (PROJECT / 'targets.cmake').write_text('''if(TARGET openvoice-bake)
  return()
endif()
set(CMAKE_CXX_STANDARD 17)
add_executable(openvoice-bake "${CRISPASR_SOURCE}/tools/ci-heavy/openvoice-bake/main.cpp"
  "${CRISPASR_SOURCE}/examples/cli/crispasr_diff.cpp")
target_include_directories(openvoice-bake PRIVATE "${CRISPASR_SOURCE}/examples/cli"
  "${CRISPASR_SOURCE}/src")
target_link_libraries(openvoice-bake PRIVATE openvoice2)
add_library(openvoice-probe SHARED "${CRISPASR_SOURCE}/src/openvoice2.cpp" "${CRISPASR_SOURCE}/src/melotts.cpp")
target_link_libraries(openvoice-probe PRIVATE melotts crispasr-core ggml ggml-base ggml-cpu)
target_include_directories(openvoice-probe PRIVATE "${CRISPASR_SOURCE}/src")
''')
    run(['cmake', '-S', ROOT, '-B', BUILD, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
         f'-DCRISPASR_SOURCE={ROOT}', f'-DCMAKE_PROJECT_crispasr_INCLUDE={PROJECT / "targets.cmake"}',
         f'-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,{links}', '-DBUILD_SHARED_LIBS=ON', '-DGGML_CUDA=ON',
         '-DCMAKE_CUDA_ARCHITECTURES=60;75', '-DGGML_NATIVE=OFF', '-DCRISPASR_BUILD_TESTS=OFF',
         '-DCRISPASR_BUILD_EXAMPLES=OFF', '-DCRISPASR_BUILD_SERVER=OFF',
         '-DCRISPASR_OPUS_FETCH=ON', '-DCMAKE_C_COMPILER_LAUNCHER=ccache',
         '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache', '-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache'], 'configure')
    run(['cmake', '--build', BUILD, '--target', 'openvoice-bake', 'openvoice-probe', '-j2'], 'build')
    bundle = OUT / 'bundle'
    bundle.mkdir()
    for name in ['openvoice-bake']:
        candidates = list(BUILD.rglob(name))
        shutil.copy2(next(p for p in candidates if p.is_file()), bundle / name)
    for path in BUILD.rglob('*.so*'):
        if path.is_file() and not path.name.startswith('libcuda.so'):
            dest = bundle / path.name
            if dest.exists():
                assert sha(dest) == sha(path)
            else:
                shutil.copy2(path, dest)
    for name in ['libcudart.so.12', 'libcublas.so.12', 'libcublasLt.so.12']:
        shutil.copy2(next(CUDA.rglob(name)), bundle / name)
    for path in bundle.iterdir():
        subprocess.run(['patchelf', '--set-rpath', '$ORIGIN', str(path)], check=True)
    run(['bash', ROOT / 'scripts/bundle-linux-runtime.sh', bundle], 'runtime-closure')
    run(['python3', ROOT / 'scripts/check-bundled-deps.py', bundle, '--allow', 'libcuda.so.*'], 'dependency-check')
    for name in ['LICENSE', 'THIRD_PARTY_NOTICES.txt']:
        shutil.copy2(ROOT / name, bundle)
    receipt = dict(source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   ggml=subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip(),
                   cuda='12.4.1',build_complete=True, gpu_executed=False, architectures=[60,75],
                   files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in bundle.iterdir()})
    (bundle / 'provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with tarfile.open(OUT / 'openvoice-cuda-bake.tar.gz','w:gz') as archive:
        archive.add(bundle,arcname='bundle')
    shutil.rmtree(bundle)

if __name__ == '__main__':
    main()
