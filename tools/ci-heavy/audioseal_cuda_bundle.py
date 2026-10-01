#!/usr/bin/env python3
"""Build a portable CUDA #482 proof on CI; GPU execution happens on Kaggle."""
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
out = repo / "audioseal-cuda-out"
out.mkdir(exist_ok=True)
wrapper = repo / "audioseal-cuda-wrapper"
wrapper.mkdir(exist_ok=True)
(wrapper / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.18)
project(audioseal_cuda_proof LANGUAGES C CXX)
add_subdirectory("{repo}" engine)
add_executable(audioseal-cuda-proof "{repo}/tools/ci-heavy/audioseal_cuda_proof.cpp")
target_link_libraries(audioseal-cuda-proof PRIVATE audioseal ggml-cuda)
''')
build = repo / "audioseal-cuda-build"
with (out / "build.log").open("w") as log:
    for command in (
        ["cmake", "-G", "Ninja", "-S", str(wrapper), "-B", str(build),
         "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON", "-DGGML_CUDA=ON",
         "-DCMAKE_CUDA_ARCHITECTURES=60;75", "-DGGML_NATIVE=OFF", "-DGGML_BLAS=OFF",
         "-DCRISPASR_BUILD_TESTS=OFF", "-DCRISPASR_BUILD_EXAMPLES=OFF", "-DCRISPASR_BUILD_SERVER=OFF",
         "-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
         "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"],
        ["cmake", "--build", str(build), "--target", "audioseal-cuda-proof", "-j2"],
    ):
        print(command, flush=True)
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)

bundle = out / "bundle"
bundle.mkdir(exist_ok=True)
shutil.copy2(build / "audioseal-cuda-proof", bundle)
for source in build.rglob("*.so*"):
    if source.is_file():
        shutil.copy2(source, bundle / source.name)
# Bundle user-space CUDA runtime dependencies, never a driver/compatibility shim.
for library in ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"):
    candidates = list(Path(os.environ.get("CUDA_PATH", "/usr/local/cuda")).rglob(library))
    if not candidates:
        raise RuntimeError(f"missing CUDA runtime {library}")
    shutil.copy2(candidates[0], bundle / library)
for binary in bundle.iterdir():
    subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(binary)], check=True)
sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
(bundle / "provenance.json").write_text(json.dumps({"sha": sha, "cuda": "12.4.1", "architectures": [60, 75]}, indent=2))
archive = out / "audioseal-cuda-proof.tar.gz"
with tarfile.open(archive, "w:gz") as tar:
    tar.add(bundle, arcname="bundle")
(out / "sha256.txt").write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + "\n")
shutil.rmtree(bundle)
print(f"Packaged {archive.stat().st_size} bytes at {sha}", flush=True)
