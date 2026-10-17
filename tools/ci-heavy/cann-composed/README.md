# Composed CANN fallback probe

This standalone diagnostic prepares the physical scheduler-fallback task in
[the CANN TODO](../../../docs/cann-todo.md). It does not alter production
backends or require model weights.

The actual graph is F32 matrix multiplication → BF16 conversion → BF16 ADD →
F32 conversion → F32 scale. Supported first/last operations are explicitly
assigned to the requested backend. The scheduler chooses the BF16 boundary
from real capabilities. On310P, `--require-cpu-fallback` additionally requires
that CANN rejects BF16 ADD and that the scheduler assigns it to CPU. A CPU
backend cannot satisfy a CANN request or a CANN fallback claim.

Three graph shapes(16→32→16 output columns) each receive six calls with both
repeated and changed inputs. An independent integer matrix product makes all
BF16 rounding exact and checks every final/intermediate value. The probe saves
actual inputs, expected values and full matrix/final arrays before acceptance.
Existing output directories are rejected. The Python runner independently
recalculates outputs and rejects missing CANN, invalid devices, false CPU
fallback claims and overwritten evidence. It retains the tested executable,
complete logs and per-file hashes.

## CPU control

From the repository root, use a storage-backed fresh build/proof path:

```sh
cmake -S tools/ci-heavy/cann-composed \
  -B ${CRISPASR_STORAGE}/crispasr/cann-composed-20261015/build-cpu \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGGML_NATIVE=OFF -DGGML_CANN=OFF -DGGML_CUDA=OFF \
  -DGGML_VULKAN=OFF -DGGML_BLAS=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build ${CRISPASR_STORAGE}/crispasr/cann-composed-20261015/build-cpu \
  --target cann-composed -j 2
PROBE_BINARY=${CRISPASR_STORAGE}/crispasr/cann-composed-20261015/build-cpu/bin/cann-composed \
PROBE_EVIDENCE_DIR=${CRISPASR_STORAGE}/crispasr/cann-composed-20261015/cpu-proof-v1 \
python3 tools/ci-heavy/cann-composed/test_probe.py -v
```

## Physical validation remains pending

The dedicated workflow builds/runs the CPU control, then compiles this same
probe in the official310P/910B SDK images and retains actual executables,
compiler metadata and loader reports. SDK compilation establishes no NPU
execution. A device run must use the matching SDK/runtime and select
`--backend cann --device N --out FRESH_PATH`; add `--require-cpu-fallback` for
the310P composed-fallback task. That flag is intentionally inappropriate for
a910B device supporting BF16 ADD. These physical paths remain unvalidated.

The standalone executable imports the CANN target's configured SDK library
search directories for static linkage. The SDK workflow retains the final
link command as well as compiler commands. The first310P attempt compiled all
backend sources but failed this final linkage; that attempt remains rejected.

This is operator/scheduler preparation, not full-model speech, ACL capture/
replay, quantization or performance acceptance. ACL build flags or repeated
correct outputs alone do not prove that capture/replay executed. Keep the
separate full-model and ACL tasks in the CANN TODO open.
