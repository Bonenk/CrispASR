# CANN composed scheduler probe: CPU proof, hardware pending

Initial candidate `81d8e279b2e739f85219c15e9ded31895e51d53d` prepares the mixed-graph fallback task after merged PR544.
It adds a standalone actual ggml graph and a dedicated immutable-source CPU/SDK
workflow. Production kernels/defaults and model behavior are unchanged.

The graph performs F32 matrix work, a BF16 ADD boundary and F32 work after that
boundary. Supported first/last operations are explicitly assigned to the exact
requested backend. The scheduler chooses the BF16 boundary using real device
capabilities. A physical310P fallback claim additionally requires that CANN
rejects BF16 ADD and that the scheduler actually assigns it to CPU. Missing
CANN, wrong device, a CPU-only fallback claim or existing evidence directory
cannot silently pass. No fake NPU or mocked execution is used by CPU controls.

A fresh source build passes all five contracts. The actual CPU graph executes
18 times over output-column shapes16→32→16, with both repeated and changed
inputs. An independent integer oracle recalculates all12,288 matrix and12,288
final values from retained actual input arrays. All results are exact; data
are deliberately representable in BF16. Repeated shapes/inputs match, while
changed inputs are checked independently. Negative requests reject. The tested
executable, all90 graph input/output payloads, rejection logs, compiler cache/
commands, actual source, notices and hash manifests are retained.

Committed probe source matches the tested source SHA256
`5a5265c9097a5f4e106843bad5d6f83cb76840e09675da38191545c39ad37b39`. The retained actual executable SHA256
is `51b33f65038c1ac56e51eec4791b035f320a4debf67c7ea6e7a6fad33777ecdf`; build logs record its fresh
compilation against ggml0a8620765. These are actual local CPU results, not an
inference about a supplied binary's source from its hash alone.

[Complete128-payload CPU proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/2eff34171877cea00d35fe867de55d16892e218a/cann-20261015-composed/complete-cpu-composed-proof.zip) is public and anonymously verified:
877,822 bytes, SHA256 `ae34420e8060f414cc015a68ce93e18ae1d880dd4b844ba56156e3038a94ed74`.

[Hosted CPU plus official SDK compilation](https://github.com/CrispStrobe/CrispASR/actions/runs/38698323227),
[general CI](https://github.com/CrispStrobe/CrispASR/actions/runs/38698330667),
[lint](https://github.com/CrispStrobe/CrispASR/actions/runs/38698334091) and
[secret scan](https://github.com/CrispStrobe/CrispASR/actions/runs/38698323280)
target this exact candidate. Hosted CPU now passes all five contracts. An
independent audit recalculates all12,288 matrix and12,288 final values from the
complete returned arrays, verifies18 scale negatives and8 rejected requests,
and hashes the actual returned ELF. Raw artifact827,670 bytes has SHA256
`6d97f14420389221bfa4703f83cf2cefb9fa47d7d340455376da775fe21e56ce`,
matching GitHub's artifact digest exactly. The full raw ZIP and job log are
retained under storage. The original secret scan passes. Both original SDK jobs
compile the backend but fail the final static executable link: the SDK library
names propagate from ggml-cann while its private search directories do not.
Both failed raw artifacts match their GitHub SHA256 digests and their full job
logs remain preserved. They do not establish successful executable compilation.

## Static consumer repair

Rebased candidate `f19d41698eccf951c7aade9769d44f9bd263146a` forwards the
backend target's configured SDK link directories to this standalone executable.
It hardcodes no SDK path and changes no production/kernel/default code. The
workflow additionally retains its final linker command.

A fresh CPU build passes all five contracts and independently recalculates
12,288 matrix/12,288 final values,18 scale negatives and8 rejected requests.
Its actual executable SHA remains
`51b33f65038c1ac56e51eec4791b035f320a4debf67c7ea6e7a6fad33777ecdf`.
An actual CMake-generation comparison confirms that the original final link
omits the configured SDK directory and only the corrected link includes it.
That configure-only control installs or executes no fake SDK; real official
SDK executable compilation remains a separate required gate.

[Complete146-payload repair proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/138b0dd326c6746854f00a80b500ed9959920fe4/cann-20261015-composed/static-link-repair-v2.zip) contains the fresh CPU arrays/executable,
compiler/source metadata, independent audit, original/fixed generated commands,
and both rejected SDK logs/raw ZIPs. It is988,939 bytes, SHA256
`7447eee8c9d0d7f53b6df73a93c68378077badeef5d5d8c809a4a9ac7a09ceaf`, fully and anonymously downloaded/rehashed.

[New composed CPU/SDK run](https://github.com/CrispStrobe/CrispASR/actions/runs/38706162449),
[general CI](https://github.com/CrispStrobe/CrispASR/actions/runs/38706180793),
[lint](https://github.com/CrispStrobe/CrispASR/actions/runs/38706184333) and
[secret scan](https://github.com/CrispStrobe/CrispASR/actions/runs/38706162315)
target the repaired candidate. These submitted jobs remain unaccepted pending
terminal artifact audits. The candidate is not integrated.

Physical310P composed fallback, full-model NPU speech,910B ACL capture/replay,
quantization and performance remain open. An ACL build flag or repeated output
does not prove capture/replay executed. See [CANN TODO](cann-todo.md) and
[the candidate source and instructions](https://github.com/CrispStrobe/CrispASR/tree/81d8e279b2e739f85219c15e9ded31895e51d53d/tools/ci-heavy/cann-composed).
[Structured receipt](cann-composed-preparation-2026-10-15.json).
