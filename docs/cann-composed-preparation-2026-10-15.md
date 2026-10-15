# CANN composed scheduler probe: CPU proof, hardware pending

Candidate `81d8e279b2e739f85219c15e9ded31895e51d53d` prepares the mixed-graph fallback task after merged PR544.
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
retained under storage. Both SDK jobs are queued; general/lint/secret gates
remain pending. SDK jobs retain actual executables, compiler
metadata and loader reports; hosted runners cannot execute the NPU path.
The candidate is not integrated pending those checks.

Physical310P composed fallback, full-model NPU speech,910B ACL capture/replay,
quantization and performance remain open. An ACL build flag or repeated output
does not prove capture/replay executed. See [CANN TODO](cann-todo.md) and
[the candidate source and instructions](https://github.com/CrispStrobe/CrispASR/tree/81d8e279b2e739f85219c15e9ded31895e51d53d/tools/ci-heavy/cann-composed).
[Structured receipt](cann-composed-preparation-2026-10-15.json).
