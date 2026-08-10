# CrispASR v0.8.28

A packaging release. v0.8.27 restored five of the six Linux tarballs it set out
to fix; **HIP still did not build one**, for a different reason that the earlier
fix could not have caught. This ships that tarball, and puts the packaging
scripts under test so the next defect in them is visible before a release
rather than after.

**If you use the HIP/ROCm build on Linux**, this is the first release since
v0.8.25 with a `crispasr-linux-x86_64-hip.tar.gz`.

## Fixed — the HIP tarball (#339)

The build always succeeded. Packaging refused it:

```
crispasr           needs  libomp.so
crispasr-quantize  needs  libomp.so
```

`scripts/bundle-linux-runtime.sh` does two things — rewrite `RUNPATH` to
`$ORIGIN`, and copy in the non-system dependency closure — and it did them in
the wrong order. `ldd` resolves through the binary's own `RUNPATH`, so erasing
it first turned exactly those dependencies into `=> not found`, which the copy
loop discarded along with the blank lines.

ROCm's clang links OpenMP against LLVM's `libomp.so` under
`/opt/rocm/lib/llvm/lib` — reachable only that way. gcc's `libgomp.so.1` is on
the default loader path, which is why the other six Linux legs never noticed and
this survived v0.8.27 untouched.

Resolve first, rewrite second; and scan a copied library at its original path,
since a `$ORIGIN`-relative `RUNPATH` means something different once the file has
moved.

An unresolved dependency is now **fatal in the bundler, by name**. It used to be
silent: the script printed `rpaths normalised, 0 librar(ies) bundled` — a
success line, with a count, over a dropped dependency — and left it to a later
gate to notice. The fatal check consults the same exclusion list the copy loop
uses, so a host-provided runtime that is legitimately absent from a build
machine (`libcuda.so.1` on the CUDA legs) still does not fail the release.

## Fixed — the same defect in two more artifact kinds (#341)

Looking for others with the same shape turned up two, neither reported, both
verified against the **published v0.8.27 assets** rather than inferred:

| artifact | needed, and not carried | had a gate? |
|---|---|---|
| `libcrispasr-linux-x86_64-hip` | `libomp.so`, `libhipblas.so.2` | no |
| `crispasr-python-linux-{x86_64,arm64}` | `libgomp.so.1`, `libblas.so.3` | no |

`import crispasr` from the Python tarball would fail in the loader on any host
without OpenBLAS and gcc's OpenMP — the exact failure mode #296 fixed for the
CLI tarballs, in an artifact that never got the fix. The Python legs did the
`RUNPATH` half inline and had no dependency-copy step and no gate at all.

The library bundles are the more instructive one. `verify-lib-bundle.sh` *did*
fail to `dlopen` on `libomp.so` — and `libomp` is on its list of externally
provided sonames, so it printed `dlopen deferred: external driver absent in CI`
and **exited 0**, skipping the remainder of the check as well. That tolerance is
sound for gcc's versioned `libgomp.so.1` on the default loader path. It is wrong
for ROCm clang's unversioned `libomp.so`, which exists only under
`/opt/rocm/lib/llvm/lib`. The comment asserting otherwise has been corrected in
place rather than left to mislead the next reader.

All five Linux library legs and both Python legs now copy the dependency closure
before rewriting rpaths, and run `check-bundled-deps.py` with the same per-leg
contracts the CLI legs use. macOS packaging is untouched.

## The packaging scripts now have a test

Both defects here shipped for the same reason: these scripts ran **only** inside
a release job, so a bug in one could not be observed without publishing a
release. v0.8.18 shipped unloadable library bundles; v0.8.27 shipped no HIP
tarball at all.

`tests/test-bundle-linux-runtime.sh` and `tests/test-package-lib-bundle.sh`
reproduce both conditions with `cc` and `-Wl,-rpath` — no ROCm, no GPU, no
release — and run in the ordinary unit tier on every push. Their acceptance
check is the packaged result *running*, relocated, with its build-time library
directory deleted; not the file being present. Presence is not resolvability,
which is precisely how v0.8.18 got out. Both were red-verified against the
scripts as shipped in v0.8.27 before being trusted, and `patchelf` was added to
the CI unit job — without it they skip, and a skip reads exactly like a pass.

## Licences travel with the binaries now

Every Windows, Android and `libcrispasr` artifact already shipped `LICENSE` and
`THIRD_PARTY_NOTICES.txt`. No CLI tarball did — not the seven Linux legs, not
macOS.

`THIRD_PARTY_NOTICES.txt` also now declares the OpenMP runtimes the Linux
archives actually bundle, which it did not mention at all:

- **libgomp** (GCC's) — GPL-3.0-or-later WITH the GCC Runtime Library Exception.
  The exception is what makes an MIT-licensed binary linking it unproblematic;
  shipping the `.so` is separately a conveyance of a GPLv3 work, so there is now
  a source pointer and a written offer, per GPLv3 §6.
- **libomp** (LLVM's, HIP only) — Apache-2.0 WITH LLVM-exception. Permissive;
  notice only.

CrispASR's own source is unaffected and remains MIT.

It also corrects a claim that had been false since #296: that OpenBLAS is "not
bundled … users must have libopenblas installed".

## Also

- A release dry run (`workflow_dispatch` with an empty `tag`) no longer fails
  `validate-version`, which compared `VERSION` against the branch name. That is
  the mode the input's own documentation recommends for inspecting packaging
  without publishing, and it was red every time.

## Known gaps

- The bundling policy for ROCm's own maths libraries is inconsistent and
  deliberately left alone here: `libamdhip64`, `librocblas` and `libhsa*` are
  treated as host-provided, while `libhipblas`, `libhipblaslt` and `librocsolver`
  — from the same ROCm install — are bundled. That is the behaviour the shipping
  CLI tarball already had; making it consistent either way is a size/contract
  decision, not a bug fix.
