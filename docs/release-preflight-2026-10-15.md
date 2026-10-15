# Accepted-source Linux package preflight

The bounded preflight freezes source
`998f8d18e536bf503fbd4c655caa01a1d461836f` on
`diag/release-preflight-source-20261015`. This accepted source excludes the
unmerged Dia intervention and OmniVoice runtime changes. It still identifies
as version 0.8.43; these are not v0.8.44 assets or the final release freeze.

[CLI run38742749435](https://github.com/CrispStrobe/CrispASR/actions/runs/38742749435)
and [library run38742753167](https://github.com/CrispStrobe/CrispASR/actions/runs/38742753167)
use the existing Release workflow's empty-tag dry run, selecting only
`build-linux` and `build-libs-linux-x86_64`. Both event metadata records match
the frozen SHA and validation branch. The CLI run passes; the library job is
still queued. No tag or
publication was requested. The v0.8.44 tag is absent.

Existing local guards pass against byte-identical frozen scripts:

- All 11 release workflow tests, with no skips.
- Runtime bundler: relocated execution, unresolved dependency rejection,
  host-runtime handling and already-staged dependency controls.
- Shared-library bundler: dependency closure, actual relocated loading/call,
  preserved directory and SOVERSION symlinks, and host-toolkit RUNPATH.
- All 21 redistributable x86 jobs pin portable CPU flags.
- Glint's sync marker exactly matches upstream main at
  `77738f3ed9b15f627196cc5bbd7f6406814ba2fb`.

The JSON receipt retains full guard output, nine exact source-file hashes and
four terminal-cancelled superseded secret scans. Current scans and required
science/lint jobs remain intact. All evidence is under
`/mnt/storage/crispasr/release-preflight-20261015`; tiny symlink-dependent test
fixtures used local temporary storage and were removed by their own traps.

The CLI artifact11980941526 passes its complete official SHA256, CRC and ten
unique safe tar-member checks. Relocated startup reports0.8.43 and998f8d18.
The quantizer has no version switch; its documented usage path executes with
exit1. Actual ldd resolves bundled libraries beside both executables, and the
dependency checker accepts all seven ELF files with `$ORIGIN` RUNPATHs.
All release attachment, creation and notes steps are terminal-skipped.

The library archive and wheel have not been validated. After that job
finishes, verify its raw hash, source, CRC and archive paths, actual relocated
C-ABI loading and dependency closure; then stage,
build and install a wheel from the same frozen Python tree and actual library
bundle. Library discovery must resolve inside the installed package from an
unrelated directory without loader-path overrides. Confirm all publication
steps skipped. These checks do not replace model/GPU acceptance, the full
platform matrix, applicable final-tip CI or the later version-bump process.
