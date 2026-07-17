# Speaker DB × diarization clusters (issue #266) — PLAN

## NOW — active work

- **Done**: worktree + branch `fix/266-speaker-db-cluster-id`; full code trace of
  the diarization / clustering / speaker-db interaction; EU AI Act compliance
  analysis (below). Issue #266's report is **confirmed accurate** on both claims.
- **In flight**: design sign-off — the compliance stance (claimed-participant
  verification only, no open 1:N DB scan) is recommended but awaits maintainer
  confirmation before implementation starts.
- **Next**: implement the shared pipeline (structured speaker labels + cluster-level
  identification stage), consent gate at the C-ABI, tests, docs.

## Confirmed findings (trace, 2026-07-17)

1. **Slice-level 1:N match, one name for the whole slice.**
   `examples/cli/crispasr_run.cpp:1053` embeds the *entire* dispatcher slice
   (`titanet_embed` over `[sl.start, sl.end)`), `:1056` runs `speaker_db_match`
   (linear scan over all `*.spkr`), `:1059-1060` writes the matched name to
   **every** segment in the slice. A mixed-speaker slice gets one identity.
2. **Global clustering overwrites DB names.** DB match runs per-slice *before*
   `merge_segments` (`crispasr_run.cpp:1315`); global clustering runs after
   (`:1322-1328`) and rewrites `seg.speaker = "(speaker N) "` unconditionally
   (`crispasr_diarize_cli.cpp:842`). So `--speaker-db` + `--diarize-speakers`
   destroys the names it just assigned. Same overwrite reachable from the
   parallel redo path (`crispasr_run.cpp:1252`) and legacy path (`cli.cpp:2668`).
3. **Labels are formatted strings with no provenance.** Both writers target the
   same `seg.speaker` string; nothing records "named from DB" vs "anonymous
   cluster", so precedence can't be enforced today.
4. **Consent gate is CLI-only.** `--speaker-db-consent` is enforced at
   enrollment (`crispasr_run.cpp:472-485`, exit 25) and at match time
   (`:1028-1064`), but the C-ABI primitives `crispasr_speaker_db_enroll/_match`
   (`src/crispasr_c_api.cpp:9756-9788`) have **no gate**; Go bindings expose them.
   The server exposes no speaker-db at all (anonymous diarization only).
5. **Identification is post-only.** The streaming/mic branch
   (`crispasr_run.cpp:2618+`) contains no diarization, clustering, or speaker-db.
   Keep it that way (see compliance).
6. `.spkr` format v1 stores only name (filename) + L2-normed embedding — no
   consent record, no enrollment metadata (`src/speaker_db.cpp:148-174`).

## EU AI Act analysis (Regulation (EU) 2024/1689)

Voiceprints are biometric data (Art 3(34)); matching them against a database of
named profiles is *biometric identification* (Art 3(35)). The classification
that matters:

- **Remote biometric identification (RBI)**, Art 3(41): identification of
  natural persons **without their active involvement**, typically at a distance,
  against a reference database. RBI (real-time *and* post) is **high-risk**
  under Annex III 1(a). High-risk status kills the open-source exemption
  (Art 2(12)) and would put full provider obligations on this repo (conformity
  assessment, EU database registration, risk management, logging, CE marking).
  **We must stay out of this category.**
- **Biometric verification** (Art 3(36)) — confirming a specific person is who
  they claim to be — is explicitly *excluded* from RBI and from Annex III 1(a).
- Anonymous diarization + clustering (`(speaker N)`) identifies no one: not
  biometric identification at all. Minimal risk, unrestricted.
- Art 5(1)(h) prohibits *real-time* RBI in public spaces for law enforcement.
  Speaker-db is post-only today; making that an explicit invariant removes any
  real-time reading.
- GDPR Art 9 applies to the deployer regardless of AI Act classification — the
  existing consent gate stays, and gets extended (below).
- Annex III high-risk obligations become applicable **2026-08-02** — this is
  timely.

**Design consequence.** The current open-ended `--speaker-db` scan ("who, out of
everyone ever enrolled, is speaking?") is 1:N identification of persons who need
not be aware of the processing — the RBI-shaped feature we must not ship. The
compliant reshape is **claimed-participant confirmation**: identification is
restricted to speakers the deployer *explicitly names for the run*, who were
*actively enrolled* (consent + provided sample). Subjects are actively involved
(enrollment + claimed presence), so it is not "remote" BI; per-cluster matching
against a small claimed set is verification-shaped ("is cluster A Alice or Bob,
whom I assert are present?"), and unmatched clusters stay anonymous.

## Target architecture (issue #266 + compliance)

```
ASR segments
  -> diarization (per slice, anonymous)
  -> merge slices
  -> global speaker clustering (anonymous, deterministic (speaker N))
  -> representative embedding per cluster (reuse clustering embeddings)
  -> OPTIONAL identification stage: match each cluster against the
     CLAIMED participant list only (--expect-speakers), consent-gated
  -> final labels: matched cluster -> (Alice); unmatched -> (speaker N)
  -> serialization (named labels never overwritten downstream)
```

Concrete changes:

1. **Structured speaker label** — replace the formatted string as source of
   truth: `{cluster_id, name?, source (diarize|cluster|db), score}` on the
   segment; format at serialization. This is what makes "later stages must not
   overwrite named clusters" enforceable.
2. **Move DB matching after global clustering**, one match per cluster
   (centroid of the embeddings clustering already computed — no new inference).
   Delete the slice-level match block (`crispasr_run.cpp:1037-1064`).
3. **`--expect-speakers "Alice,Bob"`** (name TBD): required allow-list; the
   matcher only compares against these enrolled profiles. `--speaker-db`
   without it = hard error explaining why (no open DB scan).
4. **Standalone (no diarization) path**: preserved as single-cluster
   verification — whole file is one cluster, matched against the claimed list.
5. **Consent, hardened**: `.spkr` v2 adds an enrollment consent attestation +
   timestamp; C-ABI enroll/match gain a consent parameter (or refuse) so
   bindings can't bypass the gate; match-time gate unchanged.
6. **Post-only invariant**: speaker-db remains unreachable from streaming/mic;
   add a test asserting it, and a docs statement of intended purpose
   (cooperative labeling of consenting, enrolled participants in recordings —
   e.g. meeting minutes) + out-of-scope uses (unknown-person identification,
   surveillance, law enforcement, publicly-sourced audio).
7. **Precedence rule**: db-named > anonymous cluster > slice-local diarize
   label; deterministic cluster numbering for unmatched clusters.

## Open decisions (maintainer)

- Flag name/UX for the claimed list; whether an escape hatch for full-DB scan
  should exist at all (recommendation: **no**).
- Whether `.spkr` v1 files (no consent record) keep working (recommendation:
  load with a warning, refuse enrollment in v1 format going forward).
- C-ABI shape: consent param vs session-level option.

## Test plan (sketch)

- Unit: cluster→name precedence; unmatched stays `(speaker N)`; mixed-speaker
  slice never gets one name; v1/v2 `.spkr` load; threshold behavior; dim
  mismatch skip.
- Integration: `--diarize-speakers --speaker-db --expect-speakers` on a 2-spk
  fixture → one named, one anonymous; order-of-stages regression (names survive
  clustering); parallel + sequential path parity; streaming refuses speaker-db.
- ABI: enroll/match without consent → error.
