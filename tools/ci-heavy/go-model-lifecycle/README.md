# Real-model Go lifecycle acceptance

Dispatch `heavy-cpu.yml` with `script=tools/ci-heavy/go_model_lifecycle.py` on
an immutable source branch, Linux runner, no extra pip dependencies. This is
hosted-only: it builds the real native library and ordinary public Go packages,
including generated linker directives. No substitute C headers or stub libraries
are allowed. Enable `CRISPASR_OPUS_FETCH=ON`, as the generated Go link also
requires the actual Opus/Ogg/opusfile libraries. Runtime artifacts are retained
before linking; no model is downloaded until Go compilation succeeds.
`go build -race` checks Go-visible accesses; it does not instrument
native C++ memory accesses.

The tiny.en model revision/hash and JFK audio hash match the accepted earlier
CPU metadata fixture. New scope: sequential baseline, callback close/reentry
rejection, concurrently attempted close/decode while an actual native encoder
callback is held, encoder cancellation followed by recovery, two-worker parallel
decode, and owned segment retention after close. Retain source/model/audio and
binary identities, exact segment/token outputs, logs and the receipt.

Native encoder-begin cancellation breaks the decode loop with status zero; the
probe requires no retained output and subsequent recovery rather than inventing
an error-return requirement. Concurrent Close may return ErrModelBusy; after
operations finish, repeated Close must succeed. Do not widen these contracts to
raw low-level Go pointers, which still require caller serialization.

A passing receipt accepts this CPU/high-level/greedy fixture only. It does not
establish original-reference numerical parity, WER, GPU, DTW, actual VAD-model
behavior, other-language lifetime safety or package relocation. Audit retained
outputs before archiving acceptance in HISTORY.md.
