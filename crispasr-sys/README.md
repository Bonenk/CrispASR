# crispasr-sys

Raw FFI bindings for [CrispASR](https://github.com/CrispStrobe/CrispASR) — lightweight on-device speech recognition via ggml.

This crate is the raw `extern "C"` FFI shim. Its `build.rs` builds `libcrispasr`
from source with cmake by default, or links a pre-built copy when
`CRISPASR_LIB_DIR` is set.

## Install

The crates are **not on crates.io** — depend via git so `build.rs` has the
CrispASR sources to build against:

```toml
[dependencies]
crispasr-sys = { git = "https://github.com/CrispStrobe/CrispASR" }
```

To skip the cmake build and link a **pre-built** library instead, install it
and point the linker at it:

```bash
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR && cmake -B build && cmake --build build -j && sudo cmake --install build
export CRISPASR_LIB_DIR=/path/to/lib   # e.g. /usr/local/lib
```

The legacy `libwhisper` alias also works:

```bash
export CRISPASR_LIB_NAME=whisper
```

For the safe high-level wrapper see the [`crispasr`](https://github.com/CrispStrobe/CrispASR/tree/main/crispasr) crate.

## License

MIT — see [LICENSE](LICENSE).
