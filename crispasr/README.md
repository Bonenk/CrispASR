# crispasr

Safe Rust wrapper for [CrispASR](https://github.com/CrispStrobe/CrispASR) — lightweight on-device speech recognition via ggml.

Supports 17 ASR backends including Whisper, Qwen3-ASR, FastConformer, Canary, Parakeet, Cohere, Granite-Speech, Voxtral, wav2vec2, GLM-ASR, Kyutai-STT, Moonshine, FireRed, OmniASR, and VibeVoice-ASR.

## Install

The crates are **not published on crates.io** — depend on them via git. The
`crispasr-sys` `build.rs` needs the CrispASR C/C++ sources (to build
`libcrispasr` with cmake), which only a repo checkout provides:

```toml
[dependencies]
crispasr = { git = "https://github.com/CrispStrobe/CrispASR" }
```

By default `crispasr-sys`'s `build.rs` builds `libcrispasr` from source with
cmake (needs `cmake` + a C++ toolchain; the git dependency supplies the
sources). To skip that and link a **pre-built** library instead, build/install
it once and point `CRISPASR_LIB_DIR` at it:

```bash
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR && cmake -B build && cmake --build build -j && sudo cmake --install build
# then, in your project:  export CRISPASR_LIB_DIR=/usr/local/lib
```

## Quick start

```rust
use crispasr::CrispAsr;

let model = CrispAsr::open("ggml-base.en.bin")?;
for seg in model.transcribe_file("audio.wav")? {
    println!("[{:.1}s - {:.1}s] {}", seg.start, seg.end, seg.text);
}
```

See the [main repo](https://github.com/CrispStrobe/CrispASR) for full documentation, the model registry, and the CLI.

## License

MIT — see [LICENSE](LICENSE).
