# Dia layer0 key-history producer diagnostic

This CPU component probe uses the actual original embedding loop, RMSNorm,
DenseGeneral key projection and RotaryEmbedding on979 retained shared token
rows. It loads11 physically verified F32 coefficient tensors, about80MB, and
compares the reconstructed original keys with the independently captured cache
before interpreting native error. No full model or new speech is generated.

`../dia_key_history.py` requires the isolated Torch2.6.0+cpu/NumPy environment.
Its `--inputs` folder is the complete retained `inputs-v1` fixture; `--output`
must be new. It retains all original one-step and batched intermediate arrays,
compares repeat execution, and rejects scale, head and position defects. The
complete proof additionally includes an actual original wrong-token control,
independent full-array audit, configuration/source/parameter pins and notices.

```sh
python tools/ci-heavy/dia_key_history.py \
  --inputs ${CRISPASR_STORAGE}/dia-key-proof/inputs-v1 \
  --output ${CRISPASR_STORAGE}/dia-key-proof/new-original-replay
```

`producer.cpp` compiles the literal captured-runtime embedding/RMS/RoPE helpers
against ggml0a8620765. Its small CPU graph runs the native key producer and
controlled original-input interventions, retaining every stage. Mode0 uses
native embedding/norm/projection/RoPE, mode1 supplies original normalized
inputs, and mode2 supplies original pre-RoPE keys. Only mode0 represents the
full native producer. Modes1/2 isolate arithmetic and cannot establish runtime
or speech acceptance. All input buffers are refreshed on every computation;
its CPU compute plan/workspace is reused explicitly.

The complete proof retains the exact ggml shared libraries, hashes and actual
compile command. From the matching recursive repository checkout, compile
against those retained libraries (`PROOF` is the extracted proof directory):

```sh
c++ -std=c++17 -O2 -I ggml/include \
  tools/ci-heavy/dia-key-history/producer.cpp \
  -L "$PROOF/libs" -Wl,-rpath,"$PROOF/libs" -lggml-cpu -lggml-base \
  -o "$PROOF/new-native-producer"
"$PROOF/new-native-producer" "$PROOF/inputs-v1" "$PROOF/native-inputs-v1" \
  "$PROOF/new-native-output" 0 4
```

Native stage files contain979 consecutive rows, each with2 CFG batches;
hidden/norm rows have2048 values per batch, key rows4×128. The original producer
NPY arrays use batch×time×channels (keys add the head dimension); compare after
moving the native time axis. Complete native keys reproduce the actual captured
cache byte-for-byte, including independent1/4-thread executions. RoPE accounts
for most measured key-history error in this fixture, while an earlier complete
source-RoPE speech experiment still failed. This diagnostic accepts no model,
CUDA, Q4, performance or release claim.
