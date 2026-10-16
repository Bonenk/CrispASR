# Dia broader CPU diagnostic — 2026-10-16

The numerical diagnostic passes; speech quality fails on all four cases,
including the original Python positives. This does not accept broader speech,
F16/Q4/GPU, production defaults or a release.

Frozen worker17a7aedc uses runtime5e04955e, full F32 neural weights, critical
F32 DAC output weights, serial head, F32 cross-cache and portable sampling.
Original source4a9e29b1 runs on PyTorch2.6 CPU with MKL_COMPATIBLE. Prompts,
seeds42/123, temperature1.2, top_p0.95, top_k45, CFG3 and capacity3072 were
declared before execution. No failed case was replaced.

[Hosted run38913941927](https://github.com/CrispStrobe/CrispASR/actions/runs/38913941927)
completed all four cases and correctly exited with failure. Official artifact
12057273677 contains3022635432 bytes, SHA256
`75d9f3ec9d4fa96f1e284d2cf602afcf8b288d5cbdaa5bea5bf88532eda26c90`.
Independent audit verifies1829 payloads,352 complete aligned stage comparisons,
actual original sampler replay of20556 decisions, source pins, physical tensor
audits, own-code DAC replay and CLI/Session equality. All input histories and
generated code grids exactly match original Python in all four cases.

| Case | Decoder calls | Original inputs/codes | Worst stage cosine | Largest relative L2 | Original/native ASR WER |
| --- | ---: | --- | ---: | ---: | --- |
| hello-42 | 401 | exact / exact | 0.9999741433 | 0.00719130 | 1 / 1 (empty) |
| hello-123 | 266 | exact / exact | 0.9999710707 | 0.00760665 | 1 / 1 (empty) |
| fox-42 | 455 | exact / exact | 0.9999626558 | 0.00864579 | 1 / 1 (empty) |
| fox-123 | 1162 | exact / exact | 0.9999620016 | 0.00872113 | 1 / 1 (empty) |

All352 rows satisfy cosine>=0.9995 and per-row norm error<2%. Actual ASR
also returns empty text on the original DAC decoding native codes and on CLI
PCM. This finding does not identify whether original generation, capture or
recognition causes the speech failure. Crossed recognizer replay38930760374
uses both retained library builds, the old accepted PCM, four new original
PCMs and JFK, without resynthesis. A pristine original/no-hook comparison is
the next distinct control if recognition does not explain the failure.

Each table entry below is the worst value across the four cases; extrema in
different columns can come from different cases. Full per-case arrays/metrics
and output transcripts are indexed in the [machine receipt](dia-broader-speech-results-2026-10-16.json).
Original decoder-stage comparisons use aligned startup inputs; DAC stages
compare the actual original/native decoders on each native run's own codes.
These are not cosine comparisons of diverged autoregressive trajectories.

| Intermediate | Minimum row cosine | Maximum relative L2 | Maximum row norm error |
| --- | ---: | ---: | ---: |
| cond_logits | 1.0000000000 | 9.6537897e-07 | 6.1045447e-07 |
| cross_k0 | 1.0000000000 | 5.7754703e-07 | 6.7553882e-07 |
| cross_v0 | 1.0000000000 | 6.6066462e-07 | 4.1136042e-07 |
| dac_block_0_own_codes | 0.9999973650 | 0.00029779522 | 0.0012841443 |
| dac_block_1_own_codes | 0.9999987727 | 0.00034644882 | 0.0014505793 |
| dac_block_2_own_codes | 0.9999989740 | 0.00016937629 | 0.00085563404 |
| dac_block_3_own_codes | 0.9999999671 | 0.00014413947 | 0.00030952944 |
| dac_input_conv_own_codes | 0.9999988793 | 0.00021541249 | 0.0009041444 |
| dac_latent_own_codes | 0.9999984670 | 0.00028892945 | 0.00094911579 |
| dac_output_conv_own_codes | 0.9999995765 | 0.00095582665 | 0.00025868177 |
| dac_output_snake_own_codes | 0.9999999066 | 0.00020871954 | 0.00016592919 |
| decoded_audio_own_codes | 0.9999620016 | 0.0087211278 | 0.0002884323 |
| decoder_block0_ca_residual | 1.0000000000 | 6.9610044e-07 | 1.8964575e-07 |
| decoder_block0_cross_attention | 1.0000000000 | 6.7306265e-07 | 2.3114083e-06 |
| decoder_block0_cross_output | 1.0000000000 | 7.692358e-07 | 2.9850198e-07 |
| decoder_block0_cross_q | 1.0000000000 | 4.9301207e-07 | 4.4956351e-07 |
| decoder_block0_input | 1.0000000000 | 0 | 0 |
| decoder_block0_mlp_gate | 1.0000000000 | 5.6287759e-07 | 2.9072001e-07 |
| decoder_block0_mlp_hidden | 1.0000000000 | 1.3888669e-06 | 1.360916e-06 |
| decoder_block0_mlp_output | 1.0000000000 | 2.0016278e-06 | 1.7492912e-06 |
| decoder_block0_mlp_up | 1.0000000000 | 9.8749511e-07 | 2.2565216e-07 |
| decoder_block0_output | 1.0000000000 | 1.7746813e-06 | 1.574536e-06 |
| decoder_block0_pre_ca_norm | 1.0000000000 | 5.2518083e-07 | 2.7294555e-08 |
| decoder_block0_pre_mlp_norm | 1.0000000000 | 5.9966487e-07 | 1.3833942e-07 |
| decoder_block0_pre_sa_norm | 1.0000000000 | 0 | 0 |
| decoder_block0_sa_attention | 1.0000000000 | 6.695147e-07 | 2.9017803e-07 |
| decoder_block0_sa_output | 1.0000000000 | 7.3707194e-07 | 1.6220913e-07 |
| decoder_block0_sa_residual | 1.0000000000 | 7.2680475e-07 | 2.1868524e-07 |
| decoder_block17_ca_residual | 1.0000000000 | 1.1071419e-06 | 2.4000356e-07 |
| decoder_block17_cross_attention | 1.0000000000 | 7.969088e-07 | 1.0547769e-06 |
| decoder_block17_cross_output | 1.0000000000 | 9.3679212e-07 | 2.9994332e-07 |
| decoder_block17_cross_q | 1.0000000000 | 5.8631834e-07 | 5.0451273e-07 |
| decoder_block17_input | 1.0000000000 | 1.0383495e-06 | 2.6220317e-07 |
| decoder_block17_mlp_gate | 1.0000000000 | 1.0438305e-06 | 3.9624028e-07 |
| decoder_block17_mlp_hidden | 1.0000000000 | 1.2321132e-06 | 1.2494723e-06 |
| decoder_block17_mlp_output | 1.0000000000 | 1.7032307e-06 | 7.7678345e-07 |
| decoder_block17_mlp_up | 1.0000000000 | 1.2308213e-06 | 4.2208303e-07 |
| decoder_block17_output | 1.0000000000 | 1.3063162e-06 | 5.1299304e-07 |
| decoder_block17_pre_ca_norm | 1.0000000000 | 8.572428e-07 | 7.9473558e-08 |
| decoder_block17_pre_mlp_norm | 1.0000000000 | 1.5307719e-06 | 2.4805906e-07 |
| decoder_block17_pre_sa_norm | 1.0000000000 | 1.4365122e-06 | 2.6199664e-07 |
| decoder_block17_sa_attention | 1.0000000000 | 2.2191106e-06 | 7.2950446e-07 |
| decoder_block17_sa_output | 1.0000000000 | 1.50574e-06 | 5.2097743e-07 |
| decoder_block17_sa_residual | 1.0000000000 | 1.1285995e-06 | 2.6057973e-07 |
| decoder_ca0 | 1.0000000000 | 7.692358e-07 | 2.9850198e-07 |
| decoder_hidden | 1.0000000000 | 1.8380892e-06 | 4.0691019e-07 |
| decoder_layer_0 | 1.0000000000 | 1.7746813e-06 | 1.574536e-06 |
| decoder_layer_1 | 1.0000000000 | 1.306964e-06 | 1.2273675e-06 |
| decoder_layer_10 | 1.0000000000 | 9.272386e-07 | 7.4886168e-07 |
| decoder_layer_11 | 1.0000000000 | 7.9781131e-07 | 3.7403662e-07 |
| decoder_layer_12 | 1.0000000000 | 7.983949e-07 | 3.5918458e-07 |
| decoder_layer_13 | 1.0000000000 | 9.2078181e-07 | 1.713914e-07 |
| decoder_layer_14 | 1.0000000000 | 9.9719363e-07 | 1.8674085e-07 |
| decoder_layer_15 | 1.0000000000 | 9.9379901e-07 | 2.3833043e-07 |
| decoder_layer_16 | 1.0000000000 | 1.0383495e-06 | 2.6220317e-07 |
| decoder_layer_17 | 1.0000000000 | 1.3063162e-06 | 5.1299304e-07 |
| decoder_layer_2 | 1.0000000000 | 1.2528395e-06 | 1.2492338e-06 |
| decoder_layer_3 | 1.0000000000 | 1.3620571e-06 | 1.4454731e-06 |
| decoder_layer_4 | 1.0000000000 | 1.3018485e-06 | 1.3392993e-06 |
| decoder_layer_5 | 1.0000000000 | 1.3785857e-06 | 1.4338201e-06 |
| decoder_layer_6 | 1.0000000000 | 1.2573791e-06 | 1.194366e-06 |
| decoder_layer_7 | 1.0000000000 | 1.1171336e-06 | 1.0494289e-06 |
| decoder_layer_8 | 1.0000000000 | 1.0005388e-06 | 8.291357e-07 |
| decoder_layer_9 | 1.0000000000 | 8.7868714e-07 | 6.9319724e-07 |
| decoder_sa0_attention | 1.0000000000 | 6.695147e-07 | 2.9017803e-07 |
| decoder_sa0_k | 1.0000000000 | 6.416451e-07 | 2.1665104e-07 |
| decoder_sa0_q | 1.0000000000 | 7.4945458e-07 | 7.5837883e-07 |
| decoder_sa0_v | 1.0000000000 | 6.695147e-07 | 2.9017803e-07 |
| decoder_sa17_attention | 1.0000000000 | 2.2191106e-06 | 7.2950446e-07 |
| decoder_sa17_k | 1.0000000000 | 8.6986787e-07 | 7.4593745e-07 |
| decoder_sa17_q | 1.0000000000 | 6.6979039e-07 | 6.7107188e-07 |
| decoder_sa17_v | 1.0000000000 | 2.2191106e-06 | 7.2950446e-07 |
| encoder_embedding | 1.0000000000 | 0 | 0 |
| encoder_layer_0 | 1.0000000000 | 1.481601e-07 | 1.8951192e-07 |
| encoder_layer_1 | 1.0000000000 | 1.7434047e-07 | 2.563124e-07 |
| encoder_layer_10 | 1.0000000000 | 5.0904957e-07 | 4.4571551e-07 |
| encoder_layer_11 | 1.0000000000 | 2.7332257e-07 | 1.7847408e-06 |
| encoder_layer_2 | 1.0000000000 | 2.5611121e-07 | 2.2293614e-07 |
| encoder_layer_3 | 1.0000000000 | 2.8721647e-07 | 2.433244e-07 |
| encoder_layer_4 | 1.0000000000 | 3.8390665e-07 | 2.6916194e-07 |
| encoder_layer_5 | 1.0000000000 | 3.8042075e-07 | 2.6548046e-07 |
| encoder_layer_6 | 1.0000000000 | 3.8472525e-07 | 2.9106552e-07 |
| encoder_layer_7 | 1.0000000000 | 4.0144007e-07 | 5.9063515e-07 |
| encoder_layer_8 | 1.0000000000 | 3.4536787e-07 | 4.4507828e-07 |
| encoder_layer_9 | 1.0000000000 | 4.2496899e-07 | 4.6437124e-07 |
| encoder_output | 1.0000000000 | 7.316707e-07 | 3.7107947e-07 |
| encoder_tokens | 1.0000000000 | 0 | 0 |
| uncond_logits | 1.0000000000 | 9.4143337e-07 | 6.8383328e-07 |

Complete raw evidence and auditor are retained under
`/mnt/storage/crispasr/dia-broader-speech-20261016`.
Permanent complete-proof publication remains pending; the official GitHub
artifact is retained, not replaced by this table.
