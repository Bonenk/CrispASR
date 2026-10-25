#!/usr/bin/env python3
"""Convert bodhan-ai/indic-transcribe-flex (HF safetensors) -> Canary GGUF.

The checkpoint is an IndicCanary / Canary-2 AED model: a 32-layer
FastConformer encoder (d=1024, 8 heads, FFN=4096) plus a 24-layer
pre-LN Transformer decoder (vocab 7152, tied embeddings).

The GGUF uses the existing legacy `canary.*` tensor/metadata layout so
CrispASR's `src/canary.cpp` can load it.  Two additions make the Indic
prompt work:

  canary.prompt_format = "indic"
  canary.special.*     = ids for the 10-token prompt

The prompt is, for language L and mode M:

  <|startofcontext|><|startoftranscript|><|emo:undefined|>
  <|L|><|L|><|pnc|><noitn|itn|><|noromanized|romanized|>
  <|notimestamp|><|nodiarize|>

The tokenizer is the aggregate of `tokenizer_spl_tokens.model` (ids
0..1151) followed by `tokenizer_multilingual.model` (ids 1152..7151),
exactly matching IndicCanaryTokenizer.

Example:
  python models/convert-indic-transcribe-to-gguf.py \
      --hf-dir /path/to/indic-transcribe-flex \
      --output indic-transcribe-flex-f16.gguf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")
try:
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")
try:
    import sentencepiece as spm
except ImportError:
    sys.exit("pip install sentencepiece")


SPL_FILE = "tokenizer_spl_tokens.model"
MULTI_FILE = "tokenizer_multilingual.model"
FEATURE_FILE = "feature_extractor.safetensors"
WEIGHTS_FILE = "model.safetensors"

LANGUAGES = (
    "as", "bgc", "bhb", "bho", "bn", "brx", "doi", "en", "gu", "hi", "hne",
    "kn", "kok", "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa",
    "sat", "sd", "ta", "te", "ur",
)

SPECIAL_TOKENS = {
    "nospeech_id": "<|nospeech|>",
    "pad_id": "<pad>",
    "eos_id": "<|endoftext|>",
    "bos_id": "<|startoftranscript|>",
    "pnc_id": "<|pnc|>",
    "nopnc_id": "<|nopnc|>",
    "startofcontext_id": "<|startofcontext|>",
    "itn_id": "<|itn|>",
    "noitn_id": "<|noitn|>",
    "romanized_id": "<|romanized|>",
    "noromanized_id": "<|noromanized|>",
    "notimestamp_id": "<|notimestamp|>",
    "nodiarize_id": "<|nodiarize|>",
    "emo_id": "<|emo:undefined|>",
}


def remap_encoder(name: str) -> str | None:
    if name.endswith("num_batches_tracked"):
        return None
    if name.startswith("encoder.pre_encode."):
        return name.replace("encoder.pre_encode.", "encoder.pre.")

    if not name.startswith("encoder.layers."):
        return None
    rest = name[len("encoder.layers.") :]
    layer_id, sub = rest.split(".", 1)
    sub = (
        sub.replace("feed_forward1", "ff1")
        .replace("feed_forward2", "ff2")
        .replace("norm_feed_forward1", "norm_ff1")
        .replace("norm_feed_forward2", "norm_ff2")
        .replace("norm_self_att", "norm_attn")
        .replace("self_attn.linear_q", "attn.q")
        .replace("self_attn.linear_k", "attn.k")
        .replace("self_attn.linear_v", "attn.v")
        .replace("self_attn.linear_out", "attn.out")
        .replace("self_attn.linear_pos", "attn.pos")
        .replace("self_attn.pos_bias_u", "attn.pos_bias_u")
        .replace("self_attn.pos_bias_v", "attn.pos_bias_v")
        .replace("conv.pointwise_conv1", "conv.pw1")
        .replace("conv.depthwise_conv", "conv.dw")
        .replace("conv.pointwise_conv2", "conv.pw2")
        .replace("conv.batch_norm", "conv.bn")
    )
    return f"encoder.layers.{layer_id}.{sub}"


def remap_decoder(name: str) -> str | None:
    if name == "model.decoder.embedding.token_embedding.weight":
        return "decoder.embed.weight"
    if name == "model.decoder.embedding.position_embedding.pos_enc":
        return "decoder.pos_enc"
    if name == "model.decoder.embedding.layer_norm.weight":
        return "decoder.embed_ln.weight"
    if name == "model.decoder.embedding.layer_norm.bias":
        return "decoder.embed_ln.bias"

    if name.startswith("model.decoder.layers."):
        rest = name[len("model.decoder.layers.") :]
        layer_id, sub = rest.split(".", 1)
        sub = (
            sub.replace("layer_norm_1", "norm_sa")
            .replace("layer_norm_2", "norm_ca")
            .replace("layer_norm_3", "norm_ff")
            .replace("first_sub_layer.query_net", "sa_q")
            .replace("first_sub_layer.key_net", "sa_k")
            .replace("first_sub_layer.value_net", "sa_v")
            .replace("first_sub_layer.out_projection", "sa_out")
            .replace("second_sub_layer.query_net", "ca_q")
            .replace("second_sub_layer.key_net", "ca_k")
            .replace("second_sub_layer.value_net", "ca_v")
            .replace("second_sub_layer.out_projection", "ca_out")
            .replace("third_sub_layer.dense_in", "ff_in")
            .replace("third_sub_layer.dense_out", "ff_out")
        )
        return f"decoder.layers.{layer_id}.{sub}"

    if name == "model.decoder.final_layer_norm.weight":
        return "decoder.final_norm.weight"
    if name == "model.decoder.final_layer_norm.bias":
        return "decoder.final_norm.bias"

    if name == "lm_head.weight":
        return "decoder.head.weight"
    if name == "lm_head.bias":
        return "decoder.head.bias"
    return None


def remap_name(name: str) -> str | None:
    if name.startswith("model.encoder."):
        return remap_encoder(name[len("model.") :])
    if name.startswith("model.decoder.") or name.startswith("lm_head."):
        return remap_decoder(name)
    return None


def is_f32_tensor(name: str, shape: tuple[int, ...]) -> bool:
    if name.startswith("preprocessor."):
        return True
    if name == "decoder.pos_enc":
        return True
    if name.endswith(".bias"):
        return True
    if "norm" in name or "bn" in name:
        return True
    if "pos_bias_u" in name or "pos_bias_v" in name:
        return True
    return len(shape) <= 1


def load_tokenizer(hf_dir: Path) -> tuple[list[str], dict[str, int]]:
    spl = spm.SentencePieceProcessor(model_file=str(hf_dir / SPL_FILE))
    multi = spm.SentencePieceProcessor(model_file=str(hf_dir / MULTI_FILE))
    tokens = [spl.id_to_piece(i) for i in range(spl.get_piece_size())]
    tokens += [multi.id_to_piece(i) for i in range(multi.get_piece_size())]
    ids = {}
    for key, piece in SPECIAL_TOKENS.items():
        idx = spl.piece_to_id(piece)
        if idx == spl.unk_id() and piece != spl.id_to_piece(spl.unk_id()):
            raise SystemExit(f"special token {piece} missing from {SPL_FILE}")
        ids[key] = idx
    for lang in LANGUAGES:
        idx = spl.piece_to_id(f"<|{lang}|>")
        if idx == spl.unk_id() and f"<|{lang}|>" != spl.id_to_piece(spl.unk_id()):
            raise SystemExit(f"language token <|{lang}|> missing from {SPL_FILE}")
        ids[f"lang_{lang}"] = idx
    return tokens, ids


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-dir", required=True, type=Path, help="directory with model.safetensors")
    p.add_argument("--output", required=True, type=Path, help="output GGUF path")
    p.add_argument("--f32", action="store_true", help="keep every weight F32 (debug only)")
    args = p.parse_args()

    hf_dir: Path = args.hf_dir
    weights_path = hf_dir / WEIGHTS_FILE
    if not weights_path.exists():
        raise SystemExit(f"missing {weights_path}")
    for required in (SPL_FILE, MULTI_FILE, FEATURE_FILE):
        if not (hf_dir / required).exists():
            raise SystemExit(f"missing {hf_dir / required}")

    cfg = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    tokens, special = load_tokenizer(hf_dir)

    with safe_open(str(weights_path), framework="np", device="cpu") as f:
        names = list(f.keys())
        mapped: dict[str, str] = {}
        for name in names:
            dst = remap_name(name)
            if dst is not None:
                mapped[name] = dst

        embed_name = "model.decoder.embedding.token_embedding.weight"
        if embed_name not in mapped:
            raise SystemExit(f"missing embedding tensor {embed_name}")
        embed_shape = tuple(int(x) for x in f.get_slice(embed_name).get_shape())
        vocab_size, d_model = embed_shape

        if "lm_head.weight" not in mapped:
            mapped[embed_name + "::head"] = "decoder.head.weight"

        enc_layers = sorted(
            {int(n.split(".")[3]) for n in mapped if n.startswith("model.encoder.layers.")}
        )
        dec_layers = sorted(
            {int(n.split(".")[3]) for n in mapped if n.startswith("model.decoder.layers.")}
        )
        n_enc = enc_layers[-1] + 1 if enc_layers else int(cfg["encoder_layers"])
        n_dec = dec_layers[-1] + 1 if dec_layers else int(cfg["decoder_layers"])

        enc_ff = int(
            f.get_slice("model.encoder.layers.0.feed_forward1.linear1.weight").get_shape()[0]
        )
        dec_ff = int(
            f.get_slice("model.decoder.layers.0.third_sub_layer.dense_in.weight").get_shape()[0]
        )
        pos_shape = tuple(
            int(x)
            for x in f.get_slice("model.decoder.embedding.position_embedding.pos_enc").get_shape()
        )

        print(f"Loading: {weights_path}")
        print(f"  vocab={vocab_size} d_model={d_model} enc={n_enc}L dec={n_dec}L")
        print(f"  enc_ff={enc_ff} dec_ff={dec_ff} pos_enc={pos_shape}")

        print(f"Writing: {args.output}")
        writer = gguf.GGUFWriter(str(args.output), arch="canary")
        writer.add_string("general.name", "indic-transcribe-flex")

        writer.add_uint32("canary.sample_rate", 16000)
        writer.add_uint32("canary.n_mels", int(cfg.get("num_mel_bins", 128)))
        writer.add_uint32("canary.n_fft", 512)
        writer.add_uint32("canary.win_length", 400)
        writer.add_uint32("canary.hop_length", 160)
        writer.add_uint32("canary.d_model", d_model)
        writer.add_uint32("canary.enc_n_layers", n_enc)
        writer.add_uint32("canary.dec_n_layers", n_dec)
        writer.add_uint32("canary.n_heads", int(cfg.get("encoder_attention_heads", 8)))
        writer.add_uint32("canary.head_dim", d_model // int(cfg.get("encoder_attention_heads", 8)))
        writer.add_uint32("canary.ff_dim", enc_ff)
        writer.add_uint32("canary.subsampling_factor", int(cfg.get("subsampling_factor", 8)))
        writer.add_uint32("canary.subsampling_channels", int(cfg.get("subsampling_conv_channels", 256)))
        writer.add_uint32("canary.conv_kernel", int(cfg.get("conv_kernel_size", 9)))
        writer.add_uint32("canary.vocab_size", vocab_size)
        writer.add_uint32("canary.max_dec_ctx", pos_shape[0])
        writer.add_uint32("canary.frame_dur_cs", 8)

        writer.add_string("canary.prompt_format", "indic")
        writer.add_array("canary.languages", list(LANGUAGES))
        for key, value in special.items():
            if key.startswith("lang_"):
                continue
            writer.add_int32(f"canary.special.{key}", int(value))

        writer.add_array("tokenizer.ggml.tokens", tokens)

        feats = safe_open(str(hf_dir / FEATURE_FILE), framework="np", device="cpu")
        writer.add_tensor("preprocessor.fb", feats.get_tensor("fb").squeeze(0).astype(np.float32))
        writer.add_tensor("preprocessor.window", feats.get_tensor("window").astype(np.float32))

        n_written = 0
        n_f16 = 0
        n_f32 = 0
        for src_name in sorted(mapped):
            dst_name = mapped[src_name]
            lookup = src_name
            if lookup.endswith("::head"):
                lookup = embed_name
            tensor = f.get_tensor(lookup)
            if tensor.dtype != np.float32:
                tensor = tensor.astype(np.float32)
            shape = tuple(int(x) for x in tensor.shape)
            if args.f32 or is_f32_tensor(dst_name, shape):
                tensor = tensor.astype(np.float32)
                n_f32 += 1
            else:
                tensor = tensor.astype(np.float16)
                n_f16 += 1
            writer.add_tensor(dst_name, tensor)
            n_written += 1
            if n_written <= 25 or n_written % 100 == 0:
                print(f"  {dst_name:60s} {str(tensor.shape):28s} {tensor.dtype}")

    if "decoder.head.bias" not in mapped.values():
        raise SystemExit("missing lm_head.bias (required by the Canary decoder)")
    if n_written != len(mapped):
        raise SystemExit(f"wrote {n_written} tensors, mapped {len(mapped)}")

    print(f"  total tensors: {n_written} (F16: {n_f16}, F32: {n_f32})")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Done: {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
