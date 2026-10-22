"""Adapt active weights for an unconditioned engine benchmark, without requantizing.

F16 Snake alphas are widened exactly to F32.

This is not runtime schema support. The template encoder is retained but unused;
voice cloning with this mixed companion is outside this benchmark's scope.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

ROOT_NAMES = {
    "audio_embd.weight": "audio_embeddings.weight",
    "audio_output.weight": "audio_heads.weight",
    "llm.token_embd.weight": "llm.embed_tokens.weight",
    "llm.output_norm.weight": "llm.norm.weight",
}
LAYER_NAMES = {
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}


def rename(name):
    if name in ROOT_NAMES:
        return ROOT_NAMES[name]
    match = re.fullmatch(r"llm.blk.(\d+).(.+).weight", name)
    assert match, name
    return f"llm.layers.{match[1]}.{LAYER_NAMES[match[2]]}.weight"


def decode_weight(name):
    return (
        name.startswith("acoustic_decoder.")
        or name in ("fc2.weight", "fc2.bias")
        or (
            name.startswith("quantizer.quantizers.")
            and (name.endswith(".codebook.embed") or ".project_out." in name)
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--crispasr-models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {}
    configs = [
        ("llm", "omnivoice-q8_0.gguf", "omnivoice-base-Q8_0.gguf", rename),
        ("codec", "omnivoice-tokenizer-f16.gguf", "omnivoice-tokenizer-Q8_0.gguf", lambda n: n),
    ]
    for kind, source, template, name_fn in configs:
        source_path = args.crispasr_models / source
        reader = GGUFReader(source_path)
        reference = GGUFReader(args.templates / template)
        tensors = list(reader.tensors)
        if kind == "codec":
            tensors = [t for t in reader.tensors if decode_weight(t.name)] + [
                t for t in reference.tensors if not decode_weight(t.name)
            ]
        assert {t.name for t in reference.tensors} <= {name_fn(t.name) for t in tensors}
        dest = args.output / f"same-weights-{kind}.gguf"
        writer = GGUFWriter(dest, reference.fields["general.architecture"].contents())
        for key, field in reference.fields.items():
            if key.startswith("GGUF.") or key == "general.architecture":
                continue
            writer.add_key_value(
                key, field.contents(), field.types[0], field.types[-1] if len(field.types) > 1 else None
            )
        mapping = {}
        originals = {}
        for tensor in tensors:
            name = name_fn(tensor.name)
            assert name not in mapping, name
            # The comparator's DAC alpha loader treats every non-F32 value as
            # BF16. Widen F16 alphas exactly instead of mislabeling their bytes.
            widen = (
                kind == "codec"
                and name.endswith(".alpha")
                and tensor.tensor_type == GGMLQuantizationType.F16
            )
            payload = tensor.data.astype("float32") if widen else tensor.data
            dtype = GGMLQuantizationType.F32 if widen else tensor.tensor_type
            writer.add_tensor(name, payload, raw_dtype=dtype)
            mapping[name] = {
                "source": tensor.name,
                "source_sha256": hashlib.sha256(tensor.data).hexdigest(),
                "adapted_sha256": hashlib.sha256(payload).hexdigest(),
                "exact_f16_to_f32": widen,
            }
            originals[name] = tensor
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        check = GGUFReader(dest)
        assert len(check.tensors) == len(mapping)
        for tensor in check.tensors:
            original = originals[tensor.name]
            assert (tensor.shape == original.shape).all()
            row = mapping[tensor.name]
            if row["exact_f16_to_f32"]:
                assert tensor.tensor_type == GGMLQuantizationType.F32
                assert (tensor.data == original.data.astype("float32")).all()
            else:
                assert tensor.tensor_type == original.tensor_type
            assert hashlib.sha256(tensor.data).hexdigest() == row["adapted_sha256"], tensor.name
        receipt[kind] = {
            "source_sha256": hashlib.file_digest(source_path.open("rb"), "sha256").hexdigest(),
            "adapted_sha256": hashlib.file_digest(dest.open("rb"), "sha256").hexdigest(),
            "verified_tensors": len(mapping),
            "mapping": mapping,
            "scope": "All active LLM/decoder numeric values match CrispASR; F16 alphas are widened exactly; template encoder is unused.",
        }
        print(kind, "verified payloads, shapes, types and exact alpha widening", flush=True)
    (args.output / "adapt-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
