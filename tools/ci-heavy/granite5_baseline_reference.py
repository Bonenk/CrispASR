#!/usr/bin/env python3
"""Checkpoint the unmodified Granite5 conversion, quants and original F32 oracle.

Hosted CPU only. This baseline intentionally predates the YaRN port and cannot
establish native acceptance. Large outputs are checkpointed individually to a
diagnostic model repo; references go to the regression dataset. No registry edits.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MODEL = "ibm-granite/granite-speech-5.0-2b"
REVISION = "c075cf19e041a2e4d16c5ee24763e5b037d8f479"
BASELINE_CONVERTER = "49c5d562cffa668c59cf300e61c543bb673345bd"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rope_oracle(config_path, output):
    import inspect
    import numpy as np
    import torch
    import transformers
    from transformers import GraniteConfig
    from transformers.models.granite.modeling_granite import GraniteRotaryEmbedding, apply_rotary_pos_emb
    from transformers import modeling_rope_utils

    assert transformers.__version__ == "4.57.3"
    expected = {"modeling_granite.py": "edb70e0780d16eb5e473df244cb9153e96f01fe4944010e784fcd1cbc5a0b542",
                "modeling_rope_utils.py": "55cc0c8cb76f592ab178b4662adcbdf4ad5012bb0b4249c0a96603141f18e9ac"}
    for path in (inspect.getfile(GraniteRotaryEmbedding), inspect.getfile(modeling_rope_utils)):
        assert sha(path) == expected[Path(path).name], path
    config = GraniteConfig.from_dict(json.loads(Path(config_path).read_text())["text_config"])
    rotary = GraniteRotaryEmbedding(config, device="cpu")
    positions = torch.tensor([[0, 1, 127, 4095, 4096, 8191, 16383]], dtype=torch.int64)
    d = config.hidden_size // config.num_attention_heads
    x = torch.zeros(1, positions.shape[1], d, dtype=torch.float32)
    cos, sin = rotary(x, positions)
    q = torch.linspace(-1, 1, positions.numel() * d).reshape(1, 1, -1, d)
    k = torch.flip(q, [-1])
    qr, kr = apply_rotary_pos_emb(q, k, cos, sin)
    np.savez(output, positions=positions.numpy(), inv_freq=rotary.inv_freq.numpy(),
             attention_scaling=np.array(rotary.attention_scaling), cos=cos.numpy(), sin=sin.numpy(),
             q=q.numpy(), k=k.numpy(), q_rotated=qr.numpy(), k_rotated=kr.numpy())
    evidence = {"transformers": transformers.__version__, "original_module": True,
                "rope_scaling": config.rope_scaling,
                "attention_scaling": float(rotary.attention_scaling),
                "source_files": {Path(p).name: sha(p) for p in (
                    inspect.getfile(GraniteRotaryEmbedding), inspect.getfile(modeling_rope_utils))}}
    Path(str(output) + ".json").write_text(json.dumps(evidence, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-repo")
    parser.add_argument("--public-diagnostics", action="store_true")
    parser.add_argument("--rope-oracle", nargs=2, metavar=("CONFIG", "OUTPUT"))
    args = parser.parse_args()
    if args.rope_oracle:
        rope_oracle(*args.rope_oracle)
        return
    if not args.checkpoint_repo:
        parser.error("--checkpoint-repo is required for the hosted producer")
    from huggingface_hub import HfApi, snapshot_download

    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "granite5"
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    prompt = "<|audio|>transcribe the speech with proper punctuation and capitalization."
    os.environ["GRANITE_REFERENCE_PROMPT"] = prompt
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    prefix = f"granite-speech-5.0-2b/{source}"
    receipt = {"source": source, "run": os.environ["GITHUB_RUN_ID"],
               "model": MODEL, "revision": REVISION, "native_acceptance": False,
               "known_gap": "Frozen baseline converter omits YaRN scaling; native acceptance pending",
               "complete": False, "outputs": {}, "checkpoint_repo": args.checkpoint_repo,
               "reference_recipe": {"prompt": prompt, "dtype": "float32", "device": "cpu",
                                    "audio": "samples/jfk.wav", "max_new_tokens": 96,
                                    "do_sample": False, "num_beams": 1,
                                    "changes_from_model_card": "CPU F32, JFK fixture, 96 token bound"},
               "packages": {p: importlib.metadata.version(p) for p in (
                   "torch", "transformers", "torchaudio", "huggingface_hub", "gguf")}}

    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def run(command, tag, timeout=14400):
        with (out / f"{tag}.log").open("wb") as log:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout)
        print(f"{tag}: exit {result.returncode}", flush=True)
        if result.returncode:
            print((out / f"{tag}.log").read_text(errors="replace")[-6000:], flush=True)
            raise RuntimeError(f"{tag} failed; retained log")

    api = HfApi()
    # Prove both commit and actual LFS access before paying for conversion.
    info = api.repo_info(args.checkpoint_repo)
    assert info.private == (not args.public_diagnostics)
    assert info.card_data["license"] == "apache-2.0", "Staging must preserve the model license"
    if args.public_diagnostics:
        assert info.card_data["crispasr_native_acceptance"] is False
    save()
    api.upload_file(path_or_fileobj=out / "receipt.json", path_in_repo=f"{prefix}/receipt.json",
                    repo_id=args.checkpoint_repo, commit_message="Start unaccepted Granite5 baseline")
    api.upload_file(path_or_fileobj=b"Granite5 LFS storage preflight\n" * 64,
                    path_in_repo=f"{prefix}/lfs-storage-preflight.bin", repo_id=args.checkpoint_repo,
                    commit_message="Verify actual diagnostic checkpoint storage")
    assert api.get_paths_info(args.checkpoint_repo, [f"{prefix}/lfs-storage-preflight.bin"])[0].lfs is not None
    model = Path(snapshot_download(MODEL, revision=REVISION,
                                  ignore_patterns=["*.bin", "*.onnx", "*.h5"]))
    cfg = json.loads((model / "config.json").read_text())
    assert cfg["text_config"]["rope_scaling"] == {
        "factor": 4.0, "original_max_position_embeddings": 4096, "rope_type": "yarn"}
    evidence = out / "source"
    evidence.mkdir()
    for name in ("README.md", "config.json", "preprocessor_config.json", "processor_config.json",
                 "generation_config.json", "LICENSE", "NOTICE"):
        if (model / name).is_file():
            shutil.copyfile(model / name, evidence / name)
    model_info = api.model_info(MODEL, revision=REVISION)
    assert model_info.card_data["license"] == "apache-2.0"
    (evidence / "publisher.json").write_text(json.dumps({"repo": MODEL, "sha": model_info.sha,
        "license": model_info.card_data["license"]}, indent=2) + "\n")
    receipt["source_files"] = {str(p.relative_to(model)): {"sha256": sha(p), "bytes": p.stat().st_size}
                               for p in sorted(model.rglob("*")) if p.is_file()}
    receipt["recipe_files"] = {name: sha(ROOT / name) for name in (
        "models/convert-granite-speech-to-gguf.py", "tools/dump_reference.py",
        "tools/reference_backends/granite.py", "tools/ci-heavy/granite5_baseline_reference.py")}
    save()

    def checkpoint(path, reference=False):
        repo = "cstr/crispasr-regression-fixtures" if reference else args.checkpoint_repo
        kind = "dataset" if reference else "model"
        name = f"{prefix}/{path.name}"
        record = {"sha256": sha(path), "bytes": path.stat().st_size, "repo": repo,
                  "repo_type": kind, "path": name}
        try:
            commit = api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=repo, repo_type=kind,
                                     commit_message="Checkpoint unaccepted Granite5 original/baseline")
        except Exception:
            # Preserve expensive produced bytes in the workflow's always-uploaded
            # artifact if the transfer bus fails; never silently lose conversion.
            fallback = out / "failed-transfer"
            fallback.mkdir(exist_ok=True)
            shutil.copyfile(path, fallback / path.name)
            raise
        record["revision"] = commit.oid
        receipt["outputs"][path.name] = record
        save()
        api.upload_file(path_or_fileobj=out / "receipt.json", path_in_repo=f"{prefix}/receipt.json",
                        repo_id=args.checkpoint_repo, commit_message="Record Granite5 checkpoint identity")

    f16 = scratch / "granite-speech-5.0-2b-baseline-f16.gguf"
    # Freeze the pre-YaRN producer even while the native port is developed.
    run(["git", "fetch", "--depth=1", "origin", BASELINE_CONVERTER], "fetch-baseline")
    converter = out / "baseline-converter.py"
    converter.write_bytes(subprocess.check_output([
        "git", "show", f"{BASELINE_CONVERTER}:models/convert-granite-speech-to-gguf.py"], cwd=ROOT))
    receipt["baseline_converter"] = {"source": BASELINE_CONVERTER, "sha256": sha(converter)}
    save()
    run([sys.executable, converter,
         "--input", model, "--output", f16], "convert")
    checkpoint(f16)
    build = scratch / "build"
    if not shutil.which("ccache") or not shutil.which("ninja"):
        run(["sudo", "apt-get", "update"], "apt-update", timeout=600)
        run(["sudo", "apt-get", "install", "-y", "ccache", "ninja-build"], "apt-tools", timeout=600)
    run(["cmake", "-S", ROOT, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
         "-DGGML_NATIVE=OFF", "-DGGML_BLAS=OFF", "-DCRISPASR_BUILD_TESTS=OFF",
         "-DCRISPASR_BUILD_SERVER=OFF"], "configure")
    run(["cmake", "--build", build, "--target", "crispasr-quantize", "-j4"], "build")
    for quant in ("q8_0", "q4_k"):
        path = scratch / f"granite-speech-5.0-2b-baseline-{quant}.gguf"
        run([build / "bin/crispasr-quantize", f16, path, quant], f"quantize-{quant}")
        checkpoint(path)
        path.unlink()
    f16.unlink()
    run([sys.executable, __file__, "--rope-oracle", model / "config.json", out / "original-yarn.npz"],
        "original-yarn")
    checkpoint(out / "original-yarn.npz", reference=True)
    reference = out / "jfk-ref.gguf"
    run([sys.executable, ROOT / "tools/dump_reference.py", "--backend", "granite", "--model-dir", model,
         "--audio", ROOT / "samples/jfk.wav", "--output", reference, "--max-new-tokens", "96"],
        "original-f32-reference")
    checkpoint(reference, reference=True)
    receipt["audio_sha256"] = sha(ROOT / "samples/jfk.wav")
    receipt["complete"] = True
    save()
    api.upload_folder(folder_path=out, path_in_repo=f"{prefix}/evidence", repo_id=args.checkpoint_repo,
                      ignore_patterns=["*.gguf"], commit_message="Retain Granite5 baseline evidence")
    (out / "summary.md").write_text(
        "Original F32 reference and pre-YaRN conversion baseline checkpointed.\n\n"
        "Native parity, decoded-output acceptance and CUDA remain pending.\n")


if __name__ == "__main__":
    main()
