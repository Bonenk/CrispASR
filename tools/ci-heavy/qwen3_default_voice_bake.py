#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Design fresh references, then bake model-specific Qwen Base prompt candidates.

CPU only, on the heavy GitHub runner. No input audio or preset speaker is used.
All source licensing evidence survives independently of native speech acceptance.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import urllib.request
import zipfile

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="1")
OUT = Path(os.environ["HEAVY_OUT"])
SCRATCH = Path(os.environ["HEAVY_SCRATCH"])
SOURCES = {
    "design": ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "5ecdb67327fd37bb2e042aab12ff7391903235d3"),
    "1.7b": ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "fd4b254389122332181a7c3db7f27e918eec64e3"),
    "0.6b": ("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "5d83992436eae1d760afd27aff78a71d676296fc"),
    "tokenizer": ("Qwen/Qwen3-TTS-Tokenizer-12Hz", "7dd38ad4e9bad454aae9cd937d0cd577604fe229"),
}
UPSTREAM_COMMIT = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
WHEEL_URL = "https://files.pythonhosted.org/packages/31/01/95f3ae26f5b7eb2105f69ad2c15af5a28b9ebdd1258a0b27541f5ebd6225/qwen_tts-0.1.1-py3-none-any.whl"
WHEEL_SHA = "11a290d8dabc7ef91a90c54478c8ab19b3edb1d85c0882313721892bdc4af15d"
TEXT = "The morning light comes through the window. We have time to make a good start."
VOICES = {
    "default": "An adult masculine voice with a warm medium-low pitch, clear neutral English pronunciation, and a calm conversational pace. Clean studio sound.",
    "clear": "An adult feminine voice with a clear medium pitch, neutral English pronunciation, and a steady friendly conversational pace. Clean studio sound.",
}
GENERATION = dict(language="English", do_sample=False, subtalker_dosample=False,
                  max_new_tokens=200, non_streaming_mode=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def licensing():
    """Retain publisher declarations, full license, package and exact recipe."""
    import qwen_tts
    evidence = OUT / "licensing"
    evidence.mkdir(parents=True, exist_ok=True)
    declarations = {}
    for label, (repo, revision) in SOURCES.items():
        api_url = f"https://huggingface.co/api/models/{repo}/revision/{revision}"
        body = fetch(api_url)
        metadata = json.loads(body)
        assert metadata["sha"] == revision
        assert metadata.get("cardData", {}).get("license") == "apache-2.0", label
        (evidence / f"{label}-publisher.json").write_bytes(body)
        readme_url = f"https://huggingface.co/{repo}/raw/{revision}/README.md"
        (evidence / f"{label}-README.md").write_bytes(fetch(readme_url))
        extra_notices = []
        for entry in metadata["siblings"]:
            name = entry["rfilename"]
            if Path(name).name.lower().split(".")[0] in ("license", "notice", "copying"):
                target = evidence / f"{label}-{name.replace('/', '_')}"
                target.write_bytes(fetch(f"https://huggingface.co/{repo}/resolve/{revision}/{name}"))
                extra_notices.append(target.name)
        declarations[label] = dict(repo=repo, revision=revision, publisher_api=api_url,
                                   license="Apache-2.0", supplied_notice_files=extra_notices)
    wheel = fetch(WHEEL_URL)
    assert hashlib.sha256(wheel).hexdigest() == WHEEL_SHA
    package = Path(qwen_tts.__file__).parent
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        # Check the executed package sources against the hash-pinned distribution.
        for name in archive.namelist():
            if name.startswith("qwen_tts/") and name.endswith(".py"):
                assert (package / name.removeprefix("qwen_tts/")).read_bytes() == archive.read(name), name
        (evidence / "qwen-tts-0.1.1-LICENSE").write_bytes(
            archive.read("qwen_tts-0.1.1.dist-info/licenses/LICENSE"))
        (evidence / "qwen-tts-0.1.1-METADATA").write_bytes(
            archive.read("qwen_tts-0.1.1.dist-info/METADATA"))
    (OUT / "LICENSE").write_bytes(fetch(
        f"https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/{UPSTREAM_COMMIT}/LICENSE"))
    (OUT / "recipe.py").write_bytes(Path(__file__).read_bytes())
    (OUT / "LICENSE-RECIPE.txt").write_bytes((Path(__file__).resolve().parents[2] / "LICENSE").read_bytes())
    write_json(evidence / "sources.json", dict(models=declarations, package=dict(
        name="qwen-tts", version="0.1.1", wheel_url=WHEEL_URL, wheel_sha256=WHEEL_SHA),
        upstream_code_commit=UPSTREAM_COMMIT, audit_scope="publisher declarations, not upstream training-data clearance"))
    (OUT / "NOTICE").write_text(
        "CrispASR designed Qwen Base voice prompt candidates\n"
        "Generated and encoded with Qwen3-TTS by the Qwen Team / Alibaba Cloud.\n"
        "Upstream code and model publisher declarations: Apache License 2.0.\n"
        "Changes: original generic voice descriptions and reference transcript; newly\n"
        "generated reference waveforms; extraction of Base-model speaker embeddings and\n"
        "codec tokens; GGUF packaging with provenance metadata. No upstream weights or\n"
        "third-party recordings are included. No endorsement by Qwen or Alibaba is implied.\n"
        "recipe.py is MIT-licensed CrispASR code; see LICENSE-RECIPE.txt, including\n"
        "the original copyright and permission notices retained verbatim.\n"
        "The voice assets are offered under Apache-2.0 to the extent the distributor holds\n"
        "licensable rights; no exclusive copyright in machine-generated audio is claimed.\n"
        "No claim is made that generated voices cannot resemble real people, or that\n"
        "upstream training data has been independently cleared. See README.md and licensing/.\n")
    (OUT / "README.md").write_text(
        "---\nlicense: apache-2.0\n---\n# Designed Qwen Base voice prompt candidates\n\n"
        "**AI-generated voices and reference audio. Not recordings of commissioned actors.**\n"
        "Fresh generic descriptions were supplied to the pinned original VoiceDesign model;\n"
        "no input recording, named person, imitation request, existing preset speaker or\n"
        "downloaded voice prompt was used. Names are neutral pack selectors.\n\n"
        "The transcript and descriptions were written for this bake. Base encoders process\n"
        "the saved, disclosed reference WAVs. Separate 1024/2048-dimensional embeddings\n"
        "are required for 0.6B/1.7B; matching dimensions alone do not prove model compatibility.\n\n"
        "## License and scope\n\n"
        "Apache-2.0 applies to licensable voice assets, without asserting copyright\n"
        "where none exists. recipe.py is separately MIT-licensed CrispASR code; see\n"
        "LICENSE-RECIPE.txt with its retained copyright and permission notices.\n"
        "Redistribution is governed by those licenses. Preserve required notices.\n"
        "Keeping the full provenance is recommended, not an extra license condition.\n"
        "Publisher Apache declarations and package license are preserved in licensing/.\n"
        "A model license is not a performer release. No identity or likeness rights,\n"
        "exclusivity, endorsement, universal non-infringement or upstream training-data\n"
        "clearance are represented. AI generation does not prove absence of resemblance.\n"
        "Any recognized resemblance must be reviewed before promotion; unresolved reports\n"
        "block that candidate. This audit is not a legal opinion.\n\n"
        "## Acceptance\n\n"
        "These are candidates until separate reference-ASR, native CPU/F16/Q8, real CUDA\n"
        "and listening/resemblance reviews are recorded. Neither this card nor successful\n"
        "array readback constitutes speech or legal acceptance. Full recipe, versions,\n"
        "source hashes, generation settings and reference audio accompany the packs.\n")


def load(label):
    import torch
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel
    torch.set_num_threads(4)
    repo, revision = SOURCES[label]
    model_path = Path(snapshot_download(repo, revision=revision, local_dir=SCRATCH / label,
                                      allow_patterns=["*.json", "*.safetensors", "*.txt", "speech_tokenizer/*"]))
    files = {str(p.relative_to(model_path)): dict(bytes=p.stat().st_size, sha256=sha(p))
             for p in model_path.rglob("*") if p.is_file() and ".cache" not in p.parts}
    write_json(OUT / f"{label}-source-files.json", files)
    print(f"Loading {repo}@{revision} in F32 on CPU", flush=True)
    return Qwen3TTSModel.from_pretrained(str(model_path), device_map="cpu", dtype=torch.float32,
                                        attn_implementation="eager")


def design():
    import numpy as np
    import soundfile as sf
    import torch
    model = load("design")
    for index, (name, description) in enumerate(VOICES.items()):
        torch.manual_seed(5550 + index)
        print(f"Designing {name}: {description}", flush=True)
        with torch.inference_mode():
            waves, rate = model.generate_voice_design(text=TEXT, instruct=description, **GENERATION)
        audio = np.asarray(waves[0], dtype=np.float32)
        assert rate == 24000 and audio.ndim == 1 and np.isfinite(audio).all()
        assert 2 * rate <= len(audio) <= 16 * rate and np.sqrt(np.mean(audio ** 2)) > .001
        output = OUT / f"ai-generated-{name}-reference.wav"
        sf.write(output, audio, rate, subtype="PCM_16")
        # RIFF INFO disclosure preserves PCM and the transcript used by the encoder.
        comment = b"AI-generated by Qwen3-TTS VoiceDesign; no input recording; see provenance.json.\0"
        subchunk = b"ICMT" + struct.pack("<I", len(comment)) + comment
        if len(comment) % 2:
            subchunk += b"\0"
        chunk = b"INFO" + subchunk
        data = bytearray(output.read_bytes())
        data += b"LIST" + struct.pack("<I", len(chunk)) + chunk
        struct.pack_into("<I", data, 4, len(data) - 8)
        output.write_bytes(data)
        write_json(OUT / f"{name}-generation.json", dict(name=name, description=description, text=TEXT,
                   seed=5550 + index, generation=GENERATION, source_model=SOURCES["design"],
                   reference=output.name, sha256=sha(output), samples=len(audio), rate=rate,
                   input_recordings=[], named_person_targets=[], preset_speakers=[]))
        del waves, audio
        gc.collect()


def bake(size):
    import numpy as np
    import soundfile as sf
    import torch
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Type
    model = load(size)
    dimension = {"0.6b": 1024, "1.7b": 2048}[size]
    output = OUT / f"qwen3-tts-voice-{size}-designed.gguf"
    writer = GGUFWriter(str(output), arch="qwen3tts.voicepack")
    writer.add_array("voicepack.names", list(VOICES))
    writer.add_array("voicepack.ref_texts", [TEXT] * len(VOICES))
    writer.add_string("general.license", "apache-2.0")
    writer.add_string("general.license.link", "https://www.apache.org/licenses/LICENSE-2.0")
    # Preserve notices even when a user copies only the GGUF. Sidecar copies
    # remain part of the downloadable distribution for straightforward reading.
    writer.add_string("crispasr.voice.license_text", (OUT / "LICENSE").read_text())
    writer.add_string("crispasr.voice.notice_text", (OUT / "NOTICE").read_text())
    writer.add_string("crispasr.voice.recipe_license_text", (OUT / "LICENSE-RECIPE.txt").read_text())
    writer.add_string("crispasr.voice.provenance", (OUT / "provenance.json").read_text())
    writer.add_description("AI-generated designed references; Qwen original F32 Base encoder; see accompanying provenance and NOTICE")
    writer.add_string("voicepack.source_model", SOURCES[size][0])
    writer.add_string("voicepack.source_revision", SOURCES[size][1])
    writer.add_uint32("voicepack.speaker_dimension", dimension)
    writer.add_string("crispasr.voice.reference_kind", "ai_generated_voice_design")
    writer.add_string("crispasr.voice.identity_review", "pending; no real-person target supplied")
    writer.add_string("crispasr.voice.speaker_identity", "unknown")
    writer.add_bool("crispasr.voice.cloned_from_recording", False)
    writer.add_bool("crispasr.voice.human_input_recording", False)
    arrays = {}
    records = []
    for name in VOICES:
        reference = OUT / f"ai-generated-{name}-reference.wav"
        audio, rate = sf.read(reference, dtype="float32")
        with torch.inference_mode():
            prompt = model.create_voice_clone_prompt(ref_audio=(audio, rate), ref_text=TEXT,
                                                     x_vector_only_mode=False)[0]
        speaker = np.ascontiguousarray(prompt.ref_spk_embedding.detach().cpu().float().numpy().reshape(-1))
        codes = np.ascontiguousarray(prompt.ref_code.detach().cpu().numpy().astype(np.int32))
        assert speaker.shape == (dimension,) and np.isfinite(speaker).all() and np.linalg.norm(speaker) > 0
        assert codes.ndim == 2 and codes.shape[1] == 16 and codes.shape[0] > 0
        assert codes.min() >= 0 and codes.max() < 2048
        for tensor_name, array, dtype in [(f"voicepack.spk.{name}.embd", speaker, Type.F32),
                                          (f"voicepack.code.{name}.codes", codes, Type.I32)]:
            writer.add_tensor(tensor_name, array, raw_dtype=dtype)
            arrays[tensor_name] = array
        np.save(OUT / f"{size}-{name}-speaker.npy", speaker)
        np.save(OUT / f"{size}-{name}-codes.npy", codes)
        writer.add_string(f"voicepack.reference.{name}.sha256", sha(reference))
        writer.add_string(f"voicepack.reference.{name}.description", VOICES[name])
        records.append(dict(name=name, reference_sha256=sha(reference), speaker_shape=list(speaker.shape),
                            codes_shape=list(codes.shape), speaker_norm=float(np.linalg.norm(speaker))))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    reader = GGUFReader(output)
    assert len(reader.tensors) == len(arrays)
    for tensor in reader.tensors:
        assert np.array_equal(tensor.data.reshape(arrays[tensor.name].shape), arrays[tensor.name])
    write_json(OUT / f"{size}-bake.json", dict(pack=output.name, sha256=sha(output), bytes=output.stat().st_size,
               voices=records, gguf_array_readback_exact=True, original_prompt_extraction_passed=True,
               native_speech_accepted=False, cuda_speech_accepted=False))
    print(f"Wrote {output.name}: {records}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "design", "0.6b", "1.7b"), default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    assert importlib.metadata.version("qwen-tts") == "0.1.1"
    if args.stage != "all":
        design() if args.stage == "design" else bake(args.stage)
        return
    licensing()
    write_json(OUT / "provenance.json", dict(schema=1, recipe_sha256=sha(__file__), source_models=SOURCES,
               producer_commit=os.environ["GITHUB_SHA"], producer_run=os.environ["GITHUB_RUN_ID"],
               recipe_license="MIT", recipe_license_sha256=sha(OUT / "LICENSE-RECIPE.txt"),
               text=TEXT, descriptions=VOICES, generation=GENERATION, input_recordings=[], named_person_targets=[],
               package_versions={p: importlib.metadata.version(p) for p in
                                 ["qwen-tts", "torch", "torchaudio", "transformers", "gguf", "numpy"]},
               licensing_basis="publisher Apache-2.0 declarations and retained full license",
               upstream_training_rights_independently_verified=False, listening_review="pending",
               resemblance_review="pending", native_speech_accepted=False, cuda_speech_accepted=False))
    # Process isolation guarantees that one model's allocations are freed before the next.
    for stage in ("design", "1.7b", "0.6b"):
        subprocess.run([sys.executable, __file__, "--stage", stage], check=True)
    write_json(OUT / "SHA256SUMS.json", {str(p.relative_to(OUT)): sha(p)
               for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.json"})
    print("Candidate bake complete. Native speech and listening/resemblance review remain pending.", flush=True)


if __name__ == "__main__":
    main()
