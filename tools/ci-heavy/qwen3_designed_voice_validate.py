#!/usr/bin/env python3
"""Audit designed voice candidates, then exercise native defaults and the C ABI.

Run CPU on GitHub; CUDA only on actual hardware. A successful run does not
replace a listening/resemblance review or clear upstream training-data rights.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import wave

import numpy as np

from qwen3_voice_pack_validate import PINS, sha, wer

LICENSE_SHA = "a44a6081c73ad75f0255bb2bb5cab74ef1829565a895a24e53a4f11290ab7655"
RECIPE_LICENSE_SHA = "aa918fb53ff1948e12cdae8136e045210fb6db222a1f6f0c14bad48e6de2100e"
MODELS = {
    "design": ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "5ecdb67327fd37bb2e042aab12ff7391903235d3"),
    "1.7b": ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "fd4b254389122332181a7c3db7f27e918eec64e3"),
    "0.6b": ("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "5d83992436eae1d760afd27aff78a71d676296fc"),
    "tokenizer": ("Qwen/Qwen3-TTS-Tokenizer-12Hz", "7dd38ad4e9bad454aae9cd937d0cd577604fe229"),
}
TEXT = "I found the file you asked about and opened it."
SELECTED_VOICE = "clear"  # The other candidate failed full-reference transcription.


def audit(candidate):
    """Independent readback of the generated manifest, licensing and GGUF arrays."""
    from gguf import GGUFReader
    sums = json.loads((candidate / "SHA256SUMS.json").read_text())
    actual = {str(p.relative_to(candidate)) for p in candidate.rglob("*") if p.is_file()
              and p.name != "SHA256SUMS.json"}
    assert set(sums) == actual, "Missing or unmanifested candidate files"
    for name, expected in sums.items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        assert sha(candidate / name) == expected, name
    assert sha(candidate / "LICENSE") == LICENSE_SHA
    assert sha(candidate / "licensing/qwen-tts-0.1.1-LICENSE") == LICENSE_SHA
    proof = json.loads((candidate / "provenance.json").read_text())
    assert sha(candidate / "recipe.py") == proof["recipe_sha256"]
    assert proof["recipe_license"] == "MIT"
    assert sha(candidate / "LICENSE-RECIPE.txt") == proof["recipe_license_sha256"] == RECIPE_LICENSE_SHA
    assert proof["input_recordings"] == [] and proof["named_person_targets"] == []
    assert proof["upstream_training_rights_independently_verified"] is False
    assert proof["native_speech_accepted"] is False and proof["cuda_speech_accepted"] is False
    assert proof["source_models"] == {key: list(value) for key, value in MODELS.items()}
    for label, (repo, revision) in MODELS.items():
        publisher = json.loads((candidate / f"licensing/{label}-publisher.json").read_text())
        assert publisher["id"] == repo and publisher["sha"] == revision
        assert publisher["cardData"]["license"] == "apache-2.0"
        frontmatter = (candidate / f"licensing/{label}-README.md").read_text().split("---", 2)[1]
        assert "license: apache-2.0" in [line.strip() for line in frontmatter.splitlines()]
    for size, dimension in (("0.6b", 1024), ("1.7b", 2048)):
        path = candidate / f"qwen3-tts-voice-{size}-designed.gguf"
        reader = GGUFReader(path)
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        def string(key):
            field = reader.fields[key]
            return bytes(field.parts[field.data[0]]).decode()
        assert string("general.license") == "apache-2.0"
        assert string("crispasr.voice.license_text") == (candidate / "LICENSE").read_text()
        assert string("crispasr.voice.notice_text") == (candidate / "NOTICE").read_text()
        assert string("crispasr.voice.recipe_license_text") == (candidate / "LICENSE-RECIPE.txt").read_text()
        assert string("crispasr.voice.speaker_identity") == "unknown"
        clone = reader.fields["crispasr.voice.cloned_from_recording"]
        assert not bool(clone.parts[clone.data[0]][0])
        assert json.loads(string("crispasr.voice.provenance")) == proof
        assert string("voicepack.source_model") == MODELS[size][0]
        assert string("voicepack.source_revision") == MODELS[size][1]
        for key, expected in (("voicepack.names", ["default", "clear"]),
                              ("voicepack.ref_texts", [proof["text"], proof["text"]])):
            field = reader.fields[key]
            assert [bytes(field.parts[index]).decode() for index in field.data] == expected
        bake = json.loads((candidate / f"{size}-bake.json").read_text())
        assert bake["sha256"] == sha(path) and bake["bytes"] == path.stat().st_size
        assert bake["original_prompt_extraction_passed"] and bake["gguf_array_readback_exact"]
        assert bake["native_speech_accepted"] is False and bake["cuda_speech_accepted"] is False
        assert len(tensors) == 4
        for name in ("default", "clear"):
            generation = json.loads((candidate / f"{name}-generation.json").read_text())
            reference = candidate / generation["reference"]
            assert generation["sha256"] == sha(reference)
            assert generation["description"] == proof["descriptions"][name]
            assert generation["text"] == proof["text"]
            assert generation["source_model"] == list(MODELS["design"])
            assert generation["input_recordings"] == [] and generation["preset_speakers"] == []
            assert generation["named_person_targets"] == []
            assert generation["generation"] == proof["generation"]
            assert string(f"voicepack.reference.{name}.sha256") == sha(reference)
            speaker = np.load(candidate / f"{size}-{name}-speaker.npy", allow_pickle=False)
            codes = np.load(candidate / f"{size}-{name}-codes.npy", allow_pickle=False)
            assert speaker.shape == (dimension,) and speaker.dtype == np.float32
            assert codes.ndim == 2 and codes.shape[1] == 16 and codes.dtype == np.int32
            assert np.isfinite(speaker).all() and np.linalg.norm(speaker) > 0
            assert codes.min() >= 0 and codes.max() < 2048
            for key, array in ((f"voicepack.spk.{name}.embd", speaker), (f"voicepack.code.{name}.codes", codes)):
                assert np.array_equal(tensors[key].data.reshape(array.shape), array), key
    return proof


def single_voice(candidate, size, name, output, output_name=None):
    """Select one original pair, unchanged, for the CLI's single active voice."""
    from gguf import GGUFWriter, GGUFReader, GGMLQuantizationType as Type
    proof = json.loads((candidate / "provenance.json").read_text())
    output_name = output_name or name
    writer = GGUFWriter(str(output), arch="qwen3tts.voicepack")
    writer.add_array("voicepack.names", [output_name])
    writer.add_array("voicepack.ref_texts", [proof["text"]])
    writer.add_string("general.license", "apache-2.0")
    writer.add_string("crispasr.voice.license_text", (candidate / "LICENSE").read_text())
    writer.add_string("crispasr.voice.notice_text", (candidate / "NOTICE").read_text())
    writer.add_string("crispasr.voice.recipe_license_text", (candidate / "LICENSE-RECIPE.txt").read_text())
    writer.add_string("crispasr.voice.speaker_identity", "unknown")
    writer.add_bool("crispasr.voice.cloned_from_recording", False)
    writer.add_string("crispasr.voice.provenance", (candidate / "provenance.json").read_text())
    writer.add_description(f"Selected original {name} reference as {output_name}; encoder arrays unchanged")
    writer.add_string("voicepack.source_model", MODELS[size][0])
    writer.add_string("voicepack.source_revision", MODELS[size][1])
    writer.add_string("crispasr.voice.selected_source_voice", name)
    writer.add_string("crispasr.voice.selected_description", proof["descriptions"][name])
    writer.add_string("crispasr.voice.curation_recipe_sha256", sha(__file__))
    writer.add_string("crispasr.voice.producer_manifest_sha256", sha(candidate / "SHA256SUMS.json"))
    arrays = {}
    for kind, suffix, dtype in (("spk", "speaker", Type.F32), ("code", "codes", Type.I32)):
        key = f"voicepack.{kind}.{output_name}." + ("embd" if kind == "spk" else "codes")
        arrays[key] = np.load(candidate / f"{size}-{name}-{suffix}.npy", allow_pickle=False)
        writer.add_tensor(key, arrays[key], raw_dtype=dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    reader = GGUFReader(output)
    assert len(reader.tensors) == 2
    for tensor in reader.tensors:
        assert np.array_equal(tensor.data.reshape(arrays[tensor.name].shape), arrays[tensor.name])


def session_child(args):
    sys.path.insert(0, str(args.bundle / "python"))
    from crispasr import Session
    with Session(str(args.child_model), lib_path=str(args.bundle / "libcrispasr.so"),
                 use_gpu=args.device == "cuda", verbosity=1, n_threads=4) as session:
        assert session.backend == "qwen3-tts", session.backend
        session.set_codec_path(str(args.child_codec))
        session.set_voice(str(args.child_pack))
        pcm = session.synthesize(TEXT)  # Normal marked API; never synthesize_raw.
        assert session.output_sample_rate() == 24000
    assert pcm.ndim == 1 and np.isfinite(pcm).all() and len(pcm) > 12000
    with wave.open(str(args.child_wav), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    print("C_ABI_AUTODETECT_MARKED_SPEECH_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    for field in ("model", "codec", "pack", "wav"):
        parser.add_argument("--child-" + field, type=Path)
    args = parser.parse_args()
    if args.audit_only:
        assert args.candidate
        audit(args.candidate)
        print("DESIGNED_VOICE_CANDIDATE_AUDIT_PASS; no speech or legal clearance claimed")
        return
    assert args.bundle
    if args.child_model:
        session_child(args)
        return
    from huggingface_hub import hf_hub_download
    assert args.out and args.scratch and args.candidate
    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    receipt = dict(complete=False, speech_pass=False, licensing_evidence_pass=False,
                   device=args.device, cases={}, models={}, listening_review="pending",
                   resemblance_review="pending", upstream_training_rights_independently_verified=False)
    def save():
        (args.out / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    save()
    environment = dict(os.environ, LD_LIBRARY_PATH=str(args.bundle) + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
                       GGML_BACKEND_PATH=str(args.bundle), CRISPASR_QWEN3_TTS_MAX_FRAMES="180",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4")
    def run(command, label, gpu=False, reject=False):
        print(label, flush=True)
        peak = 0
        with (args.out / f"{label}.log").open("w") as log:
            child = subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, env=environment)
            deadline = time.monotonic() + 1800
            while child.poll() is None:
                if args.device == "cuda" and gpu:
                    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                                    "--format=csv,noheader,nounits"], text=True)
                    peak = max([peak] + [int(row.split(",")[1]) for row in rows.splitlines()
                                        if row.strip() and int(row.split(",")[0]) == child.pid])
                if time.monotonic() > deadline:
                    child.kill()
                    child.wait()
                    raise TimeoutError(label)
                time.sleep(.5)
        receipt["cases"][label] = dict(returncode=child.returncode, peak_gpu_mib=peak)
        save()
        log = (args.out / f"{label}.log").read_text(errors="replace")
        if reject:
            assert child.returncode == 15 and "incompatible voice pack" in log, label
        else:
            assert child.returncode == 0, label
        if args.device == "cuda" and gpu:
            assert peak >= 512 and "CUDA" in log, label
        return log
    cli = args.bundle / "crispasr"
    common = ["-t", "4"] + (["-ng"] if args.device == "cpu" else [])
    marking = ["--no-spoken-disclaimer", "--accept-marking-responsibility"]
    def judge(path, label, expected, threshold=.1):
        with wave.open(str(path)) as wav:
            assert wav.getframerate() == 24000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float64) / 32768
        assert len(pcm) > 12000 and np.isfinite(pcm).all() and np.sqrt(np.mean(pcm ** 2)) > .001
        prefix = args.out / (label + "-asr")
        run([cli, "-m", models["judge"], "-f", path, "-ng", "-nt", "-np", "-otxt", "-of", prefix], label + "-judge")
        heard = prefix.with_suffix(".txt").read_text()
        error = wer(expected, heard)
        receipt["cases"][label].update(wer=error, expected=expected, heard=heard, wav_sha256=sha(path))
        save()
        assert error <= threshold, (label, error, heard)
    try:
        if args.device == "cuda":
            receipt["hardware"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
                                                          "--format=csv,noheader"], text=True).strip()
            assert receipt["hardware"]
        proof = audit(args.candidate)
        receipt.update(licensing_evidence_pass=True, candidate_manifest_sha256=sha(args.candidate / "SHA256SUMS.json"))
        pins = dict(PINS)
        pins["large-q8"] = (PINS["large"][0], PINS["large"][1], "qwen3-tts-12hz-1.7b-base-q8_0.gguf")
        models = {}
        directory = args.scratch / "models"
        for label, (repo, revision, name) in pins.items():
            path = Path(hf_hub_download(repo, name, revision=revision, local_dir=directory))
            models[label] = path
            receipt["models"][label] = dict(repo=repo, revision=revision, filename=name, sha256=sha(path))
            save()
        receipt["curation"] = dict(selected_source_voice=SELECTED_VOICE, output_voice="default",
                                   reason="Raw default diagnostic recovered only the first sentence; clear matched the full transcript",
                                   curation_recipe_sha256=sha(__file__), encoder_arrays_changed=False)
        for name in (SELECTED_VOICE,):
            label = "reference-" + name
            receipt["cases"][label] = {}
            judge(args.candidate / f"ai-generated-{name}-reference.wav", label, proof["text"], threshold=0)
        # Leave the legacy 1024-d generic pack alongside BOTH designed packs.
        # Successful no-voice 1.7B renders must therefore prefer metadata-sized defaults.
        for size in ("0.6b", "1.7b"):
            filename = f"qwen3-tts-voice-{size}-designed.gguf"
            # Do not install the raw two-voice candidate. Select only the vetted
            # reference, retaining the separately extracted model-sized arrays.
            curated = args.out / "defaults"
            curated.mkdir(exist_ok=True)
            single_voice(args.candidate, size, SELECTED_VOICE, curated / filename, output_name="default")
            shutil.copy2(curated / filename, directory / filename)
            single_voice(args.candidate, size, SELECTED_VOICE, directory / f"clear-{size}.gguf")
            receipt["curation"][size] = dict(filename=filename, sha256=sha(curated / filename))
            save()
        for filename in ("LICENSE", "LICENSE-RECIPE.txt", "NOTICE"):
            shutil.copy2(args.candidate / filename, curated / filename)
        shutil.copy2(__file__, curated / "curation-recipe.py")
        (curated / "curation.json").write_text(json.dumps(receipt["curation"], indent=2) + "\n")
        renamed = directory / "renamed-model.gguf"
        assert not renamed.exists(), "Use fresh scratch for each run"
        os.link(models["large"], renamed)
        for key, size in (("small", "0.6b"), ("large", "1.7b"), ("large-q8", "1.7b")):
            for suffix, text in (("hello", "Hello world."), ("sentence", TEXT), ("clear", TEXT)):
                label = f"{key}-{suffix}"
                output = args.out / f"ai-generated-{label}.wav"
                explicit = ["--voice", directory / f"clear-{size}.gguf", "--i-have-rights"] if suffix == "clear" else []
                log = run([cli, "-m", models[key], "--tts", text, "--tts-output", output,
                           *explicit, *marking, *common], label, gpu=True)
                if not explicit:
                    assert f"qwen3-tts-voice-{size}-designed.gguf" in log, label
                judge(output, label, text, threshold=0 if suffix == "hello" else .1)
        label = "renamed-large"
        output = args.out / f"ai-generated-{label}.wav"
        log = run([cli, "-m", renamed, "--tts", TEXT, "--tts-output", output, *marking, *common], label, gpu=True)
        assert "qwen3-tts-voice-1.7b-designed.gguf" in log
        judge(output, label, TEXT)
        for key, size in (("small", "0.6b"), ("large", "1.7b")):
            label = "session-" + key
            output = args.out / f"ai-generated-{label}.wav"
            log = run([sys.executable, __file__, "--bundle", args.bundle, "--device", args.device,
                       "--child-model", models[key], "--child-codec", models["codec"],
                       "--child-pack", directory / f"qwen3-tts-voice-{size}-designed.gguf", "--child-wav", output],
                      label, gpu=True)
            assert "C_ABI_AUTODETECT_MARKED_SPEECH_PASS" in log
            judge(output, label, TEXT)
        forbidden = args.out / "must-not-exist.wav"
        run([cli, "-m", models["large"], "--voice", models["voice"], "--tts", "Hello world.",
             "--i-have-rights", "--tts-output", forbidden, *common], "wrong-pack-rejection", gpu=True, reject=True)
        assert not forbidden.exists()
        receipt.update(complete=True, speech_pass=True)
    finally:
        save()


if __name__ == "__main__":
    main()
