#!/usr/bin/env python3
"""Bake a model-sized 1.7B default prompt with the original Qwen package on CPU.

The reference is an already ASR-verified AI-generated CLI output. This produces
an asset candidate; native CPU/CUDA speech acceptance is a separate gate.
"""
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import zipfile

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='1')
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
MODEL = 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'
REVISION = 'fd4b254389122332181a7c3db7f27e918eec64e3'
FIXTURE_REVISION = 'eefe7afa71e4920f4a6e9a3f48d70e21d04f292e'
FIXTURE_PATH = 'issue555-voice-pack-20261023/issue555-cpu-cuda-proof-v1.zip'
FIXTURE_SHA = 'e2de3b75b4efd5f18005128c1be80e30769347427f7be21ef31e16676d766edb'
TEXT = 'I found the file you asked about and opened it.'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    import numpy as np
    import soundfile as sf
    import torch
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Type
    from huggingface_hub import hf_hub_download, snapshot_download
    from qwen_tts import Qwen3TTSModel
    OUT.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(4)
    assert importlib.metadata.version('qwen-tts') == '0.1.1'
    archive = Path(hf_hub_download('cstr/crispasr-regression-fixtures', FIXTURE_PATH,
                                  revision=FIXTURE_REVISION, repo_type='dataset'))
    assert sha(archive) == FIXTURE_SHA
    with zipfile.ZipFile(archive) as z:
        audio_bytes = z.read('cpu/synthetic-reference.wav')
        cpu = json.loads(z.read('cpu/validation.json'))
    assert cpu['cases']['speech-small']['wer'] == 0
    assert hashlib.sha256(audio_bytes).hexdigest() == cpu['cases']['speech-small']['wav_sha256']
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
    assert sr == 24000 and audio.ndim == 1 and np.isfinite(audio).all()
    (OUT/'reference.wav').write_bytes(audio_bytes)
    model_path = snapshot_download(MODEL, revision=REVISION, local_dir=SCRATCH/'qwen17-source',
                                  allow_patterns=['*.json','*.safetensors','*.txt','speech_tokenizer/*'])
    source_files = {str(p.relative_to(model_path)):dict(bytes=p.stat().st_size,sha256=sha(p))
                    for p in Path(model_path).rglob('*') if p.is_file() and '.cache' not in p.parts}
    print('Loading pinned original 1.7B model on CPU in F32', flush=True)
    model = Qwen3TTSModel.from_pretrained(model_path, device_map='cpu', dtype=torch.float32,
                                         attn_implementation='eager')
    prompt = model.create_voice_clone_prompt(ref_audio=(audio, sr), ref_text=TEXT,
                                             x_vector_only_mode=False)[0]
    speaker = np.ascontiguousarray(prompt.ref_spk_embedding.detach().cpu().float().numpy().reshape(-1))
    codes = np.ascontiguousarray(prompt.ref_code.detach().cpu().numpy().astype(np.int32))
    assert speaker.shape == (2048,) and np.isfinite(speaker).all() and np.linalg.norm(speaker) > 0
    assert codes.ndim == 2 and codes.shape[1] == 16 and codes.shape[0] > 0
    assert codes.min() >= 0 and codes.max() < 2048
    np.save(OUT/'speaker.npy', speaker)
    np.save(OUT/'codes.npy', codes)
    output = OUT/'qwen3-tts-voice-1.7b-default.gguf'
    writer = GGUFWriter(str(output), arch='qwen3tts.voicepack')
    writer.add_description('1.7B Base default prompt from an AI-generated reference; original Qwen F32 encoder')
    writer.add_array('voicepack.names', ['default'])
    writer.add_array('voicepack.ref_texts', [TEXT])
    writer.add_string('voicepack.source_model', MODEL)
    writer.add_string('voicepack.source_revision', REVISION)
    writer.add_uint32('voicepack.speaker_dimension', 2048)
    writer.add_string('crispasr.voice.reference_kind', 'ai_generated')
    writer.add_string('crispasr.voice.speaker_identity', 'unknown')
    writer.add_bool('crispasr.voice.cloned_from_recording', False)
    writer.add_string('voicepack.reference_sha256', hashlib.sha256(audio_bytes).hexdigest())
    writer.add_tensor('voicepack.spk.default.embd', speaker, raw_dtype=Type.F32)
    writer.add_tensor('voicepack.code.default.codes', codes, raw_dtype=Type.I32)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    reader = GGUFReader(output)
    by_name = {t.name:t for t in reader.tensors}
    assert np.array_equal(by_name['voicepack.spk.default.embd'].data.reshape(-1), speaker)
    assert np.array_equal(by_name['voicepack.code.default.codes'].data.reshape(codes.shape), codes)
    receipt = dict(source_model=MODEL, source_revision=REVISION, original_dtype='float32',
                   package_versions={p:importlib.metadata.version(p) for p in ['qwen-tts','torch','torchaudio','transformers','gguf','numpy']},
                   source_files=source_files, reference=dict(repo='cstr/crispasr-regression-fixtures',revision=FIXTURE_REVISION,
                   path=FIXTURE_PATH, archive_sha256=FIXTURE_SHA, wav_sha256=hashlib.sha256(audio_bytes).hexdigest(),text=TEXT,
                   samples=len(audio),rate=sr,kind='ai_generated'), speaker_shape=list(speaker.shape),
                   speaker_norm=float(np.linalg.norm(speaker)),codes_shape=list(codes.shape),
                   pack=dict(filename=output.name,sha256=sha(output),bytes=output.stat().st_size),
                   original_prompt_extraction_passed=True, gguf_array_readback_exact=True,
                   native_speech_accepted=False, cuda_speech_accepted=False)
    (OUT/'bake.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['speaker_shape','speaker_norm','codes_shape','pack']},indent=2),flush=True)


if __name__ == '__main__':
    main()
