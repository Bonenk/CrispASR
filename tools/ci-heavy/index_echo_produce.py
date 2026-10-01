#!/usr/bin/env python3
"""Pinned #485 producer: convert, quantize, upload, independent CPU reference.

Experimental artifacts remain private until stage and decoded-output parity pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from huggingface_hub import HfApi, snapshot_download

SOURCE = 'IndexTeam/Index-Echo-S2TT-2B'
REVISION = '5d98a34d9685869b11e9c94d01dee22a8b8e53b5'
LLAMA_REVISION = '42d958167a748f2c04b1f888e84e7a58f609ddcb'
DESTINATION = 'cstr/index-echo-2b-GGUF'
ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--reference-only', action='store_true')
parser.add_argument('--convert-only', action='store_true')
parser.add_argument('--clips', nargs='+', choices=['jfk', 'zh', 'jfk-tail'], default=['jfk', 'zh', 'jfk-tail'])
args = parser.parse_args()
if args.reference_only and args.convert_only:
    parser.error('--reference-only and --convert-only are mutually exclusive')
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'index-echo'
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(SCRATCH)
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['INDEX_ECHO_REF_THREADS'] = '4'

receipt = dict(source=SOURCE, revision=REVISION, converter_revision=LLAMA_REVISION,
               destination=DESTINATION, validated=False, artifacts=[], events=[])


def event(name):
    receipt['events'].append(dict(name=name, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(name, flush=True)


def run(*args, cwd=ROOT):
    event('run: ' + ' '.join(map(str, args)))
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


api = HfApi(token=os.environ['HF_TOKEN'])
# Maintainer creates the private destination once; CI publishing credentials
# may write existing repositories without permission to create new ones.
# Refuse to overwrite a previously published validated model repository.
if not api.repo_info(DESTINATION).private:
    raise RuntimeError('Producer destination must be private until validated')
# Verify publishing rights before downloading or converting multi-GB weights.
api.upload_file(path_or_fileobj=b'---\nlicense: apache-2.0\n---\n\n# Index-Echo S2TT 2B\n\n'
                b'Private development artifacts for CrispASR issue #485. '
                b'Runtime parity and decoded-output validation are pending.\n',
                path_in_repo='README.md', repo_id=DESTINATION)


def upload(path, remote=None):
    remote = remote or path.name
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    api.upload_file(path_or_fileobj=path, path_in_repo=remote, repo_id=DESTINATION)
    receipt['artifacts'].append(dict(path=remote, bytes=path.stat().st_size, sha256=digest.hexdigest()))
    event('uploaded: ' + remote)


try:
    event('download pinned source')
    source = Path(snapshot_download(SOURCE, revision=REVISION, local_dir=SCRATCH / 'source'))
    upload(source / 'llm' / 'LICENSE', 'LICENSE')
    if not args.reference_only:
        llama = SCRATCH / 'llama-converter'
        run('git', 'init', llama)
        run('git', 'remote', 'add', 'origin', 'https://github.com/ggml-org/llama.cpp.git', cwd=llama)
        run('git', 'fetch', '--depth=1', 'origin', LLAMA_REVISION, cwd=llama)
        run('git', 'checkout', 'FETCH_HEAD', cwd=llama)
        audio = SCRATCH / 'index-echo-2b-f16.gguf'
        decoder = SCRATCH / 'index-echo-2b-decoder-f16.gguf'
        run(sys.executable, ROOT / 'models/convert-index-echo-to-gguf.py',
            '--model', source, '--output', audio)
        upload(audio)
        run(sys.executable, llama / 'convert_hf_to_gguf.py', source / 'llm',
            '--outfile', decoder, '--outtype', 'f16', '--no-mtp')
        # The released inference class uses only the 24 text layers. The HF
        # config retains an MTP layer declaration without its weights; exporting
        # that declaration creates an unloadable, fictitious 25th layer.
        import gguf
        converted_decoder = gguf.GGUFReader(str(decoder))
        assert int(converted_decoder.fields['qwen35.block_count'].contents()) == 24
        assert not any(t.name.startswith('blk.24.') for t in converted_decoder.tensors)
        del converted_decoder
        upload(decoder)
        build = SCRATCH / 'build'
        run('cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
            '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DGGML_NATIVE=OFF')
        run('cmake', '--build', build, '--target', 'crispasr-quantize', '-j', '4')
        quantizer = build / 'bin/crispasr-quantize'
        for quant in ['q8_0', 'q4_k']:
            # Each audio file points to its own quantized decoder companion.
            companion = f'index-echo-2b-decoder-{quant}.gguf'
            run(sys.executable, ROOT / 'models/convert-index-echo-to-gguf.py', '--model', source,
                '--output', audio, '--decoder-name', companion)
            for original, filename in [(audio, f'index-echo-2b-{quant}.gguf'), (decoder, companion)]:
                converted = SCRATCH / filename
                run(quantizer, original, converted, quant)
                upload(converted)
                converted.unlink()
        audio.unlink()
        decoder.unlink()
        receipt['decoder_no_mtp'] = True
        event('all conversion cohorts complete')
        upload(OUT / 'receipt.json', 'conversion-receipt.json')
    for clip in ([] if args.convert_only else args.clips):
        event('independent released Python class: CPU F32 ' + clip)
        audio_path = ROOT / 'samples' / ('paraformer_zh.wav' if clip == 'zh' else 'jfk.wav')
        if clip == 'jfk-tail':
            import wave
            with wave.open(str(audio_path), 'rb') as wav:
                params = wav.getparams()
                # Stress non-hop-aligned length and a partial final conv chunk.
                pcm = wav.readframes(wav.getnframes() - 197)
            audio_path = OUT / 'jfk-tail.wav'
            with wave.open(str(audio_path), 'wb') as wav:
                wav.setparams(params); wav.writeframes(pcm)
            upload(audio_path, 'reference/jfk-tail.wav')
        ref = OUT / f'index-echo-2b-{clip}-ref.gguf'
        run(sys.executable, ROOT / 'tools/dump_reference.py', '--backend', 'index-echo',
            '--model-dir', source, '--audio', audio_path, '--output', ref)
        upload(ref, f'reference/{clip}-ref.gguf')
    event('producer complete; runtime parity pending')
    (OUT / 'summary.md').write_text('Index-Echo 2B pinned conversion and Python reference complete. '
                                   'Artifacts are private and runtime parity is pending.\n')
except Exception:
    event('producer failed; inspect Actions log')
    raise
