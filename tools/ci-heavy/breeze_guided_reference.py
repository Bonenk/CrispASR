#!/usr/bin/env python3
"""Run the pinned upstream Breeze oracle on GH CPU, preserving all stage captures.

This adapts delivery of the reviewed contributor dumper, not its inference.
No Kaggle CPU fallback, automatic fixture overwrite, or native acceptance claim.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

UPSTREAM = '58ec70ce5fa4cc361bdebf77ec40d1365da00ab2'
MODEL = '3e28c5151381a722f1d8661b4118c298caa77aa4'
DUMPER_SHA = '09c26e31a63ec36d852f2918415929c6148e5d1c750a68f9a5cb8e49819b4761'
SCRIPT_VERSION = '2026-10-11.breeze-guided-gh-cpu.1'
CASES = ('plain', 'clone', 'design-s0', 'design-s1', 'design-s4',
         'direct-s0', 'direct-s1', 'direct-s4')
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download

    OUT.mkdir(parents=True, exist_ok=True)
    upstream = SCRATCH / 'breeze-upstream'
    subprocess.run(['git', 'clone', '--no-checkout',
                    'https://github.com/breezeblue-ai/breeze-tts.git', str(upstream)], check=True)
    subprocess.run(['git', 'checkout', '--detach', UPSTREAM], cwd=upstream, check=True)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip()
    assert actual == UPSTREAM
    dumper = ROOT / 'tools/kaggle/breeze-refdump/breeze_refdump.py'
    assert digest(dumper) == DUMPER_SHA, 'Reviewed oracle changed; re-audit before running'
    source = dumper.read_text()
    # Reuse its complete stage/guidance computation; replace only deployment.
    code = source.split('# ── checkpoint ─', 1)[1]
    code = code[code.index('from huggingface_hub import snapshot_download'):]
    code = code.split('# ── upload ─', 1)[0]
    needle = 'snapshot_download(HF_MODEL, local_dir=str(CKPT), token=HF_TOKEN or None,'
    assert code.count(needle) == 1
    code = code.replace(needle, needle + '\n                  revision=MODEL_REVISION,')
    compile(code, str(dumper), 'exec')
    (OUT / 'reviewed-oracle.py').write_text(source)
    (OUT / 'executed-oracle-section.py').write_text(code)
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    dumps = OUT / 'breeze-dumps'
    dumps.mkdir()
    def step(name, **values):
        print(f'[{time.strftime("%H:%M:%S")}] {name} ' + json.dumps(values, default=str), flush=True)
    env = dict(__name__='__main__', json=json, os=os, sys=sys, time=time,
               Path=Path, np=np, torch=torch, sf=sf, step=step,
               WORK=OUT, DUMPS=dumps, BREEZE=upstream, CLONE=ROOT,
               CKPT=SCRATCH / 'breeze-checkpoint', HF_MODEL='BreezeBlue/Breeze-TTS-2',
               HF_TOKEN=os.environ.get('HF_TOKEN', ''), MODEL_REVISION=MODEL,
               DEVICE='cpu', RUN_DEVICE='cpu', RUN_GPU=None, SEED=42, GREEDY=True,
               MAX_FRAMES=24, DUMP_TE_LAYERS=True, N_DUMP_FRAMES=5,
               SCRIPT_VERSION=SCRIPT_VERSION,
               SYN_TEXT='The quick brown fox jumps over the lazy dog.',
               REF_TEXT='And so my fellow Americans, ask not what your country can do for you, '
                        'ask what you can do for your country.')
    receipt = dict(script_version=SCRIPT_VERSION, upstream=UPSTREAM, model_revision=MODEL,
                   source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   oracle_sha256=DUMPER_SHA, sample_sha256=digest(ROOT / 'samples/jfk.wav'),
                   device='cpu', seed=42, greedy=True, max_frames=24,
                   packages={p: importlib.metadata.version(p) for p in
                             ('torch', 'torchaudio', 'transformers', 'qwen-tts', 'numpy')},
                   native_parity_tested=False, full_speech_asr_tested=False,
                   complete=False)
    try:
        exec(compile(code, str(dumper), 'exec'), env)
        assert not env['guided_errors'], env['guided_errors']
        assert set(env['guided_meta']) == set(CASES)
        for case in CASES:
            required = ['prompt_input_ids', 'backbone_logits_frame0', 'codes']
            if case.endswith(('-s0', '-s4')):
                required.append('neg_prompt_input_ids')
            for name in required:
                a = np.load(dumps / 'guided' / case / (name + '.npy'), allow_pickle=False)
                assert a.size and np.isfinite(a).all(), (case, name, a.shape)
                if name == 'codes':
                    assert a.ndim == 2 and a.shape[1] == 16 and a.dtype.kind in 'iu'
                elif name == 'backbone_logits_frame0':
                    assert a.shape == (2052,) and a.dtype.kind == 'f'
                else:
                    assert a.ndim == 1 and a.dtype.kind in 'iu'
        receipt['arrays'] = {}
        for p in sorted(dumps.rglob('*.npy')):
            a = np.load(p, allow_pickle=False)
            assert a.size and np.isfinite(a).all(), str(p)
            receipt['arrays'][str(p.relative_to(dumps))] = dict(
                shape=list(a.shape), dtype=str(a.dtype), sha256=digest(p))
        receipt['guided'] = env['guided_meta']
        receipt['complete'] = True
        print('Original oracle: all eight guidance cases and full finite stage captures COMPLETE.', flush=True)
    except BaseException as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (OUT / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
        (OUT / 'summary.md').write_text(
            f"Breeze original guidance reference: complete={receipt['complete']}. "
            'CPU oracle only; native parity and full-speech ASR remain separate gates.\n')


if __name__ == '__main__':
    main()
