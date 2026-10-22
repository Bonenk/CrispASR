#!/usr/bin/env python3
"""Actual JA Session metadata and short/long/short ownership regression.

This complements the native grouping fixtures. It is not original-model parity
or a boundary-accuracy evaluation, and does not cover generic CLI postprocessing.
"""
import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))


def compact(text):
    return ''.join(text.split())


def serialize(segments):
    rows = []
    for segment in segments:
        row = dataclasses.asdict(segment)
        for token in row['tokens']:
            token['raw_text'] = token['raw_text'].hex()
        rows.append(row)
    return rows


def validate(segments, seconds):
    if not segments or not any(s.words for s in segments):
        raise ValueError('No native words')
    count = 0
    for segment in segments:
        if not segment.tokens:
            raise ValueError('Native words lost token ownership')
        raw = b''.join(t.raw_text for t in segment.tokens).decode('utf8')
        if compact(raw) != compact(segment.text):
            raise ValueError('Lossless token stream does not reconstruct segment text')
        if compact(''.join(w.text for w in segment.words)) != compact(segment.text):
            raise ValueError('Native word text does not reconstruct segment text')
        previous = -1.0
        for token in segment.tokens:
            if not token.has_timestamps or not 0 <= token.start <= token.end <= seconds + .1:
                raise ValueError('Missing or out-of-range token time')
            # Native separator-only tokens can retain an earlier window frame.
            # Preserve those bytes/times verbatim; only content tokens establish
            # chronological ownership. Do not sort or fabricate token times.
            if token.raw_text.strip() and token.start < previous:
                raise ValueError(f'Content token times reordered: previous={previous}, current={token.start}')
            if not math.isfinite(token.confidence) or not 0 <= token.confidence <= 1:
                raise ValueError('Native token probability not retained')
            if token.raw_text.strip():
                previous = token.start
            count += 1
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'lib', 'pcm', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--use-gpu', action='store_true')
    args = p.parse_args()
    import numpy as np
    from crispasr import Session, Parakeet
    pcm = np.load(args.pcm, allow_pickle=False)
    if pcm.dtype != np.float32 or pcm.ndim != 1 or not np.isfinite(pcm).all():
        raise ValueError('Finite mono F32 fixture required')
    out = Path(args.out)
    out.mkdir(exist_ok=False)
    rows = []
    receipt = dict(passed=False, cases=rows, scope=__doc__,
                   model_sha256=hashlib.file_digest(Path(args.model).open('rb'), 'sha256').hexdigest(),
                   library_sha256=hashlib.file_digest(Path(args.lib).open('rb'), 'sha256').hexdigest())
    try:
        with Session(args.model, lib_path=args.lib, backend='parakeet', n_threads=4,
                     use_gpu=args.use_gpu) as session:
            for repeats in (1, 7, 1):
                samples = np.tile(pcm, repeats)
                start = time.monotonic()
                result = session.transcribe(samples, language='ja')
                row = dict(repeats=repeats, samples=len(samples), tokens=sum(len(s.tokens) for s in result),
                           pcm_sha256=hashlib.sha256(samples.tobytes()).hexdigest(),
                           seconds=time.monotonic() - start, result=serialize(result))
                rows.append(row)
                (out / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
                validate(result, len(samples) / 16000)
        if rows[0]['result'] != rows[2]['result']:
            raise ValueError('Short metadata changed after multi-window reuse')
        if rows[1]['tokens'] < 5 * rows[0]['tokens']:
            raise ValueError('Long input lost substantial output')
        # A distinct public low-level native route is the exact short-input oracle.
        with Parakeet(args.model, lib_path=args.lib, n_threads=4) as model:
            native = model.transcribe(pcm)
        flat = [t for s in rows[0]['result'] for t in s['tokens']]
        native_rows = [dict(raw_text=b.hex(), start=t[1] / 100, end=t[2] / 100,
                            confidence=t[3]) for t, b in zip(native['tokens'], native['token_bytes'])]
        public_rows = [{key: t[key] for key in ('raw_text', 'start', 'end', 'confidence')} for t in flat]
        if native_rows != public_rows:
            raise ValueError('Session token bytes/times/probabilities differ from native short result')
        if compact(native['text']) != compact(''.join(s['text'] for s in rows[0]['result'])):
            raise ValueError('Session and native short transcript differ')
        receipt['native_short_tokens'] = native_rows
        receipt['passed'] = True
        print('JA_METADATA_LIVE_PASS counts=' + ','.join(str(r['tokens']) for r in rows), flush=True)
    except BaseException:
        import traceback
        receipt['failure'] = traceback.format_exc()
        raise
    finally:
        (out / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
