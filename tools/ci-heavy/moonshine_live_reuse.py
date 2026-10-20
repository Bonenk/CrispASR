#!/usr/bin/env python3
"""Public Session short/long/short reuse; scalar fixture remains the short-input oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'python'))
sys.path.insert(0, str(ROOT/'tools/ci-heavy'))
import cann_model_fixture as fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'lib', 'audio', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--use-gpu', action='store_true')
    args = parser.parse_args()
    from crispasr import Session
    import numpy as np
    pcm = fixture.audio(args.audio)
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    rows = []
    # One public handle owns all three calls. The longer input must do more work.
    with Session(args.model, lib_path=args.lib, backend='moonshine', n_threads=4, use_gpu=args.use_gpu) as session:
        for index, repeats in enumerate((1, 2, 1)):
            audio = np.tile(pcm, repeats)
            start = time.perf_counter()
            segments = session.transcribe(audio)
            text = ' '.join(segment.text for segment in segments)
            rows.append(dict(index=index, repeats=repeats, samples=len(audio),
                             pcm_sha256=hashlib.sha256(audio.tobytes()).hexdigest(), text=text,
                             words=fixture.words(text), elapsed_seconds=time.perf_counter()-start))
            (out/'reuse.json').write_text(json.dumps(rows, indent=2)+'\n')
    if rows[0]['words'] != rows[2]['words'] or not rows[0]['words']:
        raise ValueError('Short input changed after longer input')
    if len(rows[1]['words']) < 1.5 * len(rows[0]['words']):
        raise ValueError('Long-input output did not establish proportional work')
    print('SESSION_REUSE_PASS calls=3; timings are diagnostic, not a performance claim')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
