#!/usr/bin/env python3
"""Independently audit retained issue550 outputs, inputs, hashes and device samples."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()



def audit_cache_release(samples, model_bytes):
    """Audit the v3 worker's public-free -> fresh CLI transition on one device.

    The pinned source calls free_aligner_cache before switching to phase `cli`.
    The new child overlaps the next sample, so this is a device-total drop, not
    an allocator balance, exact idle return, free latency or same-process reload.
    """
    if model_bytes <= 0:
        raise ValueError('Positive frozen model size required')
    cached = {}
    first_cli = None
    cli = {}
    for sample in samples:
        rows = {}
        for line in sample.get('samples', '').splitlines():
            device, memory, _ = map(int, line.split(','))
            if device < 0 or memory < 0 or device in rows:
                raise ValueError('Malformed device sample')
            rows[device] = memory
        if sample.get('phase') == 'short-after':
            for device, memory in rows.items():
                cached.setdefault(device, []).append(memory)
        elif sample.get('phase') == 'cli':
            if first_cli is None:
                first_cli = rows
            for device, memory in rows.items():
                cli.setdefault(device, []).append(memory)
    model_mib = model_bytes / 2**20
    active = [device for device, values in cached.items() if min(values) > model_mib / 2]
    if len(active) != 1:
        raise ValueError('Cache release needs one unambiguous resident device')
    device = active[0]
    if first_cli is None or device not in first_cli:
        raise ValueError('No boundary sample for the resident device')
    before = min(cached[device])
    after = first_cli[device]
    drop = before - after
    if drop < .8 * model_mib:
        raise ValueError('Samples do not establish model-sized release on the resident device')
    if max(cli[device]) < .8 * model_mib:
        raise ValueError('Fresh CLI model residency not observed')
    return dict(active_device_index=device, cached_before_min_mib=before,
                first_cli_sample_mib=after, observed_device_total_drop_mib=drop,
                cli_peak_after_reload_mib=max(cli[device]),
                scope='Sampled device-total transition after public cache free and during fresh CLI startup; '
                      'not per-process allocator balance, exact idle return, free-call latency or same-process reload')


def audit(out, fixture, entry):
    import numpy as np
    receipt = json.loads((out/'validation.json').read_text())
    assert receipt['complete'] is True and receipt['gpu_executed'] is True
    assert receipt['source'] == '191815511b85a526aace111b1bda2e28ecb7faad'
    assert receipt['bundle_sha256'] == '63474b5f034ebac850b7b87ef76eecec12d009092d4c72370aeeb6be341365c5'
    assert receipt['entry_sha256'] == sha(entry)
    assert receipt['version'] == '2026-10-20.alignment-cuda-memory.3'
    mutable = []
    for name, row in receipt['artifacts'].items():
        p = out/name
        if name == 'progress.jsonl':
            # The worker writes its terminal progress event after hashing files.
            assert p.read_bytes()[:row['bytes']] and hashlib.sha256(p.read_bytes()[:row['bytes']]).hexdigest() == row['sha256']
            mutable.append(name)
        else:
            assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], name
    cpu = json.loads((fixture/'receipt.json').read_text())
    assert cpu['pins'] == receipt['fixture_pins']
    assert cpu['pins']['model'] == receipt['model']
    pcm = np.load(fixture/'pcm.npy', allow_pickle=False)
    display = json.loads((fixture/'units.json').read_text())['display']
    text = (fixture/'transcript.txt').read_text().strip()
    assert pcm.dtype == np.float32 and pcm.shape == (101760,)
    outputs = []
    for case, (label, repeats, offset) in zip(receipt['cases'], [('short-before',7,0),('long',425,1234),('short-after',7,1234)]):
        assert case['label'] == label and case['repeats'] == repeats and case['offset_cs'] == offset
        words = json.loads((out/(label+'.json')).read_text())
        wanted = display + [s for _ in range(repeats-1) for s in [' '+display[0]] + display[1:]]
        assert len(words) == len(wanted) == repeats*21 == case['units']
        assert [w['word'] for w in words] == wanted
        previous = offset
        for w in words:
            assert type(w['t0']) is int and type(w['t1']) is int
            assert previous <= w['t0'] < w['t1'] <= offset + repeats*636 + 8
            previous = w['t1']
        h = hashlib.sha256()
        for _ in range(repeats):h.update(pcm.tobytes())
        assert h.hexdigest() == case['pcm_sha256']
        assert hashlib.sha256(' '.join([text]*repeats).encode()).hexdigest() == case['transcript_sha256']
        assert case['samples'] == repeats*len(pcm)
        outputs.append(words)
    assert len(receipt['cases']) == 3
    assert outputs[0] == [dict(w, t0=w['t0']-1234, t1=w['t1']-1234) for w in outputs[2]]
    cli = json.loads((out/'cli.json').read_text())
    assert len(cli) == len(outputs[0])
    for c, w in zip(cli, outputs[0]):
        assert c['word'] == w['word'] and abs(c['start']*100-w['t0']) < 1e-7 and abs(c['end']*100-w['t1']) < 1e-7
    native = (out/'public-native.log').read_text()
    splits = len(re.findall(r'^## SPLIT.*CUDA0', native, re.M))
    assert splits == sum(math.ceil(c['samples']/16000/30) for c in receipt['cases'])
    assert len(re.findall(r'^## SPLIT.*CUDA0', (out/'cli.log').read_text(), re.M)) == 2
    samples = [json.loads(s) for s in (out/'gpu-samples.jsonl').read_text().splitlines()]
    summaries = {}
    for label in ('idle','short-before','long','short-after','cli'):
        data = [row.split(',') for sample in samples if sample.get('phase') == label
                for row in sample.get('samples','').splitlines() if row.strip()]
        assert data
        summaries[label] = dict(sampled_device_peak_mib=max(int(row[1]) for row in data),
                                sampled_utilization_max=max(int(row[2]) for row in data))
    assert summaries == receipt['device_samples_summary']
    assert summaries['long']['sampled_utilization_max'] > 0
    return dict(passed=True, cases=3, units=[len(w) for w in outputs], cuda_splits=splits,
                short_after_exact_with_offset=True, cli_public_exact=True,
                input_hashes_recomputed=True, artifact_hashes_verified=True,
                mutable_terminal_progress_prefix_verified=mutable, device_samples_summary=summaries,
                hardware=receipt['hardware'], source=receipt['source'], bundle_sha256=receipt['bundle_sha256'],
                cache_release_evidence=audit_cache_release(samples, receipt['model']['bytes']),
                scope='repeated licensed speech; sampled device totals, not allocator maxima or RTX3090 certification')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('fixture',type=Path)
    p.add_argument('--entry',type=Path,default=Path(__file__).resolve().parents[1]/'kaggle/alignment-cuda-memory/alignment_cuda_memory.py')
    p.add_argument('--receipt',type=Path,required=True)
    a = p.parse_args()
    result = audit(a.output, a.fixture, a.entry)
    a.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print('CUDA_ALIGNMENT_INDEPENDENT_AUDIT_PASS units='+str(result['units']))


if __name__ == '__main__':
    main()
