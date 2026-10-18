#!/usr/bin/env python3
"""Measure our opt-in character CTC kernel against a frozen CrispASR baseline.

No models or third-party alignment code. --large belongs on GH heavy workers.
This measures the DP/post-processing kernel, not encoder or end-to-end alignment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
from ctc_align_harness import BASELINE_SHA256, FLAGS, Probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--large', action='store_true')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is None:
        if not os.environ.get('HEAVY_OUT'):
            parser.error('--output is required outside a heavy worker')
        args.output = Path(os.environ['HEAVY_OUT']) / 'alignment-ctc-bench.json'
    if args.large and os.environ.get('GITHUB_ACTIONS') != 'true':
        parser.error('--large requires a hosted GH worker; use tiny fixtures locally')
    if not 3 <= args.repeats <= 25:
        parser.error('--repeats must be in [3,25]')
    cases = [(256, 16, 20), (256, 64, 20), (256, 1024, 20), (256, 4096, 20)]
    if args.large:
        cases += [(2048, 1024, 128), (4096, 4096, 256), (4096, 16384, 256)]
    source = args.source_root / 'src/align.cpp'
    try:
        commit = subprocess.check_output(['git', '-C', str(args.source_root), 'rev-parse', 'HEAD'], text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    compiler = subprocess.check_output(shlex.split(os.environ.get('CXX', 'c++')) + ['--version'], text=True).splitlines()[0]
    cpu = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    receipt = {'schema': 1, 'scope': 'character CTC postprocessor only; no encoder/model/audio',
               'source_commit': commit,
               'source_file_dirty': bool(subprocess.check_output(['git', '-C', str(args.source_root),
                   'status', '--porcelain', '--', 'src/align.cpp'], text=True).strip()),
               'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'baseline_commit': '8c59ed5838861f2f2fa4650a014e7d0d42aa03f8',
               'baseline_source_sha256': BASELINE_SHA256, 'compiler': compiler, 'flags': FLAGS,
               'cpu': cpu, 'architecture': platform.machine(), 'platform': platform.system(),
               'gate': 'CRISPASR_CTC_ALIGN_OPT', 'repeats': args.repeats,
               'warmup_calls_per_arm': 1, 'cases': [],
               'memory_scope': 'peak live C++ allocations inside kernel; excludes input/model/IPC/setup'}
    probe = Probe(receipt['gate'], source_root=args.source_root, build_dir=os.environ.get('HEAVY_SCRATCH'))
    try:
        rng = np.random.default_rng(217)
        for frames, vocabulary, count in cases:
            logits = rng.normal(0, 3, size=(frames, vocabulary)).astype(np.float32)
            vocab = ['<blank>', 'a', 'b', '|'] + [f'unused{i}' for i in range(vocabulary - 4)]
            words = ['ab'] * count
            arms = [('frozen_reference', True, 0), ('gate_off', False, 0), ('gate_on', False, 1)]
            expected = probe.run(logits, words, vocab, reference=True)['bytes']
            samples = {name: [] for name, _, _ in arms}
            for name, reference, gate in arms:
                warm = probe.run(logits, words, vocab, reference=reference, gate=gate)
                if warm['bytes'] != expected:
                    raise RuntimeError(f'warm-up output mismatch in {name}')
            for _ in range(args.repeats):
                for name, reference, gate in arms:
                    result = probe.run(logits, words, vocab, reference=reference, gate=gate)
                    if result['bytes'] != expected:
                        raise RuntimeError(f'output mismatch in {name}')
                    samples[name].append(result)
            row = {'T': frames, 'V': vocabulary, 'words': count,
                   'labels': count * 2 + count - 1, 'unique_labels_including_blank': 4,
                   'output_parity': 'bit-exact word and character stamps', 'arms': {}}
            for name, values in samples.items():
                row['arms'][name] = {
                    'median_ms': statistics.median(v['kernel_ns'] for v in values) / 1e6,
                    'min_ms': min(v['kernel_ns'] for v in values) / 1e6,
                    'max_ms': max(v['kernel_ns'] for v in values) / 1e6,
                    'peak_heap_bytes': max(v['peak_heap_bytes'] for v in values),
                    'allocations': max(v['allocations'] for v in values)}
            before, after = row['arms']['gate_off'], row['arms']['gate_on']
            row['speedup_vs_gate_off'] = before['median_ms'] / after['median_ms']
            row['heap_reduction_vs_gate_off'] = before['peak_heap_bytes'] / after['peak_heap_bytes']
            if before['allocations'] <= after['allocations']:
                raise RuntimeError('Optimization is absent or does not reduce frame allocations')
            receipt['cases'].append(row)
            print(json.dumps(row), flush=True)
    finally:
        probe.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
