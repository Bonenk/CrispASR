"""Build/run our model-free alignment probe with a frozen CrispASR reference.

The child measures kernel wall time and live C++ allocations during alignment;
input tensors, transcript/vocabulary setup and result serialization are excluded.
No third-party implementation, model or fitted data is used.
"""
from pathlib import Path
import hashlib
import os
import shlex
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA256 = '3a09740cb1c4fdaa278c34e63dbd0a613479d13fd02172e436b40f76ab0e9ec8'
FLAGS = ['-std=c++17', '-O3', '-DNDEBUG', '-fno-fast-math']


class Probe:
    def __init__(self, gate, source_root=ROOT, build_dir=None):
        if sys.byteorder != 'little':
            raise RuntimeError('The test wire format requires a little-endian host')
        self.tmp = tempfile.TemporaryDirectory(dir=build_dir)
        root = Path(source_root)
        output = Path(self.tmp.name)
        baseline = ROOT / 'tests/fixtures/align-baseline-8c59ed583.cpp'
        if hashlib.sha256(baseline.read_bytes()).hexdigest() != BASELINE_SHA256:
            raise RuntimeError('Immutable CrispASR alignment baseline changed')
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        include = ['-I', str(root / 'src')]
        subprocess.run(compiler + FLAGS + include + ['-Dctc_forced_align=ctc_forced_align_reference',
            '-c', str(baseline), '-o', str(output / 'reference.o')], check=True)
        subprocess.run(compiler + FLAGS + include + [str(root / 'src/align.cpp'),
            str(ROOT / 'tests/ctc_align_probe.cpp'), str(output / 'reference.o'),
            '-o', str(output / 'probe')], check=True)
        self.proc = subprocess.Popen([str(output / 'probe'), gate], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def close(self):
        if getattr(self, 'proc', None):
            self.proc.stdin.close()
            try:
                code = self.proc.wait(timeout=10)
            finally:
                self.proc.stdout.close()
            if code:
                raise RuntimeError(f'CTC probe exited with {code}')
            self.proc = None
        self.tmp.cleanup()

    def _read(self, n):
        chunks = []
        while n:
            chunk = self.proc.stdout.read(n)
            if not chunk:
                raise RuntimeError(f'CTC probe stopped unexpectedly (exit {self.proc.poll()})')
            chunks.append(chunk)
            n -= len(chunk)
        return b''.join(chunks)

    def run(self, logits, words, vocab, *, reference=False, gate=1, blank=0, duration=.08, shape=None):
        a = np.ascontiguousarray(logits, dtype='<f4')
        t, v = shape if shape is not None else a.shape
        encoded_words = [w.encode('utf-8') for w in words]
        encoded_vocab = [w.encode('utf-8') for w in vocab]
        header = struct.pack('<8if', t, v, len(words), len(vocab), blank, int(reference), gate, a.size, duration)
        packet = bytearray(header)
        for text in encoded_words + encoded_vocab:
            packet += struct.pack('<I', len(text)) + text
        packet += a.tobytes()
        self.proc.stdin.write(packet)
        self.proc.stdin.flush()
        size, peak, allocs, ns = struct.unpack('<4Q', self._read(32))
        payload = self._read(size)
        return {'bytes': payload, 'stamps': decode(payload), 'peak_heap_bytes': peak,
                'allocations': allocs, 'kernel_ns': ns}


def decode(data):
    offset = 0

    def take(fmt):
        nonlocal offset
        values = struct.unpack_from(fmt, data, offset)
        offset += struct.calcsize(fmt)
        return values[0] if len(values) == 1 else values

    def text():
        nonlocal offset
        size = take('<I')
        value = data[offset:offset + size].decode('utf-8')
        offset += size
        return value

    out = []
    for _ in range(take('<I')):
        word = text()
        start, end = take('<2f')
        chars = []
        for _ in range(take('<I')):
            ch = text()
            a, b = take('<2f')
            chars.append((ch, a, b))
        out.append((word, start, end, chars))
    if offset != len(data):
        raise RuntimeError('CTC probe returned trailing data')
    return out
