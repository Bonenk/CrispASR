"""Execute the real CPU graph and reject false CANN/device/evidence claims."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import unittest

BINARY = Path(os.environ['PROBE_BINARY']).resolve()
EVIDENCE = Path(os.environ['PROBE_EVIDENCE_DIR']).resolve()
if EVIDENCE.exists():
    raise RuntimeError('Use a fresh PROBE_EVIDENCE_DIR; preserve prior results')
EVIDENCE.mkdir(parents=True)


def floats(path):
    data = path.read_bytes()
    return struct.unpack('<' + 'f' * (len(data) // 4), data)


class ComposedProbeTest(unittest.TestCase):
    def run_probe(self, label, *arguments):
        command = [str(BINARY), *map(str, arguments)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        (EVIDENCE / (label + '.log')).write_bytes(result.stdout)
        (EVIDENCE / (label + '.json')).write_text(json.dumps({
            'command': command, 'returncode': result.returncode,
        }, indent=2) + '\n')
        return result

    def test_cpu_math_repeats_and_shapes(self):
        target = EVIDENCE / 'cpu-control'
        result = self.run_probe('cpu', '--backend', 'cpu', '--out', target)
        self.assertEqual(result.returncode, 0, result.stdout.decode())
        self.assertEqual(result.stdout.count(b'COMPUTE_PASS '), 18)
        self.assertIn(b'COMPOSED_PROBE_PASS backend=cpu calls=18 shapes=3', result.stdout)
        count = 0
        for case, n in enumerate((16, 32, 16)):
            for iteration in range(6):
                stem = target / f'shape-{case}-iter-{iteration}'
                a = floats(Path(str(stem) + '-a.f32'))
                b = floats(Path(str(stem) + '-b.f32'))
                self.assertEqual((len(a), len(b)), (1024, 32 * n))
                matrix = [sum(a[k + 32 * m] * b[k + 32 * column] for k in range(32))
                          for column in range(n) for m in range(32)]
                expected = [value * 0.5 for value in matrix]
                self.assertEqual(floats(Path(str(stem) + '-matmul.f32')), tuple(matrix))
                self.assertEqual(floats(Path(str(stem) + '-actual.f32')), tuple(expected))
                self.assertEqual(floats(Path(str(stem) + '-expected.f32')), tuple(expected))
                if iteration in (1, 4):
                    previous = target / f'shape-{case}-iter-{iteration - 1}-actual.f32'
                    self.assertEqual(Path(str(stem) + '-actual.f32').read_bytes(), previous.read_bytes())
                if case == 2:
                    initial = target / f'shape-0-iter-{iteration}-actual.f32'
                    self.assertEqual(Path(str(stem) + '-actual.f32').read_bytes(), initial.read_bytes())
                count += len(expected)
        self.assertEqual(count, 12288)
        (EVIDENCE / 'independent-math-audit.json').write_text(json.dumps({
            'passed': True, 'actual_computes': 18, 'independently_recalculated_final_values': count,
            'independently_recalculated_matrix_values': count,
            'scope': 'CPU graph only; no CANN execution, fallback or ACL capture/replay acceptance',
        }, indent=2) + '\n')

    def test_missing_cann_cannot_pass_as_cpu(self):
        result = self.run_probe('missing-cann', '--backend', 'cann', '--out', EVIDENCE / 'missing-cann')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'CANN', result.stdout)
        self.assertNotIn(b'COMPOSED_PROBE_PASS', result.stdout)

    def test_cpu_cannot_claim_cann_fallback(self):
        result = self.run_probe('false-fallback', '--backend', 'cpu', '--require-cpu-fallback',
                                '--out', EVIDENCE / 'false-fallback')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'CANN fallback proof requires', result.stdout)
        self.assertNotIn(b'COMPOSED_PROBE_PASS', result.stdout)

    def test_invalid_device_and_backend_requests(self):
        for index in ('-1', '1', 'NaN', '0junk'):
            with self.subTest(index=index):
                result = self.run_probe('invalid-device-' + index, '--backend', 'cpu', '--device', index,
                                        '--out', EVIDENCE / ('invalid-device-' + index))
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn(b'COMPOSED_PROBE_PASS', result.stdout)
        result = self.run_probe('invalid-backend', '--backend', 'gpu', '--out', EVIDENCE / 'invalid-backend')
        self.assertNotEqual(result.returncode, 0)

    def test_previous_evidence_is_preserved(self):
        target = EVIDENCE / 'prior-evidence'; target.mkdir()
        sentinel = target / 'sentinel'; sentinel.write_bytes(b'preserve this prior result\n')
        result = self.run_probe('existing-output', '--backend', 'cpu', '--out', target)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'output directory already exists', result.stdout)
        self.assertEqual(sentinel.read_bytes(), b'preserve this prior result\n')


if __name__ == '__main__':
    shutil.copy2(BINARY, EVIDENCE / 'tested-cann-composed')
    result = unittest.main(exit=False)
    rows = []
    for path in sorted(EVIDENCE.rglob('*')):
        if path.is_file():
            rows.append({'path': path.relative_to(EVIDENCE).as_posix(), 'bytes': path.stat().st_size,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (EVIDENCE / 'manifest.json').write_text(json.dumps({'files': rows}, indent=2) + '\n')
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
