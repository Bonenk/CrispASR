"""Tiny frame-grid and actual shared-aligner inference dispatch regressions."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CtcWindows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='ctc-windows-')
        path = Path(cls.tmp.name)
        source = (ROOT / 'src/crispasr_aligner.cpp').read_text()
        start = source.index('    float* ctc_logits = nullptr;')
        end = source.index('    if (label_words.empty())', start)
        fixture = (ROOT / 'tests/fixtures/ctc-windows.cpp').read_text()
        fixture = fixture.replace('// PRODUCTION_DISPATCH', source[start:end])
        (path / 'probe.cpp').write_text(fixture)
        cls.binary = path / 'probe'
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O1',
                        '-fsanitize=undefined', '-fno-sanitize-recover=all',
                        '-I' + str(ROOT / 'src'), str(path / 'probe.cpp'), '-o', str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_case(self, case):
        run = subprocess.run([str(self.binary), case], capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_every_frame_once_with_context_and_partial_tail(self):
        self.check_case('geometry')

    def test_zero_context_and_exact_endpoint_frame(self):
        self.check_case('endpoint')

    def test_compute_failure_discards_partial_output(self):
        self.check_case('failure')

    def test_bad_shape_and_changing_vocabulary_are_rejected(self):
        self.check_case('shape')

    def test_invalid_geometry_is_rejected_before_compute(self):
        self.check_case('invalid')

    def test_actual_shared_aligner_uses_bounded_encoder_windows(self):
        self.check_case('dispatch-long')

    def test_actual_shared_aligner_short_path_is_unchanged(self):
        self.check_case('dispatch-short')

    def test_explicit_zero_keeps_full_encoder_path(self):
        self.check_case('dispatch-disabled')

    def test_dispatch_failure_returns_empty_and_frees_owned_output(self):
        self.check_case('dispatch-failure')


if __name__ == '__main__':
    unittest.main()
