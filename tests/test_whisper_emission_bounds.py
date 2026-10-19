"""Compile the literal result-emission block with controlled decoder predictions.

No model or GGML build: the fixture controls timestamp estimation/DTW/wrapping
and verifies the values delivered by the production callback path.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_session_utf8 import definition

ROOT = Path(__file__).resolve().parents[1]


class WhisperEmissionBounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='whisper-emission-')
        path = Path(cls.tmp.name)
        source = (ROOT / 'src/crispasr.cpp').read_text()
        start = source.index('        // output results through a user-provided callback')
        end = source.index('            // ref: https://github.com/CrispStrobe/CrispASR/pull/2629', start)
        emission = source[start:end] + '\n}\n'
        fixture = (ROOT / 'tests/fixtures/whisper-emission-bounds.cpp').read_text()
        fixture = fixture.replace('// PRODUCTION_SEGMENT', definition(source, 'struct whisper_segment {') + ';')
        fixture = fixture.replace('// PRODUCTION_EMISSION', emission)
        (path / 'probe.cpp').write_text(fixture)
        cls.binary = path / 'probe'
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O1',
                        '-fsanitize=undefined', '-fno-sanitize-recover=all',
                        '-I' + str(ROOT / 'include'), '-I' + str(ROOT / 'ggml/include'),
                        str(path / 'probe.cpp'), '-o', str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_case(self, case):
        result = subprocess.run([str(self.binary), case], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_timestamp_boundary_before_callback(self):
        self.check_case('timestamp')

    def test_single_segment_tail_before_callback(self):
        self.check_case('tail')

    def test_wrapped_token_times_and_alternatives(self):
        self.check_case('wrapped')

    def test_dtw_callback_after_prior_window(self):
        self.check_case('dtw')

    def test_unavailable_token_times_are_preserved(self):
        self.check_case('unavailable')

    def test_nonzero_window_start(self):
        self.check_case('offset')


if __name__ == '__main__':
    unittest.main()
