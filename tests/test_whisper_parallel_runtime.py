"""Compile the literal parallel runtime against tiny deterministic worker states.

No GGML, models or replacement implementation of the merge loop is built.
Public parameters/tokens and the private segment layout come from production.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_session_utf8 import definition

ROOT = Path(__file__).resolve().parents[1]


class WhisperParallelRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='whisper-parallel-runtime-')
        path = Path(cls.tmp.name)
        source = (ROOT / 'src/crispasr.cpp').read_text()
        fixture = (ROOT / 'tests/fixtures/whisper-parallel-runtime.cpp').read_text()
        fixture = fixture.replace('// PRODUCTION_VAD_TYPES', definition(source, 'struct vad_time_mapping {') + ';')
        fixture = fixture.replace('// PRODUCTION_VAD_SEGMENT_INFO', definition(source, 'struct vad_segment_info {') + ';')
        fixture = fixture.replace('// PRODUCTION_SEGMENT', definition(source, 'struct whisper_segment {') + ';')
        fixture = fixture.replace('// PRODUCTION_RESET_MAPPING', definition(source, 'static void whisper_reset_vad_mapping('))
        fixture = fixture.replace('// PRODUCTION_PARALLEL', definition(source, 'int whisper_full_parallel('))
        (path / 'probe.cpp').write_text(fixture)
        cls.binary = path / 'parallel-runtime-probe'
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-pthread', '-O1',
                        '-fsanitize=undefined', '-fno-sanitize-recover=all',
                        '-I' + str(ROOT / 'include'), '-I' + str(ROOT / 'ggml/include'),
                        str(path / 'probe.cpp'), '-o', str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_case(self, case):
        run = subprocess.run([str(self.binary), case], capture_output=True, text=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_later_chunk_segment_token_dtw_times_and_alternative_ownership(self):
        self.check_case('timestamps')

    def test_fractional_centisecond_offset_matches_actual_chunk_sample_boundary(self):
        self.check_case('fractional-offset')

    def test_multi_minute_offset_uses_wide_sample_arithmetic(self):
        self.check_case('offset')

    def test_long_clip_merge_and_logged_boundaries_use_wide_arithmetic(self):
        self.check_case('long')

    def test_worker_error_joins_frees_and_discards_partial_results(self):
        self.check_case('worker-error')

    def test_main_error_joins_frees_and_discards_partial_results(self):
        self.check_case('main-error')

    def test_worker_exception_is_contained_and_cleaned_up(self):
        self.check_case('worker-throw')

    def test_state_allocation_failure_frees_previously_allocated_states(self):
        self.check_case('allocation')

    def test_state_allocation_exception_frees_previously_allocated_states(self):
        self.check_case('allocation-throw')

    def test_main_exception_joins_workers_and_discards_partial_results(self):
        self.check_case('main-throw')

    def test_vad_failure_clears_previous_transcript(self):
        self.check_case('vad-failure')

    def test_vad_silence_clears_previous_transcript(self):
        self.check_case('silence')

    def test_vad_runs_once_and_preserves_parent_timeline_mapping(self):
        self.check_case('vad')

    def test_invalid_inputs_and_excess_processors(self):
        self.check_case('inputs')

    def test_bounded_duration_is_split_once_over_the_requested_interval(self):
        self.check_case('duration')

    def test_duration_at_zero_offset_ignores_the_unrequested_tail(self):
        self.check_case('duration-zero-offset')

    def test_duration_clips_to_available_audio_and_keeps_remainder(self):
        self.check_case('duration-clipped')

    def test_large_duration_uses_wide_arithmetic_before_clipping(self):
        self.check_case('duration-large')

    def test_fractional_duration_preserves_exact_sample_boundaries(self):
        self.check_case('duration-fractional')

    def test_duration_with_too_many_workers_uses_bounded_fallback(self):
        self.check_case('duration-tiny')

    def test_negative_duration_is_rejected(self):
        self.check_case('duration-negative')

    def test_single_processor_preserves_sequential_route(self):
        self.check_case('single')


if __name__ == '__main__':
    unittest.main()
