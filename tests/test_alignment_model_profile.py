"""Exercise the profiling worker's rejecting evidence contracts, without models."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('alignment_model_profile', ROOT / 'tools/ci-heavy/alignment_model_profile.py')
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class ModelProfileContracts(unittest.TestCase):
    def test_tensor_identity_rejects_close_changed_and_nonfinite_values(self):
        original = np.ones((2, 3), dtype=np.float32)
        worker.same_array(original, original.copy())
        changed = original.copy()
        changed[0, 0] = np.nextafter(np.float32(1), np.float32(2))
        self.assertTrue(np.allclose(original, changed))
        for other in (changed, original.astype(np.float64), original.reshape(3, 2),
                      np.full_like(original, np.nan), np.full_like(original, np.inf)):
            with self.subTest(dtype=other.dtype, shape=other.shape), self.assertRaises(ValueError):
                worker.same_array(original, other)

    def test_stamps_reject_drops_reorder_zero_overlap_range_and_noninteger(self):
        original = [{'word': '群', 't0': 1234, 't1': 1242}, {'word': ' 島', 't0': 1250, 't1': 1258}]
        texts = ['群', ' 島']
        worker.validate_words(original, texts, 100, offset=1234)
        mutations = [original[:1], list(reversed(original))]
        for key, value in (('word', 'x'), ('t0', 0), ('t1', 1234), ('t1', 1400), ('t0', 1234.0), ('t0', True)):
            bad = copy.deepcopy(original)
            bad[0][key] = value
            mutations.append(bad)
        overlap = copy.deepcopy(original)
        overlap[1]['t0'] = 1241
        mutations.append(overlap)
        for bad in mutations:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                worker.validate_words(bad, texts, 100, offset=1234)

    def test_decimal_outputs_cannot_hide_half_centisecond_drift(self):
        self.assertEqual(worker.centiseconds(12.34), 1234)
        for value in (0.005, float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                worker.centiseconds(value)

    def test_recording_selection_uses_filename_not_repeated_sentence_id(self):
        pin = {'id': 7, 'filename': 'wanted.wav', 'num_samples': 32000,
               'raw_transcription': '群島。', 'transcription': '群島'}
        wanted = dict(pin, audio={'path': 'test/wanted.wav', 'bytes': b'fixture'})
        other = dict(wanted, audio={'path': 'test/other.wav', 'bytes': b'other speaker'})
        self.assertIs(worker.select_recording([other, wanted], pin), wanted)
        with self.assertRaises(ValueError):
            worker.select_recording([other], pin)
        with self.assertRaises(ValueError):
            worker.select_recording([wanted, wanted], pin)
        bad = dict(wanted, num_samples=32001)
        with self.assertRaises(ValueError):
            worker.select_recording([bad], pin)

    def test_physical_file_hash_and_size_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'payload'
            path.write_bytes(b'actual bytes')
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            worker.verify_file(path, expected, 12)
            with self.assertRaises(ValueError):
                worker.verify_file(path, expected, 13)
            path.write_bytes(b'changed byte')
            with self.assertRaises(ValueError):
                worker.verify_file(path, expected, 12)

    def test_model_work_is_refused_outside_hosted_actions(self):
        from unittest.mock import patch
        with patch.dict(worker.os.environ, {'GITHUB_ACTIONS': 'false'}):
            with self.assertRaisesRegex(RuntimeError, 'hosted Linux'):
                worker.run_hosted(None)


if __name__ == '__main__':
    unittest.main()
