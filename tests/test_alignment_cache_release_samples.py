"""Cache telemetry must follow the resident GPU, not an idle second device."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'alignment_audit', Path(__file__).resolve().parents[1] / 'tools/ci-heavy/audit_alignment_cuda_memory.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def sample(phase, rows):
    return dict(phase=phase, samples='\n'.join(f'{device}, {memory}, 0' for device, memory in rows))


class CacheReleaseSamplesTest(unittest.TestCase):
    def test_release_and_reload_follow_resident_device_even_when_it_is_not_zero(self):
        samples = [sample('short-after', [(0, 3), (1, 2735)]),
                   sample('cli', [(0, 3), (1, 125)]),
                   sample('cli', [(0, 3), (1, 2823)])]
        result = audit.audit_cache_release(samples, 2 * 1024**3)
        self.assertEqual(result['active_device_index'], 1)
        self.assertEqual(result['observed_device_total_drop_mib'], 2610)
        self.assertEqual(result['cli_peak_after_reload_mib'], 2823)

    def test_idle_second_gpu_cannot_fake_release_on_resident_gpu(self):
        samples = [sample('short-after', [(0, 2735), (1, 125)]),
                   sample('cli', [(0, 2735), (1, 3)])]
        with self.assertRaisesRegex(ValueError, 'model-sized release'):
            audit.audit_cache_release(samples, 2 * 1024**3)

    def test_later_low_sample_cannot_replace_missing_boundary_drop(self):
        samples = [sample('short-after', [(0, 2735)]),
                   sample('cli', [(0, 2735)]), sample('cli', [(0, 125)]),
                   sample('cli', [(0, 2823)])]
        with self.assertRaisesRegex(ValueError, 'model-sized release'):
            audit.audit_cache_release(samples, 2 * 1024**3)

    def test_boundary_sample_from_other_gpu_is_insufficient(self):
        samples = [sample('short-after', [(0, 2735), (1, 3)]),
                   sample('cli', [(1, 3)])]
        with self.assertRaisesRegex(ValueError, 'boundary sample'):
            audit.audit_cache_release(samples, 2 * 1024**3)

    def test_multiple_resident_devices_are_ambiguous(self):
        samples = [sample('short-after', [(0, 2735), (1, 2735)]),
                   sample('cli', [(0, 125), (1, 125)])]
        with self.assertRaisesRegex(ValueError, 'unambiguous'):
            audit.audit_cache_release(samples, 2 * 1024**3)

    def test_release_without_fresh_cli_model_residency_is_incomplete(self):
        samples = [sample('short-after', [(0, 2735)]), sample('cli', [(0, 125)])]
        with self.assertRaisesRegex(ValueError, 'residency not observed'):
            audit.audit_cache_release(samples, 2 * 1024**3)


if __name__ == '__main__':
    unittest.main()
