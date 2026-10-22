"""Native separators may retain an earlier frame; content inversions must fail."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as Row
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('live', ROOT/'tools/ci-heavy/parakeet_ja_metadata_live.py')
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


def token(text, start):
    return Row(raw_text=text.encode(), start=start, end=start+.1, confidence=.9, has_timestamps=True)


class MetadataValidatorTest(unittest.TestCase):
    def test_native_separator_keeps_original_earlier_time(self):
        segment = Row(text='猫。 犬', words=[Row(text='猫。'), Row(text='犬')],
                      tokens=[token('猫', 1), token('。', 2), token(' ', 1), token('犬', 3)])
        self.assertEqual(live.validate([segment], 4), 4)
        self.assertEqual(segment.tokens[2].start, 1)

    def test_actual_content_inversion_is_rejected(self):
        segment = Row(text='猫犬', words=[Row(text='猫犬')], tokens=[token('猫', 2), token('犬', 1)])
        with self.assertRaisesRegex(ValueError, 'Content token times reordered'):
            live.validate([segment], 4)

    def test_missing_owned_token_is_rejected(self):
        segment = Row(text='猫犬', words=[Row(text='猫犬')], tokens=[token('猫', 1)])
        with self.assertRaisesRegex(ValueError, 'reconstruct'):
            live.validate([segment], 4)


if __name__ == '__main__':
    unittest.main()
