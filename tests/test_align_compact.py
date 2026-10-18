"""Independent acceptance for our opt-in CTC workspace optimization (#217)."""
import os
from pathlib import Path
import unittest

import numpy as np

from ctc_align_harness import Probe, ROOT

GATE = 'CRISPASR_CTC_ALIGN_OPT'


def exhaustive_paths(labels, frames, blank=0):
    """Enumerate legal CTC states, independently of the implementation's DP."""
    seq = [blank]
    for label in labels:
        seq.extend([label, blank])
    paths = [(0,), (1,)]
    for _ in range(1, frames):
        following = []
        for path in paths:
            old = path[-1]
            for new in (old, old + 1, old + 2):
                if new >= len(seq):
                    continue
                if new == old + 2 and (seq[new] == blank or seq[new] == seq[old]):
                    continue
                following.append(path + (new,))
        paths = following
    return seq, [p for p in paths if p[-1] >= len(seq) - 2]


class CompactAlignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = Probe(GATE, source_root=Path(os.environ.get('CRISPASR_ALIGN_SOURCE', ROOT)))

    @classmethod
    def tearDownClass(cls):
        cls.probe.close()

    def compare(self, logits, words, vocab, **kwargs):
        expected = self.probe.run(logits, words, vocab, reference=True, **kwargs)
        for gate in (-1, 0, 1):
            actual = self.probe.run(logits, words, vocab, gate=gate, **kwargs)
            self.assertEqual(actual['bytes'], expected['bytes'], (gate, words, kwargs))
        return expected

    def test_randomized_exact_word_and_character_stamps(self):
        rng = np.random.default_rng(217)
        pool = ['a', 'b', 'aa', 'ab', '日', '本', 'a!', '???', '']
        vocab = ['<blank>', 'a', 'b', '|', '日', '本']
        for case in range(240):
            frames = int(rng.integers(1, 65))
            logits = rng.normal(0, 4, (frames, len(vocab))).astype(np.float32)
            words = rng.choice(pool, size=int(rng.integers(0, 8))).tolist()
            with self.subTest(case=case):
                self.compare(logits, words, vocab)

    def test_ties_repeats_unicode_and_vocab_edge_cases(self):
        cases = [(['aa'], ['<blank>', 'a']), (['ab', 'aa'], ['<blank>', 'a', 'b', '|']),
                 (['日本', 'a!'], ['<blank>', '日', '本', 'a', '|']),
                 (['???', 'a', ''], ['<blank>', 'a', '|']),
                 (['a', 'b'], ['|', 'a', 'b']), (['a', 'b'], ['a', 'a', 'b', '<blank>'])]
        for words, vocab in cases:
            for value in (0., -10000., 10000.):
                blank = vocab.index('<blank>') if '<blank>' in vocab else 0
                self.compare(np.full((32, len(vocab)), value, dtype=np.float32), words, vocab, blank=blank)
        # Equal starts/end alternatives and frame clocks must remain bit-identical.
        self.compare(np.zeros((32, 6), dtype=np.float32), ['a', 'aa'], ['_', 'a'], duration=.02)

    def test_rejected_and_infeasible_inputs_keep_failure_contract(self):
        logits = np.zeros((2, 4), dtype=np.float32)
        vocab = ['<blank>', 'a', 'b', '|']
        for kwargs in ({'blank': -1}, {'blank': 4}, {'duration': 0}, {'duration': float('nan')},
                       {'shape': (0, 4)}, {'shape': (-1, 4)}, {'shape': (2, 0)}):
            self.assertEqual(self.compare(logits, ['a'], vocab, **kwargs)['stamps'], [])
        self.assertEqual(self.compare(logits, ['a'], vocab + ['extra'])['stamps'], [])
        self.assertEqual(self.compare(logits, ['aaaaaa'], vocab)['stamps'], [])

    def test_nonfinite_scores_and_unlisted_blank_preserve_baseline(self):
        vocab = ['<blank>', 'a', 'b', '|']
        for value in (float('nan'), float('inf'), -float('inf')):
            scores = np.zeros((8, 4), dtype=np.float32)
            scores[2, 1] = value
            self.compare(scores, ['a', 'b'], vocab)
        self.compare(np.zeros((8, 8), dtype=np.float32), ['a'], ['unused', 'a'], blank=7)

    def test_unique_optimum_against_exhaustive_path_oracle(self):
        rng = np.random.default_rng(218)
        vocab = ['<blank>', 'a', 'b', '|']
        for labels, words, word_ranges in [([1], ['a'], [(0, 0)]),
                                           ([1, 1], ['aa'], [(0, 1)]),
                                           ([1, 3, 2], ['a', 'b'], [(0, 0), (2, 2)])]:
            seq, paths = exhaustive_paths(labels, 6)
            accepted = 0
            for _ in range(100):
                raw = rng.integers(-25, 26, size=(6, 4)).astype(np.float32)
                scores = [sum(int(raw[t, seq[j]]) for t, j in enumerate(path)) for path in paths]
                maximum = max(scores)
                if scores.count(maximum) != 1:
                    continue  # Ties have a separately checked frozen-baseline contract.
                path = paths[scores.index(maximum)]
                result = self.compare(raw, words, vocab)['stamps']
                self.assertEqual(len(result), len(words))
                duration = np.float32(.08)
                for wi, (first, last) in enumerate(word_ranges):
                    hit = [t for t, state in enumerate(path) if 2 * first + 1 <= state <= 2 * last + 1]
                    expected = (float(np.float32(hit[0]) * duration),
                                float(np.float32(hit[-1] + 1) * duration))
                    self.assertEqual(result[wi][1:3], expected)
                accepted += 1
                if accepted == 20:
                    break
            self.assertEqual(accepted, 20)

    def test_workspace_does_not_expand_with_repeated_transcript_labels(self):
        rng = np.random.default_rng(219)
        vocab = ['<blank>', 'a', 'b', '|'] + [f'unused{i}' for i in range(12)]
        logits = rng.normal(size=(256, len(vocab))).astype(np.float32)
        words = ['ab'] * 20
        expected = self.probe.run(logits, words, vocab, reference=True)
        actual = self.probe.run(logits, words, vocab)
        self.assertEqual(actual['bytes'], expected['bytes'])
        self.assertLessEqual(actual['peak_heap_bytes'], expected['peak_heap_bytes'])
        self.assertLess(actual['allocations'], expected['allocations'])


if __name__ == '__main__':
    unittest.main()
