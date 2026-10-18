"""Exact-output and independent-path acceptance for our subword workspace gate."""
import unittest

import numpy as np

from subword_align_harness import Probe


class SubwordCompactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = Probe()

    @classmethod
    def tearDownClass(cls):
        cls.probe.close()

    def parity(self, scores, words, vocab, **kwargs):
        expected = self.probe.run(scores, words, vocab, reference=True, **kwargs)
        for gate in (-1, 0, 1, 2):
            actual = self.probe.run(scores, words, vocab, gate=gate, **kwargs)
            self.assertEqual(actual['bytes'], expected['bytes'], (gate, words, scores.shape))
        return expected

    def test_seeded_random_exact_outputs(self):
        rng = np.random.default_rng(217549)
        vocab = ['<blank>', '▁a', 'b', '▁b', 'a', '▁猫', '猫', '▁ab']
        for i in range(240):
            frames = int(rng.integers(1, 40))
            vocabulary = int(rng.choice([8, 16, 64]))
            words = rng.choice(['a', 'ab', 'aba', 'b', 'ba', '猫', '猫猫', 'missing', ''],
                               size=int(rng.integers(1, 9))).tolist()
            scores = rng.normal(size=(frames, vocabulary)).astype(np.float32) * (0.001, 1, 1000)[i % 3]
            with self.subTest(case=i):
                self.parity(scores, words, vocab + [f'unused{k}' for k in range(vocabulary - len(vocab))])

    def test_ties_repeats_unknowns_and_nonfinite_scores(self):
        vocab = ['<blank>', '▁a', 'b', '▁猫', '猫']
        for words in (['a'], ['ab'], ['a', 'a'], ['猫猫'], ['missing', 'a', '', 'ab']):
            for frames in (1, 2, 3, 16):
                self.parity(np.zeros((frames, len(vocab)), np.float32), words, vocab)
        for value in (np.nan, np.inf, -np.inf):
            scores = np.zeros((8, len(vocab)), np.float32)
            for token in range(len(vocab)):
                scores[:, token] = value
                self.parity(scores, ['a', 'ab'], vocab)
                scores[:, token] = 0
        # Existing unknown/byte-fallback behavior is frozen, not retokenized.
        self.parity(np.zeros((32, 4), np.float32), ['猫', 'unknown'], ['<blank>', '<unk>', 'a', 'b'])

    def test_nonzero_blank_duration_and_long_text(self):
        rng = np.random.default_rng(7)
        vocab = ['▁a', 'b', '<blank>', '▁猫', '猫']
        self.parity(rng.normal(size=(48, 5)), ['a', 'ab', '猫猫'], vocab, blank=2, duration=13)
        vocab = ['<blank>', '▁a', 'a']
        # Public output text truncation and repeated-label feasibility are unchanged.
        self.parity(np.zeros((256, 3)), ['a' * 100], vocab)
        self.parity(np.zeros((2, 3)), ['a'], vocab, blank=3)

    def test_independent_exhaustive_path_oracle(self):
        rng = np.random.default_rng(549)
        vocab = ['<blank>', '▁a', 'b']
        for words, labels, ranges in ((['a'], [1], [(1, 1)]),
                                      (['ab'], [1, 2], [(1, 3)]),
                                      (['a', 'a'], [1, 1], [(1, 1), (3, 3)])):
            seq = [0]
            for label in labels:
                seq += [label, 0]
            frames = 6
            # Enumerate legal CTC state paths independently of the DP/traceback.
            paths = []
            def extend(path):
                if len(path) == frames:
                    if path[-1] in (len(seq) - 2, len(seq) - 1):
                        paths.append(tuple(path))
                    return
                current = path[-1]
                for nxt in range(current, min(current + 3, len(seq))):
                    if nxt == current + 2 and (seq[nxt] == 0 or seq[nxt] == seq[current]):
                        continue
                    extend(path + [nxt])
            extend([0])
            extend([1])
            accepted = 0
            for _ in range(200):
                scores = rng.integers(-1000, 1000, size=(frames, 3)).astype(np.float32)
                ranked = sorted(((sum(float(scores[t, seq[j]]) for t, j in enumerate(path)), path)
                                 for path in paths), reverse=True)
                if ranked[0][0] == ranked[1][0]:
                    continue  # Tie convention is covered by frozen-reference fixtures.
                path = ranked[0][1]
                expected = []
                for word, (first, last) in zip(words, ranges):
                    times = [t for t, state in enumerate(path) if first <= state <= last]
                    expected.append((word.encode(), times[0] * 8, (times[-1] + 1) * 8))
                result = self.parity(scores, words, vocab)
                self.assertEqual(result['rc'], 0)
                self.assertEqual(result['stamps'], expected)
                accepted += 1
                if accepted == 20:
                    break
            self.assertEqual(accepted, 20)

    def test_workspace_and_allocation_reduction(self):
        scores = np.random.default_rng(217).normal(size=(512, 1024)).astype(np.float32)
        vocab = ['<blank>', '▁a', 'b'] + [f'unused{k}' for k in range(1021)]
        before = self.probe.run(scores, ['ab'] * 32, vocab, gate=0)
        after = self.probe.run(scores, ['ab'] * 32, vocab, gate=1)
        self.assertEqual(before['bytes'], after['bytes'])
        self.assertEqual(after['rc'], 0)
        self.assertLess(after['peak_heap_bytes'], before['peak_heap_bytes'] / 4)
        self.assertLess(after['allocations'], before['allocations'] - 400)


if __name__ == '__main__':
    unittest.main()
