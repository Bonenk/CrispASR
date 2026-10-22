#!/usr/bin/env python3
"""Exercise literal selection/checkpoint ASTs without downloads or synthesis."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/ci-heavy/omnivoice_audio_acceptance.py'


def assignment(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name for t in node.targets)


class AcceptancePolicy(unittest.TestCase):
    def test_every_eligible_reference_participates(self):
        tree = ast.parse(SOURCE.read_text())
        node = next(n for n in tree.body if assignment(n, 'CLONE_REFS'))
        namespace = dict(fixtures=dict.fromkeys(('fox', 'window', 'station')),
                         integrity={'fox': {'min_wer': 2/9},
                                    'window': {'min_wer': 0},
                                    'station': {'min_wer': 1/9}}, WER_GATE=0.2)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
        self.assertEqual(namespace['CLONE_REFS'], ('window', 'station'))

    def checkpoint(self, wer, missing=False):
        tree = ast.parse(SOURCE.read_text())
        start = next(i for i, n in enumerate(tree.body) if assignment(n, 'passed'))
        tail = compile(ast.Module(body=tree.body[start:], type_ignores=[]), str(SOURCE), 'exec')
        with tempfile.TemporaryDirectory() as directory:
            namespace = dict(OUT=Path(directory), ROOT=ROOT, json=json,
                             results={'station': {'wer': wer}},
                             expected_outputs={'station', 'missing'} if missing else {'station'},
                             WER_GATE=0.2, MODEL_PIN='model-pin', FIXTURE_PIN='fixture-pin',
                             ASR_MODELS={}, CLONE_REFS=('station',), integrity={}, error=0, code_arrays={},
                             subprocess=SimpleNamespace(check_output=lambda *a, **k: 'test-source\n'))
            failed = False
            try:
                exec(tail, namespace)
            except AssertionError:
                failed = True
            # These files must exist even when the final gate raises.
            receipt = json.loads((Path(directory) / 'omnivoice-518-acceptance.json').read_text())
            summary = (Path(directory) / 'summary.md').read_text()
            return receipt, summary, failed

    def test_success_receipt(self):
        receipt, summary, failed = self.checkpoint(0)
        self.assertTrue(receipt['passed'])
        self.assertFalse(failed)
        self.assertIn('PASS', summary)

    def test_failed_receipt_survives_gate_failure(self):
        receipt, summary, failed = self.checkpoint(2/9)
        self.assertFalse(receipt['passed'])
        self.assertTrue(failed)
        self.assertIn('FAIL', summary)

    def test_missing_arm_cannot_pass(self):
        receipt, _, failed = self.checkpoint(0, missing=True)
        self.assertFalse(receipt['passed'])
        self.assertTrue(failed)


if __name__ == '__main__':
    unittest.main()
