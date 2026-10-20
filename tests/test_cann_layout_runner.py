"""Reject vacuous device receipts; successful arithmetic is audited from the real CPU probe separately."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('layout_runner', ROOT/'tools/ci-heavy/cann-weight-layout/run_device.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def log(backend='CANN0', captures=0, replays=0, selections=0):
    rows = [f'backend={backend} device={backend} NZ=off CACHE=on',
            f'ACL captures={captures} replays={replays}', 'PASS computes=576 cases=48']
    for i in range(48):
        rows += [f'case={i} type=f32 k=32 name=arbitrary.linear eligible=1 completed',
                 f'cache_case={i} bytes={4096 if selections else 0} selections={selections}']
    return '\n'.join(rows)+'\n'


class DeviceReceipt(unittest.TestCase):
    def reject(self, text, nz=False, acl=False):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            runner.assess_probe(text, Path(directory), nz, acl)

    def test_empty_and_zero_work(self):
        self.reject('')
        self.reject(log().replace('computes=576', 'computes=0'))

    def test_cpu_cannot_satisfy_cann(self):
        self.reject(log(backend='CPU'))

    def test_no_cache_selection_cannot_satisfy_on(self):
        self.reject(log(), nz=True)
        self.reject(log(selections=1).replace('selections=1', 'selections=0'), nz=True)

    def test_compile_flag_alone_cannot_prove_acl(self):
        self.reject(log(captures=1, replays=0), acl=True)

    def test_missing_and_duplicate_case_evidence(self):
        self.reject(log().replace('case=47 type=', 'case=46 type='))
        self.reject(log().replace('cache_case=47 ', 'cache_case=46 '))

    def test_pass_marker_without_payloads_is_rejected(self):
        self.reject(log())


if __name__ == '__main__':
    unittest.main()
