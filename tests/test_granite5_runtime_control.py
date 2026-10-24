"""The runtime positive control cannot substitute partial CPU or Q4 records."""
import copy
import importlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/ci-heavy'))
control = importlib.import_module('granite5_runtime_control')


class RuntimeControlInput(unittest.TestCase):
    def test_requires_exact_complete_f16_cpu_input(self):
        pins = json.loads((ROOT / 'tests/regression/granite5.json').read_text())
        pins['profiles'] = ['f16']
        pins['models'] = {'f16': pins['models']['f16']}
        cpu = dict(pins_sha256='0' * 64, complete=True, cpu_acceptance=True, cuda_acceptance=False,
                   profiles={'f16': dict(accepted=True, complete=True, device='cpu')})
        manifest = dict(source=control.CPU_SOURCE, run=control.CPU_RUN, receipt_sha256=control.CPU_SHA,
                        source_and_terminal_status_verified=True)
        control.check_cpu_control(cpu, pins, manifest, control.CPU_SHA, '0' * 64)
        for mutate in (lambda c, p, m: c.update(complete=False),
                       lambda c, p, m: c['profiles']['f16'].update(device='cuda'),
                       lambda c, p, m: p.update(profiles=['q4_k']),
                       lambda c, p, m: m.update(source='0' * 40),
                       lambda c, p, m: m.update(source_and_terminal_status_verified=False)):
            c, p, m = copy.deepcopy((cpu, pins, manifest))
            mutate(c, p, m)
            with self.assertRaises(ValueError):
                control.check_cpu_control(c, p, m, control.CPU_SHA, '0' * 64)
        with self.assertRaises(ValueError):
            control.check_cpu_control(cpu, pins, manifest, '0' * 64, '0' * 64)


if __name__ == '__main__':
    unittest.main()
