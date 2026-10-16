"""Calibrate independent fork-prefix and scale-sensitive metric assertions."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('original_trace_audit', Path(__file__).with_name('audit.py'))
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def reject(call):
    try: call()
    except AssertionError: return
    raise AssertionError('Corrupted evidence accepted')


def main():
    base = np.arange(981*9, dtype='f4').reshape(981,9); base[979,2] = 877
    changed = base.copy(); changed[979,2] = 904
    logits = np.arange(981*2*9, dtype='f4').reshape(981,2,9)
    codes = np.arange(966*9, dtype='f4').reshape(966,9)
    values = [base, changed, logits, logits.copy(), codes, codes.copy()]
    audit.fork_prefix(*values)
    rejects = 0
    for slot, index in [(1,(978,2)), (1,(979,1)), (3,(978,1,2)), (5,(962,1))]:
        broken = [a.copy() for a in values]; broken[slot][index] += 1
        reject(lambda: audit.fork_prefix(*broken)); rejects += 1
    for token in [877, 905]:
        broken = [a.copy() for a in values]; broken[1][979,2] = token
        reject(lambda: audit.fork_prefix(*broken)); rejects += 1
    # Values after the controlled prefix may differ: this is the phenomenon measured.
    broken = [a.copy() for a in values]; broken[1][980,0] += 1; broken[3][979,0,0] += 1; broken[5][963,0] += 1
    audit.fork_prefix(*broken)
    x = np.array([[1,2,3,4]], dtype='f4'); row = audit.measure(2*x, x)
    assert abs(row['cos_min']-1) < 1e-15 and row['relative_l2'] == 1 and row['max_abs'] == 4
    assert row['native_rms'] == 2*row['original_rms']
    for value in [np.nan, np.inf, -np.inf]:
        bad = x.copy(); bad[0,0] = value
        reject(lambda: audit.measure(bad, x)); rejects += 1
    for name in ['../escape', '/absolute', 'a/../b', 'a\\b', '.', 'a//b']:
        reject(lambda: audit.safe(name)); rejects += 1
    print('DIA_ORIGINAL_TRACE_AUDIT_SELFTEST_PASS:', rejects, 'rejecting controls, exact controlled prefix and scale-sensitive metrics')


if __name__ == '__main__': main()
