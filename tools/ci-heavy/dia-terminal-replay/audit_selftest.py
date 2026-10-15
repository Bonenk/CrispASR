"""Weight-free calibration of terminal capture ownership and margin accounting."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('terminal_audit', HERE/'audit.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def reject(call):
    try:
        call()
    except AssertionError:
        return
    raise AssertionError('Corrupted evidence accepted')


def main():
    subprocess.run([sys.executable, HERE.parent/'dia-suffix-replay/audit_selftest.py'], check=True)
    # Signed, non-monotonic exact binary fractions have known integer deltas.
    unit = 2.0**-20
    row = audit.margin_steps(-2*unit, unit, 0., 5*unit)
    assert row['conditional_post17_change'] == 3*unit
    assert row['conditional_final_norm_change'] == -unit
    assert row['remaining_output_head_difference'] == 5*unit
    assert row['telescope_residual'] == 0
    logits = np.zeros((2, 9, 1028), dtype='<f4')
    logits[1, 2, 46], logits[1, 2, 904] = 3, 1
    logits[0, 2, 46], logits[0, 2, 904] = 2, 1
    assert audit.cfg_terms(logits)['float64_cfg_margin'] == 5
    rejects = 0
    for bad in [float('nan'), float('inf'), -float('inf')]:
        reject(lambda: audit.margin_steps(0, bad, 0, 0)); rejects += 1
    held = np.arange(4096, dtype='<f4').tobytes()
    injection = (np.arange(4096, dtype='<f4')+1).tobytes()
    h = audit.hashlib.sha256
    hashes = dict(computed_sha256=h(held).hexdigest(), activation_sha256=h(injection).hexdigest(),
                  consumed_sha256=h(injection).hexdigest())
    values = [held, injection, injection, held, injection]
    audit.verify_activation(*values, hashes)
    for index in range(5):
        wrong = values.copy()
        wrong[index] = bytes([wrong[index][0] ^ 1])+wrong[index][1:]
        reject(lambda: audit.verify_activation(*wrong, hashes)); rejects += 1
    for key in hashes:
        wrong = dict(hashes); wrong[key] = '0'*64
        reject(lambda: audit.verify_activation(*values, wrong)); rejects += 1
    wrong = values.copy(); wrong[0] = held[:-4]
    reject(lambda: audit.verify_activation(*wrong, hashes)); rejects += 1
    print(f'DIA_TERMINAL_AUDIT_SELFTEST_PASS: calibrated byte/metric helpers, {rejects} terminal negatives and exact signed margin accounting')


if __name__ == '__main__':
    main()
