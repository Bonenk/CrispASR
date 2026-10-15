"""Calibrate independent reconstruction and numerical guards without a model."""
import importlib.util
from pathlib import Path
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import dia_suffix_replay as worker
spec = importlib.util.spec_from_file_location('independent_suffix_audit', HERE/'audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def rejected(call):
    try:
        call()
    except AssertionError:
        return
    raise AssertionError('Broken evidence accepted')


def main():
    count = 0
    with tempfile.TemporaryDirectory(prefix='dia-independent-audit-') as tmp:
        root = Path(tmp); baseline = root/'baseline'; baseline.mkdir()
        arm = root/'original-layer-00'; arm.mkdir()
        name = 'cpp_decoder_sa1_k.f32'
        base = np.arange(2*979*512, dtype='<f4').reshape(2, 979, 512)
        base.tofile(baseline/name)
        replacement = base.copy(); replacement[:, -1] *= -1
        replacement.tofile(arm/name)
        expected = replacement.tobytes()
        rows = worker.compact_captures(arm, baseline)
        assert len(rows) == 1 and audit.reconstruct(rows[0], arm, baseline) == expected
        row = rows[0]
        for key, wrong in [('path', '../escape.f32'), ('kind', 'wrong'), ('baseline', 'baseline/other.f32'),
                           ('shape', [979, 2, 512]), ('position', 977), ('bytes', len(expected)-4),
                           ('sha256', '0'*64), ('current', '../outside.f32'), ('current_sha256', '0'*64)]:
            bad = dict(row, **{key: wrong})
            rejected(lambda: audit.reconstruct(bad, arm, baseline)); count += 1
        tail = arm/row['current']; saved = tail.read_bytes()
        tail.write_bytes(b'\0'*len(saved))
        rejected(lambda: audit.reconstruct(row, arm, baseline)); count += 1
        tail.write_bytes(saved)
        for name, values in [('cpp_final_hidden.f32', np.array([3, 4], dtype='<f4')),
                             ('cpp_decoder_layer_1.f32', np.array([1, 2], dtype='<f4'))]:
            values.tofile(baseline/name)
            (values if 'hidden' in name else -values).tofile(arm/name)
        # Remove the already compacted tail before exercising a second set.
        tail.unlink()
        next_rows = worker.compact_captures(arm, baseline)
        assert {row['kind'] for row in next_rows} == {'baseline', 'full'}
        for row in next_rows:
            rebuilt = audit.reconstruct(row, arm, baseline)
            expected = (baseline/row['path']).read_bytes() if row['kind'] == 'baseline' else (arm/row['path']).read_bytes()
            assert rebuilt == expected
    x = np.array([[3, 4]], dtype='<f4'); y = np.array([[0, 5]], dtype='<f4')
    # These values follow directly from the3-4-5 triangle, not worker metrics.
    exact = dict(cosine_min=.8, mine_norm=5., ref_norm=5., max_row_norm_error=0.,
                 relative_l2=np.sqrt(10)/5, max_abs=3., byte_exact=False)
    audit.measured(x, y, exact)
    for key in exact:
        bad = dict(exact); bad[key] = True if key == 'byte_exact' else exact[key]+.01
        rejected(lambda: audit.measured(x, y, bad)); count += 1
    rejected(lambda: audit.measured(x*2, y, exact)); count += 1
    for name in ['../x', '/x', 'x/../y', 'x\\y', '.', 'x//y']:
        rejected(lambda: audit.safe(name)); count += 1
    print(f'DIA_SUFFIX_INDEPENDENT_AUDIT_SELFTEST_PASS: all3 encodings, exact byte reconstruction, {count} rejecting controls and hand-derived numerical values')


if __name__ == '__main__':
    main()
