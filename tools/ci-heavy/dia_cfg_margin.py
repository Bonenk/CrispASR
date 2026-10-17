#!/usr/bin/env python3
"""Explain the retained Dia token46/904 rank margin using four actual logits.

This does not rerun inference or change CFG, filtering, random noise or defaults.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True, help='Audited key-intervention output directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    prior = json.loads((args.native/'receipt.json').read_text())
    assert prior['conditional_diagnostic_complete'] and prior['step'] == 978
    rows = {r['path']: r for r in prior['payloads']}
    arrays, verified = {}, []
    for owner, name in [('original', 'inputs/decoder_step_logits.npy'), ('native', 'inputs/native-logits.npy')]:
        file = args.native/name; row = rows[name]
        assert file.stat().st_size == row['bytes'] and sha(file) == row['sha256']
        z = np.load(file, allow_pickle=False, mmap_mode='r')[-1].copy()
        assert z.shape == (2, 9, 1028) and z.dtype == np.dtype('<f4') and np.isfinite(z).all()
        np.save(out/(owner+'-step978-logits.npy'), z)
        arrays[owner] = z
        verified.append(dict(owner=owner, **row))
    margins, four_logits = {}, []
    for owner, z in arrays.items():
        cfg = z[1]+np.float32(3)*(z[1]-z[0])
        c, u = z[1, 2].astype('f8'), z[0, 2].astype('f8')
        margin = float(4*(c[46]-c[904])-3*(u[46]-u[904]))
        margins[owner] = dict(float32=float(cfg[2, 46]-cfg[2, 904]), float64=margin,
                              rank46=int((cfg[2] > cfg[2, 46]).sum())+1,
                              rank904=int((cfg[2] > cfg[2, 904]).sum())+1)
        assert margins[owner]['float32'] == margin
        expected = prior['original_sampling' if owner == 'original' else 'retained_native_sampling']
        assert margins[owner]['float32'] == expected['margin46_minus904']
        assert all(margins[owner][key] == expected[key] for key in ['rank46', 'rank904'])
    error = arrays['native'].astype('f8')-arrays['original'].astype('f8')
    conditional = float(4*(error[1, 2, 46]-error[1, 2, 904]))
    unconditional = float(-3*(error[0, 2, 46]-error[0, 2, 904]))
    assert margins['original']['float64']+conditional+unconditional == margins['native']['float64']
    assert margins['original']['float64'] > 0 and margins['native']['float64'] < 0
    for branch, batch in [('unconditional', 0), ('conditional', 1)]:
        for token in [46, 904]:
            four_logits.append(dict(branch=branch, token=token,
                                    original=float(arrays['original'][batch, 2, token]),
                                    native=float(arrays['native'][batch, 2, token]), error=float(error[batch, 2, token])))
    report = dict(scope=__doc__, source_receipt_sha256=sha(args.native/'receipt.json'), verified_inputs=verified,
                  worker_sha256=sha(__file__), formula='g = c + 3*(c-u); margin = 4*(c46-c904)-3*(u46-u904)',
                  four_logits=four_logits, margins=margins, conditional_error_contribution=conditional,
                  unconditional_error_contribution=unconditional, decomposition_residual=0.0,
                  cfg_float32_rounding_changes_this_margin=False,
                  limitation='Exact incoming-logit explanation, not attribution to an upstream operator or proof of speech repair.')
    report['payloads'] = [dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.glob('*.npy'))]
    (out/'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('DIA_CFG_MARGIN_EXACT; no inference or sampling change', flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
