#!/usr/bin/env python3
"""Independently verify the tiny cancellation, scalar reduction and CFG equations."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['proof', 'reduction', 'margin', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    folder = args.proof/'tiny-cancellation'
    v = np.fromfile(folder/'v.f32', dtype='<f4')
    p = np.fromfile(folder/'mode-0-threads-4/probabilities.f32', dtype='<f4')
    native = np.fromfile(folder/'mode-0-threads-4/weighted.f32', dtype='<f4').item()
    original = np.load(folder/'actual-original-sdpa.npy', allow_pickle=False).item()
    assert np.array_equal(v, [2**24, 1, -2**24, 1]) and np.array_equal(p, [.25]*4)
    exact = math.fsum(float(x)*float(y) for x, y in zip(v, p))
    sequential = np.float32(0)
    for x, y in zip(v, p): sequential = np.float32(sequential+np.float32(x*y))
    assert exact == original == .5 and native == float(sequential) == .25

    reduction = json.loads((args.reduction/'receipt.json').read_text())
    values = np.fromfile(args.reduction/'input/v.f32', dtype='<f4')
    probabilities = np.fromfile(args.reduction/'input/shadow-probabilities.f32', dtype='<f4')
    assert values.shape == probabilities.shape == (979,)
    high_precision = math.fsum(float(x)*float(y) for x, y in zip(values, probabilities))
    assert high_precision == reduction['exact_float64_product_fsum']
    cell = tuple(reduction['full_output_index'])
    full = args.proof/'original-module-replay'
    actual = np.fromfile(full/'mode-2-threads-4/weighted.f32', dtype='<f4').reshape(2, 1, 16, 128)[cell].item()
    isolated = np.fromfile(args.reduction/'native/weighted.f32', dtype='<f4').item()
    assert actual == isolated == reduction['isolated_ggml_dot']
    assert actual-high_precision == reduction['native_absolute_error']
    original_full = np.load(full/'shadow-weighted.npy', allow_pickle=False)[cell].item()
    assert original_full-high_precision == reduction['original_full_shape_absolute_error']

    receipt = json.loads((args.margin/'receipt.json').read_text())
    for row in receipt['payloads']:
        f = args.margin/row['path']; assert f.stat().st_size == row['bytes'] and sha(f) == row['sha256']
    logits = {owner: np.load(args.margin/(owner+'-step978-logits.npy'), allow_pickle=False)
              for owner in ['original', 'native']}
    observed = {}
    for owner, z in logits.items():
        u, c = z[0, 2].astype('f8'), z[1, 2].astype('f8')
        observed[owner] = float(4*(c[46]-c[904])-3*(u[46]-u[904]))
        assert observed[owner] == receipt['margins'][owner]['float64'] == receipt['margins'][owner]['float32']
    errors = logits['native'].astype('f8')-logits['original'].astype('f8')
    conditional = float(4*(errors[1, 2, 46]-errors[1, 2, 904]))
    unconditional = float(-3*(errors[0, 2, 46]-errors[0, 2, 904]))
    assert conditional == receipt['conditional_error_contribution']
    assert unconditional == receipt['unconditional_error_contribution']
    assert observed['original']+conditional+unconditional == observed['native']
    result = dict(independent_small_audit_pass=True, tiny_exact=.5, tiny_native=.25,
                  native_four_term_float32_accumulation_reproduced=True,
                  full_native_component_reduced_to_one_dot_exact=True,
                  four_logit_cfg_margin_decomposition_exact=True,
                  scope='Specific retained CPU arithmetic; no global rank repair or speech acceptance.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
