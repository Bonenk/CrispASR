#!/usr/bin/env python3
"""Capture actual pinned original self-attention Q with owned-input controls.

This loads five layer0 tensors, never the full neural model. The original
self-attention forward and RoPE are executed unchanged; hooks only own copies.
A module replay is explicitly distinguished from the earlier full-model run.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import types

import numpy as np

SOURCE = '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'
WEIGHTS = 'caba289b60f6d7d1e58fc744f4dc25aae88995fcca46be3d05e220b971486a26'
CONFIG_SHA = '9140e85fd15b82d7f268c5681a41fb1de68573ec206b87cb3a5862fe5c598d6e'
COEFFICIENTS = {
    'pre_sa_norm.weight': 'aa94916f5b9bb77e33c3df398519de3dc6c448c9dccbaa99d6b465dfea5da920',
    'self_attention.q_proj.weight': '63da7f82eeb19a61b192a5ef6ee6c33b36f380583f120a7a7943f1dbc9a1e1e3',
    'self_attention.k_proj.weight': '48d15ad15b626b6edceb966177e5458c44b074d0d58892fbf2f0981f551178b2',
    'self_attention.v_proj.weight': 'dbe09879b470c7bb4f3acea36f4cc00afd517ae18ad8faa7105247ba3f3bc573',
    'self_attention.o_proj.weight': 'fd7728e6fe7a235dd8b72e12e8262aa9efff1f3e57a537c69c08fdec0c63df4c',
}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def metrics(x, y):
    assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
    a, b = x.astype('f8').reshape(-1, x.shape[-1]), y.astype('f8').reshape(-1, y.shape[-1])
    an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    assert (an > 0).all() and (bn > 0).all()
    return dict(byte_exact=x.tobytes() == y.tobytes(),
                cosine_min=float(np.min(np.sum(a*b, axis=1)/(an*bn))),
                mine_norm=float(np.linalg.norm(a)), ref_norm=float(np.linalg.norm(b)),
                max_row_norm_error=float(abs(an/bn-1).max()),
                relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                max_abs=float(abs(a-b).max()))


def reconstruction_gate(row):
    # Same strict gate as the accepted original key-producer reconstruction.
    return (row['relative_l2'] <= 2e-6 and row['max_row_norm_error'] <= 2e-6
            and row['cosine_min'] >= .9999999999)


def main():
    import torch
    from gguf import GGUFReader
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['blueprint', 'coefficients', 'prior', 'canonical', 'native', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    assert torch.__version__ == '2.6.0+cpu'
    torch.set_num_threads(4)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    consumed = []

    def retain(path, name, expected=None):
        if expected:
            assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], path
        dest = out/'inputs'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        consumed.append(dict(path=dest.relative_to(out).as_posix(), bytes=dest.stat().st_size, sha256=sha(dest)))
        return dest

    prior = json.loads((args.prior/'receipt.json').read_text())
    assert prior['original_source'] == SOURCE and prior['torch_version'] == torch.__version__
    prior_files = {r['path']: r for r in prior['files']}
    fetch = json.loads((args.coefficients/'fetch-receipt.json').read_text())
    assert fetch['checkpoint_linked_sha256'] == WEIGHTS
    retain(args.coefficients/'fetch-receipt.json', 'physical-coefficients.json')
    retain(args.prior/'receipt.json', 'prior-receipt.json')
    rows = {r['original_name'].removeprefix('decoder.layers.0.'): r for r in fetch['tensors']}
    arrays = {}
    for name, expected in COEFFICIENTS.items():
        row = rows[name]
        assert row['sha256'] == expected
        path = retain(args.coefficients/row['path'], row['path'], row)
        arrays[name] = np.fromfile(path, dtype='<f4').reshape(row['shape']).copy()
    config_path = retain(args.prior/'original-config.json', 'config.json', prior_files['original-config.json'])
    assert sha(config_path) == CONFIG_SHA

    def objectify(value):
        return types.SimpleNamespace(**{k: objectify(v) for k, v in value.items()}) if isinstance(value, dict) else value

    config = objectify(json.loads(config_path.read_text()))
    assert (config.model.decoder.gqa_query_heads, config.model.decoder.kv_heads,
            config.model.decoder.gqa_head_dim, config.model.decoder.n_embd) == (16, 4, 128, 2048)
    assert config.model.normalization_layer_epsilon == 1e-5
    ns = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, Tensor=torch.Tensor, DiaConfig=object)
    for filename, selected in [('state.py', {'KVCache'}),
                               ('layers.py', {'_normalize_axes', 'DenseGeneral', 'RotaryEmbedding', 'Attention'})]:
        path = args.blueprint/'dia'/filename
        assert path.read_bytes() == subprocess.check_output(['git', '-C', args.blueprint, 'show', SOURCE+':dia/'+filename])
        retained = retain(path, 'original-'+filename)
        nodes = [n for n in ast.parse(retained.read_text()).body if getattr(n, 'name', None) in selected]
        assert len(nodes) == len(selected)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(retained), 'exec'), ns)
    attention = ns['Attention'](config, 2048, 2048, 16, 4, 128, torch.float32).eval()
    state_dict = {n.removeprefix('self_attention.'): torch.from_numpy(a) for n, a in arrays.items() if n.startswith('self_attention.')}
    attention.load_state_dict(state_dict, strict=True)
    norm = torch.nn.RMSNorm(2048, eps=1e-5, dtype=torch.float32)
    with torch.no_grad():
        norm.weight.copy_(torch.from_numpy(arrays['pre_sa_norm.weight']))
    hidden = np.load(retain(args.prior/'original-original-input.npy', 'hidden.npy', prior_files['original-original-input.npy']))
    assert hidden.shape == (2, 1, 2048) and hidden.dtype == np.dtype('<f4')
    expected = {}
    for part in ['pre_sa_norm', 'sa_attention', 'sa_output']:
        name = 'original-original-'+part+'.npy'
        expected[part] = np.load(retain(args.prior/name, part+'.npy', prior_files[name]))
    cache_rows = {Path(r['path']).name: r for r in prior['verified_inputs']}
    past = {}
    for part in ['k', 'v']:
        row = cache_rows['layer-0-original-input-'+part+'.f32']
        path = retain(Path(row['path']), 'original-full-'+part+'.f32', row)
        past[part] = np.fromfile(path, dtype='<f4').reshape(2, 979, 4, 128)
    delivery = json.loads((args.canonical/'delivery.json').read_text())
    row = next(r for r in delivery['files'] if r['path'] == 'output/original-step978-ref.gguf')
    reference = args.canonical/row['path']
    assert reference.stat().st_size == row['bytes'] and sha(reference) == row['sha256']
    reader = GGUFReader(str(reference))
    tensor_map = {t.name: t for t in reader.tensors}
    assert 'decoder_block0_cross_q' in tensor_map
    # This fixture genuinely has no self-Q; never relabel its cross-Q.
    assert not any('sa0_q' in name or 'self_q' in name for name in tensor_map)
    full_expected = {}
    for part in ['pre_sa_norm', 'sa_attention', 'sa_output']:
        t = tensor_map['decoder_block0_'+part]
        full_expected[part] = t.data.reshape(tuple(map(int, reversed(t.shape)))).copy()
        np.save(out/('full-reference-'+part+'.npy'), full_expected[part])
    native_receipt = json.loads((args.native/'receipt.json').read_text())
    row = next(r for r in native_receipt['payloads'] if r['path'] == 'baseline/cpp_decoder_sa0_q.f32')
    native_q = np.fromfile(retain(args.native/row['path'], 'native-query.f32', row), dtype='<f4').reshape(2, 1, 16, 128)
    assert np.isfinite(native_q).all()
    np.save(out/'native-query.npy', native_q)

    def execute(capacity, query_control=None):
        cache = ns['KVCache'](4, capacity, 128, torch.float32, torch.device('cpu'))
        cache.k[:, :, :978] = torch.from_numpy(past['k'][:, :978].copy()).transpose(1, 2)
        cache.v[:, :, :978] = torch.from_numpy(past['v'][:, :978].copy()).transpose(1, 2)
        cache.current_idx = torch.tensor(978)
        captured, handles = {}, []

        def q_projection(_module, _args, value):
            captured['pre_rope_query'] = value.detach().numpy().copy()

        def rotary(_module, args, kwargs, value):
            position = kwargs['position']
            assert position.shape == (2, 1) and bool((position == 978).all())
            if value.shape == (2, 1, 16, 128):
                assert 'query' not in captured
                captured['query'] = value.detach().numpy().copy()
                captured['query_positions'] = position.detach().numpy().copy()
                if query_control == 'scale': return value*2
                if query_control == 'head-order': return value.flip(2)
            else:
                assert value.shape == (2, 1, 4, 128)
                captured['current_key'] = value.detach().numpy().copy()

        def before_output(_module, args):
            captured['sa_attention'] = args[0].detach().numpy().copy()

        try:
            handles.append(attention.q_proj.register_forward_hook(q_projection))
            handles.append(attention.rotary_emb.register_forward_hook(rotary, with_kwargs=True))
            handles.append(attention.o_proj.register_forward_pre_hook(before_output))
            with torch.inference_mode():
                normalized = norm(torch.from_numpy(hidden.copy())).to(torch.float32)
                captured['pre_sa_norm'] = normalized.numpy().copy()
                y = attention(Xq=normalized, Xkv=normalized,
                              q_positions=torch.full((2, 1), 978, dtype=torch.float32),
                              kv_positions=torch.full((2, 1), 978, dtype=torch.float32),
                              attn_mask=None, cache=cache, prefill=False, is_causal=False)
                captured['sa_output'] = y.numpy().copy()
                assert int(cache.current_idx) == 979
        finally:
            for handle in handles: handle.remove()
        return captured

    runs = {}
    for name, capacity, control in [('prior-capacity', 979, None), ('original-capacity', config.data.audio_length, None),
                                    ('scale-control', config.data.audio_length, 'scale'),
                                    ('head-order-control', config.data.audio_length, 'head-order')]:
        captured = execute(capacity, control)
        for key, value in captured.items(): np.save(out/(name+'-'+key+'.npy'), value)
        runs[name] = captured
    repeated = execute(config.data.audio_length)
    assert all(np.array_equal(v, repeated[k]) for k, v in runs['original-capacity'].items())
    prior_metrics = {k: metrics(runs['prior-capacity'][k].reshape(expected[k].shape), expected[k]) for k in expected}
    full_metrics = {k: metrics(runs['original-capacity'][k].reshape(full_expected[k].shape), full_expected[k]) for k in expected}
    negatives = {name: metrics(runs[name]['sa_output'], runs['original-capacity']['sa_output'])
                 for name in ['scale-control', 'head-order-control']}
    assert all(not reconstruction_gate(v) for v in negatives.values())
    query = runs['original-capacity']['query']
    assert query.shape == (2, 1, 16, 128) and query.dtype == np.dtype('<f4')
    assert query.tobytes() == runs['prior-capacity']['query'].tobytes()
    report = dict(scope=__doc__, source=SOURCE, producer_sha256=sha(__file__),
                  numpy_version=np.__version__, torch_version=torch.__version__,
                  torch_build_configuration=torch.__config__.show(), coefficient_bytes=sum(a.nbytes for a in arrays.values()),
                  verified_inputs=consumed, canonical_reference_sha256=sha(reference),
                  prior_module_replay=prior_metrics, full_original_reference=full_metrics,
                  negative_controls=negatives, repeats_exact=True,
                  native_vs_replayed_original_query=metrics(native_q, query),
                  cache_capacity_does_not_change_query=True,
                  full_original_self_query_previously_captured=False,
                  full_reference_reconstruction_gate_pass=all(reconstruction_gate(v) for v in full_metrics.values()),
                  full_reference_all_stages_byte_exact=all(v['byte_exact'] for v in full_metrics.values()),
                  prior_module_replay_byte_exact=all(v['byte_exact'] for v in prior_metrics.values()),
                  speech_acceptance=False, gpu_acceptance=False,
                  interpretation_limit='Observed Q belongs to actual pinned original module replay. It is not relabeled as an earlier full-model captured query.')
    report['files'] = [dict(path=f.relative_to(out).as_posix(), bytes=f.stat().st_size, sha256=sha(f))
                       for f in sorted(out.rglob('*')) if f.is_file() and f.name != 'receipt.json']
    (out/'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['verified_inputs','files']}, indent=2), flush=True)
    assert report['prior_module_replay_byte_exact'], 'Actual original module replay must reproduce retained component stages exactly'
    assert report['full_reference_reconstruction_gate_pass'], 'Stop: original self-attention replay does not reconstruct full reference at the predeclared strict gate'
    print('DIA_ORIGINAL_SELF_QUERY_REPLAY_PASS; not an earlier full-model query capture', flush=True)


if __name__ == '__main__':
    main()
