#!/usr/bin/env python3
"""Tiny analytic invariants and exact-input Dia attention arithmetic decomposition.

No full model is loaded. Original SDPA remains distinct from an explicit shadow
matmul/softmax decomposition. All arrays, failures, sources and libraries survive.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = '8d0ad8862d2cad5d305a48a57f2a99b5024bc39a'
GGML = 'c36dab89b662838f0f5d4826c399198c0b90bbfc'
ORIGINAL = '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def metrics(mine, reference):
    assert mine.shape == reference.shape and np.isfinite(mine).all() and np.isfinite(reference).all()
    x, y = mine.astype('f8').reshape(-1, mine.shape[-1]), reference.astype('f8').reshape(-1, mine.shape[-1])
    xn, yn = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    nonzero = (xn > 0) & (yn > 0)
    return dict(byte_exact=mine.tobytes() == reference.tobytes(),
                cosine_min=float(np.min(np.sum(x[nonzero]*y[nonzero], axis=1)/(xn[nonzero]*yn[nonzero]))) if nonzero.any() else None,
                mine_norm=float(np.linalg.norm(x)), ref_norm=float(np.linalg.norm(y)),
                relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(y)) if np.linalg.norm(y) else None,
                max_abs=float(abs(x-y).max()),
                max_row_norm_error=float(np.max(abs(xn[yn > 0]/yn[yn > 0]-1))) if (yn > 0).any() else None)


def verify_tree(directory, key):
    report = json.loads((directory/'receipt.json').read_text())
    rows = report[key]
    assert len({r['path'] for r in rows}) == len(rows)
    for row in rows:
        path = Path(row['path'])
        assert not path.is_absolute() and '..' not in path.parts
        file = directory/path
        assert file.stat().st_size == row['bytes'] and sha(file) == row['sha256'], file
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['query-proof', 'native', 'native-query-proof', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    assert torch.__version__ == '2.6.0+cpu'
    torch.set_num_threads(4)
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    report = dict(scope=__doc__, runtime_source=SOURCE, ggml_source=GGML, original_source=ORIGINAL,
                  worker_sha256=sha(__file__), torch_version=torch.__version__, cases=[],
                  speech_acceptance=False, gpu_acceptance=False, defaults_changed=False)

    def checkpoint():
        (out/'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    checkpoint()
    query = verify_tree(args.query_proof, 'files')
    native = json.loads((args.native/'receipt.json').read_text())
    proof = verify_tree(args.native_query_proof, 'payloads')
    assert proof['native_capture_reproduced'] and proof['ggml_source'] == GGML
    assert native['runtime_source'] == proof['runtime_source'] == SOURCE
    assert native['conditional_diagnostic_complete'] and query['full_reference_reconstruction_gate_pass']
    assert query['prior_module_replay_byte_exact'] and not query['full_original_self_query_previously_captured']
    inputs = out/'inputs'; inputs.mkdir()

    def retain(path, name, row=None):
        if row:
            assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], path
        target = inputs/name; shutil.copyfile(path, target); return target

    for name, path in [('query-receipt.json', args.query_proof/'receipt.json'),
                       ('native-receipt.json', args.native/'receipt.json'),
                       ('native-query-receipt.json', args.native_query_proof/'receipt.json')]:
        retain(path, name)
    for folder in ['libs', 'headers']:
        shutil.copytree(args.native_query_proof/folder, out/folder)
    for name, value in proof['actual_library_hashes'].items():
        assert sha(out/'libs'/name) == value
    source = ROOT/'tools/ci-heavy/dia-attention-arithmetic/bridge.cpp'
    shutil.copyfile(source, out/'bridge.cpp')
    runtime = subprocess.check_output(['git', 'show', SOURCE+':src/dia_tts.cpp'], cwd=ROOT, text=True)
    start = runtime.index('                    int n_rep = n_heads / n_kv_heads;', runtime.index('// --- Decoder layers ---'))
    end = runtime.index('                    cur = ggml_mul_mat(ctx0, layer.self_o_proj, cur);', start)
    (out/'literal-runtime-attention.cpp').write_text(runtime[start:end])
    command = ['c++', '-std=c++17', '-O2', '-I', out/'headers', out/'bridge.cpp',
               '-Wl,-rpath,$ORIGIN/libs', out/'libs/libggml-cpu.so.0', out/'libs/libggml-base.so.0', '-o', out/'bridge']
    (out/'build-command.json').write_text(json.dumps(list(map(str, command)), indent=2)+'\n')
    with (out/'build.log').open('w') as log:
        result = subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, (out/'build.log').read_text()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('LD_', 'GGML_', 'DIA_', 'CRISPASR_'))}
    closure = subprocess.check_output(['ldd', out/'bridge'], env=env, text=True)
    (out/'ldd.log').write_text(closure); assert 'not found' not in closure
    for line in closure.splitlines():
        if 'libggml' in line:
            assert Path(line.split('=>')[1].strip().split()[0]).resolve().parent == (out/'libs').resolve()
    report.update(binary_sha256=sha(out/'bridge'), actual_library_hashes=proof['actual_library_hashes'])
    checkpoint()

    # Execute the unchanged driving call and transpose from original Attention.
    original_layers = args.query_proof/'inputs/original-layers.py'
    retain(original_layers, 'original-layers.py')
    tree = ast.parse(original_layers.read_text())
    attention = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Attention')
    forward = next(n for n in attention.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    index = next(i for i, n in enumerate(forward.body) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                 and getattr(n.value.func, 'attr', None) == 'scaled_dot_product_attention')
    expression = ast.Module(body=forward.body[index:index+2], type_ignores=[])
    (out/'actual-original-sdpa.py').write_text(ast.unparse(expression)+'\n')
    code = compile(expression, 'unchanged-original-sdpa', 'exec')

    def original_sdpa(q, k, v):
        batch, time, kvheads, dim = k.shape
        # Preserve original noncompact cache strides (capacity3072), not just values.
        keys = torch.zeros((batch, kvheads, 3072, dim), dtype=torch.float32)
        values = torch.zeros_like(keys)
        keys[:, :, :time] = torch.from_numpy(k.copy()).transpose(1, 2)
        values[:, :, :time] = torch.from_numpy(v.copy()).transpose(1, 2)
        ns = dict(F=torch.nn.functional, self=types.SimpleNamespace(num_gqa_groups=q.shape[2]//kvheads),
                  Xq_BxNxTxH=torch.from_numpy(q.copy()).transpose(1, 2), attn_k=keys[:, :, :time],
                  attn_v=values[:, :, :time], attn_mask=None, is_causal=False)
        with torch.inference_mode(): exec(code, ns)
        return ns['attn_output'].numpy().copy()

    def shadow(q, k, v):
        qt = torch.from_numpy(q.copy()).transpose(1, 2)
        kt = torch.from_numpy(k.copy()).transpose(1, 2).repeat_interleave(q.shape[2]//k.shape[2], dim=1)
        vt = torch.from_numpy(v.copy()).transpose(1, 2).repeat_interleave(q.shape[2]//k.shape[2], dim=1)
        with torch.inference_mode():
            scores = qt @ kt.transpose(-2, -1)
            probabilities = torch.softmax(scores, dim=-1)
            weighted = (probabilities @ vt).transpose(1, 2).contiguous()
        return scores.numpy(), probabilities.numpy(), weighted.numpy()

    def case(name, q, k, v, expected=None, analytic=None):
        folder = out/name; folder.mkdir()
        assert q.dtype == k.dtype == v.dtype == np.dtype('<f4')
        assert q.shape[:2] == (k.shape[0], 1) and q.shape[-1] == k.shape[-1] and k.shape == v.shape
        for part, value in [('q', q), ('k', k), ('v', v)]: value.tofile(folder/(part+'.f32'))
        score, probability, shadow_value = shadow(q, k, v)
        score.tofile(folder/'shadow-scores.f32'); probability.tofile(folder/'shadow-probabilities.f32')
        np.save(folder/'shadow-weighted.npy', shadow_value)
        actual = original_sdpa(q, k, v)
        assert actual.tobytes() == original_sdpa(q, k, v).tobytes()
        np.save(folder/'actual-original-sdpa.npy', actual)
        if expected is not None:
            np.save(folder/'retained-attention.npy', expected)
        expanded_k = np.repeat(k, q.shape[2]//k.shape[2], axis=2).transpose(0, 2, 1, 3).astype('f8')
        expanded_v = np.repeat(v, q.shape[2]//v.shape[2], axis=2).transpose(0, 2, 1, 3).astype('f8')
        exact_scores = q.transpose(0, 2, 1, 3).astype('f8') @ expanded_k.transpose(0, 1, 3, 2)
        # Evaluate the reduction on identical rounded probabilities; do not
        # conflate softmax differences with the weighted-value accumulator.
        weighted64 = (probability.astype('f8') @ expanded_v).transpose(0, 2, 1, 3)
        np.save(folder/'scores-float64.npy', exact_scores)
        np.save(folder/'fixed-probability-weighted-float64.npy', weighted64)
        row = dict(case=name, shape_q=list(q.shape), shape_kv=list(k.shape),
                   shadow_vs_actual_sdpa=metrics(shadow_value, actual),
                   shadow_scores_vs_float64=metrics(score, exact_scores),
                   shadow_fixed_probability_vs_float64=metrics(shadow_value, weighted64), modes=[])
        report['cases'].append(row); checkpoint()
        outputs = {}
        for mode in [0, 1, 2]:
            runs = {}
            for threads in [1, 4]:
                for suffix in ['', '-repeat']:
                    dest = folder/f'mode-{mode}-threads-{threads}{suffix}'
                    command = [out/'bridge', folder, dest, q.shape[-1], q.shape[2], k.shape[2], k.shape[1], k.shape[0], mode, threads]
                    result = subprocess.run(list(map(str, command)), env=env, capture_output=True, text=True)
                    (folder/(dest.name+'.log')).write_text(result.stdout+result.stderr)
                    assert result.returncode == 0 and 'DIA_ATTENTION_ARITHMETIC_PASS' in result.stdout
                    values = {part: np.fromfile(dest/(part+'.f32'), dtype='<f4').reshape(shape)
                              for part, shape in [('scores', score.shape), ('used-scores', score.shape),
                                                  ('probabilities', score.shape), ('used-probabilities', score.shape), ('weighted', q.shape)]}
                    assert all(np.isfinite(x).all() for x in values.values())
                    runs[threads, suffix] = values
            first = runs[1, '']
            assert all(first[part].tobytes() == values[part].tobytes() for values in runs.values() for part in first)
            if mode == 1: assert first['used-scores'].tobytes() == score.tobytes()
            if mode == 2: assert first['used-probabilities'].tobytes() == probability.tobytes()
            outputs[mode] = first
            row['modes'].append(dict(mode=mode, repeats_one_four_threads_exact=True,
                                     scores_vs_shadow=metrics(first['scores'], score),
                                     probabilities_vs_shadow=metrics(first['probabilities'], probability),
                                     weighted_vs_actual_sdpa=metrics(first['weighted'], actual),
                                     weighted_vs_shadow=metrics(first['weighted'], shadow_value),
                                     weighted_vs_fixed_probability_float64=metrics(first['weighted'], weighted64) if mode == 2 else None))
            checkpoint()
        if name == 'captured-native':
            assert outputs[0]['weighted'].tobytes() == expected.tobytes(), 'native hosted attention must reproduce exactly'
            row['native_hosted_attention_byte_exact'] = True
        if name == 'original-module-replay':
            assert actual.tobytes() == expected.tobytes(), 'must reproduce retained original module attention exactly'
            row['original_module_attention_byte_exact'] = True
        if analytic:
            analytic(folder, outputs[0], actual)
            row['analytic_invariants_pass'] = True
        checkpoint()
        return outputs

    # Tiny exact oracle: four equal weights; values distinguish every batch,
    # KV head and component. Each pair of query heads must select the same KV head.
    q = np.zeros((2, 1, 4, 8), dtype='<f4')
    k = np.arange(2*4*2*8, dtype='<f4').reshape(2, 4, 2, 8)/16
    b, t, h, d = np.indices(k.shape)
    v = (100*b+10*h+2*t+d).astype('<f4')

    def uniform(folder, native, actual):
        expected = np.repeat(v.mean(axis=1, keepdims=True), 2, axis=2)
        assert np.array_equal(native['scores'], np.zeros((2, 4, 1, 4), dtype='<f4'))
        assert np.array_equal(native['probabilities'], np.full((2, 4, 1, 4), .25, dtype='<f4'))
        assert native['weighted'].tobytes() == expected.tobytes() == actual.tobytes()
        wrong_tiled = np.tile(v.mean(axis=1, keepdims=True), (1, 1, 2, 1))
        assert np.linalg.norm(wrong_tiled-expected) > 1 and np.linalg.norm(expected[::-1]-expected) > 1
        np.save(folder/'analytic-expected.npy', expected)
        np.save(folder/'wrong-tiled-heads.npy', wrong_tiled)
        np.save(folder/'wrong-batch-order.npy', expected[::-1])

    case('tiny-uniform', q, k, v, analytic=uniform)
    q.fill(0); q[..., 0] = 1
    k.fill(0)
    k[:, :, :, 0] = np.array([-2, -1, 1, 2], dtype='<f4')[None, :, None]

    def nonuniform(folder, native, actual):
        scores = np.broadcast_to(np.array([-2., -1., 1., 2.]), (2, 4, 1, 4)).copy()
        p = np.exp(scores-scores.max(axis=-1, keepdims=True)); p /= p.sum(axis=-1, keepdims=True)
        values = np.repeat(v, 2, axis=2).transpose(0, 2, 1, 3).astype('f8')
        expected = (p@values).transpose(0, 2, 1, 3)
        np.save(folder/'analytic-scores.npy', scores); np.save(folder/'analytic-probabilities.npy', p)
        np.save(folder/'analytic-weighted.npy', expected)
        assert np.array_equal(native['scores'], scores)
        assert abs(native['probabilities']-p).max() < 1e-7
        assert abs(native['weighted']-expected).max() < 2e-5
        wrong = np.exp(scores/np.sqrt(8)); wrong /= wrong.sum(axis=-1, keepdims=True)
        assert abs(wrong-p).max() > .1
        np.save(folder/'wrong-sqrt-scaled-probabilities.npy', wrong)

    positive = case('tiny-nonuniform', q.copy(), k.copy(), v.copy(), analytic=nonuniform)
    # Actual bad-input executions demonstrate sensitivity of the same bridge.
    negatives = []
    for name, nq, nk, nv in [('sqrt-scale', q/np.float32(np.sqrt(8)), k, v),
                             ('head-order', q, k[:, :, ::-1].copy(), v[:, :, ::-1].copy()),
                             ('double-scale', q, k, v*2)]:
        folder = out/('negative-'+name); folder.mkdir()
        for part, value in [('q', nq), ('k', nk), ('v', nv)]: value.tofile(folder/(part+'.f32'))
        command = [out/'bridge', folder, folder/'actual', 8, 4, 2, 4, 2, 0, 4]
        run = subprocess.run(list(map(str, command)), env=env, capture_output=True, text=True)
        (folder/'run.log').write_text(run.stdout+run.stderr)
        assert run.returncode == 0 and 'DIA_ATTENTION_ARITHMETIC_PASS' in run.stdout
        bad = np.fromfile(folder/'actual/weighted.f32', dtype='<f4').reshape(q.shape)
        m = metrics(bad, positive[0]['weighted'])
        assert m['relative_l2'] > 1e-4
        negatives.append(dict(control=name, **m))
    report['actual_negative_controls'] = negatives
    checkpoint()

    # Four values suffice to expose non-associative cancellation. This is a
    # diagnostic measurement against the exact answer, not a relaxed parity gate.
    cq = np.zeros((1, 1, 1, 1), dtype='<f4')
    ck = np.zeros((1, 4, 1, 1), dtype='<f4')
    cv = np.array([2**24, 1, -2**24, 1], dtype='<f4').reshape(ck.shape)
    def cancellation(folder, native, actual):
        exact = math.fsum(float(x)*.25 for x in cv.ravel())
        assert exact == .5 and np.array_equal(native['probabilities'], np.full((1, 1, 1, 4), .25, dtype='<f4'))
        (folder/'exact-cancellation.json').write_text(json.dumps(dict(exact=exact, native=float(native['weighted'].item()),
                                                                     original=float(actual.item())), indent=2)+'\n')
    case('tiny-cancellation', cq, ck, cv, analytic=cancellation)
    # Verify and retain exactly the native arrays from the prior successful full run.
    rows = {r['path']: r for r in native['payloads']}
    arrays = []
    for part, shape in [('q', (2, 1, 16, 128)), ('k', (2, 979, 4, 128)), ('v', (2, 979, 4, 128)), ('attention', (2, 1, 16, 128))]:
        name = 'baseline/cpp_decoder_sa0_'+part+'.f32'
        path = retain(args.native/name, 'native-'+part+'.f32', rows[name])
        arrays.append(np.fromfile(path, dtype='<f4').reshape(shape))
    case('captured-native', *arrays[:3], expected=arrays[3])

    # Reconstruct the exact original module's consumed current V and K, not a
    # mixture of its replay Q with the earlier full-model current-cache entry.
    ns = dict(torch=torch, nn=torch.nn, Tensor=torch.Tensor)
    nodes = [n for n in tree.body if getattr(n, 'name', None) in {'_normalize_axes', 'DenseGeneral'}]
    assert len(nodes) == 2
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual-original-dense', 'exec'), ns)
    norm = np.load(args.query_proof/'original-capacity-pre_sa_norm.npy', allow_pickle=False)
    physical = json.loads((args.query_proof/'inputs/physical-coefficients.json').read_text())
    vr = next(r for r in physical['tensors'] if r['original_name'] == 'decoder.layers.0.self_attention.v_proj.weight')
    vp = retain(args.query_proof/'inputs'/vr['path'], 'original-v-weight.f32', vr)
    projection = ns['DenseGeneral']((2048,), (4, 128), weight_dtype=torch.float32)
    with torch.inference_mode():
        projection.weight.copy_(torch.from_numpy(np.fromfile(vp, dtype='<f4').reshape(vr['shape']).copy()))
        current_v = projection(torch.from_numpy(norm.copy())).numpy().copy()
    retain(args.query_proof/'receipt.json', 'original-current-v-provenance.json')
    q = np.load(args.query_proof/'original-capacity-query.npy', allow_pickle=False)
    k = np.fromfile(args.query_proof/'inputs/original-full-k.f32', dtype='<f4').reshape(2, 979, 4, 128)
    v = np.fromfile(args.query_proof/'inputs/original-full-v.f32', dtype='<f4').reshape(k.shape)
    k[:, -1:] = np.load(args.query_proof/'original-capacity-current_key.npy', allow_pickle=False)
    v[:, -1:] = current_v
    expected = np.load(args.query_proof/'original-capacity-sa_attention.npy', allow_pickle=False)
    case('original-module-replay', q, k, v, expected=expected)
    # This final arm preserves captured original full K/V; its Q still comes
    # from qualified module replay, explicitly not a full-model Q capture.
    k = np.fromfile(args.query_proof/'inputs/original-full-k.f32', dtype='<f4').reshape(k.shape)
    v = np.fromfile(args.query_proof/'inputs/original-full-v.f32', dtype='<f4').reshape(k.shape)
    case('original-captured-cache', q, k, v)
    report.update(complete=True, native_hosted_attention_byte_exact=True,
                  original_module_attention_byte_exact=True,
                  limitation='Explicit shadow scores/probabilities are not internal tensors of fused SDPA. Original full-model self-Q was not retained. No speech or global rank causality from component error.')
    report['payloads'] = [dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                          for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'receipt.json']
    checkpoint()
    print('DIA_ATTENTION_ARITHMETIC_COMPLETE; tiny invariants and exact native/original controls pass', flush=True)
    print(json.dumps(report['cases'], indent=2), flush=True)


if __name__ == '__main__':
    main()
