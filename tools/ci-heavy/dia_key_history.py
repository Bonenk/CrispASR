#!/usr/bin/env python3
"""Replay the actual original layer0 key producer on retained shared history.

Only embeddings, normalization, key projection and RoPE are loaded. Synthetic
or batched computation never substitutes for the original one-step positive
control. This is a component diagnostic, not new model/speech acceptance.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import types

import numpy as np


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def metric(actual, expected):
    assert actual.shape == expected.shape and actual.size
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    a = actual.astype('f8').reshape(-1, actual.shape[-1])
    b = expected.astype('f8').reshape(-1, expected.shape[-1])
    an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    assert (an > 0).all() and (bn > 0).all()
    return dict(relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                cosine_min=float(np.min(np.sum(a*b, axis=1)/(an*bn))),
                max_row_norm_error=float(np.max(np.abs(an/bn-1))),
                mine_norm=float(np.linalg.norm(a)), ref_norm=float(np.linalg.norm(b)),
                max_abs=float(np.max(np.abs(a-b))), byte_exact=actual.tobytes() == expected.tobytes())


def reconstruction_gate(row):
    return (row['relative_l2'] <= 2e-6 and row['max_row_norm_error'] <= 2e-6
            and row['cosine_min'] >= .9999999999)


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert torch.__version__ == '2.6.0+cpu', torch.__version__
    torch.set_num_threads(4)
    out, inputs = args.output, args.inputs
    out.mkdir(parents=True, exist_ok=False)
    preparation = json.loads((inputs/'preparation.json').read_text())
    assert preparation['passed']
    for row in preparation['files']:
        name = Path(row['path'])
        assert not name.is_absolute() and '..' not in name.parts
        file = inputs/name
        assert file.stat().st_size == row['bytes'] and digest(file) == row['sha256']
    text = (inputs/'original-layers.py').read_text()
    parsed = ast.parse(text)
    names = {'_normalize_axes', 'DenseGeneral', 'RotaryEmbedding'}
    nodes = [node for node in parsed.body if getattr(node, 'name', None) in names]
    assert len(nodes) == len(names)
    namespace = dict(torch=torch, nn=torch.nn, Tensor=torch.Tensor)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual-original-layers', 'exec'), namespace)
    decoder = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == 'Decoder')
    step = next(node for node in decoder.body if isinstance(node, ast.FunctionDef) and node.name == 'decode_step')
    # Keep the actual embedding-loop statements, stopping before decoder layers.
    body = []
    for node in step.body:
        if isinstance(node, ast.For) and ast.unparse(node.iter) == 'enumerate(self.layers)':
            break
        body.append(node)
    else:
        raise AssertionError('original decoder boundary missing')
    body.append(ast.Return(value=ast.Name(id='x', ctx=ast.Load())))
    function = ast.FunctionDef(name='original_embedding', args=step.args, body=body,
                              decorator_list=[], returns=step.returns, type_comment=None)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace['DecoderInferenceState'] = object
    exec(compile(module, 'actual-original-embedding-prefix', 'exec'), namespace)
    (out/'actual-original-expressions.py').write_text(
        '\n\n'.join(ast.unparse(node) for node in nodes)+'\n\n'+ast.unparse(module)+'\n')
    coefficients = {row['name']: row for row in preparation['coefficients']}

    def array(name):
        row = coefficients[name]
        return np.fromfile(inputs/row['path'], dtype='<f4').reshape(row['shape']).copy()

    embeddings = []
    for channel in range(9):
        weight = torch.from_numpy(array(f'decoder.embeddings.{channel}.weight'))
        embeddings.append(torch.nn.Embedding.from_pretrained(weight, freeze=True))
    owner = types.SimpleNamespace(num_channels=9, embeddings=embeddings)
    norm = torch.nn.RMSNorm(2048, eps=1e-5, dtype=torch.float32)
    projection = namespace['DenseGeneral']((2048,), (4, 128), weight_dtype=torch.float32)
    with torch.no_grad():
        norm.weight.copy_(torch.from_numpy(array('decoder.layers.0.pre_sa_norm.weight')))
        projection.weight.copy_(torch.from_numpy(array('decoder.layers.0.self_attention.k_proj.weight')))
    rotary = namespace['RotaryEmbedding'](128, dtype=torch.float32)
    tokens = np.load(inputs/'shared-input-tokens.npy')
    assert tokens.shape == (979, 9) and ((tokens >= 0) & (tokens < 1028)).all()

    def produce(ids, positions):
        hidden = namespace['original_embedding'](owner, ids, None)
        normalized = norm(hidden).to(torch.float32)
        key = projection(normalized)
        rotated = rotary(key, position=positions)
        return dict(hidden=hidden, normalized=normalized, pre_rope=key, post_rope=rotated)

    stages = {name: [] for name in ['hidden', 'normalized', 'pre_rope', 'post_rope']}
    with torch.inference_mode():
        for position, row in enumerate(tokens):
            ids = torch.from_numpy(row.astype('i8')).reshape(1, 1, 9).expand(2, -1, -1)
            values = produce(ids, torch.full((2, 1), position, dtype=torch.int64))
            for name, value in values.items():
                stages[name].append(value.numpy().copy())
    stages = {name: np.concatenate(rows, axis=1) for name, rows in stages.items()}
    for name, value in stages.items():
        np.save(out/(name+'.npy'), value)
    captured = np.load(inputs/'original-captured-keys.npy')
    positive = metric(stages['post_rope'], captured)
    hidden = metric(stages['hidden'][:, 978:979], np.load(inputs/'decoder_block0_input.npy').reshape(2, 1, 2048))
    report = dict(scope=__doc__, torch_version=torch.__version__, input_manifest_sha256=digest(inputs/'preparation.json'),
                  original_reconstruction=positive, last_hidden_reconstruction=hidden,
                  original_reconstruction_pass=reconstruction_gate(positive), speech_acceptance=False)
    (out/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    # A failed reconstruction is evidence, never attributed to native arithmetic.
    assert reconstruction_gate(positive), positive
    assert hidden['byte_exact'], hidden
    report['captured_native_vs_reconstructed_original'] = metric(
        np.load(inputs/'native-captured-keys.npy'), stages['post_rope'])
    with torch.inference_mode():
        ids = torch.from_numpy(tokens.astype('i8')).unsqueeze(0).expand(2, -1, -1)
        batched = produce(ids, torch.arange(979).unsqueeze(0).expand(2, -1))
        report['batch_vs_step'] = {}
        for name, value in batched.items():
            values = value.numpy().copy()
            np.save(out/('batched-'+name+'.npy'), values)
            report['batch_vs_step'][name] = metric(values, stages[name])
        report['repeats_exact'] = True
        for position in [0, 1, 8, 64, 512, 978]:
            ids = torch.from_numpy(tokens[position].astype('i8')).reshape(1, 1, 9).expand(2, -1, -1)
            repeated = produce(ids, torch.full((2, 1), position, dtype=torch.int64))
            for name, value in repeated.items():
                assert np.array_equal(value.numpy(), stages[name][:, position:position+1])
        shifted = produce(torch.from_numpy(tokens.astype('i8')).unsqueeze(0).expand(2, -1, -1),
                          (torch.arange(979)+1).unsqueeze(0).expand(2, -1))['post_rope'].numpy().copy()
    controls = dict(scale=metric(2*stages['post_rope'], captured),
                    reverse_heads=metric(stages['post_rope'][:, :, ::-1].copy(), captured),
                    shifted_position=metric(shifted, captured))
    for name, row in controls.items():
        assert not reconstruction_gate(row) and row['relative_l2'] > .01, name
    np.save(out/'negative-shifted-position.npy', shifted)
    report['rejecting_controls'] = controls
    report['files'] = [dict(path=file.name, bytes=file.stat().st_size, sha256=digest(file))
                       for file in sorted(out.iterdir()) if file.is_file() and file.name != 'receipt.json']
    (out/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
