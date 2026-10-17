#!/usr/bin/env python3
"""Independent offline audit of complete hosted Dia suffix artifacts.

Requires the official artifact/run JSON, raw.zip, the separately audited prior
out directory and original blueprint checkout. Never runs a full model.
"""
import argparse
import ast
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
import zipfile

import numpy as np

RUNTIME = '8d0ad8862d2cad5d305a48a57f2a99b5024bc39a'
GGML = 'c36dab89b662838f0f5d4826c399198c0b90bbfc'
MODEL = '8128132c538e1aef83f2fb5257b4f71d28a67c2a5a0d54d0a1f0a3b0f5d6b73d'
ORIGINAL = '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'
PRIOR_SHA = '4e444752df59e76f965822e826282e07dbfac893215bc12e223facae4ba520de'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe(name):
    p = PurePosixPath(name)
    assert name and not p.is_absolute() and '..' not in p.parts and '\\' not in name
    assert p.as_posix() == name and name != '.'
    return name


@lru_cache(maxsize=160)
def baseline_bytes(path):
    return Path(path).read_bytes()


def reconstruct(row, arm, baseline):
    name = row['path']
    assert safe(name) == Path(name).name and re.fullmatch(r'cpp_[a-zA-Z0-9_]+\.f32', name)
    kind = row['kind']
    if kind == 'full':
        data = (arm/name).read_bytes()
    else:
        assert row['baseline'] == 'baseline/'+name
        data = baseline_bytes(baseline/name)
        assert not (arm/name).exists()
        if kind == 'current-row':
            assert re.fullmatch(r'cpp_decoder_sa(?:[0-9]|1[0-7])_[kv]\.f32', name)
            assert row['shape'] == [2, 979, 512] and row['position'] == 978
            tail = name[:-4]+'.current.f32'
            assert row['current'] == tail and sha(arm/tail) == row['current_sha256']
            current = np.fromfile(arm/tail, dtype='<f4')
            assert current.size == 1024 and np.isfinite(current).all()
            # Reconstruct using explicit per-batch byte slices, independent of
            # the worker's numpy assignment and its host-prefix transposition.
            batch_bytes, past_bytes = 979*512*4, 978*512*4
            parts = []
            tail_bytes = current.tobytes()
            for batch in range(2):
                begin = batch*batch_bytes
                parts += [data[begin:begin+past_bytes], tail_bytes[batch*512*4:(batch+1)*512*4]]
            data = b''.join(parts)
        else:
            assert kind == 'baseline'
    assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], name
    return data


def measured(x, y, reported):
    assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
    a = x.astype('f8').reshape(-1, x.shape[-1]); b = y.astype('f8').reshape(a.shape)
    an = np.sqrt(np.sum(a*a, axis=1)); bn = np.sqrt(np.sum(b*b, axis=1))
    assert (an > 0).all() and (bn > 0).all()
    delta = a-b
    values = dict(cosine_min=float(np.min(np.sum(a*b, axis=1)/(an*bn))),
                  mine_norm=float(np.sqrt(np.sum(a*a))), ref_norm=float(np.sqrt(np.sum(b*b))),
                  max_row_norm_error=float(np.max(np.abs(an/bn-1))),
                  relative_l2=float(np.sqrt(np.sum(delta*delta)/np.sum(b*b))),
                  max_abs=float(np.max(np.abs(delta))), byte_exact=x.tobytes() == y.tobytes())
    for key, value in values.items():
        if key == 'byte_exact':
            assert reported[key] is value
        else:
            assert np.isclose(value, reported[key], rtol=1e-13, atol=1e-14), (key, value, reported[key])
    return values


def sampler(original_text):
    import torch
    assert torch.__version__ == '2.6.0+cpu'
    torch.set_num_threads(4)
    node = next(n for n in ast.parse(original_text).body if isinstance(n, ast.FunctionDef) and n.name == '_sample_next_token')
    observed = []

    class Observe:
        def __getattr__(self, name):
            return getattr(torch, name)
        def multinomial(self, values, num_samples):
            assert num_samples == 1
            observed.append(values.clone())
            return torch.argmax(values, dim=-1, keepdim=True)

    namespace = {'torch': Observe()}
    code = ast.get_source_segment(original_text, node)
    exec(compile(code, 'unchanged-original-filter', 'exec'), namespace)

    def evaluate(logits):
        x = torch.from_numpy(logits.copy())
        guided = x[1]+3.0*(x[1]-x[0]); raw = guided[2].clone()
        guided[:, 1025:] = -torch.inf; guided[1:, 1024:] = -torch.inf
        namespace['_sample_next_token'](guided.float(), 1.2, .95, 45)
        probabilities = observed.pop()[2]
        return dict(cfg46=float(raw[46]), cfg904=float(raw[904]),
                    margin46_minus904=float(raw[46]-raw[904]),
                    rank46=int((guided[2] > guided[2, 46]).sum())+1,
                    rank904=int((guided[2] > guided[2, 904]).sum())+1,
                    support=torch.nonzero(probabilities > 0).flatten().tolist())
    return evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--prior-out', type=Path, required=True)
    parser.add_argument('--original-repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    bundle = args.bundle; out = bundle/'out'; prior = args.prior_out
    run = json.loads((bundle/'run.json').read_text())
    assert run['status'] == 'completed' and run['conclusion'] == 'success'
    assert run['jobs'] and all(job['conclusion'] == 'success' for job in run['jobs'])
    source = run['headSha']
    artifacts = json.loads((bundle/'artifacts.json').read_text())['artifacts']
    assert len(artifacts) == 1
    artifact = artifacts[0]; archive = bundle/'raw.zip'
    assert artifact['workflow_run']['head_sha'] == source and not artifact['expired']
    assert artifact['size_in_bytes'] == archive.stat().st_size
    assert artifact['digest'] == 'sha256:'+sha(archive)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names = z.namelist(); assert len(names) == len(set(names))
        for info in z.infolist():
            safe(info.filename)
            assert not stat.S_ISLNK(info.external_attr >> 16) and not info.is_dir()
        if not out.exists():
            z.extractall(out)
        assert set(names) == {f.relative_to(out).as_posix() for f in out.rglob('*') if f.is_file()}
        # Verify the extracted root receipt against the actual archive too.
        assert z.read('receipt.json') == (out/'receipt.json').read_bytes()
    r = json.loads((out/'receipt.json').read_text())
    assert r['worker_source'] == source and r['runtime_source'] == RUNTIME and r['ggml_source'] == GGML
    assert r['model_sha256'] == MODEL and r['step'] == 978 and r['conditional_diagnostic_complete']
    for key in ['speech_acceptance', 'gpu_acceptance', 'production_defaults_changed', 'rng_state_reproduced', 'original_full_query_capture']:
        assert r[key] is False
    payloads = {row['path']: row for row in r['payloads']}
    assert len(payloads) == len(r['payloads'])
    assert set(payloads) == set(names)-{'receipt.json'}
    for name, row in payloads.items():
        f = out/safe(name)
        assert f.stat().st_size == row['bytes'] and sha(f) == row['sha256'], name
    for row in r['runs']:
        assert row['exit'] == 0 and sha(out/(row['label']+'.log')) == row['log_sha256']
    assert 'DIA_SUFFIX_GUARD_PASS' in (out/'actual-ggml-suffix-guard.log').read_text()
    assert 'DIA_SUFFIX_SELFTEST_PASS' in (out/'method-selftest.log').read_text()
    prior_r = json.loads((prior/'receipt.json').read_text())
    assert (out/'prior-receipt.json').read_bytes() == (prior/'receipt.json').read_bytes()
    assert r['prior']['sha256'] == PRIOR_SHA and r['prior']['worker_source'] == prior_r['worker_source']
    prior_payloads = {row['path']: row for row in prior_r['payloads']}
    inputs = out/'inputs'
    for f in inputs.iterdir():
        if re.fullmatch(r'original-layer-\d{2}\.f32', f.name):
            continue
        row = prior_payloads['inputs/'+f.name]
        assert f.stat().st_size == row['bytes'] and sha(f) == row['sha256'], f.name
    method = out/'method'; verified_methods = []
    for f in method.rglob('*'):
        if not f.is_file():
            continue
        name = f.relative_to(method).as_posix()
        if name in ['dia_suffix_replay.py', 'dia_query_key_intervention.py'] or name.startswith(('dia-suffix-replay/', 'dia-cache-intervention/', 'dia-query-key-intervention/')):
            expected = subprocess.check_output(['git', 'show', source+':tools/ci-heavy/'+name], cwd=args.repo)
            assert f.read_bytes() == expected, name
            verified_methods.append(name)
    assert sha(method/'dia_suffix_replay.py') == r['worker_sha256']
    assert sha(method/'actual-dia-suffix-replay') == r['runner_sha256']
    assert (method/'actual-dia-suffix-replay').read_bytes().startswith(b'\x7fELF')
    for row in r['actual_linked_libraries']:
        name = Path(row['path']).name
        if name.startswith('libggml'):
            assert sha(out/'libs'/name) == row['sha256']
    guard = method/'actual-dia-suffix-guard'
    assert guard.read_bytes().startswith(b'\x7fELF')
    guard.chmod(guard.stat().st_mode | stat.S_IXUSR)
    guard_env = dict(os.environ, LD_LIBRARY_PATH=str((out/'libs').resolve()))
    fresh_guard = subprocess.run([str(guard.resolve())], env=guard_env, capture_output=True, text=True, timeout=60)
    (bundle/'fresh-ggml-suffix-guard.log').write_text(fresh_guard.stdout+fresh_guard.stderr)
    assert fresh_guard.returncode == 0 and 'DIA_SUFFIX_GUARD_PASS' in fresh_guard.stdout
    pristine = subprocess.check_output(['git', 'show', RUNTIME+':src/dia_tts.cpp'], cwd=args.repo)
    assert hashlib.sha256(pristine).hexdigest() == r['unpatched_runtime_sha256']
    # Execute only the patch whose bytes were independently checked against the
    # requested immutable repository object, never an unverified artifact module.
    patch_namespace = {'__name__': 'trusted_immutable_patch'}
    exec(compile((method/'dia-suffix-replay/patch.py').read_text(), 'verified-suffix-patch', 'exec'), patch_namespace)
    with tempfile.TemporaryDirectory(dir=bundle, prefix='audit-patch-') as tmp:
        cpp = Path(tmp)/'dia_tts.cpp'; cpp.write_bytes(pristine)
        patch_namespace['patch'](cpp)
        assert cpp.read_bytes() == (method/'patched-dia_tts.cpp').read_bytes()
        assert sha(cpp) == r['patched_runtime_sha256']
    original_bytes = subprocess.check_output(['git', 'show', ORIGINAL+':dia/model.py'], cwd=args.original_repo)
    assert (method/'original-model.py').read_bytes() == original_bytes
    evaluate = sampler(original_bytes.decode())
    native_logits = np.load(inputs/'native-logits.npy', allow_pickle=False)
    assert native_logits.shape == (979, 2, 9, 1028)
    original_logits = np.load(inputs/'decoder_step_logits.npy', allow_pickle=False)[-1]
    assert evaluate(original_logits) == r['original_sampling']
    assert evaluate(native_logits[-1]) == r['retained_native_sampling']
    tokens = np.fromfile(inputs/'tokens.i32', dtype='<i4').reshape(979, 9)
    assert tokens.tobytes() == np.load(inputs/'decoder_input_tokens.npy', allow_pickle=False).astype('<i4').tobytes()
    baseline = out/'baseline'
    stages = {f.name for f in baseline.glob('cpp_*.f32')}; assert len(stages) == 127
    for name in stages:
        f = baseline/name
        assert f.read_bytes() == (prior/'baseline'/name).read_bytes(), name
    cross = baseline/'suffix-cross-cache.f32'
    assert cross.stat().st_size == 18*2*2*39*2048*4
    cross_hash = sha(cross)
    names_expected = ['baseline']+[f'native-layer-{l:02}' for l in range(18)]+[f'original-layer-{l:02}' for l in range(18)]
    assert [row['arm'] for row in r['arms']] == names_expected
    results = []
    for row in r['arms']:
        name = row['arm']; layer = row['layer']; arm = out/name
        assert (arm/'feedback.i32').read_bytes() == tokens[-1].tobytes()
        assert (arm/'capture-step.txt').read_text().strip() == '978'
        log = (out/(name+'.log')).read_text()
        assert log.count('DIA_SUFFIX_PREFIX_READY') == 1 and 'DIA_SUFFIX_INVALID' not in log
        assert log.count('DIA_SUFFIX_ACTIVATION_APPLIED') == (name != 'baseline')
        assert row['cross_cache_sha256'] == cross_hash
        if name == 'baseline':
            assert layer is None
            actual = {key: baseline_bytes(baseline/key) for key in stages}
        else:
            assert layer == int(name.rsplit('-', 1)[-1])
            captures = json.loads((arm/'captures.json').read_text())
            assert captures == row['captures'] and len(captures) == 127
            assert {v['path'] for v in captures} == stages
            actual = {v['path']: reconstruct(v, arm, baseline) for v in captures}
            held = (baseline/f'cpp_decoder_layer_{layer}.f32').read_bytes()
            injection = held if name.startswith('native-') else np.load(inputs/f'decoder_layer_{layer}.npy', allow_pickle=False).tobytes()
            assert (arm/'suffix-computed.f32').read_bytes() == held
            assert (arm/'suffix-used.f32').read_bytes() == actual[f'cpp_decoder_layer_{layer}.f32'] == injection
            assert hashlib.sha256(injection).hexdigest() == row['activation_sha256'] == row['consumed_sha256']
            assert hashlib.sha256(held).hexdigest() == row['computed_sha256']
            if name.startswith('original-'):
                assert (inputs/f'original-layer-{layer:02}.f32').read_bytes() == injection
            for key in stages:
                match = re.fullmatch(r'cpp_decoder_sa(\d+)_(q|k|v|attention)\.f32', key)
                earlier = re.fullmatch(r'cpp_decoder_layer_(\d+)\.f32', key)
                if key.startswith(('cpp_encoder_', 'cpp_cross_', 'cpp_ca0_')) or (key.startswith('cpp_decoder_block0_') and (layer > 0 or key != 'cpp_decoder_block0_output.f32')) or (match and int(match[1]) <= layer) or (earlier and int(earlier[1]) < layer):
                    assert actual[key] == baseline_bytes(baseline/key), (name, key)
        logits = np.fromfile(arm/'logits.f32', dtype='<f4').reshape(2, 9, 1028)
        assert np.isfinite(logits).all() and sha(arm/'logits.f32') == row['logits_sha256']
        assert actual['cpp_step978_uncond.f32'] == logits[0].tobytes()
        assert actual['cpp_step978_cond.f32'] == logits[1].tobytes()
        if name == 'baseline' or name.startswith('native-'):
            assert logits.tobytes() == native_logits[-1].tobytes()
            assert all(actual[key] == baseline_bytes(baseline/key) for key in stages)
        for l in range(18):
            for part in ['k', 'v']:
                key = f'cpp_decoder_sa{l}_{part}.f32'
                x = np.frombuffer(actual[key], dtype='<f4').reshape(2, 979, 512)
                y = np.frombuffer(baseline_bytes(baseline/key), dtype='<f4').reshape(x.shape)
                assert x[:, :978].tobytes() == y[:, :978].tobytes(), (name, key)
        sampling = evaluate(logits); assert sampling == row['sampling']
        assert [v['layer'] for v in row['layers']] == list(range(18))
        layers = [dict(layer=l, **measured(np.frombuffer(actual[f'cpp_decoder_layer_{l}.f32'], dtype='<f4').reshape(2, 2048),
                    np.load(inputs/f'decoder_layer_{l}.npy', allow_pickle=False), row['layers'][l])) for l in range(18)]
        hidden = measured(np.frombuffer(actual['cpp_final_hidden.f32'], dtype='<f4').reshape(2, 2048),
                          np.load(inputs/'decoder_hidden.npy', allow_pickle=False), row['final_hidden'])
        lm = measured(logits, original_logits, row['logits_vs_original'])
        results.append(dict(arm=name, sampling=sampling, layers=layers, final_hidden=hidden, logits=lm))
    result = dict(independent_audit_pass=True, source=source, official_artifact_id=artifact['id'],
                  official_artifact_sha256=sha(archive), payloads_verified=len(payloads),
                  method_files_verified=verified_methods, arms=results,
                  all18_native_controls_byte_exact=True, all_capture_reconstructions_verified=True,
                  original_filter_replayed=True, fresh_actual_ggml_guard_pass=True, speech_acceptance=False, gpu_acceptance=False)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('INDEPENDENT_DIA_SUFFIX_AUDIT_COMPLETE', len(payloads), 'payloads,37 arms,666 layer comparisons')


if __name__ == '__main__':
    main()
