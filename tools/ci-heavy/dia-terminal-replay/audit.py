#!/usr/bin/env python3
"""Independent official-artifact audit for the five hosted terminal arms.

Uses the separately calibrated suffix auditor's raw-byte and numerical helpers,
never the worker's comparisons. No full model is executed locally.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('calibrated_suffix_audit', HERE.parent/'dia-suffix-replay/audit.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
sha, safe = helpers.sha, helpers.safe
reconstruct, measured, sampler = helpers.reconstruct, helpers.measured, helpers.sampler
baseline_bytes = helpers.baseline_bytes
RUNTIME, GGML, MODEL, ORIGINAL = helpers.RUNTIME, helpers.GGML, helpers.MODEL, helpers.ORIGINAL
PRIOR_SHA = '162e4e25317e60b5b9ee703feb51aa7112c8e73e0ba89a49c55da493c56aeb61'
NAMES = ['baseline', 'native-layer-17', 'native-final-hidden', 'original-layer-17', 'original-final-hidden']


def margin_steps(native, post17, hidden, original):
    assert all(math.isfinite(x) for x in [native, post17, hidden, original])
    row = dict(native=native, original_post17=post17, original_hidden=hidden, original=original,
               conditional_post17_change=post17-native,
               conditional_final_norm_change=hidden-post17,
               remaining_output_head_difference=original-hidden)
    row['telescope_residual'] = (row['conditional_post17_change']+row['conditional_final_norm_change']+
                                row['remaining_output_head_difference'])-(original-native)
    assert row['telescope_residual'] == 0
    return row


def cfg_terms(logits):
    assert logits.shape == (2, 9, 1028)
    row = dict(c46=float(logits[1, 2, 46]), c904=float(logits[1, 2, 904]),
               u46=float(logits[0, 2, 46]), u904=float(logits[0, 2, 904]))
    assert all(math.isfinite(value) for value in row.values())
    row['float64_cfg_margin'] = 4*(row['c46']-row['c904'])-3*(row['u46']-row['u904'])
    return row


def verify_activation(computed, used, actual, held, injection, row):
    assert all(len(values) == 4096*4 for values in [computed, used, actual, held, injection])
    assert computed == held and used == actual == injection
    assert hashlib.sha256(injection).hexdigest() == row['activation_sha256'] == row['consumed_sha256']
    assert hashlib.sha256(held).hexdigest() == row['computed_sha256']


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
    assert 'DIA_SUFFIX_GUARD_PASS' in (out/'actual-ggml-terminal-guard.log').read_text()
    assert 'DIA_TERMINAL_SELFTEST_PASS' in (out/'method-selftest.log').read_text()
    prior_r = json.loads((prior/'receipt.json').read_text())
    assert (out/'prior-receipt.json').read_bytes() == (prior/'receipt.json').read_bytes()
    assert r['prior']['sha256'] == PRIOR_SHA and r['prior']['worker_source'] == prior_r['worker_source']
    prior_payloads = {row['path']: row for row in prior_r['payloads']}
    inputs = out/'inputs'
    for f in inputs.iterdir():
        if f.name in ['terminal-original-layer17.f32', 'terminal-original-hidden.f32']:
            continue
        row = prior_payloads['inputs/'+f.name]
        assert f.stat().st_size == row['bytes'] and sha(f) == row['sha256'], f.name
    method = out/'method'; verified_methods = []
    for f in method.rglob('*'):
        if not f.is_file():
            continue
        name = f.relative_to(method).as_posix()
        if name in ['dia_terminal_replay.py', 'dia_suffix_replay.py', 'dia_query_key_intervention.py'] or name.startswith(('dia-terminal-replay/', 'dia-suffix-replay/', 'dia-cache-intervention/', 'dia-query-key-intervention/')):
            expected = subprocess.check_output(['git', 'show', source+':tools/ci-heavy/'+name], cwd=args.repo)
            assert f.read_bytes() == expected, name
            verified_methods.append(name)
    assert sha(method/'dia_terminal_replay.py') == r['worker_sha256']
    assert sha(method/'actual-dia-terminal-replay') == r['runner_sha256']
    assert (method/'actual-dia-terminal-replay').read_bytes().startswith(b'\x7fELF')
    for row in r['actual_linked_libraries']:
        name = Path(row['path']).name
        if name.startswith('libggml'):
            assert sha(out/'libs'/name) == row['sha256']
    guard = method/'actual-dia-terminal-guard'
    assert guard.read_bytes().startswith(b'\x7fELF')
    guard.chmod(guard.stat().st_mode | stat.S_IXUSR)
    guard_env = dict(os.environ, LD_LIBRARY_PATH=str((out/'libs').resolve()))
    fresh_guard = subprocess.run([str(guard.resolve())], env=guard_env, capture_output=True, text=True, timeout=60)
    (bundle/'fresh-ggml-terminal-guard.log').write_text(fresh_guard.stdout+fresh_guard.stderr)
    assert fresh_guard.returncode == 0 and 'DIA_SUFFIX_GUARD_PASS' in fresh_guard.stdout
    pristine = subprocess.check_output(['git', 'show', RUNTIME+':src/dia_tts.cpp'], cwd=args.repo)
    assert hashlib.sha256(pristine).hexdigest() == r['unpatched_runtime_sha256']
    # Execute only the patch whose bytes were independently checked against the
    # requested immutable repository object, never an unverified artifact module.
    spec = importlib.util.spec_from_file_location('verified_terminal_patch', method/'dia-terminal-replay/patch.py')
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)
    with tempfile.TemporaryDirectory(dir=bundle, prefix='audit-patch-') as tmp:
        cpp = Path(tmp)/'dia_tts.cpp'; cpp.write_bytes(pristine)
        patcher.patch(cpp)
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
    assert [row['arm'] for row in r['arms']] == NAMES
    # The separately accepted anchor's logical full captures are reconstructed
    # independently, including its baseline references.
    anchor_rows = json.loads((prior/'original-layer-17/captures.json').read_text())
    assert len(anchor_rows) == 127 and {v['path'] for v in anchor_rows} == stages
    anchor = {v['path']: reconstruct(v, prior/'original-layer-17', prior/'baseline') for v in anchor_rows}
    results = []
    for row in r['arms']:
        name = row['arm']; arm = out/name
        post = name.endswith('layer-17')
        terminal = name.endswith('final-hidden')
        assert row['layer'] == (17 if post else None)
        assert row['clamp'] == ('post-layer' if post else ('final-hidden' if terminal else None))
        assert (arm/'feedback.i32').read_bytes() == tokens[-1].tobytes()
        assert (arm/'capture-step.txt').read_text().strip() == '978'
        log = (out/(name+'.log')).read_text()
        assert log.count('DIA_SUFFIX_PREFIX_READY') == 1 and 'DIA_SUFFIX_INVALID' not in log
        assert log.count('DIA_SUFFIX_ACTIVATION_APPLIED') == post
        assert log.count('DIA_TERMINAL_HIDDEN_APPLIED') == terminal and 'DIA_TERMINAL_INVALID' not in log
        assert row['cross_cache_sha256'] == cross_hash
        if name == 'baseline':
            actual = {key: baseline_bytes(baseline/key) for key in stages}
        else:
            captures = json.loads((arm/'captures.json').read_text())
            assert captures == row['captures'] and len(captures) == 127
            assert {v['path'] for v in captures} == stages
            actual = {v['path']: reconstruct(v, arm, baseline) for v in captures}
            target = 'cpp_decoder_layer_17.f32' if post else 'cpp_final_hidden.f32'
            held = baseline_bytes(baseline/target)
            array = 'decoder_layer_17.npy' if post else 'decoder_hidden.npy'
            injection = held if name.startswith('native-') else np.load(inputs/array, allow_pickle=False).tobytes()
            prefix = 'suffix' if post else 'terminal'
            verify_activation((arm/(prefix+'-computed.f32')).read_bytes(),
                              (arm/(prefix+'-used.f32')).read_bytes(), actual[target], held, injection, row)
            if name.startswith('original-'):
                filename = 'terminal-original-layer17.f32' if post else 'terminal-original-hidden.f32'
                assert (inputs/filename).read_bytes() == injection
            changed = {'cpp_final_hidden.f32', 'cpp_step978_uncond.f32', 'cpp_step978_cond.f32'}
            if post:
                changed.add('cpp_decoder_layer_17.f32')
            for key in stages-changed:
                assert actual[key] == baseline_bytes(baseline/key), (name, key)
        logits = np.fromfile(arm/'logits.f32', dtype='<f4').reshape(2, 9, 1028)
        assert np.isfinite(logits).all() and sha(arm/'logits.f32') == row['logits_sha256']
        assert actual['cpp_step978_uncond.f32'] == logits[0].tobytes()
        assert actual['cpp_step978_cond.f32'] == logits[1].tobytes()
        if name == 'baseline' or name.startswith('native-'):
            assert logits.tobytes() == native_logits[-1].tobytes()
            assert all(actual[key] == baseline_bytes(baseline/key) for key in stages)
        if name == 'original-layer-17':
            assert all(actual[key] == anchor[key] for key in stages)
            assert (arm/'logits.f32').read_bytes() == (prior/'original-layer-17/logits.f32').read_bytes()
        sampling = evaluate(logits); assert sampling == row['sampling']
        assert [v['layer'] for v in row['layers']] == list(range(18))
        layers = [dict(layer=l, **measured(np.frombuffer(actual[f'cpp_decoder_layer_{l}.f32'], dtype='<f4').reshape(2, 2048),
                    np.load(inputs/f'decoder_layer_{l}.npy', allow_pickle=False), row['layers'][l])) for l in range(18)]
        hidden = measured(np.frombuffer(actual['cpp_final_hidden.f32'], dtype='<f4').reshape(2, 2048),
                          np.load(inputs/'decoder_hidden.npy', allow_pickle=False), row['final_hidden'])
        lm = measured(logits, original_logits, row['logits_vs_original'])
        if name == 'original-final-hidden':
            assert hidden['byte_exact'] and hidden['relative_l2'] == 0
        operands = cfg_terms(logits)
        results.append(dict(arm=name, sampling=sampling, cfg_operands=operands,
                            layers=layers, final_hidden=hidden, logits=lm))
    margins = {row['arm']: row['sampling']['margin46_minus904'] for row in results}
    sequence = margin_steps(margins['baseline'], margins['original-layer-17'], margins['original-final-hidden'],
                            r['original_sampling']['margin46_minus904'])
    f64 = {row['arm']: row['cfg_operands']['float64_cfg_margin'] for row in results}
    original_operands = cfg_terms(original_logits)
    sequence_f64 = margin_steps(f64['baseline'], f64['original-layer-17'], f64['original-final-hidden'],
                                original_operands['float64_cfg_margin'])
    result = dict(independent_audit_pass=True, source=source, official_artifact_id=artifact['id'],
                  official_artifact_sha256=sha(archive), payloads_verified=len(payloads),
                  method_files_verified=verified_methods, arms=results, conditional_margin_steps=sequence,
                  float64_cfg_margin_steps=sequence_f64, original_cfg_operands=original_operands,
                  all2_native_controls_byte_exact=True, original_layer17_anchor_exact=True,
                  all_capture_reconstructions_verified=True, original_filter_replayed=True,
                  fresh_actual_ggml_guard_pass=True, speech_acceptance=False, gpu_acceptance=False,
                  interpretation='Sequential fixed-input terminal interventions; no additive global causal percentages or speech acceptance.')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('INDEPENDENT_DIA_TERMINAL_AUDIT_COMPLETE', len(payloads), 'payloads,5 arms,90 layer comparisons')


if __name__ == '__main__':
    main()
