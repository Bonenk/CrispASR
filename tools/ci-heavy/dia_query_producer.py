#!/usr/bin/env python3
"""One-step native Q projection/RoPE attribution with exact captured control.

Uses only the 16 MiB physically audited query matrix and small owned arrays.
No full model, speech, GPU, optimization or default acceptance is performed.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = '8d0ad8862d2cad5d305a48a57f2a99b5024bc39a'
GGML = 'c36dab89b662838f0f5d4826c399198c0b90bbfc'
WEIGHT_SHA = '17d07c13004926d7c7f99c1df2ee57e33b584ae01105dfef77ec08462e7e7575'
HERE = Path(__file__).with_name('dia-query-key-intervention')
spec = importlib.util.spec_from_file_location('original_query_metrics', Path(__file__).with_name('dia_self_query.py'))
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['query-proof', 'native', 'headers', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    report = dict(scope=__doc__, runtime_source=SOURCE, ggml_source=GGML,
                  producer_sha256=sha(__file__), runs=[], native_capture_reproduced=False,
                  speech_acceptance=False, gpu_acceptance=False, defaults_changed=False)

    def checkpoint():
        (out/'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    def retain(file, name, row=None):
        if row:
            assert file.stat().st_size == row['bytes'] and sha(file) == row['sha256'], file
        target = out/'inputs'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
        return target

    checkpoint()
    query = json.loads((args.query_proof/'receipt.json').read_text())
    assert query['prior_module_replay_byte_exact'] and query['full_reference_reconstruction_gate_pass']
    assert not query['full_original_self_query_previously_captured']
    assert query['producer_sha256'] == sha(Path(__file__).with_name('dia_self_query.py'))
    paths = {r['path']: r for r in query['files']}
    for name, row in paths.items():
        file = args.query_proof/name
        assert file.stat().st_size == row['bytes'] and sha(file) == row['sha256'], name
    retain(args.query_proof/'receipt.json', 'original-query-receipt.json')
    physical = json.loads((args.query_proof/'inputs/physical-coefficients.json').read_text())
    retain(args.query_proof/'inputs/physical-coefficients.json', 'physical-coefficients.json')
    qrow = next(r for r in physical['tensors'] if r['original_name'] == 'decoder.layers.0.self_attention.q_proj.weight')
    mapped = next(r for r in physical['mapped'] if r['name'] == 'dia.decoder.layers.0.self_q_proj')
    assert mapped['sha256'] == WEIGHT_SHA and mapped['shape'] == [2048, 2048]
    original_weight = args.query_proof/'inputs'/qrow['path']
    assert sha(original_weight) == qrow['sha256']
    retain(original_weight, 'original-query-weight.f32', qrow)
    weight = np.fromfile(original_weight, dtype='<f4').reshape(2048, 2048).T.copy()
    assert np.isfinite(weight).all()
    weight_path = out/'inputs/native-query-weight.f32'
    weight.tofile(weight_path)
    assert sha(weight_path) == WEIGHT_SHA

    native = json.loads((args.native/'receipt.json').read_text())
    assert native['conditional_diagnostic_complete'] and native['ggml_source'] == GGML
    assert native['runtime_source'] == SOURCE
    files = {r['path']: r for r in native['payloads']}
    retain(args.native/'receipt.json', 'native-capture-receipt.json')
    norm_name, query_name = 'baseline/cpp_decoder_block0_pre_sa_norm.f32', 'baseline/cpp_decoder_sa0_q.f32'
    norm_path = retain(args.native/norm_name, 'normalized.f32', files[norm_name])
    expected_path = retain(args.native/query_name, 'captured-native-query.f32', files[query_name])
    normalized = np.fromfile(norm_path, dtype='<f4').reshape(2, 1, 2048)
    expected = np.fromfile(expected_path, dtype='<f4').reshape(2, 1, 16, 128)
    assert np.isfinite(normalized).all() and np.isfinite(expected).all()
    assert normalized.tobytes() == np.load(args.query_proof/'inputs/pre_sa_norm.npy').tobytes()
    original_pre = np.load(args.query_proof/'original-capacity-pre_rope_query.npy', allow_pickle=False)
    original_post = np.load(args.query_proof/'original-capacity-query.npy', allow_pickle=False)
    assert original_pre.shape == original_post.shape == expected.shape
    assert original_pre.dtype == original_post.dtype == np.dtype('<f4')
    original_pre.tofile(out/'inputs/original-pre-query.f32')
    retain(args.query_proof/'original-capacity-query.npy', 'original-post-query.npy')

    # Verify the literal helper against the immutable actual runtime, not memory.
    runtime = subprocess.check_output(['git', 'show', SOURCE+':src/dia_tts.cpp'], cwd=ROOT, text=True)
    start = runtime.index('static ggml_tensor* dia_rotary(')
    helper = runtime[start:runtime.index('\nstatic ggml_tensor* build_dia_encoder', start)]
    source = HERE/'query_producer.cpp'
    text = source.read_text(); begin = text.index('static ggml_tensor* dia_rotary(')
    copied = text[begin:text.index('\nstatic void require', begin)]
    assert re.sub(r'\s+', '', helper) == re.sub(r'\s+', '', copied)
    shutil.copyfile(source, out/'query_producer.cpp')
    (out/'literal-runtime-helper.cpp').write_text(helper)
    headers = out/'headers'; headers.mkdir()
    for header in args.headers.glob('*.h'):
        actual = subprocess.check_output(['git', '-C', ROOT/'ggml', 'show', GGML+':include/'+header.name])
        assert header.read_bytes() == actual, header.name
        shutil.copyfile(header, headers/header.name)
    libs = out/'libs'; libs.mkdir()
    expected_libs = {Path(r['path']).name: r for r in native['actual_linked_libraries'] if Path(r['path']).name.startswith('libggml')}
    for name, row in expected_libs.items():
        path = args.native/'libs'/name
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        shutil.copyfile(path, libs/name)
    command = ['c++', '-std=c++17', '-O2', '-I', headers, out/'query_producer.cpp',
               '-Wl,-rpath,$ORIGIN/libs', libs/'libggml-cpu.so.0', libs/'libggml-base.so.0', '-o', out/'producer']
    (out/'build-command.json').write_text(json.dumps(list(map(str, command)), indent=2)+'\n')
    with (out/'build.log').open('w') as log:
        result = subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, (out/'build.log').read_text()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('LD_', 'GGML_', 'CRISPASR_', 'DIA_'))}
    closure = subprocess.check_output(['ldd', out/'producer'], env=env, text=True)
    (out/'ldd.log').write_text(closure)
    assert 'not found' not in closure
    for line in closure.splitlines():
        if 'libggml' in line:
            path = Path(line.split('=>')[1].strip().split()[0])
            assert path.resolve().parent == libs.resolve(), line
    report.update(actual_binary_sha256=sha(out/'producer'), native_weight_sha256=WEIGHT_SHA,
                  actual_library_hashes={name: sha(libs/name) for name in expected_libs},
                  normalized_input_byte_exact_to_original=True)
    checkpoint()

    def execute(name, mode, threads, owned):
        command = [out/'producer', weight_path, norm_path, owned, out/name, str(mode), str(threads)]
        run = subprocess.run(list(map(str, command)), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (out/(name+'.log')).write_text(run.stdout)
        report['runs'].append(dict(name=name, exit=run.returncode, command=list(map(str, command))))
        checkpoint()
        assert run.returncode == 0 and 'DIA_QUERY_PRODUCER_PASS' in run.stdout, run.stdout
        return {part: np.fromfile(out/name/(part+'.f32'), dtype='<f4').reshape(2, 1, 16, 128)
                for part in ['pre_rope', 'used_pre_rope', 'post_rope']}

    streams = {}
    for mode in [0, 1]:
        for threads in [1, 4]:
            name = f'mode-{mode}-threads-{threads}'
            first = execute(name, mode, threads, out/'inputs/original-pre-query.f32')
            repeat = execute(name+'-repeat', mode, threads, out/'inputs/original-pre-query.f32')
            assert all(first[part].tobytes() == repeat[part].tobytes() for part in first)
            streams[mode, threads] = first
        assert all(streams[mode, 1][part].tobytes() == streams[mode, 4][part].tobytes() for part in first)
    baseline = streams[0, 4]
    report['native_capture_reproduced'] = baseline['post_rope'].tobytes() == expected.tobytes()
    checkpoint()
    assert report['native_capture_reproduced'], 'Stop: native one-step graph must reproduce the full captured query exactly'
    assert baseline['used_pre_rope'].tobytes() == baseline['pre_rope'].tobytes()
    assert streams[1, 4]['pre_rope'].tobytes() == baseline['pre_rope'].tobytes()
    assert streams[1, 4]['used_pre_rope'].tobytes() == original_pre.tobytes()
    baseline['pre_rope'].tofile(out/'inputs/native-pre-query.f32')
    control = execute('native-owned-control', 1, 4, out/'inputs/native-pre-query.f32')
    assert control['post_rope'].tobytes() == expected.tobytes()
    negatives = {}
    for name, value in [('scale', baseline['pre_rope']*2), ('head-order', baseline['pre_rope'][:, :, ::-1].copy())]:
        path = out/'inputs'/(name+'-pre-query.f32'); value.tofile(path)
        bad = execute('negative-'+name, 1, 4, path)
        row = original.metrics(bad['post_rope'], expected)
        assert not original.reconstruction_gate(row)
        negatives[name] = row
    report.update(repeats_and_threads_exact=True, native_owned_control_exact=True,
                  projection_vs_original=original.metrics(baseline['pre_rope'], original_pre),
                  full_native_query_vs_original=original.metrics(baseline['post_rope'], original_post),
                  original_pre_query_through_native_rope=original.metrics(streams[1, 4]['post_rope'], original_post),
                  negative_controls=negatives,
                  interpretation_limit='One-step conditional arithmetic attribution only; original Q is module replay, not an earlier full-model capture. No additive causal percentages or speech conclusions.')
    report['payloads'] = [dict(path=f.relative_to(out).as_posix(), bytes=f.stat().st_size, sha256=sha(f))
                          for f in sorted(out.rglob('*')) if f.is_file() and f.name != 'receipt.json']
    checkpoint()
    print(json.dumps({k: v for k, v in report.items() if k not in ['payloads', 'runs']}, indent=2), flush=True)
    print('DIA_NATIVE_QUERY_PRODUCER_COMPLETE; exact native control, no speech acceptance', flush=True)


if __name__ == '__main__':
    main()
