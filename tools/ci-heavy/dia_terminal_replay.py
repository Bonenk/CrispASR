#!/usr/bin/env python3
"""One-step Dia hosted final-hidden clamp with accepted layer17 anchors.

Pinned scratch runtime only. A complete byte-exact baseline is mandatory before
any replacement; skipped prefix RNG means this is not a generated trajectory.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np
from dia_query_key_intervention import digest, metrics

ROOT = Path(__file__).resolve().parents[2]
METHOD = Path(__file__).with_name('dia-terminal-replay')
BASE_METHOD = Path(__file__).with_name('dia-cache-intervention')
spec = importlib.util.spec_from_file_location('suffix_patch', METHOD/'patch.py')
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)
PRIOR_REV = '049d93d4d97ce7a542a030b4a20366df8b372699'
PRIOR_PATH = 'dia-20261015-suffix-replay/complete-hosted-suffix-replay-v1/raw.zip'
PRIOR_SHA = '162e4e25317e60b5b9ee703feb51aa7112c8e73e0ba89a49c55da493c56aeb61'
MODEL_SHA = '8128132c538e1aef83f2fb5257b4f71d28a67c2a5a0d54d0a1f0a3b0f5d6b73d'
GGML_SOURCE = 'c36dab89b662838f0f5d4826c399198c0b90bbfc'


def unpack_prior(archive, dest):
    """Check complete ZIP structure/CRC and all extracted payload hashes."""
    assert digest(archive) == PRIOR_SHA
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert len(names) == len(set(names))
        for name in names:
            p = Path(name)
            assert not p.is_absolute() and '..' not in p.parts and p.as_posix() == name
            assert ((z.getinfo(name).external_attr >> 16) & 0o170000) != 0o120000
        receipt = json.loads(z.read('receipt.json'))
        rows = {r['path']: r for r in receipt['payloads']}
        assert len(rows) == len(receipt['payloads'])
        assert set(names) == set(rows) | {'receipt.json'}
        assert receipt['runtime_source'] == patcher.SOURCE and receipt['ggml_source'] == GGML_SOURCE
        assert receipt['conditional_diagnostic_complete'] and receipt['step'] == 978
        selected = [name for name in names if name.startswith(('baseline/', 'inputs/', 'original-layer-17/')) or
                    name in ['method/original-model.py', 'receipt.json']]
        for name in selected:
            target = dest/name
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            if name in rows:
                assert target.stat().st_size == rows[name]['bytes'] and digest(target) == rows[name]['sha256'], name
    return receipt


def compact_captures(dest, baseline):
    """Lossless references/current-row deltas, validated by reconstruction.

    All36 cache prefixes must remain exact. Baseline files remain physically
    present. Every removed file has original full-byte SHA and an exact recipe.
    """
    rows = []
    for file in sorted(dest.glob('cpp_*.f32')):
        if file.name.endswith('.current.f32'):
            continue
        original_sha = digest(file)
        row = dict(path=file.name, bytes=file.stat().st_size, sha256=original_sha)
        ref = baseline/file.name
        if original_sha == digest(ref):
            row.update(kind='baseline', baseline='baseline/'+file.name)
            reconstructed = ref.read_bytes()
        elif re.fullmatch(r'cpp_decoder_sa\d+_[kv]\.f32', file.name):
            x = np.fromfile(file, dtype='<f4').reshape(2, 979, 512)
            y = np.fromfile(ref, dtype='<f4').reshape(x.shape)
            assert x[:, :978].tobytes() == y[:, :978].tobytes(), file.name
            tail = file.with_suffix('.current.f32')
            x[:, 978].copy().tofile(tail)
            rebuilt = y.copy()
            rebuilt[:, 978] = np.fromfile(tail, dtype='<f4').reshape(2, 512)
            reconstructed = rebuilt.tobytes()
            row.update(kind='current-row', baseline='baseline/'+file.name,
                       shape=[2, 979, 512], position=978, current=tail.name,
                       current_sha256=digest(tail))
        else:
            row['kind'] = 'full'
            rows.append(row)
            continue
        assert len(reconstructed) == row['bytes'] and hashlib.sha256(reconstructed).hexdigest() == original_sha
        file.unlink()
        rows.append(row)
    (dest/'captures.json').write_text(json.dumps(rows, indent=2)+'\n')
    return rows


def main():
    out, scratch = Path(os.environ['HEAVY_OUT']), Path(os.environ['HEAVY_SCRATCH'])
    out.mkdir(parents=True, exist_ok=True); scratch.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(scratch)
    report = dict(worker_source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  runtime_source=patcher.SOURCE, worker_sha256=digest(__file__), step=978, runs=[], arms=[],
                  scope='Hosted terminal hidden-state replacement with exact native and accepted original-layer17 anchors.',
                  speech_acceptance=False, gpu_acceptance=False, production_defaults_changed=False,
                  rng_state_reproduced=False, original_full_query_capture=False)
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)
    if os.environ.get('GITHUB_SHA'):
        assert report['worker_source'] == os.environ['GITHUB_SHA']

    def checkpoint():
        (out/'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    def run(command, label, env=None):
        with (out/(label+'.log')).open('w') as log:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=10800)
        report['runs'].append(dict(label=label, exit=result.returncode, log_sha256=digest(out/(label+'.log'))))
        checkpoint()
        print(label, result.returncode, (out/(label+'.log')).read_text()[-1200:], flush=True)
        assert result.returncode == 0, label

    checkpoint()
    (out/'method').mkdir()
    for name in ['dia_terminal_replay.py', 'dia_query_key_intervention.py', 'dia-terminal-replay', 'dia_suffix_replay.py', 'dia-suffix-replay',
                 'dia-cache-intervention', 'dia-query-key-intervention']:
        src = Path(__file__).parent/name
        if src.is_dir():
            shutil.copytree(src, out/'method'/name, ignore=shutil.ignore_patterns('__pycache__'))
        else:
            shutil.copyfile(src, out/'method'/name)
    from huggingface_hub import hf_hub_download, snapshot_download
    archive = hf_hub_download('cstr/crispasr-regression-fixtures', PRIOR_PATH, revision=PRIOR_REV, repo_type='dataset')
    prior = scratch/'retained-query-key'
    prior.mkdir()
    prior_receipt = unpack_prior(archive, prior)
    shutil.copyfile(prior/'receipt.json', out/'prior-receipt.json')
    report['prior'] = dict(revision=PRIOR_REV, path=PRIOR_PATH, sha256=PRIOR_SHA,
                           worker_source=prior_receipt['worker_source'])
    inputs = out/'inputs'
    shutil.copytree(prior/'inputs', inputs)
    checkpoint()
    source = scratch / 'pinned-runtime'
    run(['git', 'fetch', 'origin', patcher.SOURCE], 'fetch-runtime')
    run([sys.executable, METHOD/'selftest.py'], 'method-selftest')
    run(['git', 'worktree', 'add', '--detach', source, patcher.SOURCE], 'checkout-runtime')
    run(['git', '-C', source, 'submodule', 'update', '--init', '--recursive'], 'runtime-submodules')
    report['ggml_source'] = subprocess.check_output(['git', '-C', source / 'ggml', 'rev-parse', 'HEAD'], text=True).strip()
    assert report['ggml_source'] == GGML_SOURCE
    report['unpatched_runtime_sha256'] = digest(source / 'src/dia_tts.cpp')
    patcher.patch(source / 'src/dia_tts.cpp')
    shutil.copyfile(BASE_METHOD / 'key_input.h', source / 'src/dia_intervention_key_input.h')
    shutil.copyfile(METHOD.parent / 'dia-suffix-replay/prefix.h', source / 'src/dia_suffix_prefix.h')
    with (source / 'CMakeLists.txt').open('a') as f:
        f.write(f'\nadd_subdirectory("{METHOD}" "dia-terminal-replay")\n')
    (out / 'method/runtime.patch').write_bytes(subprocess.check_output(['git', '-C', source, 'diff']))
    shutil.copyfile(source / 'src/dia_tts.cpp', out / 'method/patched-dia_tts.cpp')
    report['patched_runtime_sha256'] = digest(source / 'src/dia_tts.cpp')
    run(['sudo', 'bash', ROOT / 'tools/ci-apt.sh', 'update'], 'apt-update')
    run(['sudo', 'bash', ROOT / 'tools/ci-apt.sh', 'install', '-y', 'ninja-build', 'ccache'], 'apt-install')
    build = scratch / 'native-build'
    run(['cmake', '-G', 'Ninja', '-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
         '-DBUILD_SHARED_LIBS=ON', '-DCMAKE_C_COMPILER_LAUNCHER=ccache', '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
         '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF',
         '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_OPUS=OFF', '-DCRISPASR_AMR=OFF'], 'configure')
    run(['cmake', '--build', build, '--target', 'dia-terminal-replay', 'dia-terminal-guard', '-j4'], 'build')
    guard = next(p for p in build.rglob('dia-terminal-guard') if p.is_file() and os.access(p, os.X_OK))
    run([guard], 'actual-ggml-terminal-guard')
    assert 'DIA_SUFFIX_GUARD_PASS' in (out/'actual-ggml-terminal-guard.log').read_text()
    shutil.copyfile(guard, out/'method/actual-dia-terminal-guard')
    runner = next(p for p in build.rglob('dia-terminal-replay') if p.is_file() and os.access(p, os.X_OK))
    report['runner_sha256'] = digest(runner)
    run(['ldd', runner], 'actual-runner-ldd')
    shutil.copyfile(runner, out / 'method/actual-dia-terminal-replay')
    library_rows = []
    for line in (out / 'actual-runner-ldd.log').read_text().splitlines():
        for token in line.split():
            library = Path(token)
            if token.startswith('/') and library.is_file():
                library_rows.append(dict(path=token, bytes=library.stat().st_size, sha256=digest(library)))
                if library.name.startswith('libggml'):
                    (out / 'libs').mkdir(exist_ok=True)
                    shutil.copyfile(library, out / 'libs' / library.name)
    assert library_rows
    report['actual_linked_libraries'] = library_rows
    shutil.copyfile(source / 'ggml/LICENSE', out / 'method/GGML-LICENSE')


    # The F32 model is exported exactly as in the retained boundary run; no DAC
    # is required for forced logits and no full original model is instantiated.
    sys.path.insert(0, str(source / 'tools'))
    from dia_model_precision import WEIGHTS_SHA
    from reference_backends.dia_tts_reference import MODEL_PIN
    location = Path(snapshot_download('nari-labs/Dia-1.6B', revision=MODEL_PIN,
                                     allow_patterns=['config.json', 'model.safetensors']))
    weights = location / 'model.safetensors'
    assert digest(weights) == WEIGHTS_SHA
    template = hf_hub_download('cstr/dia-1.6b-GGUF', 'dia-1.6b-f16.gguf', revision='3233fbcb32be47761d2e736857b6d1a075b9ba7e')
    model = scratch / 'dia-1.6b-f32.gguf'
    run([sys.executable, source / 'tools/ci-heavy/dia_neural_precision_control.py', '--export', 'f32',
         '--weights', weights, '--template', template, '--output', model], 'neural-export')
    assert digest(model) == MODEL_SHA
    report['model_sha256'] = MODEL_SHA
    eos = json.loads((location / 'config.json').read_text())['data']['audio_eos_value']
    assert eos == 1024

    # Execute the actual original filter, observing its multinomial input only.
    # This does not sample or claim a complete generated trajectory.
    import torch
    assert torch.__version__ == '2.6.0+cpu'
    torch.set_num_threads(4)
    original_bytes = (prior/'method/original-model.py').read_bytes()
    (out/'method/original-model.py').write_bytes(original_bytes)
    original_text = original_bytes.decode()
    node = next(n for n in ast.parse(original_text).body if isinstance(n, ast.FunctionDef) and n.name == '_sample_next_token')
    sampler_source = ast.get_source_segment(original_text, node)
    (out / 'method/original-sampler.py').write_text(sampler_source + '\n')
    probabilities = []

    class ObserveTorch:
        def __getattr__(self, name): return getattr(torch, name)
        def multinomial(self, values, num_samples):
            assert num_samples == 1
            probabilities.append(values.clone())
            return torch.argmax(values, dim=-1, keepdim=True)

    namespace = {'torch': ObserveTorch()}
    exec(compile(sampler_source, 'pinned-original-sampler', 'exec'), namespace)

    def sampling(logits):
        logits = torch.from_numpy(logits.copy())
        cfg = logits[1] + 3.0 * (logits[1] - logits[0])
        raw = cfg[2].clone()
        cfg[:, eos+1:] = -torch.inf; cfg[1:, eos:] = -torch.inf
        namespace['_sample_next_token'](cfg.float(), 1.2, .95, 45)
        probs = probabilities.pop()[2]
        return dict(cfg46=float(raw[46]), cfg904=float(raw[904]), margin46_minus904=float(raw[46]-raw[904]),
                    rank46=int((cfg[2] > cfg[2, 46]).sum())+1, rank904=int((cfg[2] > cfg[2, 904]).sum())+1,
                    support=torch.nonzero(probs > 0).flatten().tolist())

    expected_logits = np.load(inputs/'native-logits.npy', allow_pickle=False)[-1]
    original_logits = np.load(inputs/'decoder_step_logits.npy', allow_pickle=False)[-1]
    tokens = np.fromfile(inputs/'tokens.i32', dtype='<i4').reshape(979, 9)
    report['original_sampling'] = sampling(original_logits)
    report['retained_native_sampling'] = sampling(expected_logits)
    assert report['original_sampling']['margin46_minus904'] > 0
    assert report['retained_native_sampling']['margin46_minus904'] < 0
    retained = prior/'baseline'
    baseline = out/'baseline'
    captures = sorted(f.name for f in retained.glob('cpp_*.f32'))
    assert len(captures) == 127, len(captures)
    base_env = {k: v for k, v in os.environ.items() if not k.startswith(('CRISPASR_DIA_', 'DIA_'))}
    base_env.update(CRISPASR_DIA_CROSS_CACHE_F32='1', CRISPASR_DIA_PORTABLE_RNG='1',
                    CRISPASR_DIA_TTS_GPU='0', CRISPASR_DIA_SOURCE_ROPE='0',
                    CRISPASR_DIA_FORCE_TOKENS=str(inputs/'tokens.i32'), CRISPASR_DIA_DUMP_STEP='978',
                    CRISPASR_DIA_DUMP_SELF_ATTN='1', CRISPASR_DIA_DUMP_DECODER_BOUNDARIES='1',
                    DIA_SUFFIX_INPUT_DIR=str(retained))
    original_post = np.load(inputs/'decoder_layer_17.npy', allow_pickle=False)
    original_hidden = np.load(inputs/'decoder_hidden.npy', allow_pickle=False)
    for name, values in [('terminal-original-layer17.f32', original_post), ('terminal-original-hidden.f32', original_hidden)]:
        assert values.shape == (2, 2048) and values.dtype == np.dtype('<f4') and np.isfinite(values).all()
        values.tofile(inputs/name)
    recipe = [('baseline', None, None, None),
              ('native-layer-17', 17, retained/'cpp_decoder_layer_17.f32', None),
              ('native-final-hidden', None, None, retained/'cpp_final_hidden.f32'),
              ('original-layer-17', 17, inputs/'terminal-original-layer17.f32', None),
              ('original-final-hidden', None, None, inputs/'terminal-original-hidden.f32')]
    anchor_rows = json.loads((prior/'original-layer-17/captures.json').read_text())
    anchor = {}
    for row in anchor_rows:
        assert row['kind'] in ['baseline', 'full']
        path = prior/row['baseline'] if row['kind'] == 'baseline' else prior/'original-layer-17'/row['path']
        assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
        anchor[row['path']] = path
    checkpoint()
    for name, layer, activation, final_hidden in recipe:
        dest = out/name
        dest.mkdir()
        env = dict(base_env, CRISPASR_DIA_DUMP_DIR=str(dest),
                   CRISPASR_DIA_DUMP_STEPLOGITS=str(dest/'logits.f32'),
                   CRISPASR_DIA_DUMP_INPUT_TOKENS=str(dest/'feedback.i32'))
        if activation is not None:
            env.update(DIA_SUFFIX_LAYER=str(layer), DIA_SUFFIX_ACTIVATION=str(activation))
        if final_hidden is not None:
            env['DIA_TERMINAL_HIDDEN'] = str(final_hidden)
        run([runner, model], name, env)
        log = (out/(name+'.log')).read_text()
        assert log.count('DIA_SUFFIX_PREFIX_READY') == 1 and 'DIA_SUFFIX_INVALID' not in log
        assert log.count('DIA_SUFFIX_ACTIVATION_APPLIED') == (activation is not None)
        assert log.count('DIA_TERMINAL_HIDDEN_APPLIED') == (final_hidden is not None)
        assert 'DIA_TERMINAL_INVALID' not in log
        assert (dest/'feedback.i32').read_bytes() == tokens[-1].tobytes()
        assert (dest/'capture-step.txt').read_text().strip() == '978'
        cross = dest/'suffix-cross-cache.f32'
        assert cross.stat().st_size == 18*2*2*39*2048*4
        cross_sha = digest(cross)
        if name != 'baseline':
            assert cross.read_bytes() == (baseline/'suffix-cross-cache.f32').read_bytes(), name
            cross.unlink()

        assert sorted(f.name for f in dest.glob('cpp_*.f32')) == captures
        logits = np.fromfile(dest/'logits.f32', dtype='<f4')
        assert logits.size == 2*9*1028
        logits = logits.reshape(2, 9, 1028)
        assert np.isfinite(logits).all()
        if name == 'baseline' or name.startswith('native-'):
            assert logits.tobytes() == expected_logits.tobytes(), name+' logits not byte-exact'
            for file in captures:
                assert (dest/file).read_bytes() == (retained/file).read_bytes(), (name, file)
        elif final_hidden is not None:
            for file in captures:
                if file not in ['cpp_final_hidden.f32', 'cpp_step978_cond.f32', 'cpp_step978_uncond.f32']:
                    assert (dest/file).read_bytes() == (baseline/file).read_bytes(), (name, file)
        else:
            assert name == 'original-layer-17'
            assert (dest/'logits.f32').read_bytes() == (prior/'original-layer-17/logits.f32').read_bytes()
            assert set(anchor) == set(captures)
            for file in captures:
                assert (dest/file).read_bytes() == anchor[file].read_bytes(), file
        if activation is not None:
            assert (dest/'suffix-used.f32').read_bytes() == activation.read_bytes(), name
            assert (dest/'suffix-computed.f32').read_bytes() == (baseline/f'cpp_decoder_layer_{layer}.f32').read_bytes(), name
            assert (dest/f'cpp_decoder_layer_{layer}.f32').read_bytes() == activation.read_bytes(), name
        if final_hidden is not None:
            assert (dest/'terminal-used.f32').read_bytes() == final_hidden.read_bytes()
            assert (dest/'cpp_final_hidden.f32').read_bytes() == final_hidden.read_bytes()
            assert (dest/'terminal-computed.f32').read_bytes() == (baseline/'cpp_final_hidden.f32').read_bytes()
        assert (dest/'cpp_step978_uncond.f32').read_bytes() == logits[0].tobytes()
        assert (dest/'cpp_step978_cond.f32').read_bytes() == logits[1].tobytes()
        row = dict(arm=name, layer=layer, clamp='final-hidden' if final_hidden is not None else ('post-layer' if activation is not None else None), sampling=sampling(logits), cross_cache_sha256=cross_sha,
                   logits_vs_original=metrics(logits, original_logits),
                   logits_sha256=digest(dest/'logits.f32'), layers=[])
        if activation is not None:
            row['activation_sha256'] = digest(activation)
            row['computed_sha256'] = digest(dest/'suffix-computed.f32')
            row['consumed_sha256'] = digest(dest/'suffix-used.f32')
        if final_hidden is not None:
            row['activation_sha256'] = digest(final_hidden)
            row['computed_sha256'] = digest(dest/'terminal-computed.f32')
            row['consumed_sha256'] = digest(dest/'terminal-used.f32')
        for l in range(18):
            actual = np.fromfile(dest/f'cpp_decoder_layer_{l}.f32', dtype='<f4').reshape(2, 2048)
            row['layers'].append(dict(layer=l, **metrics(actual, np.load(inputs/f'decoder_layer_{l}.npy', allow_pickle=False))))
        hidden = np.fromfile(dest/'cpp_final_hidden.f32', dtype='<f4').reshape(2, 2048)
        row['final_hidden'] = metrics(hidden, np.load(inputs/'decoder_hidden.npy', allow_pickle=False))
        if name != 'baseline':
            row['captures'] = compact_captures(dest, baseline)
        report['arms'].append(row)
        checkpoint()
    report['conditional_diagnostic_complete'] = True
    report['interpretation_limit'] = ('All caches hold native positions0..977, cross producer and token row fixed. '
        'Only one current post-layer or terminal activation changes per arm. This is conditional terminal sensitivity, '
        'not an original-cache replay, sampled trajectory, full original query capture, speech repair or GPU evidence.')
    report['payloads'] = [dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size, sha256=digest(p))
                          for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'receipt.json']
    checkpoint()
    print('DIA_TERMINAL_REPLAY_COMPLETE;5 arms, no speech/GPU/default acceptance', flush=True)


if __name__ == '__main__':
    main()
