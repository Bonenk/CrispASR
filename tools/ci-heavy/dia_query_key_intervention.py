#!/usr/bin/env python3
"""Conditional layer0 original-module-replay Q and captured K intervention.

CPU full model belongs on GH. Null PCM is deliberate: completion requires every
forced input, every logit and the actual attention input, never runner exit alone.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHOD = Path(__file__).with_name('dia-query-key-intervention')
BASE_METHOD = Path(__file__).with_name('dia-cache-intervention')
spec = importlib.util.spec_from_file_location('intervention_patch', METHOD / 'patch.py')
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)
INPUT_REV = '5e2fc2430413357a21807cd0350960aeeb6fe183'
INPUT_PATH = 'dia-20261015-cache-intervention/fixed-history-inputs-v1.zip'
INPUT_SHA = 'a0c7e2e876394469e07a455095cf79f6e004c1d2087f3ce4fb671dedbb567d52'
QUERY_REV = 'dce9d23fc048c78e57729e403af04ed97dd78adb'
QUERY_PATH = 'dia-20261015-query-key/query-inputs-v1.zip'
QUERY_SHA = '7b20c5f07250621ae9fde8e4aec9e8ba1ab114f420dd95b30d70aea27381affb'
MODEL_SHA = '8128132c538e1aef83f2fb5257b4f71d28a67c2a5a0d54d0a1f0a3b0f5d6b73d'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unpack(archive, dest):
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert len(names) == len(set(names))
        for name in names:
            assert Path(name).name == name and name not in ('.', '..')
            assert not ((z.getinfo(name).external_attr >> 16) & 0o170000) == 0o120000
        z.extractall(dest)
    manifest = json.loads((dest / 'manifest.json').read_text())
    assert set(names) == {'manifest.json'} | {r['path'] for r in manifest['files']}
    for row in manifest['files']:
        p = dest / row['path']
        assert p.stat().st_size == row['bytes'] and digest(p) == row['sha256'], row['path']
    assert manifest['source'] == patcher.SOURCE and manifest['step'] == 978
    return manifest


def arms(native, original, native_q, original_q):
    assert native.shape == original.shape == (2, 979, 4, 128)
    assert native_q.shape == original_q.shape == (2, 1, 16, 128)
    assert all(a.dtype == np.dtype('<f4') and np.isfinite(a).all()
               for a in [native, original, native_q, original_q])
    return {'native-control': (native.copy(), native_q.copy()),
            'original-query': (native.copy(), original_q.copy()),
            'original-keys': (original.copy(), native_q.copy()),
            'original-query-and-keys': (original.copy(), original_q.copy())}


def unpack_query(archive, dest):
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names = z.namelist(); assert len(names) == len(set(names))
        for name in names:
            assert Path(name).name == name and name not in ('.', '..')
            assert not ((z.getinfo(name).external_attr >> 16) & 0o170000) == 0o120000
        z.extractall(dest)
    manifest = json.loads((dest/'manifest.json').read_text())
    assert set(names) == {'manifest.json'} | {row['path'] for row in manifest['files']}
    for row in manifest['files']:
        file = dest/row['path']; assert file.stat().st_size == row['bytes'] and digest(file) == row['sha256']
    assert manifest['step'] == 978 and manifest['query_shape'] == [2, 1, 16, 128]
    assert manifest['query_kind'] == 'actual pinned original module replay; not earlier full-model capture'
    receipt = json.loads((dest/'producer-receipt.json').read_text())
    assert receipt['prior_module_replay_byte_exact'] and receipt['full_reference_reconstruction_gate_pass']
    assert not receipt['full_original_self_query_previously_captured']
    assert receipt['source'] == '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'
    return manifest, receipt


def metrics(actual, expected):
    assert actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
    x = actual.astype('f8').reshape(-1, actual.shape[-1])
    y = expected.astype('f8').reshape(x.shape)
    xn, yn = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    assert (xn > 0).all() and (yn > 0).all()
    return dict(cosine_min=float(np.min(np.sum(x*y, axis=1)/(xn*yn))),
                mine_norm=float(np.linalg.norm(x)), ref_norm=float(np.linalg.norm(y)),
                max_row_norm_error=float(np.max(abs(xn/yn-1))),
                relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(y)),
                max_abs=float(abs(x-y).max()), byte_exact=actual.tobytes() == expected.tobytes())


def main():
    out, scratch = Path(os.environ['HEAVY_OUT']), Path(os.environ['HEAVY_SCRATCH'])
    out.mkdir(parents=True, exist_ok=True); scratch.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(scratch)
    report = dict(worker_source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  runtime_source=patcher.SOURCE, worker_sha256=digest(__file__), step=978, runs=[], arms=[],
                  scope='Conditional fixed-history full-decoder layer0 replayed-original Q/captured K factorial; other caches stay native.',
                  speech_acceptance=False, gpu_acceptance=False, production_defaults_changed=False)
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)
    if os.environ.get('GITHUB_SHA'):
        assert report['worker_source'] == os.environ['GITHUB_SHA']

    def checkpoint():
        (out / 'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    def run(command, label, env=None):
        with (out / (label + '.log')).open('w') as log:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=10800)
        report['runs'].append(dict(label=label, exit=result.returncode, log_sha256=digest(out / (label + '.log'))))
        checkpoint()
        print(label, result.returncode, (out / (label + '.log')).read_text()[-1200:], flush=True)
        assert result.returncode == 0, label

    checkpoint()
    run([sys.executable, METHOD / 'selftest.py'], 'method-selftest')
    for name in ['dia_query_key_intervention.py', 'dia_cache_intervention.py', 'dia-query-key-intervention', 'dia-cache-intervention']:
        source = Path(__file__).parent / name
        if source.is_dir(): shutil.copytree(source, out / 'method' / name, ignore=shutil.ignore_patterns('__pycache__'))
        else:
            (out / 'method').mkdir(exist_ok=True); shutil.copyfile(source, out / 'method' / name)
    from huggingface_hub import hf_hub_download, snapshot_download
    archive = hf_hub_download('cstr/crispasr-regression-fixtures', INPUT_PATH, revision=INPUT_REV, repo_type='dataset')
    assert digest(archive) == INPUT_SHA
    inputs = out / 'inputs'; inputs.mkdir()
    report['input_manifest'] = unpack(archive, inputs)
    checkpoint()
    native = np.load(inputs / 'native-captured-keys.npy')
    original = np.load(inputs / 'original-captured-keys.npy')
    query_archive = hf_hub_download('cstr/crispasr-regression-fixtures', QUERY_PATH, revision=QUERY_REV, repo_type='dataset')
    assert digest(query_archive) == QUERY_SHA
    query_inputs = out/'query-inputs'; query_inputs.mkdir()
    report['query_manifest'], report['query_producer_receipt'] = unpack_query(query_archive, query_inputs)
    native_q = np.load(query_inputs/'native-query.npy', allow_pickle=False)
    original_q = np.load(query_inputs/'original-query.npy', allow_pickle=False)
    replacements = arms(native, original, native_q, original_q)
    for name, (keys, query) in replacements.items():
        keys.tofile(inputs/(name+'.f32'))
        query.tofile(inputs/(name+'-query.f32'))

    source = scratch / 'pinned-runtime'
    run(['git', 'fetch', 'origin', patcher.SOURCE], 'fetch-runtime')
    run(['git', 'worktree', 'add', '--detach', source, patcher.SOURCE], 'checkout-runtime')
    run(['git', '-C', source, 'submodule', 'update', '--init', '--recursive'], 'runtime-submodules')
    report['ggml_source'] = subprocess.check_output(['git', '-C', source / 'ggml', 'rev-parse', 'HEAD'], text=True).strip()
    report['unpatched_runtime_sha256'] = digest(source / 'src/dia_tts.cpp')
    patcher.patch(source / 'src/dia_tts.cpp')
    shutil.copyfile(BASE_METHOD / 'key_input.h', source / 'src/dia_intervention_key_input.h')
    with (source / 'CMakeLists.txt').open('a') as f:
        f.write(f'\nadd_subdirectory("{METHOD}" "dia-key-intervention")\n')
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
    run(['cmake', '--build', build, '--target', 'dia-key-intervention', 'dia-key-graph-guard', 'dia-query-key-graph-guard', '-j4'], 'build')
    guard = next(p for p in build.rglob('dia-key-graph-guard') if p.is_file() and os.access(p, os.X_OK))
    run([guard], 'actual-ggml-graph-guard')
    shutil.copyfile(guard, out / 'method/actual-dia-key-graph-guard')
    query_guard = next(p for p in build.rglob('dia-query-key-graph-guard') if p.is_file() and os.access(p, os.X_OK))
    run([query_guard], 'actual-ggml-query-key-graph-guard')
    shutil.copyfile(query_guard, out/'method/actual-dia-query-key-graph-guard')
    runner = next(p for p in build.rglob('dia-key-intervention') if p.is_file() and os.access(p, os.X_OK))
    report['runner_sha256'] = digest(runner)
    run(['ldd', runner], 'actual-runner-ldd')
    shutil.copyfile(runner, out / 'method/actual-dia-key-intervention')
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
    import requests
    import torch
    assert torch.__version__ == '2.6.0+cpu'
    torch.set_num_threads(4)
    original_url = 'https://raw.githubusercontent.com/nari-labs/dia/4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8/dia/model.py'
    response = requests.get(original_url, timeout=60); response.raise_for_status()
    (out / 'method/original-model.py').write_bytes(response.content)
    node = next(n for n in ast.parse(response.text).body if isinstance(n, ast.FunctionDef) and n.name == '_sample_next_token')
    sampler_source = ast.get_source_segment(response.text, node)
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

    expected_logits = np.load(inputs / 'native-logits.npy')
    original_logits = np.load(inputs / 'decoder_step_logits.npy')[-1]
    report['original_sampling'] = sampling(original_logits)
    report['retained_native_sampling'] = sampling(expected_logits[-1])
    assert report['original_sampling']['margin46_minus904'] > 0
    assert report['retained_native_sampling']['margin46_minus904'] < 0
    checkpoint()
    base_env = {k: v for k, v in os.environ.items() if not k.startswith(('CRISPASR_DIA_', 'DIA_'))}
    base_env.update(CRISPASR_DIA_CROSS_CACHE_F32='1', CRISPASR_DIA_PORTABLE_RNG='1',
                    CRISPASR_DIA_TTS_GPU='0', CRISPASR_DIA_SOURCE_ROPE='0',
                    CRISPASR_DIA_FORCE_TOKENS=str(inputs / 'tokens.i32'), CRISPASR_DIA_DUMP_STEP='978',
                    CRISPASR_DIA_DUMP_SELF_ATTN='1', CRISPASR_DIA_DUMP_DECODER_BOUNDARIES='1')
    baseline = None
    for name in ['baseline', *replacements]:
        dest = out / name; dest.mkdir()
        env = dict(base_env, CRISPASR_DIA_DUMP_DIR=str(dest),
                   CRISPASR_DIA_DUMP_STEPLOGITS=str(dest / 'logits.f32'),
                   CRISPASR_DIA_DUMP_INPUT_TOKENS=str(dest / 'feedback.i32'))
        if name != 'baseline':
            env['DIA_KEY_INTERVENTION_FILE'] = str(inputs/(name+'.f32'))
            env['DIA_QUERY_INTERVENTION_FILE'] = str(inputs/(name+'-query.f32'))
        run([runner, model], name, env)
        log = (out / (name + '.log')).read_text()
        assert log.count('DIA_KEY_INTERVENTION_APPLIED') == (name != 'baseline')
        assert 'DIA_KEY_INTERVENTION_INVALID_FILE' not in log
        assert log.count('DIA_QUERY_INTERVENTION_APPLIED') == (name != 'baseline')
        assert 'DIA_QUERY_INTERVENTION_INVALID_FILE' not in log
        assert (dest / 'feedback.i32').read_bytes() == (inputs / 'tokens.i32').read_bytes()
        assert (dest / 'capture-step.txt').read_text().strip() == '978'
        logits = np.fromfile(dest / 'logits.f32', dtype='<f4')
        assert logits.size == expected_logits.size
        logits = logits.reshape(expected_logits.shape)
        assert np.isfinite(logits).all()
        if name == 'baseline':
            assert logits.tobytes() == expected_logits.tobytes(), 'Baseline must reproduce all979 retained logits exactly'
            baseline = dest
        else:
            assert logits[:978].tobytes() == expected_logits[:978].tobytes(), 'Intervention changed earlier steps'
        used_keys = np.fromfile(dest / 'cpp_decoder_sa0_k.f32', dtype='<f4').reshape(native.shape)
        assert used_keys.tobytes() == (native if name == 'baseline' else replacements[name][0]).tobytes()
        used_query = np.fromfile(dest/'cpp_decoder_sa0_q.f32', dtype='<f4').reshape(native_q.shape)
        assert used_query.tobytes() == (native_q if name == 'baseline' else replacements[name][1]).tobytes()
        if name != 'baseline':
            for file in ['cpp_decoder_sa0_v.f32', 'cpp_decoder_block0_input.f32',
                         'cpp_decoder_block0_pre_sa_norm.f32', 'cpp_cross_k0.f32', 'cpp_cross_v0.f32']:
                assert (dest / file).read_bytes() == (baseline / file).read_bytes(), file
        if name == 'native-control':
            assert logits.tobytes() == expected_logits.tobytes(), 'Replacing with native keys must be byte-exact'
            for file in baseline.glob('cpp_*.f32'):
                assert file.read_bytes() == (dest / file.name).read_bytes(), file.name
        row = dict(arm=name, input_sha256=digest(dest / 'feedback.i32'), logits_sha256=digest(dest / 'logits.f32'),
                   sampling=sampling(logits[-1]), logits_vs_original=metrics(logits[-1], original_logits), layers=[])
        for layer in range(18):
            actual = np.fromfile(dest / f'cpp_decoder_layer_{layer}.f32', dtype='<f4').reshape(2, 2048)
            row['layers'].append(dict(layer=layer, **metrics(actual, np.load(inputs / f'decoder_layer_{layer}.npy'))))
        hidden = np.fromfile(dest / 'cpp_final_hidden.f32', dtype='<f4').reshape(2, 2048)
        row['final_hidden'] = metrics(hidden, np.load(inputs / 'decoder_hidden.npy'))
        report['arms'].append(row); checkpoint()
    report['conditional_diagnostic_complete'] = True
    report['interpretation_limit'] = 'Q is captured from actual pinned original module replay whose stages reconstruct the full reference within2e-6, but are not byte-exact. The factorial tests that conditional Q/K substitution, not an earlier full-model query or speech repair.'
    report['payloads'] = [dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size, sha256=digest(p))
                          for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'receipt.json']
    checkpoint()
    print('DIA_QUERY_KEY_INTERVENTION_COMPLETE; no speech/GPU/default acceptance', flush=True)


if __name__ == '__main__':
    main()
