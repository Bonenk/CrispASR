#!/usr/bin/env python3
"""Full-original Dia observations and one explicit sampled-token counterfactual.

Heavy CPU work runs on GH. No native Dia arithmetic/default is changed.
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
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHOD = Path(__file__).with_name('dia-original-trace')
REFERENCE = '8d0ad8862d2cad5d305a48a57f2a99b5024bc39a'
ORIGINAL = '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'
TEXT = '[S1] We are committed to making it secure.'
STEP = 978


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array(path):
    return np.load(path, allow_pickle=False, mmap_mode='r')


def recognize(args, out):
    sys.path.insert(0, str(ROOT/'python'))
    from crispasr import Session
    pcm = np.load(out/'decoded_audio.npy', allow_pickle=False)
    assert pcm.dtype == np.float32 and pcm.ndim == 1 and len(pcm) and np.isfinite(pcm).all()
    with Session(str(args.model), lib_path=str(args.lib), n_threads=4) as session:
        assert 'nemotron' in session.backend
        backend = session.backend
        transcript = ' '.join(segment.text for segment in session.transcribe(pcm, sample_rate=44100, language='en')).strip()
    words = lambda text: re.findall('[a-z]+', re.sub(r'<[^>]*>', '', text).lower())
    expected, actual = words(TEXT[5:]), words(transcript)
    dp = list(range(len(actual)+1))
    for i, word in enumerate(expected, 1):
        row = [i]
        for j, value in enumerate(actual, 1):
            row.append(min(row[-1]+1, dp[j]+1, dp[j-1]+(word != value)))
        dp = row
    result = dict(transcript=transcript, reference=TEXT[5:], wer=dp[-1]/len(expected), backend=backend,
                  model_sha256=digest(args.model), library_sha256=digest(args.lib), sample_rate=44100)
    (out/'asr.json').write_text(json.dumps(result, indent=2)+'\n')
    if args.arm == 'off':
        assert result['wer'] == 0, result
    print('ORIGINAL_TRACE_ASR_RESULT', json.dumps(result), flush=True)


def original_child(args, out):
    import torch
    from gguf import GGUFReader
    assert torch.__version__ == '2.6.0+cpu'; torch.set_num_threads(4)
    sys.path.insert(0, str(args.reference/'tools'))
    from reference_backends import dia_tts_reference as reference
    import dia_attention_capture as attention_capture
    from dia_attention_capture import attention_stages
    from dia_decoder_capture import boundary_stages
    reference.verify_source(args.blueprint)
    sys.path.insert(0, str(args.blueprint))
    import dia.model as model_module
    import dia.layers as layers_module
    assert Path(model_module.__file__).resolve() == (args.blueprint/'dia/model.py').resolve()
    spec = importlib.util.spec_from_file_location('owned_observers', METHOD/'observe.py')
    observe = importlib.util.module_from_spec(spec); spec.loader.exec_module(observe)
    canonical = {t.name: np.array(t.data).reshape(tuple(reversed(t.shape))) for t in GGUFReader(str(args.canonical)).tensors}
    valid = canonical['encoder_tokens'].shape[1]
    assert valid == 39
    enabled = args.arm == 'on'
    selected = tuple(range(18)) if enabled else ()
    os.environ.update(DIA_SOURCE=str(args.blueprint), DIA_DEVICE='cpu', DIA_TEXT=TEXT,
                      DIA_MAX_TOKENS='3072', DIA_SEED='42', DIA_TEMPERATURE='1.2', DIA_TOP_P='.95',
                      DIA_TOP_K='45', DIA_CFG_SCALE='3', DIA_CAPTURE_STEP=str(STEP),
                      DIA_CAPTURE_ATTENTION_LAYERS=','.join(map(str, selected)),
                      DIA_CAPTURE_BOUNDARY_LAYERS=','.join(map(str, selected)))
    extras = {}
    def save(name, value):
        assert name not in extras and value.dtype == torch.float32
        owned = value.detach().cpu().numpy().copy()
        assert owned.size and np.isfinite(owned).all()
        extras[name] = owned
    old_install = attention_capture.install_attention_hooks
    old_cross = layers_module.Decoder.precompute_cross_attn_cache
    old_sample = model_module._sample_next_token
    witnessed, witness = observe.sample_witness(old_sample)
    sampled, fork = observe.single_sample_fork(witnessed) if args.arm == 'fork' else (witnessed, None)
    try:
        model_module._sample_next_token = sampled
        if enabled:
            attention_capture.install_attention_hooks = observe.extend_attention_installer(old_install, save)
            layers_module.Decoder.precompute_cross_attn_cache = observe.cross_cache_observer(old_cross, valid, save)
        stages = set(reference.BOUNDARY_STAGES)
        stages.update(f'encoder_layer_{i}' for i in range(12))
        stages.update(f'decoder_layer_{i}' for i in range(18))
        stages.update(attention_stages(selected)); stages.update(boundary_stages(selected))
        result = reference.dump('nari-labs/Dia-1.6B', None, stages=stages)
    finally:
        attention_capture.install_attention_hooks = old_install
        layers_module.Decoder.precompute_cross_attn_cache = old_cross
        model_module._sample_next_token = old_sample
    assert result['dia_source_pin'] == ORIGINAL and result['dia_model_revision'] == reference.MODEL_PIN
    assert result['dia_torch_version'] == '2.6.0+cpu'
    assert result['dia_weights_sha256'] == 'caba289b60f6d7d1e58fc744f4dc25aae88995fcca46be3d05e220b971486a26'
    assert result['dia_dac_sha256'] == 'a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa'
    assert len(witness['events']) == 1 and witness['events'][0]['sampled'][2] == 877
    if args.arm != 'fork':
        for name in ['decoder_input_tokens', 'decoder_codes_rev']:
            assert result[name].tobytes() == canonical[name].tobytes(), name
    else:
        assert fork['calls'] == witness['calls'] and len(fork['events']) == 1
        assert fork['events'][0]['rng_after_original_sha256'] == witness['events'][0]['rng_after_sample_sha256']
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    assert not (set(arrays) & set(extras)); arrays.update(extras)
    rows = []
    for name, values in sorted(arrays.items()):
        assert values.size and np.isfinite(values).all()
        path = out/(name+'.npy'); np.save(path, values, allow_pickle=False)
        rows.append(dict(name=name, shape=list(values.shape), dtype=str(values.dtype), path=path.name,
                         bytes=path.stat().st_size, sha256=digest(path)))
    metadata = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (out/'arrays.json').write_text(json.dumps(rows, indent=2)+'\n')
    event = dict(arm=args.arm, witness=witness, fork=fork, extra_observations=sorted(extras),
                 final_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
    (out/'sampling-witness.json').write_text(json.dumps(event, indent=2)+'\n')
    print('DIA_ACTUAL_ORIGINAL_TRACE_ARM_COMPLETE', args.arm, len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', choices=['original', 'asr'])
    parser.add_argument('--arm', choices=['off', 'on', 'repeat', 'fork'])
    for name in ['reference', 'blueprint', 'canonical', 'model', 'lib']:
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    out, scratch = Path(os.environ['HEAVY_OUT']), Path(os.environ['HEAVY_SCRATCH'])
    out.mkdir(parents=True, exist_ok=True); scratch.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(scratch)
    if args.child:
        dest = out/'arms'/args.arm; dest.mkdir(parents=True, exist_ok=True)
        (original_child if args.child == 'original' else recognize)(args, dest)
        return
    report = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  reference_source=REFERENCE, original_source=ORIGINAL, worker_sha256=digest(__file__),
                  step=STEP, runs=[], arms=[], speech_acceptance=False, native_speech_acceptance=False,
                  gpu_acceptance=False, production_defaults_changed=False)
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)
    if os.environ.get('GITHUB_SHA'): assert report['source'] == os.environ['GITHUB_SHA']
    def checkpoint():
        (out/'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    def run(command, label):
        with (out/(label+'.log')).open('w') as log:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=10800)
        report['runs'].append(dict(label=label, exit=result.returncode, log_sha256=digest(out/(label+'.log')))); checkpoint()
        print(label, result.returncode, (out/(label+'.log')).read_text()[-1400:], flush=True)
        assert result.returncode == 0, label
    checkpoint()
    (out/'method').mkdir()
    shutil.copyfile(__file__, out/'method/dia_original_trace.py')
    shutil.copytree(METHOD, out/'method/dia-original-trace', ignore=shutil.ignore_patterns('__pycache__'))
    reference = scratch/'pinned-reference'
    run(['git', 'fetch', 'origin', REFERENCE], 'fetch-reference')
    run(['git', 'worktree', 'add', '--detach', reference, REFERENCE], 'reference-checkout')
    run(['git', '-C', reference, 'submodule', 'update', '--init', '--recursive'], 'reference-submodules')
    blueprint = scratch/'original-dia'
    run(['git', 'clone', '--filter=blob:none', 'https://github.com/nari-labs/dia', blueprint], 'original-clone')
    run(['git', '-C', blueprint, 'checkout', '--detach', ORIGINAL], 'original-pin')
    os.environ.update(DIA_SOURCE=str(blueprint), DIA_DEVICE='cpu')
    run([sys.executable, METHOD/'selftest.py', '--blueprint', blueprint, '--reference', reference], 'tiny-original-contracts')
    run([sys.executable, reference/'tests/test_dia_reference_capture.py', '-v'], 'existing-original-contracts')
    proof = scratch/'canonical-proof'
    run([sys.executable, reference/'tools/dia_proof_archives.py', 'canonical-original', proof], 'canonical-download')
    canonical = next(proof.glob('heavy-*'))/'ref.gguf'
    assert digest(canonical) == '3db6f8923abffd612e6c83d7036debebd10ed28b77c3eb9ec3d91fec1ce1a542'
    report['canonical_sha256'] = digest(canonical)
    # Retain the exact observer implementation and original blueprint sources.
    for prefix, source, names in [('pinned-reference', reference/'tools', ['reference_backends/dia_tts_reference.py', 'dia_attention_capture.py', 'dia_decoder_capture.py', 'dia_proof_archives.py']),
                                  ('original-dia', blueprint, ['dia/model.py', 'dia/layers.py', 'dia/state.py', 'dia/config.py', 'LICENSE'])]:
        for name in names:
            dest = out/'method'/prefix/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source/name, dest)
    for arm in ['off', 'on', 'repeat']:
        run([sys.executable, __file__, '--child', 'original', '--arm', arm, '--reference', reference,
             '--blueprint', blueprint, '--canonical', canonical], 'original-'+arm)
    off = out/'arms/off'; on = out/'arms/on'; repeat = out/'arms/repeat'
    off_rows = json.loads((off/'arrays.json').read_text())
    common = {row['name'] for row in off_rows}
    for other in [on, repeat]:
        rows = {row['name']: row for row in json.loads((other/'arrays.json').read_text())}
        assert common <= set(rows)
        for row in off_rows:
            assert row['sha256'] == rows[row['name']]['sha256'], (other.name, row['name'])
        base_witness = json.loads((off/'sampling-witness.json').read_text())
        other_witness = json.loads((other/'sampling-witness.json').read_text())
        assert base_witness['witness'] == other_witness['witness']
        assert base_witness['final_rng_sha256'] == other_witness['final_rng_sha256']
    for layer in range(18):
        for part, shape in [('q', (2, 1, 16, 128)), ('k', (2, 979, 4, 128)), ('v', (2, 979, 4, 128)), ('attention', (2, 1, 16, 128)),
                            ('pre_q', (2, 1, 16, 128)), ('pre_k', (2, 1, 4, 128)), ('pre_v', (2, 1, 4, 128))]:
            assert array(on/f'decoder_sa{layer}_{part}.npy').shape == shape
        for part in ['k', 'v']:
            assert array(on/f'decoder_cross{layer}_{part}.npy').shape == (2, 39, 16, 128)
    report['off_on_off_all_common_arrays_and_rng_exact'] = True; checkpoint()
    # Build the recognizer once, with no original full model resident.
    run(['sudo', 'bash', ROOT/'tools/ci-apt.sh', 'update'], 'apt-update')
    run(['sudo', 'bash', ROOT/'tools/ci-apt.sh', 'install', '-y', 'ninja-build', 'ccache'], 'apt-install')
    build = scratch/'asr-build'
    run(['cmake', '-G', 'Ninja', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
         '-DCMAKE_C_COMPILER_LAUNCHER=ccache', '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache', '-DGGML_NATIVE=OFF',
         '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF', '-DGGML_CANN=OFF', '-DGGML_WEBGPU=OFF',
         '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_OPUS=OFF', '-DCRISPASR_AMR=OFF'], 'asr-configure')
    run(['cmake', '--build', build, '--target', 'crispasr-lib', '-j4'], 'asr-build')
    lib = next(build.rglob('libcrispasr.so'))
    report['asr_library_sha256'] = digest(lib)
    run(['ldd', lib], 'asr-library-dependencies')
    (out/'asr-libraries').mkdir()
    for binary in sorted(build.rglob('*.so*')):
        if binary.is_file() and not binary.is_symlink():
            shutil.copyfile(binary, out/'asr-libraries'/binary.name)
    from huggingface_hub import hf_hub_download, get_hf_file_metadata, hf_hub_url
    asr_repo, asr_name, asr_rev = 'cstr/nemotron-3.5-asr-streaming-GGUF', 'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf', 'bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2'
    model = Path(hf_hub_download(asr_repo, asr_name, revision=asr_rev))
    assert digest(model) == get_hf_file_metadata(hf_hub_url(asr_repo, asr_name, revision=asr_rev)).etag.strip('"')
    report['asr_model_sha256'] = digest(model)
    run([sys.executable, __file__, '--child', 'asr', '--arm', 'off', '--model', model, '--lib', lib], 'original-positive-asr')
    report['original_positive'] = json.loads((off/'asr.json').read_text()); assert report['original_positive']['wer'] == 0
    checkpoint()
    run([sys.executable, __file__, '--child', 'original', '--arm', 'fork', '--reference', reference,
         '--blueprint', blueprint, '--canonical', canonical], 'original-fork')
    fork = out/'arms/fork'
    base_inputs, fork_inputs = array(off/'decoder_input_tokens.npy'), array(fork/'decoder_input_tokens.npy')
    assert base_inputs[:STEP+1].tobytes() == fork_inputs[:STEP+1].tobytes()
    assert array(off/'decoder_step_logits.npy')[:STEP+1].tobytes() == array(fork/'decoder_step_logits.npy')[:STEP+1].tobytes()
    next_row = base_inputs[STEP+1].copy(); assert next_row[2] == 877; next_row[2] = 904
    assert fork_inputs[STEP+1].tobytes() == next_row.tobytes()
    assert array(off/'decoder_codes_rev.npy')[:STEP-15].tobytes() == array(fork/'decoder_codes_rev.npy')[:STEP-15].tobytes()
    fork_witness = json.loads((fork/'sampling-witness.json').read_text())
    event = fork_witness['fork']['events'][0]
    assert event['rng_after_original_sha256'] == base_witness['witness']['events'][0]['rng_after_sample_sha256']
    assert event['original'] == base_witness['witness']['events'][0]['sampled']
    assert [i for i, (a, b) in enumerate(zip(event['original'], event['used'])) if a != b] == [2]
    report['fork_prefix_and_rng_exact'] = True; report['fork_event'] = event; checkpoint()
    run([sys.executable, __file__, '--child', 'asr', '--arm', 'fork', '--model', model, '--lib', lib], 'fork-asr')
    report['fork_asr'] = json.loads((fork/'asr.json').read_text())
    # Lossless deduplication: full NPY hashes/shape/dtype remain in each manifest.
    base_rows = {r['name']: r for r in off_rows}
    for arm in ['off', 'on', 'repeat', 'fork']:
        dest = out/'arms'/arm; storage = []
        for row in json.loads((dest/'arrays.json').read_text()):
            same = arm != 'off' and row['name'] in base_rows and row['sha256'] == base_rows[row['name']]['sha256']
            target = off/row['path'] if same else dest/row['path']
            assert target.stat().st_size == row['bytes'] and digest(target) == row['sha256']
            storage.append(dict(**row, storage=target.relative_to(out).as_posix()))
            if same: (dest/row['path']).unlink()
        (dest/'array-storage.json').write_text(json.dumps(storage, indent=2)+'\n')
        report['arms'].append(dict(arm=arm, arrays=len(storage), storage_manifest_sha256=digest(dest/'array-storage.json')))
    report['full_original_trace_complete'] = True
    report['interpretation_limit'] = 'Actual original full-model observations with exact OFF/ON/OFF controls; the fork is one explicit sampled-token intervention, not a native runtime/default or speech fix.'
    report['payloads'] = [dict(path=f.relative_to(out).as_posix(), bytes=f.stat().st_size, sha256=digest(f))
                          for f in sorted(out.rglob('*')) if f.is_file() and f.name != 'receipt.json']
    checkpoint(); print('DIA_FULL_ORIGINAL_TRACE_COMPLETE; fork quality reported without threshold changes', flush=True)


if __name__ == '__main__':
    main()
