#!/usr/bin/env python3
"""Independent complete-artifact audit; no model inference or worker imports."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import zipfile

import numpy as np

SOURCE = '9ed6cff6f12627fddb61ea0ff7410b50950fd2b3'
REFERENCE = '8d0ad8862d2cad5d305a48a57f2a99b5024bc39a'
ORIGINAL = '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'
ARCHIVE = 'c57ddb8fba5fd6c2c96eca39192c3cb38f4079924a5efbc788c60a7462c8ced4'
PARTS = ('input', 'pre_sa_norm', 'sa_attention', 'sa_output', 'sa_residual', 'pre_ca_norm',
         'cross_q', 'cross_attention', 'cross_output', 'ca_residual', 'pre_mlp_norm',
         'mlp_gate', 'mlp_up', 'mlp_hidden', 'mlp_output', 'output')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe(name):
    path = PurePosixPath(name)
    assert name and path.as_posix() == name and name != '.'
    assert not path.is_absolute() and '..' not in path.parts and '\\' not in name
    return name


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def fork_prefix(base, changed, logits, fork_logits, codes, fork_codes):
    assert base.ndim == changed.ndim == 2 and base.shape[1] == changed.shape[1] == 9
    assert len(base) > 979 and len(changed) > 979
    exact(base[:979], changed[:979]); exact(logits[:979], fork_logits[:979])
    assert base[979, 2] == 877 and changed[979, 2] == 904
    assert np.flatnonzero(base[979] != changed[979]).tolist() == [2]
    assert len(codes) >= 963 and len(fork_codes) >= 963
    exact(codes[:963], fork_codes[:963])


def measure(a, b):
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    a = np.asarray(a, dtype='f8').reshape(-1, a.shape[-1]); b = np.asarray(b, dtype='f8').reshape(a.shape)
    x, y = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    denominator = x*y
    cos = np.divide(np.sum(a*b, axis=1), denominator, out=np.zeros_like(x), where=denominator != 0)
    cos[(x == 0) & (y == 0)] = 1
    return dict(cos_min=float(cos.min()), native_rms=float(np.sqrt(np.mean(a*a))),
                original_rms=float(np.sqrt(np.mean(b*b))), max_abs=float(np.max(np.abs(a-b))),
                relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-300)))


def load_arm(out, name):
    directory = out/'arms'/name
    manifest = json.loads((directory/'arrays.json').read_text())
    storage = json.loads((directory/'array-storage.json').read_text())
    assert len(manifest) == len(storage)
    expected = {row['name']: row for row in manifest}; assert len(expected) == len(manifest)
    arrays = {}
    for row in storage:
        key = row['name']; assert re.fullmatch('[a-z0-9_]+', key) and key not in arrays
        assert {k: v for k, v in row.items() if k != 'storage'} == expected[key]
        assert row['path'] == key+'.npy'
        target = out/safe(row['storage'])
        assert row['storage'] in [f'arms/{name}/{key}.npy', f'arms/off/{key}.npy']
        assert target.stat().st_size == row['bytes'] and sha(target) == row['sha256']
        value = np.load(target, mmap_mode='r', allow_pickle=False)
        assert list(value.shape) == row['shape'] and str(value.dtype) == row['dtype'] == 'float32'
        assert value.size and np.isfinite(value).all()
        arrays[key] = value
    assert set(arrays) == set(expected)
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle', 'repo', 'original-repo', 'canonical', 'terminal-bundle', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists()
    bundle = args.bundle; archive = bundle/'raw.zip'; out = bundle/'out'
    assert archive.stat().st_size == 338692054 and sha(archive) == ARCHIVE
    run = json.loads((bundle/'run.json').read_text())
    assert run['headSha'] == SOURCE and run['status'] == 'completed' and run['conclusion'] == 'success'
    assert run['jobs'] and all(j['conclusion'] == 'success' for j in run['jobs'])
    lint = json.loads((bundle/'lint.json').read_text())
    assert lint['headSha'] == SOURCE and lint['conclusion'] == 'success'
    artifacts = json.loads((bundle/'artifacts.json').read_text())['artifacts']; assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact['id'] == 12014433279 and artifact['workflow_run']['head_sha'] == SOURCE
    assert artifact['digest'] == 'sha256:'+ARCHIVE and artifact['size_in_bytes'] == archive.stat().st_size
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names = z.namelist(); assert len(names) == len(set(names))
        for info in z.infolist():
            safe(info.filename); assert not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16)
        if not out.exists(): z.extractall(out)
        assert z.read('receipt.json') == (out/'receipt.json').read_bytes()
    assert set(names) == {f.relative_to(out).as_posix() for f in out.rglob('*') if f.is_file()}
    receipt = json.loads((out/'receipt.json').read_text())
    assert receipt['source'] == SOURCE and receipt['reference_source'] == REFERENCE and receipt['original_source'] == ORIGINAL
    for key in ['full_original_trace_complete', 'off_on_off_all_common_arrays_and_rng_exact', 'fork_prefix_and_rng_exact']:
        assert receipt[key] is True
    for key in ['speech_acceptance', 'native_speech_acceptance', 'gpu_acceptance', 'production_defaults_changed']:
        assert receipt[key] is False
    payloads = {r['path']: r for r in receipt['payloads']}
    assert len(payloads) == len(receipt['payloads']) and set(payloads) == set(names)-{'receipt.json'}
    for name, row in payloads.items():
        file = out/safe(name); assert file.stat().st_size == row['bytes'] and sha(file) == row['sha256'], name
    for row in receipt['runs']:
        assert row['exit'] == 0 and sha(out/(row['label']+'.log')) == row['log_sha256']
    method = out/'method'
    for file in method.rglob('*'):
        if not file.is_file(): continue
        name = file.relative_to(method).as_posix()
        if name.startswith('original-dia/'):
            repo, ref, relative = args.original_repo, ORIGINAL, name[len('original-dia/'):]
        elif name.startswith('pinned-reference/'):
            repo, ref, relative = args.repo, REFERENCE, 'tools/'+name[len('pinned-reference/'):]
        else:
            repo, ref, relative = args.repo, SOURCE, 'tools/ci-heavy/'+name
        assert file.read_bytes() == subprocess.check_output(['git', 'show', ref+':'+relative], cwd=repo), name
    assert sha(method/'dia_original_trace.py') == receipt['worker_sha256']
    assert 'DIA_ORIGINAL_TRACE_WIRING_PASS' in (out/'tiny-original-contracts.log').read_text()
    assert '6 rejected controls' in (out/'tiny-original-contracts.log').read_text()
    assert 'OK' in (out/'existing-original-contracts.log').read_text()
    arms = {name: load_arm(out, name) for name in ['off', 'on', 'repeat', 'fork']}
    assert [len(arms[n]) for n in arms] == [41, 491, 41, 41]
    for name in ['on', 'repeat']:
        for key, value in arms['off'].items(): exact(value, arms[name][key])
    witnesses = {name: json.loads((out/'arms'/name/'sampling-witness.json').read_text()) for name in arms}
    for name in ['on', 'repeat']:
        assert witnesses[name]['witness'] == witnesses['off']['witness']
        assert witnesses[name]['final_rng_sha256'] == witnesses['off']['final_rng_sha256']
    for name in arms:
        metadata = json.loads((out/'arms'/name/'metadata.json').read_text())
        assert metadata['dia_source_pin'] == ORIGINAL and metadata['dia_torch_version'] == '2.6.0+cpu' and metadata['dia_device'] == 'cpu'
        assert metadata['dia_weights_sha256'] == 'caba289b60f6d7d1e58fc744f4dc25aae88995fcca46be3d05e220b971486a26'
        assert metadata['dia_model_revision'] == '257bc72f9b78182ccc6fa07675a9ae4c1a44e2cd'
        assert metadata['dia_dac_sha256'] == 'a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa'
        settings = json.loads(metadata['dia_generation_settings'])
        for key, expected in dict(max_tokens=3072, seed=42, temperature=1.2, top_p=.95, top_k=45, cfg_scale=3., capture_step=978).items():
            assert settings[key] == expected
    from gguf import GGUFReader
    assert sha(args.canonical) == '3db6f8923abffd612e6c83d7036debebd10ed28b77c3eb9ec3d91fec1ce1a542'
    canonical = {t.name: np.asarray(t.data).reshape(tuple(reversed(t.shape))) for t in GGUFReader(str(args.canonical)).tensors}
    for key in ['decoder_input_tokens', 'decoder_codes_rev', 'encoder_tokens']: exact(arms['off'][key], canonical[key])
    on = arms['on']
    for i in range(18):
        for part, shape in [('q', (2,1,16,128)), ('k', (2,979,4,128)), ('v', (2,979,4,128)),
                            ('attention', (2,1,16,128)), ('pre_q', (2,1,16,128)), ('pre_k', (2,1,4,128)), ('pre_v', (2,1,4,128))]:
            assert on[f'decoder_sa{i}_{part}'].shape == shape
        for part in ['k', 'v']: assert on[f'decoder_cross{i}_{part}'].shape == (2,39,16,128)
        assert all(f'decoder_block{i}_{part}' in on for part in PARTS)
        exact(on[f'decoder_block{i}_sa_attention'], on[f'decoder_sa{i}_attention'])
        exact(on[f'decoder_block{i}_output'][:,0], on[f'decoder_layer_{i}'])
        if i: exact(on[f'decoder_block{i}_input'], on[f'decoder_block{i-1}_output'])
    for part in ['k', 'v']: exact(on[f'decoder_cross0_{part}'].reshape(2,39,2048), on[f'cross_{part}0'])
    a, b = arms['off'], arms['fork']
    fork_prefix(a['decoder_input_tokens'], b['decoder_input_tokens'], a['decoder_step_logits'], b['decoder_step_logits'],
                a['decoder_codes_rev'], b['decoder_codes_rev'])
    event = witnesses['fork']['fork']['events']; assert len(event) == 1; event = event[0]
    original_event = witnesses['off']['witness']['events']; assert len(original_event) == 1; original_event = original_event[0]
    assert event['step'] == 978 and event['channel'] == 2
    assert event['original'] == original_event['sampled'] == witnesses['fork']['witness']['events'][0]['sampled']
    assert event['original'][2] == 877 and event['used'][2] == 904
    assert [i for i in range(9) if event['original'][i] != event['used'][i]] == [2]
    assert event['rng_after_original_sha256'] == event['rng_after_fork_sha256'] == original_event['rng_after_sample_sha256']
    assert event == receipt['fork_event']
    asr = {name: json.loads((out/'arms'/name/'asr.json').read_text()) for name in ['off', 'fork']}
    assert asr['off']['transcript'] == 'We are committed to making it secure.' and asr['off']['wer'] == 0
    assert asr['fork']['transcript'] == '' and asr['fork']['wer'] == 1
    libraries = list((out/'asr-libraries').glob('libcrispasr.so*'))
    assert len(libraries) == 1 and libraries[0].read_bytes().startswith(b'\x7fELF')
    for result in asr.values():
        assert result['backend'] == 'nemotron' and result['sample_rate'] == 44100
        assert result['model_sha256'] == '60a2b074cdf46e81d72bc15d3eb79505dafbb697135f558659d9068f651854a2'
        assert result['library_sha256'] == receipt['asr_library_sha256'] == sha(libraries[0])
    assert asr['off'] == receipt['original_positive'] and asr['fork'] == receipt['fork_asr']
    # Compare actual new full-model observations against accepted native captures.
    terminal = args.terminal_bundle
    assert sha(terminal/'raw.zip') == 'a2d4c27e7ef6351e9b0026c1389ec9fd409c33421694ac1e596b090a0ec2cc21'
    metrics = []
    with zipfile.ZipFile(terminal/'raw.zip') as z:
        old_receipt = json.loads(z.read('receipt.json')); old_rows = {r['path']: r for r in old_receipt['payloads']}
        for key in ['decoder_hidden']+[f'decoder_layer_{i}' for i in range(18)]:
            raw = z.read('inputs/'+key+'.npy')
            assert hashlib.sha256(raw).hexdigest() == old_rows['inputs/'+key+'.npy']['sha256']
            exact(on[key], np.load(io.BytesIO(raw), allow_pickle=False))
        mapping = {'ca0_out':'decoder_ca0', 'final_hidden':'decoder_hidden'}
        for name in sorted(n for n in z.namelist() if n.startswith('baseline/cpp_') and n.endswith('.f32')):
            key = Path(name).name[4:-4]; key = mapping.get(key, key)
            if key not in on: continue
            raw = z.read(name); row = old_rows[name]
            assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
            original = on[key]; native = np.frombuffer(raw, dtype='<f4').reshape(original.shape)
            metrics.append(dict(stage=key, **measure(native, original)))
    result = dict(source=SOURCE, archive_sha256=ARCHIVE, payloads=len(payloads), arms={n:len(a) for n,a in arms.items()},
                  all_common_arrays_and_rng_exact=True, canonical_exact=True, all18_actual_full_model_observations=True,
                  one_sample_fork_prefix_exact=True, original_wer=0, fork_wer=1, fork_transcript='',
                  metrics_vs_accepted_native=metrics, native_speech_acceptance=False, production_defaults_changed=False)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('DIA_ORIGINAL_TRACE_INDEPENDENT_AUDIT_PASS', len(payloads), 'payloads;', len(metrics), 'native comparison rows; original WER0 / fork WER1')


if __name__ == '__main__':
    main()
