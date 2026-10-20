#!/usr/bin/env python3
"""Validate the pinned Moonshine F32 fixture on CPU, then an exact CANN device.

Requires a supplied Linux build; no model downloads, compilation or changes to
upstream quality thresholds. Retains complete logs, arrays, binary snapshots
and incremental receipts. This does not validate quantization, ACL or speed.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SHA256 = '7d63d2a5b90eb77843aff25107b1d1f70bafcfc1c8f73075e2f5a9a3f87d5e84'
FIXTURE_BYTES = 108495214
spec = importlib.util.spec_from_file_location('moonshine_fixture', ROOT/'tools/ci-heavy/cann_model_fixture.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def unpack(archive, destination):
    """Validate all member paths/counts before writing any payload."""
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate fixture members')
        total = 0
        for member in z.infolist():
            path = Path(member.filename)
            total += member.file_size
            if (path.is_absolute() or '..' in path.parts or '\\' in member.filename
                    or stat.S_ISLNK(member.external_attr >> 16) or total > 192*1024*1024):
                raise ValueError('Unsafe or oversized fixture member')
        for member in z.infolist():
            if member.is_dir():
                continue
            path = destination/member.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, path.open('wb') as dest:
                shutil.copyfileobj(src, dest, 4194304)


def assess_placement(text, backend, device, minimum_encodes, require_nz=None):
    rows = re.findall(r'backend_validation\[moonshine.encoder\]: required_matmuls=(\d+) unassigned=(\d+) compute_success=(\d+)', text)
    if len(rows) < minimum_encodes or any(int(m) <= 0 or int(u) != 0 or int(ok) != 1 for m, u, ok in rows):
        raise ValueError('Missing, incomplete or unsuccessful encoder execution evidence')
    assigned = re.findall(r'assigned=([^\s/]+)/MUL_MAT nodes=(\d+)', text)
    selected = re.findall(r'backend_validation: registry=(\S+) device=(\S+) ordinal=(\d+) initialized=(\d+)', text)
    wanted = [name for name, count in assigned if int(count) > 0 and
              (name == 'CPU' if backend == 'cpu' else name in {row[1] for row in selected})]
    if not wanted:
        raise ValueError('No matmuls on the requested backend')
    if backend != 'cpu':
        registries = {'cann'} if backend == 'cann' else {'metal', 'mtl'}
        if not selected or any(reg.lower() not in registries or int(index) != device or int(ok) != 1
                               for reg, name, index, ok in selected):
            raise ValueError('Required device was not initialized')
        if not set(wanted).issubset({name for reg, name, index, ok in selected}):
            raise ValueError('Scheduled device differs from selected device')
    nz = []
    if require_nz is not None:
        matches = re.findall(r'nz_cached_matmuls=(\d+) nz_unique_weights=(\d+) nz_bytes=(\d+) nz_selections=(\d+) nz_selected_weights=(\d+)', text)
        if len(matches) != len(rows):
            raise ValueError('Missing or incomplete NZ usage evidence')
        for match in matches:
            m, w, size, selections, used = map(int, match)
            if require_nz == 'on':
                if backend != 'cann' or not (m >= w >= used > 0 and size > 0 and selections >= used):
                    raise ValueError('NZ on did not select actual cached model weights')
            elif any((m, w, size, selections, used)):
                raise ValueError('ND control unexpectedly used NZ cache')
            nz.append(dict(cached_matmuls=m, unique_weights=w, bytes=size,
                           host_encode_selections=selections, selected_weights=used))
    return dict(encoder_computes=len(rows), required_matmuls=[int(row[0]) for row in rows],
                assigned_devices=sorted(set(wanted)), selected_devices=selected, nz_usage=nz)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-zip', required=True, type=Path)
    parser.add_argument('--build-dir', required=True, type=Path)
    parser.add_argument('--backend', choices=('cpu', 'cann', 'metal'), default='cann')
    parser.add_argument('--exercise-reuse', action='store_true')
    parser.add_argument('--nz-cache', choices=('unchanged', 'off', 'on'), default='unchanged')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    if args.device < 0 or args.timeout <= 0 or (args.backend == 'cpu' and args.device != 0):
        parser.error('Use a nonnegative device, positive timeout and CPU device0')
    if args.nz_cache == 'on' and args.backend != 'cann':
        parser.error('--nz-cache on requires --backend cann')
    # Fresh outputs prevent stale arrays/logs from passing after a failed run.
    args.out.mkdir(parents=True, exist_ok=False)
    out = args.out.resolve()
    receipt = dict(passed=False, requested_backend=args.backend, requested_device=args.device,
                   validator_sha256=fixture.digest(__file__), architecture=platform.machine(),
                   scope='Pinned F32 JFK full-model encoder/decoded-output validation; no quant/ACL/performance acceptance.',
                   nz_cache_mode=args.nz_cache,
                   runs=[], binaries=[], arms={})
    receipt_path = out/'receipt.json'

    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')

    def run(label, command, env, backend=None, encodes=0):
        path = out/(label+'.log')
        with path.open('w') as log:
            try:
                code = subprocess.run([str(x) for x in command], env=env, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code = 124; log.write('\nVALIDATOR_TIMEOUT\n')
            except OSError as error:
                code = 127; log.write(str(error)+'\n')
        row = dict(label=label, command=[str(x) for x in command], exit=code,
                   log_sha256=fixture.digest(path), passed=False)
        receipt['runs'].append(row); save()
        if code != 0:
            raise ValueError(f'{label} failed with exit{code}')
        text = path.read_text(errors='replace')
        if backend:
            row['placement'] = assess_placement(text, backend, args.device if backend == args.backend else 0, encodes,
                                                       ('on' if backend == 'cann' and args.nz_cache == 'on' else 'off')
                                                       if args.nz_cache != 'unchanged' else None)
        row['passed'] = True; save()
        return text

    save()
    try:
        archive = args.fixture_zip.resolve(strict=True)
        if archive.stat().st_size != FIXTURE_BYTES or fixture.digest(archive) != FIXTURE_SHA256:
            raise ValueError('Fixture archive differs from accepted immutable bytes')
        receipt['fixture'] = dict(bytes=FIXTURE_BYTES, sha256=FIXTURE_SHA256); save()
        data = out/'fixture'; unpack(archive, data)
        sys.path.insert(0, str(ROOT/'ggml/gguf-py'))
        import numpy as np
        from gguf import GGUFReader
        original = json.loads((data/'receipt.json').read_text())
        model = data/'moonshine-tiny.gguf'; audio = data/'audio.wav'
        if fixture.digest(model) != original['model']['sha256'] or fixture.digest(audio) != original['sample']['sha256']:
            raise ValueError('Extracted model/sample identity mismatch')
        reference = {tensor.name: tensor for tensor in GGUFReader(str(data/'reference.gguf')).tensors}
        pcm = fixture.audio(audio)
        expected = np.load(data/'encoder-original.npy')
        if expected.shape != (457, 288) or not np.array_equal(reference['raw_audio'].data.reshape(-1), pcm):
            raise ValueError('Reference shape/PCM mismatch')
        if not np.array_equal(reference['encoder_output'].data.reshape(expected.shape), expected):
            raise ValueError('Original reference arrays disagree')
        build = args.build_dir.resolve(strict=True)
        lib = build/('src/libcrispasr.dylib' if sys.platform == 'darwin' else 'src/libcrispasr.so'); cli = build/'bin/crispasr'; diff = build/'bin/crispasr-diff'
        cache = build/'CMakeCache.txt'; commands = build/'compile_commands.json'
        for path in (lib, cli, diff, cache, commands):
            if not path.is_file():
                raise ValueError(f'Required build evidence missing: {path.name}')
        for path in (cache, commands):
            shutil.copyfile(path, out/path.name)
        # Preserve actual executable/library bytes, deduplicated by full SHA.
        library_glob = '*.dylib*' if sys.platform == 'darwin' else '*.so*'
        binaries = list(build.rglob(library_glob))+[cli, diff]
        blobs = out/'binaries'; blobs.mkdir()
        for path in binaries:
            if not path.is_file():
                continue
            digest = fixture.digest(path); target = blobs/digest
            if not target.exists():
                shutil.copyfile(path, target)
            if fixture.digest(target) != digest:
                raise ValueError('Binary snapshot identity mismatch')
            receipt['binaries'].append(dict(path=str(path.relative_to(build)), bytes=path.stat().st_size,
                                             sha256=digest, snapshot=str(target.relative_to(out))))
        receipt['harness_source'] = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip()
        receipt['build_source_scope'] = 'Actual binaries/cache/commands/ldd retained; checkout revision is harness identity, not proof a binary was compiled from that revision.'
        save()
        env = dict(os.environ, TMPDIR=str(out), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        dirs = sorted({str(path.parent) for path in build.rglob(library_glob) if path.is_file()})
        env['LD_LIBRARY_PATH'] = ':'.join(dirs+([env['LD_LIBRARY_PATH']] if env.get('LD_LIBRARY_PATH') else []))
        for name in list(env):
            if name.startswith('CRISPASR_MOONSHINE_') or name in ('CRISPASR_SCHED_PROFILE','GGML_SCHED_DEBUG'):
                env.pop(name)
        if args.nz_cache != 'unchanged':
            env['GGML_CANN_WEIGHT_NZ_CACHE'] = 'on'
            env['GGML_CANN_WEIGHT_NZ'] = args.nz_cache
        for label, binary in (('lib',lib),('cli',cli),('diff',diff)):
            loader = ['otool','-L'] if sys.platform == 'darwin' else ['ldd']
            text = run(('otool-' if sys.platform == 'darwin' else 'ldd-')+label, loader+[binary], env)
            if 'not found' in text:
                raise ValueError('Unresolved native library dependency')
        for backend in (['cpu'] if args.backend == 'cpu' else ['cpu',args.backend]):
            arm = out/backend; arm.mkdir()
            selected = dict(env, CRISPASR_MOONSHINE_REQUIRE_BACKEND=backend,
                            CRISPASR_MOONSHINE_REQUIRE_DEVICE=str(args.device if backend == args.backend else 0))
            if backend == 'cann' and args.nz_cache == 'on':
                selected['CRISPASR_MOONSHINE_REQUIRE_NZ_CACHE'] = '1'
            if backend == 'cann':
                if not re.search(r'^GGML_CANN:(?:BOOL|UNINITIALIZED)=ON$', cache.read_text(), re.M):
                    raise ValueError('Supplied build does not enable CANN')
                # Read-only device metadata; absence is retained, not invented.
                try:
                    with (out/'npu-smi.log').open('w') as log:
                        code = subprocess.run(['npu-smi','info'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=30).returncode
                    receipt['npu_smi_exit'] = code
                except (OSError, subprocess.TimeoutExpired) as error:
                    receipt['npu_smi_error'] = str(error)
                save()
            diff_log = run(backend+'-diff', [diff,'moonshine',model,data/'reference.gguf',audio], selected, backend, 1)
            if not re.search(r'summary: 1 pass, 0 fail, 0 skip', diff_log) or 'magnitude=PASS' not in diff_log:
                raise ValueError('Incomplete encoder diff acceptance')
            native = [sys.executable,fixture.__file__,'--native','--model',model,'--lib',lib,'--audio',audio,'--output',arm]
            if backend != 'cpu':
                native.append('--use-gpu')
            run(backend+'-native', native, selected, backend, 6)
            actual = np.load(arm/'encoder-native-0.npy')
            if not np.array_equal(actual, np.load(arm/'encoder-native-1.npy')):
                raise ValueError('Encoder repetitions disagree')
            metrics = fixture.compare(expected, actual)
            if backend != 'cpu':
                receipt['cpu_'+backend+'_encoder_metrics'] = fixture.compare(np.load(out/'cpu/encoder-native-0.npy'), actual)
            sessions = json.loads((arm/'session-outputs.json').read_text())
            if set(sessions) != {'auto','explicit'} or any(len(texts) != 2 for texts in sessions.values()):
                raise ValueError('Incomplete Session dispatch/repetition')
            if any(fixture.words(text) != fixture.words(original['original_transcript']) for texts in sessions.values() for text in texts):
                raise ValueError('Session transcript differs from original')
            cli_texts = []
            for i in range(2):
                prefix = arm/f'cli-{i}'
                command = [cli,'--backend','moonshine','-m',model,'-f',audio,'-t','4','-l','en','--no-punctuation','-osrt','-of',prefix]
                command += ['--no-gpu'] if backend == 'cpu' else ['--gpu-backend',backend]
                run(f'{backend}-cli-{i}', command, selected, backend, 1)
                text = ' '.join(line for line in prefix.with_suffix('.srt').read_text().splitlines()
                                if line.strip() and not line.strip().isdigit() and '-->' not in line)
                if fixture.words(text) != fixture.words(original['original_transcript']):
                    raise ValueError('CLI transcript differs from original')
                cli_texts.append(text)
            reuse_rows = []
            if args.exercise_reuse:
                command = [sys.executable, ROOT/'tools/ci-heavy/moonshine_live_reuse.py', '--model',model,
                           '--lib',lib, '--audio',audio, '--output',arm/'reuse']
                if backend != 'cpu':
                    command.append('--use-gpu')
                run(backend+'-reuse', command, selected, backend, 3)
                reuse_rows = json.loads((arm/'reuse/reuse.json').read_text())
                if reuse_rows[0]['words'] != fixture.words(original['original_transcript']):
                    raise ValueError('Reuse short transcript differs from original')
                if backend != 'cpu':
                    cpu_reuse = json.loads((out/'cpu/reuse/reuse.json').read_text())
                    if [row['words'] for row in reuse_rows] != [row['words'] for row in cpu_reuse]:
                        raise ValueError('CPU/device short-long-short transcripts differ')
            receipt['arms'][backend] = dict(passed=True, encoder_metrics=metrics, sessions=sessions,
                                           cli_transcripts=cli_texts, reuse=reuse_rows, session_bytes_exact=all(text == original['original_transcript'] for texts in sessions.values() for text in texts))
            save()
        receipt['passed'] = True; save(); print('FULL_MODEL_VALIDATION_PASS', args.backend)
        return 0
    except Exception as error:
        receipt['error'] = str(error); save(); print('FULL_MODEL_VALIDATION_FAIL', str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
