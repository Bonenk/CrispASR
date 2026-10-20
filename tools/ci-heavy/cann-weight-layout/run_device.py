#!/usr/bin/env python3
"""Build an exact clean revision and retain bounded physical Ascend layout proof.

No CPU fallback can satisfy this request. Full-model runs are optional, after the
small probe. CPU/Metal controls use validate_cann_model.py separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_URL = ('https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/'
               '9cf29ec347a91d7a96177cfdc048bb235f8c3289/'
               'cann-20261014-model-strict/complete-hosted-strict-fixture.zip')
FIXTURE_SHA = '7d63d2a5b90eb77843aff25107b1d1f70bafcfc1c8f73075e2f5a9a3f87d5e84'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4194304), b''):
            result.update(chunk)
    return result.hexdigest()


def values(path, code):
    raw = Path(path).read_bytes()
    size = struct.calcsize('<'+code)
    if not raw or len(raw) % size:
        raise ValueError('Empty or partial numeric payload: '+Path(path).name)
    return struct.unpack('<'+str(len(raw)//size)+code, raw)


def assess_probe(log, directory, require_nz, require_acl=False, backend='cann'):
    """Recompute every matrix product independently from retained inputs."""
    if 'PASS computes=576 cases=48' not in log or 'FAIL:' in log:
        raise ValueError('Incomplete or failed layout probe')
    if backend == 'cann' and not re.search(r'^backend=CANN\d+ device=\S+', log, re.M):
        raise ValueError('Missing actual CANN backend identity')
    if backend == 'cpu' and not re.search(r'^backend=CPU device=\S+', log, re.M):
        raise ValueError('Missing CPU control identity')
    captures = re.findall(r'^ACL captures=(\d+) replays=(\d+)$', log, re.M)
    if len(captures) != 1 or (require_acl and not all(int(n) > 0 for n in captures[0])):
        raise ValueError('Required ACL capture/replay not established')
    cases = re.findall(r'^case=(\d+) type=(f16|f32) k=(\d+) name=(\S+) eligible=([01]) completed$', log, re.M)
    caches = re.findall(r'^cache_case=(\d+) bytes=(\d+) selections=(\d+)$', log, re.M)
    if len(cases) != 48 or {int(row[0]) for row in cases} != set(range(48)) or len(caches) != 48:
        raise ValueError('Missing or duplicate case evidence')
    cache_by_case = {int(index): (int(size), int(selections)) for index, size, selections in caches}
    if set(cache_by_case) != set(range(48)):
        raise ValueError('Missing cache readiness/selection evidence')
    metadata = {}
    for index, dtype, k, name, eligible in cases:
        index, k = int(index), int(k)
        size, selections = cache_by_case[index]
        expected = require_nz and eligible == '1'
        if (expected and (size <= 0 or selections <= 0)) or (not expected and (size or selections)):
            raise ValueError('NZ arm silently missed cache selection or ND arm used it')
        metadata[index] = (dtype, k)
    counts = {index: [] for index in range(48)}
    payloads = sorted(Path(directory).glob('*-actual.f32'))
    if len(payloads) != 576:
        raise ValueError('Missing actual matrix payloads')
    for path in payloads:
        stem = path.name.removesuffix('-actual.f32')
        index = int(stem.split('-')[0])
        dtype, k = metadata[index]
        weights = values(path.parent/(stem+'-weights.bin'), 'f' if dtype == 'f32' else 'e')
        inputs = values(path.parent/(stem+'-input.f32'), 'f')
        actual = values(path, 'f')
        declared = values(path.parent/(stem+'-expected.f32'), 'f')
        if len(weights) != k*48 or len(inputs) % k:
            raise ValueError('Invalid independent-oracle input shape')
        columns = len(inputs)//k
        if columns not in (1, 5) or len(actual) != 48*columns:
            raise ValueError('Invalid output shape')
        oracle = tuple(sum(weights[j+k*r]*inputs[j+k*n] for j in range(k))
                       for n in range(columns) for r in range(48))
        if actual != oracle or declared != oracle:
            raise ValueError('Independent scalar oracle mismatch: '+stem)
        counts[index].append(columns)
    if any(sorted(columns) != [1]*8+[5]*4 for columns in counts.values()):
        raise ValueError('Missing repeated/change-shape matrix cases')
    return dict(passed=True, independently_checked_computes=576, cases=48,
                acl_captures=int(captures[0][0]), acl_replays=int(captures[0][1]),
                nz_required=require_nz, scope='Physical probe' if backend == 'cann' else 'CPU harness control')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Exact immutable 40-hex CrispASR commit')
    parser.add_argument('--chip', required=True, choices=('310p', '910b'))
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--with-model', action='store_true', help='Also run the pinned F32 Moonshine CLI/Session/diff matrix')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.source) or args.device < 0 or args.jobs < 1:
        parser.error('Exact 40-hex source, nonnegative device and positive jobs required')
    if platform.system() != 'Linux':
        parser.error('Physical CANN procedure requires a Linux Ascend host')
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error('Output/build evidence must live outside the clean source checkout')
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(passed=False, requested_source=args.source, chip=args.chip, device=args.device,
                   runner_sha256=digest(__file__), commands=[], arms={},
                   scope='Bounded NZ cache correctness; no speed claim. Full-model/ACL scopes are recorded separately.')
    def save():
        (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    def run(label, command, env=None):
        with (out/(label+'.log')).open('w') as log:
            code = subprocess.run([str(x) for x in command], env=env, cwd=ROOT,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        receipt['commands'].append(dict(label=label, exit=code, log_sha256=digest(out/(label+'.log'))))
        save()
        if code:
            raise ValueError(label+' failed; retain the complete proof directory')
        return (out/(label+'.log')).read_text(errors='replace')
    save()
    try:
        source = run('source', ['git','rev-parse','HEAD']).strip()
        clean = run('source-status', ['git','status','--porcelain']).strip()
        module = run('ggml-source', ['git','-C',ROOT/'ggml','rev-parse','HEAD']).strip()
        module_clean = run('ggml-status', ['git','-C',ROOT/'ggml','status','--porcelain']).strip()
        expected_module = run('ggml-pin', ['git','ls-tree','HEAD','ggml']).split()[2]
        if source != args.source or clean or module_clean or module != expected_module:
            raise ValueError('Checkout/submodule do not match the exact clean source request')
        receipt.update(source=source, ggml=module)
        run('host-lifecycle', [sys.executable,ROOT/'tests/test_cann_nz_lifecycle.py','-v'])
        run('host-policy', [sys.executable,ROOT/'tests/test_cann_weight_layout.py','-v'])
        run('device-info', ['npu-smi','info'])
        build = out/'build'
        source_dir = ROOT if args.with_model else ROOT/'tools/ci-heavy/cann-weight-layout'
        command = ['cmake','-S',source_dir,'-B',build,'-DCMAKE_BUILD_TYPE=Release',
                   '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DGGML_CANN=ON',
                   '-DSOC_TYPE=ascend'+args.chip,'-DUSE_ACL_GRAPH='+('on' if args.chip=='910b' else 'off')]
        if args.with_model:
            command += ['-DCRISPASR_BUILD_CANN_LAYOUT_PROBE=ON','-DCRISPASR_BUILD_TESTS=OFF','-DCRISPASR_BUILD_SERVER=OFF']
        run('configure', command)
        targets = ['cann-weight-layout'] + (['crispasr-cli','crispasr-lib','crispasr-diff'] if args.with_model else [])
        run('build', ['cmake','--build',build,'--target',*targets,'-j',args.jobs])
        compiler_commands = json.loads((build/'compile_commands.json').read_text())
        cann_commands = [row for row in compiler_commands if Path(row['file']).name == 'ggml-cann.cpp']
        chip_define = 'ASCEND_310P' if args.chip == '310p' else 'ASCEND_910B'
        if not cann_commands:
            raise ValueError('No actual CANN compiler command retained')
        for row in cann_commands:
            command_text = row.get('command') or ' '.join(row.get('arguments', []))
            if not re.search(r'(?:^|\s)-D'+chip_define+r'(?:=1)?(?:\s|$)', command_text):
                raise ValueError('Compiler command lacks requested chip definition')
            acl_defined = bool(re.search(r'(?:^|\s)-DUSE_ACL_GRAPH(?:=1)?(?:\s|$)', command_text))
            if acl_defined != (args.chip == '910b'):
                raise ValueError('Actual ACL compiler definition differs from requested mode')
        receipt['compiler_definitions_checked'] = True
        binary = build/'bin/cann-weight-layout'
        receipt['probe_binary_sha256'] = digest(binary)
        for graph in (['off','on'] if args.chip == '910b' else ['off']):
            for nz in ('off','on'):
                label = 'probe-'+nz+'-acl-'+graph
                env = dict(os.environ, GGML_CANN_WEIGHT_NZ_CACHE='on', GGML_CANN_WEIGHT_NZ=nz, GGML_CANN_ACL_GRAPH=graph)
                command = [binary,'--backend','cann','--device',args.device,'--out',out/label,
                           '--require-nz' if nz=='on' else '--require-nd']
                if graph=='on':
                    command.append('--require-acl-replay')
                log = run(label, command, env)
                receipt['arms'][label] = assess_probe(log, out/label, nz=='on', graph=='on')
                save()
        if args.with_model:
            archive = out/'moonshine-fixture.zip'
            with urllib.request.urlopen(FIXTURE_URL,timeout=120) as stream, archive.open('wb') as destination:
                while data := stream.read(4194304):
                    destination.write(data)
            if archive.stat().st_size != 108495214 or digest(archive) != FIXTURE_SHA:
                raise ValueError('Immutable Moonshine fixture identity mismatch')
            for nz in ('off','on'):
                label = 'model-'+nz
                env = dict(os.environ, GGML_CANN_ACL_GRAPH='off')
                run(label, [sys.executable,ROOT/'tools/validate_cann_model.py','--fixture-zip',archive,
                            '--build-dir',build,'--backend','cann','--device',args.device,'--nz-cache',nz,
                            '--exercise-reuse','--out',out/label],env)
                result = json.loads((out/label/'receipt.json').read_text())
                if not result['passed']:
                    raise ValueError('Full-model arm did not pass')
                receipt['arms'][label] = dict(passed=True, receipt_sha256=digest(out/label/'receipt.json'))
        receipt['files'] = {str(path.relative_to(out)): dict(bytes=path.stat().st_size, sha256=digest(path))
                            for path in out.rglob('*') if path.is_file() and build not in path.parents and path != out/'receipt.json'}
        receipt['passed'] = True
        save()
        print('ASCEND_LAYOUT_VALIDATION_PASS; retain complete proof, not only receipt.json')
        return 0
    except Exception as error:
        receipt['error'] = str(error)
        save()
        print('ASCEND_LAYOUT_VALIDATION_FAIL', error)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
