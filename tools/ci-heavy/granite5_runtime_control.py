#!/usr/bin/env python3
"""Actual GPU F16 positive control for a new source-matched runtime capsule.

Uses an already accepted immutable CPU regression and published F16 weights.
This establishes no Q4 acceptance and publishes no production model.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from granite5_cuda_acceptance import qualify_runtime
from granite5_cuda_preflight import run_preflight
from granite5_execution import validate_case
from granite5_native_acceptance import sha
from granite5_regression import check_pins, check_published_profiles

ROOT = Path(__file__).resolve().parents[2]
CPU_SOURCE = '5b5278a0f030a8b57e93a38a41bdb8fb1a4f7cca'
CPU_RUN = 40012494531
CPU_REVISION = '0fa4dbaf1041069dbf5fecacbb41ac1149af8767'
CPU_PREFIX = f'regression/{CPU_SOURCE}/{CPU_RUN}/cpu'
CPU_SHA = '98bc4fdee1283f9ed1df358583deb78e02b5dd10b6a54de05e7b14fa92df3ccb'
STAGING = 'cstr/crispasr-granite5-validation-public'


def check_cpu_control(cpu, pins, manifest, receipt_hash, pins_hash):
    if not (receipt_hash == CPU_SHA and pins_hash == cpu['pins_sha256'] and
            cpu['complete'] and cpu['cpu_acceptance'] and not cpu['cuda_acceptance'] and
            set(cpu['profiles']) == {'f16'} and cpu['profiles']['f16']['accepted'] and
            cpu['profiles']['f16']['complete'] and cpu['profiles']['f16']['device'] == 'cpu' and
            pins['profiles'] == ['f16'] and manifest['source'] == CPU_SOURCE and manifest['run'] == CPU_RUN and
            manifest['receipt_sha256'] == CPU_SHA and manifest['source_and_terminal_status_verified']):
        raise ValueError('The complete immutable CPU F16 control is required')
    check_pins(pins)


def main():
    from huggingface_hub import HfApi, hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'scratch', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    def evidence(name):
        return Path(hf_hub_download(STAGING, CPU_PREFIX + '/' + name,
                    revision=CPU_REVISION, local_dir=args.scratch / 'cpu'))
    cpu_path, pins_path, manifest_path = (evidence(name) for name in
                                         ('receipt.json', 'pins.json', 'source-manifest.json'))
    cpu, pins, manifest = (json.loads(path.read_text()) for path in (cpu_path, pins_path, manifest_path))
    check_cpu_control(cpu, pins, manifest, sha(cpu_path), sha(pins_path))
    # The original regression predates the runtime manifest field. Compare
    # entire Git trees against its pinned source, including the GGML gitlink.
    subprocess.run(['git', 'fetch', '--depth=1', 'origin', CPU_SOURCE], cwd=ROOT, check=True)
    for directory in ('src', 'examples/cli', 'ggml', 'python/crispasr', 'samples/jfk.wav'):
        before = subprocess.check_output(['git', 'rev-parse', CPU_SOURCE + ':' + directory], cwd=ROOT)
        after = subprocess.check_output(['git', 'rev-parse', 'HEAD:' + directory], cwd=ROOT)
        if before != after:
            raise ValueError('Native control sources differ from the accepted CPU source')
    actual = {str(p.relative_to(ROOT)): sha(p) for directory in ('src', 'examples/cli')
              for p in (ROOT / directory).rglob('*') if p.is_file() and p.suffix in {'.cpp', '.c', '.h', '.hpp'}}
    provenance = qualify_runtime(args.bundle, dict(runtime_sources=actual))
    check_published_profiles(HfApi().model_info(pins['repo'], revision=pins['revision']).card_data, ['f16'])
    receipt = dict(complete=False, runtime_control_pass=False, q4_acceptance=False,
        production_release=False, cpu_source=CPU_SOURCE, cpu_run=CPU_RUN,
        cpu_revision=CPU_REVISION, cpu_receipt_sha256=CPU_SHA,
        runtime_source=provenance['source'], runtime_provenance_sha256=sha(args.bundle / 'provenance.json'),
        native_sources_match_cpu=True, native_source_files=len(actual))
    def save():
        (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    preflight = run_preflight(args.bundle, args.out / 'runtime-preflight.json')
    if not preflight['runtime_preflight_pass']:
        raise ValueError('Actual GPU loader preflight failed')
    receipt['hardware'] = preflight['gpu0']
    save()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', LD_LIBRARY_PATH=preflight['ld_library_path'],
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
    if any(k.startswith(('CRISPASR_GRANITE_', 'GRANITE_')) or k in
           ('CRISPASR_N_GPU_LAYERS', 'CRISPASR_IMATRIX_OUT', 'CRISPASR_ACTDUMP_OUT') for k in environment):
        raise ValueError('Inherited runtime overrides invalidate the F16 control')
    def fetch(record):
        path = Path(hf_hub_download(record['repo'], record['path'], repo_type=record.get('repo_type'),
                                   revision=record['revision'], local_dir=args.scratch / 'models'))
        if sha(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError('Pinned control input differs')
        return path
    model_record = dict(repo=pins['repo'], revision=pins['revision'],
                        path=pins['models']['f16']['filename'], **pins['models']['f16'])
    model, reference = fetch(model_record), fetch(pins['reference'])
    receipt['validation'] = validate_case(model, reference, pins['instruction'], args.bundle,
        args.bundle / 'python', ROOT / 'samples/jfk.wav', args.out / 'f16', 'cuda', environment, 'f16')
    receipt.update(complete=True, runtime_control_pass=True)
    save()
    print('New runtime passes the full actual GPU F16 control; Q4 remains unqualified', flush=True)


if __name__ == '__main__':
    main()
