#!/usr/bin/env python3
"""Full actual-CUDA acceptance of one immutable CPU-approved Q4_K recipe."""
import argparse
import json
import os
from pathlib import Path
import shutil

from granite5_cuda_acceptance import ORACLE, ORACLE_METADATA, qualify_runtime
from granite5_cuda_preflight import run_preflight
from granite5_execution import resources, validate_case
from granite5_native_acceptance import sha
from granite5_q4_recipe import RECIPES

ROOT = Path(__file__).resolve().parents[2]


def approved_case(cpu, name):
    row = cpu['cases'][name]
    if not (cpu['complete'] and cpu['diagnostic_only'] and not cpu['production_release'] and
            not cpu['cuda_acceptance'] and row['cpu_acceptance'] and
            row['validation']['accepted'] and row['validation']['complete'] and
            row['validation']['device'] == 'cpu' and row['elements_by_type'].get('Q4_K', 0) > 0 and
            row['rules'] == RECIPES[name]):
        raise ValueError('A complete CPU-approved, actual Q4_K recipe is required')
    return row


def main():
    from huggingface_hub import hf_hub_download
    import subprocess
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'cpu-receipt', 'scratch', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--case', choices=RECIPES, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    cpu = json.loads(args.cpu_receipt.read_text())
    selected = approved_case(cpu, args.case)
    provenance = qualify_runtime(args.bundle, cpu)
    receipt = dict(complete=False, cuda_acceptance=False, production_release=False,
                   selected_recipe=args.case, cpu_source=cpu['source'], cpu_run=cpu['run'],
                   cpu_receipt_sha256=sha(args.cpu_receipt), runtime_source=provenance['source'],
                   runtime_provenance_sha256=sha(args.bundle / 'provenance.json'),
                   compiler_header_components=provenance['compiler_header_components'], profiles={})
    def save():
        (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    def fetch(record):
        path = Path(hf_hub_download(record['repo'], record['path'], repo_type=record['repo_type'],
                                   revision=record['revision'], local_dir=args.scratch / 'inputs'))
        if sha(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError('Pinned CPU-approved input bytes differ')
        return path
    save()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    preflight = run_preflight(args.bundle, args.out / 'runtime-preflight.json')
    if not preflight['runtime_preflight_pass']:
        raise ValueError('Actual CUDA loader preflight failed')
    environment = dict(os.environ, LD_LIBRARY_PATH=preflight['ld_library_path'],
        GGML_BACKEND_PATH=str(args.bundle), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
    if any(k.startswith(('CRISPASR_GRANITE_', 'GRANITE_')) or k in
           ('CRISPASR_N_GPU_LAYERS', 'CRISPASR_IMATRIX_OUT', 'CRISPASR_ACTDUMP_OUT') for k in environment):
        raise ValueError('Inherited runtime overrides invalidate production-path parity')
    receipt['hardware'] = preflight['gpu0']
    receipt['integer_mmq_hardware_capable'] = float(preflight['gpu0']['compute_capability']) >= 6.1
    receipt['mmq_limit'] = 'Actual GPU qualification; no isolated MMQ performance A/B claim'
    save()
    # Keep the same original rotary oracle and four wrong-convention controls.
    oracle, metadata = fetch(ORACLE), fetch(ORACLE_METADATA)
    shutil.copyfile(metadata, Path(str(oracle) + '.json'))
    snapshot = resources('cuda')
    receipt['rope_resources'] = snapshot
    save()
    if (snapshot['memory_available_kib'] < 5 * 1024 * 1024 or
            snapshot['load_average'][0] > 2 * snapshot['cpus']):
        raise RuntimeError('Insufficient headroom before CUDA component inference')
    command = [sys.executable, ROOT / 'tools/ci-heavy/granite5_rope_parity.py',
               '--probe', args.bundle / 'granite5-rope-probe', '--oracle', oracle,
               '--out', args.out / 'rope', '--device', 'cuda']
    # Mirror the established CUDA acceptance invocation; this probe has no
    # trained weights and does not substitute for full model execution below.
    with (args.out / 'rope.log').open('w') as log:
        subprocess.run(list(map(str, command)), env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=600)
    receipt['rope_parity'] = json.loads((args.out / 'rope/receipt.json').read_text())
    if not (receipt['rope_parity']['complete'] and receipt['rope_parity']['device'] == 'cuda'):
        raise ValueError('Actual CUDA rotary controls did not complete')
    save()
    model = fetch(selected['file'])
    reference = fetch(cpu['reference'])
    receipt['profiles']['q4_k'] = dict(file=selected['file'], rules=selected['rules'], accepted=False)
    save()
    validation = validate_case(model, reference, cpu['instruction'], args.bundle, args.bundle / 'python',
                               ROOT / 'samples/jfk.wav', args.out / 'q4_k', 'cuda', environment, 'q4_k')
    receipt['profiles']['q4_k'].update(validation=validation, accepted=True)
    receipt.update(complete=True, cuda_acceptance=True)
    save()
    print('Granite5 Q4_K: full original-reference stages, tokens, Session and CLI pass actual CUDA', flush=True)


if __name__ == '__main__':
    main()
