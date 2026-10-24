#!/usr/bin/env python3
"""Hosted Q4_K recipe diagnostics using fresh crispasr-quantize output.

Fixed precision arms and disjoint CC0 calibration; no numerical gate changes.
Candidates are checkpointed before execution and never published as accepted
models by this worker. CPU acceptance is per case; actual CUDA remains separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import wave
import zipfile

import numpy as np

from granite5_execution import run, validate_case
from granite5_native_acceptance import sha
from granite5_regression import check_pins
from granite5_result_audit import metrics, token_comparison

ROOT = Path(__file__).resolve().parents[2]
STAGING = 'cstr/crispasr-granite5-validation-public'
HEAD = r'^output\.weight$=f16'
RECIPES = {
    'body': [HEAD],
    'body-imatrix': [HEAD],
    'attention-f16': [HEAD, r'^blk\.[0-9]+\.attn_(q|k|v|output)\.weight$=f16'],
    'ffn-f16': [HEAD, r'^blk\.[0-9]+\.ffn_(gate|up|down)\.weight$=f16'],
    'value-down-q6': [HEAD, r'^blk\.[0-9]+\.(attn_v|ffn_down)\.weight$=q6_k'],
    'value-down-f16': [HEAD, r'^blk\.[0-9]+\.(attn_v|ffn_down)\.weight$=f16'],
    'edges-f16': [HEAD, r'^blk\.(0|1|2|3|36|37|38|39)\..*\.weight$=f16'],
}
RUNTIME = dict(repo=STAGING, repo_type='model',
    path='runtime/e2368b9e6ba94368a31798d003dca3b61e0bda10/granite5-cuda-validation.tar.gz',
    revision='d597fd6b5cde7126f181fdce08b5d85499c54cb9', bytes=230938776,
    sha256='132ba8f3e82abe6dcc6b66de808883d5f634997f65210a435ef88689ef7473d9')
CALIBRATION = dict(repo='cstr/crispasr-imatrix-calib', repo_type='dataset',
    revision='06258ecef2dfed2be52ced573ec0dffc2443fabc',
    path='index-echo-en-zh-20261009/index-echo-en-zh-calibration.zip', bytes=6446200,
    sha256='44680f20e390cc0c6bf5bfcf3e284a1a3feabf118bd1a96e0ab510c682bf2188')
MANIFEST = dict(repo=CALIBRATION['repo'], repo_type='dataset', revision=CALIBRATION['revision'],
    path='index-echo-en-zh-20261009/corpus-manifest.json',
    sha256='d05c4483414ebf55f9a84a3322bb9e308469e73817970371e0abbc358f71e471')


def pcm_sha(path):
    with wave.open(str(path), 'rb') as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
            raise ValueError('Calibration must be mono 16 kHz PCM16')
        return hashlib.sha256(audio.readframes(audio.getnframes())).hexdigest()


def tensor_inventory(original, candidate, rules):
    """Verify overrides, unchanged frontend/norms and actual Q4_K payloads."""
    from gguf import GGMLQuantizationType as Q, GGUFReader
    before, after = GGUFReader(original), GGUFReader(candidate)
    expected = {t.name: t for t in before.tensors}
    if set(expected) != {t.name for t in after.tensors}:
        raise ValueError('Candidate tensor names differ from the qualified F16')
    inventory, counts = {}, {}
    def logical_shape(shape):
        # ggml_n_dims() canonicalizes trailing unit axes when the C++
        # quantizer creates its output descriptors. They carry no stride or
        # element-count change; actual non-unit dimension changes still fail.
        result = list(shape)
        while len(result) > 1 and result[-1] == 1:
            result.pop()
        return result
    for tensor in after.tensors:
        source = expected[tensor.name]
        if logical_shape(tensor.shape) != logical_shape(source.shape):
            raise ValueError(f'Candidate tensor shape differs: {tensor.name}: '
                             f'{source.shape.tolist()} -> {tensor.shape.tolist()}')
        kind = tensor.tensor_type.name
        counts[kind] = counts.get(kind, 0) + tensor.n_elements
        matched = next((rule.rsplit('=', 1)[1] for rule in rules
                        if re.search(rule.rsplit('=', 1)[0], tensor.name)), None)
        if matched and len(tensor.shape) >= 2 and kind.lower() != matched:
            raise ValueError(f'Quantizer override not applied to {tensor.name}')
        preserved = (tensor.name.startswith(('enc.', 'proj.')) or
                     tensor.name == 'token_embd.weight' or len(tensor.shape) < 2)
        if preserved and (tensor.tensor_type != source.tensor_type or
                          hashlib.sha256(tensor.data).digest() != hashlib.sha256(source.data).digest()):
            raise ValueError(f'Protected source payload changed: {tensor.name}')
        inventory[tensor.name] = dict(type=kind, shape=tensor.shape.tolist(),
                                      sha256=hashlib.sha256(tensor.data).hexdigest())
    if counts.get(Q.Q4_K.name, 0) == 0:
        raise ValueError('A Q4_K candidate must contain real Q4_K weights')
    return dict(tensors=inventory, elements_by_type=counts)


def build_quantizer(bundle, scratch, execute):
    build = scratch / 'build'
    project = scratch / 'quantizer-project'
    project.mkdir()
    # Compile the actual quantizer and its actual format-parser helper. The
    # top-level "common" dependency also builds the full C ABI unnecessarily
    # for this standalone tool; the qualified GGML base exports every symbol
    # these two sources use. Match the repository's tensor-name ABI explicitly.
    (project / 'CMakeLists.txt').write_text(f'''cmake_minimum_required(VERSION 3.20)
project(granite_q4_quantizer LANGUAGES CXX)
find_package(Threads REQUIRED)
add_executable(crispasr-quantize "{ROOT}/examples/crispasr-quantize/main.cpp" "{ROOT}/examples/common-ggml.cpp")
target_include_directories(crispasr-quantize PRIVATE "{ROOT}/examples" "{ROOT}/ggml/include")
target_compile_features(crispasr-quantize PRIVATE cxx_std_17)
target_compile_definitions(crispasr-quantize PRIVATE GGML_MAX_NAME=128)
target_link_libraries(crispasr-quantize PRIVATE "{bundle}/libggml-base.so" Threads::Threads)
set_target_properties(crispasr-quantize PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/bin")
''')
    execute(['cmake', '-S', project, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache'], 'configure-quantizer')
    execute(['cmake', '--build', build, '--target', 'crispasr-quantize', '-j2'], 'build-quantizer')
    return build / 'bin/crispasr-quantize'


def main():
    from gguf import GGUFReader
    from huggingface_hub import HfApi, hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=RECIPES, default=list(RECIPES))
    parser.add_argument('--reuse-calibration-run', type=int)
    parser.add_argument('--reuse-calibration-source')
    args = parser.parse_args()
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'granite5-q4-recipe'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
    environment = dict(os.environ)
    if any(k.startswith(('CRISPASR_GRANITE_', 'GRANITE_')) or k in
           ('CRISPASR_QUANT_LMHEAD', 'CRISPASR_IMATRIX_OUT', 'CRISPASR_ACTDUMP_OUT') for k in environment):
        raise ValueError('Inherited quantizer/runtime overrides invalidate the controlled recipes')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    receipt = dict(source=source, run=int(os.environ['GITHUB_RUN_ID']), complete=False,
                   diagnostic_only=True, production_release=False, cuda_acceptance=False,
                   requested_cases=args.cases, cases={})
    def save():
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    def execute(command, label, env=None):
        return run(command, label, out, env or environment, 'cpu', timeout=7200)
    def fetch(record, directory='inputs'):
        path = Path(hf_hub_download(record['repo'], record['path'], repo_type=record.get('repo_type'),
                                   revision=record['revision'], local_dir=scratch / directory))
        if sha(path) != record['sha256'] or ('bytes' in record and path.stat().st_size != record['bytes']):
            raise ValueError('Immutable input content differs')
        return path
    save()
    pins = json.loads((ROOT / 'tests/regression/granite5.json').read_text())
    check_pins(pins)
    receipt.update(reference=pins['reference'], instruction=pins['instruction'],
                   quantizer_source_sha256=sha(ROOT / 'examples/crispasr-quantize/main.cpp'),
                   runtime_sources={str(p.relative_to(ROOT)): sha(p)
                       for folder in ('src', 'examples/cli') for p in (ROOT / folder).rglob('*')
                       if p.is_file() and p.suffix in {'.cpp', '.c', '.h', '.hpp'}})
    f16_record = dict(repo=pins['repo'], revision=pins['revision'],
                      path=pins['models']['f16']['filename'], **pins['models']['f16'])
    f16 = fetch(f16_record)
    reference = fetch(pins['reference'])
    receipt['f16'] = f16_record
    api = HfApi()
    card = api.model_info(STAGING).card_data
    if card['license'] != 'apache-2.0' or card['crispasr_native_acceptance'] is not False:
        raise ValueError('Diagnostic license/scope differs')
    if not shutil.which('ccache') or not shutil.which('ninja'):
        execute(['sudo', 'apt-get', 'update'], 'apt-update')
        execute(['sudo', 'apt-get', 'install', '-y', 'ccache', 'ninja-build'], 'apt-tools')
    # The already qualified capsule contains the same CPU backend and native
    # sources. Reuse it on this CPU runner; no GPU execution is requested.
    # Only the quantizer needs a fresh build, avoiding another full runtime build.
    runtime = fetch(RUNTIME, 'runtime-input')
    with tarfile.open(runtime) as archive:
        archive.extractall(scratch / 'runtime', filter='data')
    bundle = scratch / 'runtime/bundle'
    from granite5_cuda_acceptance import qualify_runtime
    provenance = qualify_runtime(bundle, receipt)
    ggml_source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT / 'ggml', text=True).strip()
    if ggml_source != provenance['ggml']:
        raise ValueError('Quantizer headers and qualified GGML library source differ')
    receipt['runtime'] = dict(file=RUNTIME, source=provenance['source'],
                              native_sources_verified=len(provenance['runtime_sources']), device='cpu')
    environment['LD_LIBRARY_PATH'] = str(bundle)
    # GGML_BACKEND_PATH names one module file, not a directory. This also
    # bootstraps the CPU module when a Python process embeds the C ABI.
    environment['GGML_BACKEND_PATH'] = str(bundle / 'libggml-cpu.so')
    save()
    quantizer = build_quantizer(bundle, scratch, execute)
    receipt['quantizer_build'] = dict(main_source_sha256=receipt['quantizer_source_sha256'],
        helper_source_sha256=sha(ROOT / 'examples/common-ggml.cpp'), tensor_name_abi=128,
        ggml_source=ggml_source, ggml_base_sha256=sha(bundle / 'libggml-base.so'),
        runtime_source=provenance['source'])
    save()
    imatrix = scratch / 'calibration.imatrix.gguf'
    if bool(args.reuse_calibration_run) != bool(args.reuse_calibration_source):
        raise ValueError('Both immutable calibration run and source are required')
    if 'body-imatrix' in args.cases and args.reuse_calibration_run:
        if not re.fullmatch('[0-9a-f]{40}', args.reuse_calibration_source):
            raise ValueError('Immutable calibration source required')
        previous_run = json.loads(subprocess.check_output(['gh', 'api',
            f'repos/CrispStrobe/CrispASR/actions/runs/{args.reuse_calibration_run}']))
        if not (previous_run['head_sha'] == args.reuse_calibration_source and previous_run['status'] == 'completed' and
                previous_run['event'] == 'workflow_dispatch'):
            raise ValueError('Unexpected calibration input run')
        old = scratch / 'prior-calibration'
        execute(['gh', 'run', 'download', args.reuse_calibration_run, '--repo', 'CrispStrobe/CrispASR',
                 '--name', f'heavy-{args.reuse_calibration_run}', '--dir', old], 'calibration-input')
        previous = json.loads((old / 'receipt.json').read_text())
        calibration = previous['calibration']
        if not (previous['source'] == args.reuse_calibration_source and previous['f16'] == f16_record and
                previous['instruction'] == pins['instruction'] and
                previous['runtime_sources'] == receipt['runtime_sources'] and
                calibration['manifest'] == MANIFEST and calibration['archive'] == CALIBRATION and
                len(calibration['clips']) == 8 and sha(old / imatrix.name) == calibration['imatrix_sha256']):
            raise ValueError('Retained calibration source, corpus or statistics differ')
        shutil.copyfile(old / imatrix.name, imatrix)
        shutil.copyfile(imatrix, out / imatrix.name)
        for name in ('calibration-source-card.md', 'commonvoice-source-card.md'):
            shutil.copyfile(old / name, out / name)
        receipt['calibration'] = dict(calibration, reused_from=dict(run=args.reuse_calibration_run,
                                                                   source=args.reuse_calibration_source))
        save()
    elif 'body-imatrix' in args.cases:
        archive = fetch(CALIBRATION, 'calibration')
        manifest = json.loads(fetch(MANIFEST, 'calibration').read_text())
        selected = sorted((c for c in manifest['clips'] if c['language'] == 'en'), key=lambda c: c['path'])[:8]
        if len(selected) != 8 or manifest['license'] != 'CC0-1.0':
            raise ValueError('Fixed CC0 calibration selection differs')
        receipt['calibration'] = dict(archive=CALIBRATION, manifest=MANIFEST, clips=selected,
            selection='First eight English clips by filename, fixed before outcomes; JFK excluded')
        for name, repo, revision, digest in [
            ('calibration-source-card.md', 'cstr/crispasr-imatrix-calib',
             '1ccafd189b801db25fbb94cacf493b6cfa02a35b',
             '8889b7876209a35cb5ab633cbd8ee0cc5b7d70f01d32b2db87ad8ef0f36bd88c'),
            ('commonvoice-source-card.md', 'fsicoli/common_voice_17_0',
             '8262c16bf297c87a9cd88c51997c4758ed7a8ba2',
             '5d7f13a790c3f4de73ec28608570d7a4619bce675f37bdd42af47dfa6bfb0281')]:
            path = fetch(dict(repo=repo, repo_type='dataset', revision=revision, path='README.md', sha256=digest))
            if 'license: cc0-1.0' not in path.read_text():
                raise ValueError('Pinned calibration publisher license differs')
            shutil.copyfile(path, out / name)
        with zipfile.ZipFile(archive) as zipped:
            for index, clip in enumerate(selected):
                wav = scratch / clip['path']
                wav.write_bytes(zipped.read(clip['path']))
                if sha(wav) != clip['wav_sha256'] or pcm_sha(wav) != clip['pcm_sha256'] or pcm_sha(wav) == pcm_sha(ROOT / 'samples/jfk.wav'):
                    raise ValueError('Calibration content/disjointness check failed')
                execute([bundle / 'crispasr', '-m', f16, '-f', wav, '-l', 'en', '--no-gpu',
                         '--ask', pins['instruction'], '--max-new-tokens', '1', '-t', '4'], f'calibrate-{index}',
                        dict(environment, CRISPASR_IMATRIX_OUT=str(imatrix)))
        collected = GGUFReader(imatrix)
        names = {t.name for t in collected.tensors}
        required = {t.name for t in GGUFReader(f16).tensors if re.fullmatch(r'blk\.[0-9]+\.(attn_(q|k|v|output)|ffn_(gate|up|down))\.weight', t.name)}
        if not required or not required <= names:
            raise ValueError('Calibration missed decoder weight inputs')
        shutil.copyfile(imatrix, out / imatrix.name)
        receipt['calibration'].update(imatrix_sha256=sha(imatrix), decoder_tensors=len(required))
        save()
    ref = GGUFReader(reference)
    arrays = {t.name: t.data.reshape(-1).copy() for t in ref.tensors}
    for case in args.cases:
        model = scratch / f'granite-speech-5.0-2b-q4_k-{case}.gguf'
        command = [quantizer, f16, model, 'q4_k']
        for rule in RECIPES[case]:
            command += ['--tensor-type', rule]
        if case == 'body-imatrix':
            command += ['--imatrix', imatrix]
        execute(command, f'quantize-{case}', dict(environment, CRISPASR_GRANITE_Q4_LEGACY='1'))
        row = dict(rules=RECIPES[case], cpu_acceptance=False)
        record = dict(repo=STAGING, repo_type='model', path=f'q4-recipes/{source}/{model.name}',
                      sha256=sha(model), bytes=model.stat().st_size)
        try:
            commit = api.upload_file(path_or_fileobj=model, path_in_repo=record['path'], repo_id=STAGING,
                                     commit_message='Checkpoint unaccepted Granite5 Q4_K recipe')
        except Exception:
            # Retain a produced artifact even if the remote checkpoint fails.
            fallback = out / 'failed-transfer'
            fallback.mkdir(exist_ok=True)
            shutil.copyfile(model, fallback / model.name)
            raise
        record['revision'] = commit.oid
        row['file'] = record
        receipt['cases'][case] = row
        save()
        inventory = tensor_inventory(f16, model, RECIPES[case])
        (out / f'{case}-tensors.json').write_text(json.dumps(inventory, indent=2) + '\n')
        row['elements_by_type'] = inventory['elements_by_type']
        save()
        case_out = out / case
        try:
            row['validation'] = validate_case(model, reference, pins['instruction'], bundle, bundle / 'python',
                                               ROOT / 'samples/jfk.wav', case_out, 'cpu', environment, 'q4_k')
            row['cpu_acceptance'] = True
        except ValueError as error:
            # A measured numerical/text rejection is a diagnostic result, not
            # permission to change the gate or skip later independent arms.
            if not (case_out / 'acceptance.json').exists():
                raise
            row['validation_error'] = str(error)
            row['validation'] = json.loads((case_out / 'acceptance.json').read_text())
        row['all_stage_metrics'] = {name: metrics(arrays['llm_prefill_logits' if name == 'pipeline_prefill_logits' else name],
            np.fromfile(case_out / (name + '.bin'), dtype='<f4'))
            for name in ('mel_spectrogram', 'encoder_out', 'projector_out', 'llm_inputs_embeds',
                         'llm_prefill_logits', 'pipeline_prefill_logits')}
        expected = arrays['llm_argmax'][len(arrays['prompt_ids']):]
        row['bucketed'] = token_comparison(expected, np.fromfile(case_out / 'generated_ids.bin', dtype='<i4'))
        row['dynamic'] = token_comparison(expected, np.fromfile(case_out / 'dynamic_generated_ids.bin', dtype='<i4'))
        print(case, 'full CPU acceptance:', row['cpu_acceptance'], row['all_stage_metrics']['llm_prefill_logits'],
              row['all_stage_metrics']['pipeline_prefill_logits'], flush=True)
        save()
        # Checkpoint raw receipts before the next candidate; no ELF payloads.
        api.upload_folder(folder_path=out, path_in_repo=f'q4-recipes/{source}/runs/{receipt["run"]}/cpu-evidence', repo_id=STAGING)
        model.unlink()
    receipt['complete'] = True
    save()
    api.upload_folder(folder_path=out, path_in_repo=f'q4-recipes/{source}/runs/{receipt["run"]}/cpu-evidence', repo_id=STAGING)
    (out / 'summary.md').write_text('Q4_K precision diagnostics completed; see individual CPU gates. CUDA/publication remain separate.\n')


if __name__ == '__main__':
    main()
