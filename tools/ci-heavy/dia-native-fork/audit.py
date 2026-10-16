#!/usr/bin/env python3
"""Independent reciprocal-fork audit. Reads complete raw evidence, not worker comparisons."""
import argparse
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

RUNTIME = '8d0ad8862d2cad5d305a48a57f2a99b5024bc39a'
ORIGINAL = '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'
GGML = 'c36dab89b662838f0f5d4826c399198c0b90bbfc'
# Imported retained methods are evidence: never add bytecode to that tree.
sys.dont_write_bytecode = True


def sha(file):
    with Path(file).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def safe(name):
    p = PurePosixPath(name)
    assert name and name != '.' and not p.is_absolute() and '..' not in p.parts and '\\' not in name and p.as_posix() == name
    return name


def exact(a,b):
    assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def verify_fork(base,changed,base_logits,changed_logits,base_codes,changed_codes):
    assert base.ndim == changed.ndim == 2 and base.shape[1] == changed.shape[1] == 9
    assert min(len(base),len(changed)) > 979
    exact(base[:979],changed[:979]); exact(base_logits[:979],changed_logits[:979])
    assert base[979,2] == 904 and changed[979,2] == 877
    assert np.flatnonzero(base[979] != changed[979]).tolist() == [2]
    assert len(base_codes) >= 963 and len(changed_codes) >= 963
    exact(base_codes[:963],changed_codes[:963])


def verify_rng(before,after,baseline):
    assert before == after == baseline
    words = before.split()
    assert len(words) == 625 and all(w.isdigit() for w in words)
    assert all(0 <= int(w) <= 0xffffffff for w in words[:-1]) and int(words[-1]) <= 624


def check_asr(row,pcm,pcm_path,library_sha):
    assert pcm.dtype == np.float32 and pcm.ndim == 1 and len(pcm) and np.isfinite(pcm).all()
    words = re.findall('[a-z]+',row['transcript'].lower())
    expected = 'we are committed to making it secure'.split()
    # Independent edit-distance table (rather than the producer's rolling row).
    d = np.zeros((len(expected)+1,len(words)+1),dtype='i4')
    d[:,0] = np.arange(len(expected)+1); d[0] = np.arange(len(words)+1)
    for i in range(1,len(expected)+1):
        for j in range(1,len(words)+1):
            d[i,j] = min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(expected[i-1] != words[j-1]))
    assert row['wer'] == int(d[-1,-1])/len(expected)
    assert row['pcm_sha256'] == sha(pcm_path) and row['library_sha256'] == library_sha
    assert row['asr_sha256'] == '60a2b074cdf46e81d72bc15d3eb79505dafbb697135f558659d9068f651854a2'
    assert row['seconds'] == len(pcm)/44100
    assert math.isclose(row['rms'],float(np.sqrt(np.mean(pcm.astype('f8')**2))),rel_tol=1e-12,abs_tol=1e-15)


def check_native_log(log):
    assert 'DIA_CROSS_CACHE: type=F32' in log and 'DIA_ROTARY: source=0' in log
    assert 'DIA_FORCE_TOKENS:' not in log and 'DIA_FORCE_OUTPUT_TOKENS:' not in log
    # Pinned runtime prints capacity only at verbosity >= 1. Capacity/config
    # provenance comes from the exact worker/runtime and physical export hashes;
    # optional debug messages must agree, but silence is not evidence of a cap.
    for value in re.findall(r'max_gen(?:=| )(\d+)',log):
        assert int(value) == 3072


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','repo','original-repo','canonical','terminal-bundle','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker-source',required=True,help='Expected immutable full worker git SHA')
    parser.add_argument('--run-id',type=int,required=True,help='Expected official heavy run ID')
    args = parser.parse_args(); assert not args.output.exists()
    assert re.fullmatch('[0-9a-f]{40}',args.worker_source) and args.run_id > 0
    source = args.worker_source
    bundle = args.bundle; archive = bundle/'raw.zip'; out = bundle/'out'
    run = json.loads((bundle/'run.json').read_text()); lint = json.loads((bundle/'lint.json').read_text())
    for record in [run,lint]:
        assert record['headSha'] == source and record['status'] == 'completed' and record['conclusion'] == 'success'
        assert record['jobs'] and all(j['conclusion'] == 'success' for j in record['jobs'])
    artifacts = json.loads((bundle/'artifacts.json').read_text())['artifacts']; assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact['workflow_run']['head_sha'] == source and artifact['workflow_run']['id'] == args.run_id
    assert artifact['digest'] == 'sha256:'+sha(archive) and artifact['size_in_bytes'] == archive.stat().st_size
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names = z.namelist(); assert len(names) == len(set(names))
        for info in z.infolist():
            safe(info.filename); assert not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16)
        if not out.exists(): z.extractall(out)
        assert z.read('receipt.json') == (out/'receipt.json').read_bytes()
    assert set(names) == {f.relative_to(out).as_posix() for f in out.rglob('*') if f.is_file()}
    receipt = json.loads((out/'receipt.json').read_text())
    assert receipt['worker_source'] == source and receipt['runtime_source'] == RUNTIME and receipt['ggml_source'] == GGML
    for key in ['reciprocal_diagnostic_complete','pristine_off_identity_exact','fork_prefix_and_rng_exact','accepted_native_prefix_exact']:
        assert receipt[key] is True
    for key in ['production_defaults_changed','native_speech_acceptance','gpu_acceptance','q4_acceptance']:
        assert receipt[key] is False
    assert receipt['model_sha256'] == '8128132c538e1aef83f2fb5257b4f71d28a67c2a5a0d54d0a1f0a3b0f5d6b73d'
    assert receipt['codec_sha256'] == '6db51da2333868918b6d476008d92036a53400bc816ff7e1d2b1e973499e4135'
    neural = json.loads((out/'neural-f32-physical-audit.json').read_text())
    assert neural['export_sha256'] == receipt['model_sha256'] and neural['precision'] == 'f32'
    assert neural['original_weights_sha256'] == 'caba289b60f6d7d1e58fc744f4dc25aae88995fcca46be3d05e220b971486a26'
    assert len(neural['tensors']) == 381 and len({r['name'] for r in neural['tensors']}) == 381
    assert all(r['selected'] is True and r['exported_type'] == 'F32' for r in neural['tensors'])
    codec_audit = json.loads((out/'physical-codec-audit.json').read_text())
    assert codec_audit['export_sha256'] == receipt['codec_sha256'] and codec_audit['precision'] == 'output-f32'
    assert codec_audit['original_checkpoint_sha256'] == 'a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa'
    payloads = {r['path']:r for r in receipt['payloads']}; assert len(payloads) == len(receipt['payloads'])
    assert set(payloads) == set(names)-{'receipt.json'}
    for name,row in payloads.items():
        file = out/safe(name); assert file.stat().st_size == row['bytes'] and sha(file) == row['sha256'],name
    for row in receipt['runs']:
        assert row['exit'] == 0 and sha(out/(row['label']+'.log')) == row['sha256']
    assert 'dia_tts.cpp.o' in (out/'patched-build.log').read_text() and 'Linking' in (out/'patched-build.log').read_text()
    method = out/'method'
    for file in method.rglob('*'):
        if not file.is_file(): continue
        name = file.relative_to(method).as_posix(); repo = args.repo
        if name == 'patched-dia_tts.cpp' or name == 'runtime.patch': continue
        if name.startswith('original-dia/'):
            repo, ref, relative = args.original_repo,ORIGINAL,'dia/'+name[len('original-dia/'):]
        elif name == 'Dia-LICENSE': repo,ref,relative = args.original_repo,ORIGINAL,'LICENSE'
        elif name == 'GGML-LICENSE': continue
        elif name == 'pristine-dia_tts.cpp': ref,relative = RUNTIME,'src/dia_tts.cpp'
        elif name in ['dia_sampling.h','dia_exponential_rng.h','dia_dump_step.h']: ref,relative = RUNTIME,'src/'+name
        else: ref,relative = source,'tools/ci-heavy/'+name
        assert file.read_bytes() == subprocess.check_output(['git','show',ref+':'+relative],cwd=repo),name
    # Regenerate the patch from source bytes validated against the immutable git object.
    spec = importlib.util.spec_from_file_location('verified_fork_patch',method/'dia-native-fork/patch.py')
    patcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(patcher)
    audit_workdir = Path(tempfile.mkdtemp(prefix='audit-native-fork-',dir=bundle))
    patch_dir = audit_workdir/'fresh-patch'; patch_dir.mkdir()
    cpp = patch_dir/'dia_tts.cpp'; cpp.write_bytes((method/'pristine-dia_tts.cpp').read_bytes()); patcher.patch(cpp)
    assert cpp.read_bytes() == (method/'patched-dia_tts.cpp').read_bytes()
    with (audit_workdir/'fresh-sampler-guard.log').open('w') as log:
        subprocess.run([sys.executable,method/'dia-native-fork/selftest.py','--repo',args.repo,
                        '--scratch',audit_workdir/'fresh-sampler-guard'],stdout=log,stderr=subprocess.STDOUT,check=True)
    assert 'DIA_NATIVE_FORK_GUARD_PASS' in (audit_workdir/'fresh-sampler-guard.log').read_text()
    libraries = {}
    for label in ['pristine','patched']:
        candidates = list((out/'libs'/label).glob('libcrispasr.so*')); assert len(candidates) == 1
        assert candidates[0].read_bytes().startswith(b'\x7fELF')
        libraries[label] = sha(candidates[0]); assert libraries[label] == receipt[label+'_lib_sha256']
    assert libraries['pristine'] != libraries['patched']
    for file in (out/'libs/pristine').glob('libggml*'): assert sha(file) == sha(out/'libs/patched'/file.name)
    arm_names = ['pristine','off','identity','restore']; dirs = {n:out/'arms'/n for n in arm_names}
    arrays = {}
    for arm,directory in dirs.items():
        log = (out/(arm+'-native.log')).read_text()
        check_native_log(log)
        x = np.fromfile(directory/'native-inputs.i32',dtype='<i4').reshape(-1,9)
        l = np.memmap(directory/'native-logits.f32',dtype='<f4',mode='r').reshape(-1,2,9,1028)
        assert len(x) == len(l) > 979 and np.isfinite(l).all() and ((x >= 0) & (x <= 1026)).all()
        arrays[arm] = (x,l)
    pristine = dirs['pristine']; off = dirs['off']; identity = dirs['identity']; restore = dirs['restore']
    for arm in ['off','identity']:
        for name in ['native-inputs.i32','native-logits.f32','native-pcm.npy']:
            assert sha(dirs[arm]/name) == sha(pristine/name), (arm,name)
    stage_names = {f.name for f in (pristine/'stages').iterdir()}; assert len([n for n in stage_names if n.startswith('cpp_')]) == 127
    for directory in [off,identity,restore]:
        assert stage_names == {f.name for f in (directory/'stages').iterdir()}
        for name in stage_names: assert sha(directory/'stages'/name) == sha(pristine/'stages'/name),name
    for name in ['native-delayed.u32','native-codes.u32','rng-final.txt','sample-original.u32','sample-used.u32','rng-before.txt','rng-after.txt']:
        assert sha(off/name) == sha(identity/name),name
    grids = {}
    for arm in ['off','identity','restore']:
        directory = dirs[arm]
        grid = np.fromfile(directory/'native-codes.u32',dtype='<u4').reshape(-1,9)
        assert len(grid) >= 963 and (grid <= 1023).all(); grids[arm] = grid
        before = (directory/'rng-before.txt').read_bytes(); after = (directory/'rng-after.txt').read_bytes()
        verify_rng(before,after,(off/'rng-before.txt').read_bytes())
        sampled = np.fromfile(directory/'sample-original.u32',dtype='<u4'); used = np.fromfile(directory/'sample-used.u32',dtype='<u4')
        assert sampled.shape == used.shape == (9,) and sampled[2] == 904
        exact(sampled,np.fromfile(off/'sample-original.u32',dtype='<u4'))
        expected = sampled.copy()
        if arm == 'restore': expected[2] = 877
        exact(used,expected)
    verify_fork(arrays['off'][0],arrays['restore'][0],arrays['off'][1],arrays['restore'][1],grids['off'],grids['restore'])
    from gguf import GGUFReader
    assert sha(args.canonical) == '3db6f8923abffd612e6c83d7036debebd10ed28b77c3eb9ec3d91fec1ce1a542'
    wanted = {'decoder_input_tokens','decoder_codes_rev','decoded_audio'}
    original = {t.name:np.asarray(t.data).reshape(tuple(reversed(t.shape))) for t in GGUFReader(str(args.canonical)).tensors if t.name in wanted}
    exact(arrays['off'][0][:979],original['decoder_input_tokens'][:979].astype('<i4'))
    exact(arrays['restore'][0][979],original['decoder_input_tokens'][979].astype('<i4'))
    exact(np.load(out/'original-codes.npy',allow_pickle=False),original['decoder_codes_rev'])
    original_dir = out/'arms/original'
    exact(np.load(original_dir/'canonical-pcm.npy',allow_pickle=False),original['decoded_audio'].reshape(-1))
    canonical_pcm = original['decoded_audio'].reshape(-1).astype('f8')
    reconstructed = np.load(original_dir/'original-dac-pcm.npy',allow_pickle=False)
    assert reconstructed.shape == canonical_pcm.shape and np.isfinite(reconstructed).all()
    error = float(np.linalg.norm(reconstructed-canonical_pcm)/np.linalg.norm(canonical_pcm))
    assert error < 1e-4 and math.isclose(error,receipt['original_dac_positive_relative_l2'],rel_tol=1e-12,abs_tol=1e-15)
    terminal_archive = args.terminal_bundle/'raw.zip'
    assert sha(terminal_archive) == 'a2d4c27e7ef6351e9b0026c1389ec9fd409c33421694ac1e596b090a0ec2cc21'
    with zipfile.ZipFile(terminal_archive) as z:
        assert neural['tensors'] == json.loads(z.read('neural-f32-physical-audit.json'))['tensors']
        exact(arrays['off'][1][:979],np.load(io.BytesIO(z.read('inputs/native-logits.npy')),allow_pickle=False))
        for name in stage_names:
            path = 'baseline/'+name
            if path in z.namelist(): assert (off/'stages'/name).read_bytes() == z.read(path)
    results = {}
    for arm in ['original']+arm_names:
        directory = out/'arms'/arm
        audio_names = ['canonical-pcm.npy','original-dac-pcm.npy'] if arm == 'original' else ['native-pcm.npy']+(['original-dac-pcm.npy'] if arm != 'pristine' else [])
        results[arm] = {}
        for name in audio_names:
            pcm = np.load(directory/name,allow_pickle=False); row = json.loads((directory/(name+'.asr.json')).read_text())
            check_asr(row,pcm,directory/name,libraries['patched']); results[arm][name] = row
            if arm == 'original': assert row['wer'] == 0
        if arm in ['original','off','identity','restore']:
            provenance = json.loads((directory/'dac-provenance.json').read_text())
            assert provenance['torch_version'] == '2.6.0+cpu' and provenance['threads'] == 4 and provenance['package'] == '1.0.0'
            assert provenance['checkpoint_sha256'] == 'a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa'
            for name,digest in provenance['blueprint_hashes'].items():
                raw = subprocess.check_output(['git','show',ORIGINAL+':'+name],cwd=args.original_repo)
                assert hashlib.sha256(raw).hexdigest() == digest
            for name in ['dac-model.py','dac-quantize.py','dac-layers.py']: assert sha(directory/name) == sha(original_dir/name)
    exact(np.load(off/'original-dac-pcm.npy'),np.load(identity/'original-dac-pcm.npy'))
    for arm in ['off','identity']:
        for key in ['transcript','wer']: assert results[arm]['native-pcm.npy'][key] == results['pristine']['native-pcm.npy'][key]
    for key in ['transcript','wer']: assert results['off']['original-dac-pcm.npy'][key] == results['identity']['original-dac-pcm.npy'][key]
    assert [r['arm'] for r in receipt['arms']] == arm_names
    for row in receipt['arms']:
        arm = row['arm']; assert row['steps'] == len(arrays[arm][0]) and row['native_asr'] == results[arm]['native-pcm.npy']
        assert row['original_dac_asr'] == results[arm].get('original-dac-pcm.npy')
    result = dict(source=source,run_id=args.run_id,official_artifact_sha256=sha(archive),payloads_verified=len(payloads),
                  audit_workdir=audit_workdir.name,fresh_sampler_guard_log_sha256=sha(audit_workdir/'fresh-sampler-guard.log'),
                  pristine_off_identity_exact=True,fork_prefix_and_rng_exact=True,accepted_native_prefix_exact=True,
                  fresh_actual_sampler_guard=True,original_dac_relative_l2=error,results=results,
                  native_speech_acceptance=False,production_defaults_changed=False)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('DIA_NATIVE_FORK_INDEPENDENT_AUDIT_PASS',len(payloads),'payloads; all speech outcomes reported unchanged')


if __name__ == '__main__': main()
