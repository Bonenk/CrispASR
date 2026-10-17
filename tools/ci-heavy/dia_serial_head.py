#!/usr/bin/env python3
"""All-step serial F32 head graph diagnostic; unchanged sampler/defaults."""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHOD = Path(__file__).with_name('dia-serial-head')
SOURCE = '8d0ad8862d2cad5d305a48a57f2a99b5024bc39a'
GGML = 'c36dab89b662838f0f5d4826c399198c0b90bbfc'
TEXT = '[S1] We are committed to making it secure.'
sys.dont_write_bytecode = True
MODEL_SHA = '8128132c538e1aef83f2fb5257b4f71d28a67c2a5a0d54d0a1f0a3b0f5d6b73d'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def inputs(directory):
    return np.fromfile(directory/'native-inputs.i32', dtype='<i4').reshape(-1,9)


def logits(directory):
    return np.memmap(directory/'native-logits.f32', mode='r', dtype='<f4').reshape(-1,2,9,1028)


def codes(directory):
    return np.fromfile(directory/'native-codes.u32', dtype='<u4').reshape(-1,9)


def child(args, out):
    dest = out/'arms'/args.arm; dest.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.runtime/'python'))
    from crispasr import Session
    if args.child == 'native':
        if args.arm != 'pristine':
            os.environ.update(CRISPASR_DIA_SERIAL_HEAD=args.arm, CRISPASR_DIA_SERIAL_HEAD_DIR=str(dest))
        else:
            assert not os.environ.get('CRISPASR_DIA_SERIAL_HEAD')
        os.environ.update(CRISPASR_DIA_CROSS_CACHE_F32='1', CRISPASR_DIA_PORTABLE_RNG='1',
                          CRISPASR_DIA_SOURCE_ROPE='0', CRISPASR_DIA_TTS_GPU='0',
                          CRISPASR_DIA_DUMP_INPUT_TOKENS=str(dest/'native-inputs.i32'),
                          CRISPASR_DIA_DUMP_STEPLOGITS=str(dest/'native-logits.f32'),
                          CRISPASR_DIA_DUMP_TOKENS='1', CRISPASR_DIA_DUMP_DIR=str(dest/'stages'),
                          CRISPASR_DIA_DUMP_STEP='978', CRISPASR_DIA_DUMP_SELF_ATTN='1',
                          CRISPASR_DIA_DUMP_DECODER_BOUNDARIES='1')
        for suffix in ['FORCE_TOKENS','FORCE_OUTPUT_TOKENS','DECODE_CODES','GREEDY','MAX_STEPS']:
            assert not any(os.environ.get(p+suffix) for p in ['DIA_','CRISPASR_DIA_'])
        (dest/'stages').mkdir()
        with Session(str(args.model), lib_path=str(args.lib), backend='dia', n_threads=4) as session:
            session.set_temperature(1.2, seed=42)
            pcm = session.synthesize(TEXT)
            assert session.output_sample_rate() == 44100
        pcm = np.asarray(pcm, dtype='f4'); assert pcm.ndim == 1 and len(pcm) and np.isfinite(pcm).all()
        np.save(dest/'native-pcm.npy', pcm, allow_pickle=False)
        print('DIA_SERIAL_HEAD_ARM_COMPLETE', args.arm, len(pcm), flush=True)
    elif args.child == 'dac':
        import importlib.metadata
        import inspect
        import torch
        import dac
        assert torch.__version__ == '2.6.0+cpu'; torch.set_num_threads(4)
        sys.path.insert(0, str(args.runtime/'tools'))
        from reference_backends import dia_tts_reference as reference
        _, source_hashes = reference.verify_source(args.blueprint)
        sys.path.insert(0, str(args.blueprint))
        from dia.audio import decode
        checkpoint = Path(dac.utils.download())
        assert sha(checkpoint) == 'a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa'
        codec = dac.DAC.load(str(checkpoint)).cpu().eval()
        import dac.model.dac as dac_model
        import dac.nn.quantize as dac_quantize
        import dac.nn.layers as dac_layers
        for module, name in [(dac_model,'dac-model.py'),(dac_quantize,'dac-quantize.py'),(dac_layers,'dac-layers.py')]:
            shutil.copyfile(inspect.getsourcefile(module),dest/name)
        (dest/'dac-provenance.json').write_text(json.dumps(dict(blueprint_hashes=source_hashes,
            torch_version=torch.__version__,threads=torch.get_num_threads(),checkpoint_sha256=sha(checkpoint),
            package=importlib.metadata.version('descript-audio-codec')),indent=2)+'\n')
        grid = np.load(out/'original-codes.npy', allow_pickle=False) if args.arm == 'original' else codes(dest)
        assert grid.ndim == 2 and grid.shape[1] == 9 and len(grid) and ((grid >= 0) & (grid <= 1023)).all()
        with torch.inference_mode():
            audio = decode(codec, torch.from_numpy(grid.astype('i8').T.copy()).unsqueeze(0)).squeeze().numpy().copy()
        assert audio.ndim == 1 and len(audio) and audio.dtype == np.float32 and np.isfinite(audio).all()
        np.save(dest/'original-dac-pcm.npy', audio, allow_pickle=False)
        print('DIA_ACTUAL_ORIGINAL_DAC_COMPLETE', args.arm, len(audio), flush=True)
    else:
        pcm = np.load(dest/args.audio, allow_pickle=False)
        with Session(str(args.model), lib_path=str(args.lib), backend='nemotron', n_threads=4) as session:
            text = ' '.join(s.text for s in session.transcribe(pcm, sample_rate=44100, language='en')).strip()
        word = lambda s: re.findall('[a-z]+', s.lower())
        expected, actual = word(TEXT[5:]), word(text); dp = list(range(len(actual)+1))
        for i, a in enumerate(expected, 1):
            row = [i]
            for j, b in enumerate(actual, 1): row.append(min(row[-1]+1, dp[j]+1, dp[j-1]+(a != b)))
            dp = row
        result = dict(transcript=text, wer=dp[-1]/len(expected), pcm_sha256=sha(dest/args.audio),
                      asr_sha256=sha(args.model), library_sha256=sha(args.lib),
                      seconds=len(pcm)/44100, rms=float(np.sqrt(np.mean(pcm.astype('f8')**2))))
        (dest/(args.audio+'.asr.json')).write_text(json.dumps(result, indent=2)+'\n')
        if args.arm == 'original': assert result['wer'] == 0
        print('DIA_SERIAL_HEAD_ASR_RESULT', args.arm, args.audio, json.dumps(result), flush=True)


def preflight(runtime,build,out,scratch,run):
    from huggingface_hub import hf_hub_download
    archive = Path(hf_hub_download('cstr/crispasr-regression-fixtures',
        'dia-20261016-reduction-cause/complete-reduction-cause-v1.zip',
        revision='81d18136133e7bad61961813f45afa58301fe32e',repo_type='dataset'))
    assert sha(archive) == 'b092ed027e26480ae4ad6c59d30d216662c7c5afe949f5300c9afafe86163397'
    fixture = scratch/'head-fixture'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist(): assert not Path(name).is_absolute() and '..' not in Path(name).parts
        z.extractall(fixture)
    manifest = json.loads((fixture/'manifest.json').read_text())
    for row in manifest['files']:
        p = fixture/row['path']; assert sha(p) == row['sha256'] and p.stat().st_size == row['bytes']
    shutil.copyfile(fixture/'manifest.json',out/'head-fixture-manifest.json')
    probe = out/'head-probe'
    cpu,base = next(build.rglob('libggml-cpu.so')),next(build.rglob('libggml-base.so'))
    run(['c++','-std=c++17','-O3','-I',runtime/'ggml/include','-I',METHOD,METHOD/'probe.cpp',
         cpu,base,'-Wl,-rpath,'+str(cpu.parent)+':'+str(base.parent),'-o',probe],'head-probe-build')
    run(['ldd',probe],'head-probe-ldd')
    dest = out/'head-preflight'; dest.mkdir()
    w = np.fromfile(fixture/'dispatch-v1/inputs/head-weight.f32',dtype='<f4').reshape(2048,9,1028)
    weight = scratch/'native-head-weight.f32'; np.ascontiguousarray(w.transpose(1,2,0)).tofile(weight)
    for hidden in ['original-hidden','native-hidden']:
        x = np.load(fixture/'dispatch-v1/inputs'/(hidden+'.npy'),allow_pickle=False)
        path = scratch/(hidden+'.f32'); x.tofile(path)
        expected = np.load(fixture/'order-v3'/(hidden+'-serial.npy'),allow_pickle=False)
        off = None
        for threads in [1,4]:
            for mode in ['off','identity','serial']:
                name = hidden+'-'+mode+'-'+str(threads); output = dest/(name+'.f32')
                run([probe,weight,path,mode,str(threads),output],'preflight-'+name)
                actual = np.fromfile(output,dtype='<f4').reshape(2,9,1028)
                if mode == 'off':
                    if off is not None: exact(actual,off)
                    off = actual
                elif mode == 'identity': exact(actual,off)
                else: exact(actual,expected)
    print('DIA_SERIAL_HEAD_PREFLIGHT_EXACT: actual graph, both captured hidden states, one/four threads',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', choices=['native','dac','asr'])
    parser.add_argument('--arm', choices=['pristine','off','identity','serial','original'])
    parser.add_argument('--audio')
    for name in ['runtime','blueprint','model','lib']: parser.add_argument('--'+name,type=Path)
    args = parser.parse_args()
    out, scratch = Path(os.environ['HEAVY_OUT']), Path(os.environ['HEAVY_SCRATCH'])
    out.mkdir(parents=True, exist_ok=True); scratch.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(scratch)
    if args.child: child(args,out); return
    report = dict(worker_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  runtime_source=SOURCE, ggml_source=GGML, runs=[], arms=[], production_defaults_changed=False,
                  native_speech_acceptance=False, gpu_acceptance=False, q4_acceptance=False)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
    if os.environ.get('GITHUB_SHA'): assert report['worker_source'] == os.environ['GITHUB_SHA']
    def checkpoint(): (out/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def run(command,label,env=None):
        with (out/(label+'.log')).open('w') as log:
            result = subprocess.run(list(map(str,command)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=10800)
        report['runs'].append(dict(label=label,exit=result.returncode,sha256=sha(out/(label+'.log')))); checkpoint()
        print(label,result.returncode,(out/(label+'.log')).read_text()[-1200:],flush=True)
        assert result.returncode == 0,label
    checkpoint()
    (out/'method').mkdir(); shutil.copyfile(__file__,out/'method/dia_serial_head.py')
    shutil.copytree(METHOD,out/'method/dia-serial-head',ignore=shutil.ignore_patterns('__pycache__'))
    runtime = scratch/'pinned-runtime'
    run(['git','fetch','origin',SOURCE],'fetch-runtime')
    run(['git','worktree','add','--detach',runtime,SOURCE],'checkout-runtime')
    run(['git','-C',runtime,'submodule','update','--init','--recursive'],'runtime-submodules')
    assert subprocess.check_output(['git','-C',runtime/'ggml','rev-parse','HEAD'],text=True).strip() == GGML
    shutil.copyfile(runtime/'src/dia_tts.cpp',out/'method/pristine-dia_tts.cpp')
    for name in ['dia_sampling.h','dia_exponential_rng.h','dia_dump_step.h']:
        shutil.copyfile(runtime/'src'/name,out/'method'/name)
    shutil.copyfile(runtime/'ggml/LICENSE',out/'method/GGML-LICENSE')
    run(['sudo','bash',ROOT/'tools/ci-apt.sh','update'],'apt-update')
    run(['sudo','bash',ROOT/'tools/ci-apt.sh','install','-y','ninja-build','ccache'],'apt-install')
    build = scratch/'native-build'
    run(['cmake','-G','Ninja','-S',runtime,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DBUILD_SHARED_LIBS=ON',
         '-DCMAKE_C_COMPILER_LAUNCHER=ccache','-DCMAKE_CXX_COMPILER_LAUNCHER=ccache','-DGGML_NATIVE=OFF',
         '-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF','-DGGML_BLAS=OFF','-DCRISPASR_BUILD_TESTS=OFF',
         '-DCRISPASR_BUILD_SERVER=OFF','-DCRISPASR_OPUS=OFF','-DCRISPASR_AMR=OFF'],'configure')
    run(['cmake','--build',build,'--target','crispasr-lib','-j4'],'pristine-build')
    lib = next(build.rglob('libcrispasr.so'))
    def retain_libraries(label):
        dest = out/'libs'/label; dest.mkdir(parents=True)
        for file in build.rglob('*.so*'):
            if file.is_file() and not file.is_symlink(): shutil.copyfile(file,dest/file.name)
        run(['ldd',lib],label+'-ldd')
        report[label+'_lib_sha256'] = sha(lib); checkpoint()
    retain_libraries('pristine')
    preflight(runtime,build,out,scratch,run)
    from huggingface_hub import hf_hub_download,snapshot_download
    sys.path.insert(0,str(runtime/'tools'))
    canonical_proof = scratch/'canonical-proof'
    run([sys.executable,runtime/'tools/dia_proof_archives.py','canonical-original',canonical_proof],'canonical-download')
    canonical = next(canonical_proof.glob('heavy-*'))/'ref.gguf'
    assert sha(canonical) == '3db6f8923abffd612e6c83d7036debebd10ed28b77c3eb9ec3d91fec1ce1a542'
    from gguf import GGUFReader
    selected = {'decoded_audio','decoder_codes_rev','decoder_input_tokens'}
    reference = {t.name:np.array(t.data).reshape(tuple(reversed(t.shape))) for t in GGUFReader(str(canonical)).tensors if t.name in selected}
    np.save(out/'original-codes.npy',reference['decoder_codes_rev'],allow_pickle=False)
    original = out/'arms/original'; original.mkdir(parents=True)
    np.save(original/'canonical-pcm.npy',reference['decoded_audio'].reshape(-1),allow_pickle=False)
    location = Path(snapshot_download('nari-labs/Dia-1.6B',revision='257bc72f9b78182ccc6fa07675a9ae4c1a44e2cd',allow_patterns=['config.json','model.safetensors']))
    template = hf_hub_download('cstr/dia-1.6b-GGUF','dia-1.6b-f16.gguf',revision='3233fbcb32be47761d2e736857b6d1a075b9ba7e')
    model_dir = scratch/'model'; model_dir.mkdir(); model = model_dir/'dia-1.6b-f32.gguf'
    run([sys.executable,runtime/'tools/ci-heavy/dia_neural_precision_control.py','--export','f32',
         '--weights',location/'model.safetensors','--template',template,'--output',model],'neural-export')
    assert sha(model) == MODEL_SHA
    codec_template = hf_hub_download('cstr/dia-1.6b-GGUF','dac-44khz.gguf',revision='3233fbcb32be47761d2e736857b6d1a075b9ba7e')
    codec = model_dir/'dac-44khz.gguf'
    run([sys.executable,runtime/'models/convert-dia-dac-to-gguf.py','--template',codec_template,'--output',codec,
         '--precision','output-f32','--receipt',out/'physical-codec-audit.json'],'codec-export')
    assert sha(codec) == '6db51da2333868918b6d476008d92036a53400bc816ff7e1d2b1e973499e4135'
    asr = Path(hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF','nemotron-3.5-asr-streaming-0.6b-q4_k.gguf',revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2'))
    assert sha(asr) == '60a2b074cdf46e81d72bc15d3eb79505dafbb697135f558659d9068f651854a2'
    report.update(model_sha256=sha(model),codec_sha256=sha(codec),asr_sha256=sha(asr)); checkpoint()
    blueprint = scratch/'original-dia'
    run(['git','clone','--filter=blob:none','https://github.com/nari-labs/dia',blueprint],'clone-original')
    run(['git','-C',blueprint,'checkout','--detach','4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'],'pin-original')
    shutil.copytree(blueprint/'dia',out/'method/original-dia',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(blueprint/'LICENSE',out/'method/Dia-LICENSE')
    clean_env = {k:v for k,v in os.environ.items() if not k.startswith(('DIA_','CRISPASR_DIA_'))}
    def invoke(kind,arm,weights=model,audio=None):
        command = [sys.executable,__file__,'--child',kind,'--arm',arm,'--runtime',runtime,
                   '--blueprint',blueprint,'--model',weights,'--lib',lib]
        if audio: command += ['--audio',audio]
        run(command,arm+'-'+kind+('-'+audio if audio else ''),clean_env)
    invoke('dac','original')
    original_pcm = np.load(original/'original-dac-pcm.npy',allow_pickle=False)
    expected_pcm = reference['decoded_audio'].reshape(-1)
    assert original_pcm.shape == expected_pcm.shape
    error = np.linalg.norm(original_pcm.astype('f8')-expected_pcm)/np.linalg.norm(expected_pcm.astype('f8'))
    assert error < 1e-4; report['original_dac_positive_relative_l2'] = float(error)
    invoke('native','pristine')
    spec = importlib.util.spec_from_file_location('scratch_serial_head_patch',METHOD/'patch.py')
    patcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(patcher)
    patcher.patch(runtime/'src/dia_tts.cpp'); shutil.copyfile(METHOD/'head.h',runtime/'src/dia_serial_head.h')
    shutil.copyfile(runtime/'src/dia_tts.cpp',out/'method/patched-dia_tts.cpp')
    (out/'method/runtime.patch').write_bytes(subprocess.check_output(['git','-C',runtime,'diff']))
    run(['cmake','--build',build,'--target','crispasr-lib','-j4'],'patched-build'); retain_libraries('patched')
    # Every ASR result uses this same final library, including both positives.
    invoke('asr','original',asr,'canonical-pcm.npy')
    invoke('asr','original',asr,'original-dac-pcm.npy')
    for arm in ['off','identity','serial']: invoke('native',arm)
    dirs = {arm:out/'arms'/arm for arm in ['pristine','off','identity','serial']}
    pristine = dirs['pristine']; baseline = dirs['off']; identity = dirs['identity']; serial = dirs['serial']
    for directory in [baseline,identity]:
        for name in ['native-inputs.i32','native-logits.f32','native-pcm.npy']:
            assert sha(directory/name) == sha(pristine/name), (directory.name,name)
        assert set(p.name for p in (directory/'stages').iterdir()) == set(p.name for p in (pristine/'stages').iterdir())
        for file in (directory/'stages').iterdir(): assert sha(file) == sha(pristine/'stages'/file.name), file.name
    for name in ['native-delayed.u32','native-codes.u32','rng-final.txt']:
        assert sha(baseline/name) == sha(identity/name), name
    exact(inputs(baseline)[:979],reference['decoder_input_tokens'][:979].astype('<i4'))
    report['pristine_off_identity_exact'] = True; checkpoint()
    # Retain all-step serial differences without assuming the first changed row.
    a,b = inputs(baseline),inputs(serial); common = min(len(a),len(b))
    changed = np.flatnonzero(np.any(a[:common] != b[:common],axis=1))
    report['first_serial_input_difference'] = int(changed[0]) if len(changed) else (common if len(a) != len(b) else None)
    report['serial_steps'] = len(b); report['baseline_steps'] = len(a)
    original_inputs = reference['decoder_input_tokens'].astype('<i4')
    report['serial_original_inputs_exact'] = b.shape == original_inputs.shape and b.tobytes() == original_inputs.tobytes()
    prior = Path(hf_hub_download('cstr/crispasr-regression-fixtures','dia-20261015-terminal-replay/complete-hosted-terminal-replay-v1/raw.zip',revision='6b0b9fdf167e4a01bcdc2a397f27f246f0eadaf7',repo_type='dataset'))
    assert sha(prior) == 'a2d4c27e7ef6351e9b0026c1389ec9fd409c33421694ac1e596b090a0ec2cc21'
    with zipfile.ZipFile(prior) as z:
        exact(logits(baseline)[:979],np.load(io.BytesIO(z.read('inputs/native-logits.npy')),allow_pickle=False))
        for file in (baseline/'stages').glob('cpp_*.f32'):
            name = 'baseline/'+file.name
            if name in z.namelist(): assert file.read_bytes() == z.read(name),name
    report['accepted_native_prefix_exact'] = True; checkpoint()
    prior_full = Path(hf_hub_download('cstr/crispasr-regression-fixtures',
        'dia-20261016-native-fork/complete-hosted-native-fork-v1/raw.zip',
        revision='f0a412c02c23bf2625ec91011ed962493db66763',repo_type='dataset'))
    assert sha(prior_full) == '0cdd76a38bdae5be82b7a854b7d0c3f1194e47237afee4b68fb1164baca2ad47'
    with zipfile.ZipFile(prior_full) as z:
        for name in ['native-inputs.i32','native-logits.f32','native-pcm.npy']:
            assert (baseline/name).read_bytes() == z.read('arms/pristine/'+name),name
    report['accepted_native_full_trajectory_exact'] = True; checkpoint()
    for arm in ['pristine','off','identity','serial']:
        invoke('asr',arm,asr,'native-pcm.npy')
        if arm != 'pristine':
            invoke('dac',arm); invoke('asr',arm,asr,'original-dac-pcm.npy')
        report['arms'].append(dict(arm=arm,steps=len(inputs(dirs[arm])),
            native_asr=json.loads((dirs[arm]/'native-pcm.npy.asr.json').read_text()),
            original_dac_asr=json.loads((dirs[arm]/'original-dac-pcm.npy.asr.json').read_text()) if arm != 'pristine' else None))
        checkpoint()
    exact(np.load(baseline/'original-dac-pcm.npy'),np.load(identity/'original-dac-pcm.npy'))
    results = {r['arm']:r for r in report['arms']}
    for arm in ['off','identity']:
        for key in ['transcript','wer']:
            assert results[arm]['native_asr'][key] == results['pristine']['native_asr'][key]
    report['all_step_diagnostic_complete'] = True
    report['serial_both_decoders_wer_zero'] = all(results['serial'][key]['wer'] == 0 for key in ['native_asr','original_dac_asr'])
    report['interpretation'] = 'All-step generic serial F32 graph head; sampler/noise/weights/recipe fixed, no reference input or token override. Pinned runtime only; no production promotion.'
    report['payloads'] = [dict(path=f.relative_to(out).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(out.rglob('*')) if f.is_file() and f.name != 'receipt.json']
    checkpoint()
    print('DIA_ALL_STEP_SERIAL_HEAD_COMPLETE',json.dumps(results['serial']),flush=True)
    assert report['serial_both_decoders_wer_zero'],'Unforced serial head did not restore speech; complete raw evidence retained'


if __name__ == '__main__': main()
