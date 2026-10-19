#!/usr/bin/env python3
"""Real GPU voice conversion against pinned original modules.

Reuse the independently audited CUDA runtime; compile only changed model/driver
translation units against its exact ggml headers. No ggml/CUDA/full SDK build,
no CPU-only Kaggle job, no hardware fishing or cross-account retry.
"""
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request

VERSION = '2026-10-19.openvoice-bake.1'
SOURCE = '71e329ccfbb33dc99fa53743565292b1dba13b1e'
UPSTREAM = '74a1d147b17a8c3092dd5430504bd83ef6c7eb23'
BASE_SOURCE = '191815511b85a526aace111b1bda2e28ecb7faad'
BASE_REV = '425c555beb63de4665cba604c47ea38232f5d080'
BASE_SHA = '63474b5f034ebac850b7b87ef76eecec12d009092d4c72370aeeb6be341365c5'
OUT = Path('/kaggle/working')
WORK = Path('/kaggle/temp/openvoice-bake')
ROOT = WORK / 'sdk'
BUNDLE = WORK / 'bundle'
RECEIPT = {'schema': 1, 'version': VERSION, 'source': SOURCE, 'upstream': UPSTREAM,
           'complete': False, 'model_acceptance': False, 'gpu_executed': False,
           'cases': [], 'limitations': ['native reference encoder and flows are CPU; vocoder uses CUDA',
                                      'native-rate replay isolates resampling; raw reference without VAD',
                                      'deterministic tau=0; upstream tau=0.3 quality separate',
                                      'closed-loop synthetic speakers, not human speaker verification']}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save():
    (OUT / 'validation.json').write_text(json.dumps(RECEIPT, indent=2) + '\n')


def run(command, name, **kwargs):
    print(name, flush=True)
    with (OUT / (name + '.log')).open('w') as log:
        subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=kwargs.pop('timeout', 1200), **kwargs)


def download(url, path, expected=None):
    with urllib.request.urlopen(url, timeout=120) as stream, Path(path).open('wb') as output:
        shutil.copyfileobj(stream, output, 4*1024*1024)
    if expected and sha(path) != expected:
        raise ValueError('Checksum mismatch: ' + Path(path).name)


def bootstrap():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    RECEIPT['entry_sha256']=sha(Path(__file__))
    os.environ.update(TMPDIR=str(WORK), HF_HOME=str(WORK/'hf'), OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='1', CRISPASR_OV2_GPU='1', GGML_SCHED_DEBUG='1')
    RECEIPT['hardware'] = subprocess.check_output(
        ['nvidia-smi','--query-gpu=name,compute_cap,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
    caps = [float(row.split(',')[1]) for row in RECEIPT['hardware'].splitlines()]
    if not caps or any(cap not in (6.0,7.5) for cap in caps):
        raise RuntimeError('Frozen CUDA runtime supports sm60/sm75 only')
    for command in (['git','init',ROOT],['git','-C',ROOT,'remote','add','origin','https://github.com/CrispStrobe/CrispASR.git'],
                    ['git','-C',ROOT,'fetch','--depth=1','origin',SOURCE],['git','-C',ROOT,'checkout','FETCH_HEAD'],
                    ['git','-C',ROOT,'submodule','update','--init','--depth=1','ggml']):
        subprocess.run(list(map(str,command)),check=True)
    sys.path.insert(0,str(ROOT/'tools/kaggle'))
    import kaggle_harness as kh
    kh.init_progress()
    kh.provenance(VERSION,ROOT)
    os.environ['HF_TOKEN'] = kh.resolve_hf_token(require=True)
    runtime=WORK/'runtime.tar.gz'
    download(f'https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/{BASE_REV}/alignment-long-cuda-20261019/v2/runtime.tar.gz',runtime,BASE_SHA)
    with tarfile.open(runtime) as archive:
        archive.extractall(WORK,filter='data')
    provenance=json.loads((BUNDLE/'provenance.json').read_text())
    assert provenance['source']==BASE_SOURCE and provenance['build_complete']
    for name,row in provenance['files'].items():
        path=BUNDLE/name
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],name
    ggml=subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()
    assert ggml=='0a862076567afd8468822ccc512c97ac6a69b340'
    RECEIPT.update(base_runtime_source=BASE_SOURCE,base_runtime_sha256=BASE_SHA,ggml=ggml)
    if os.environ.get('CRISPASR_OV2_BUNDLE_READY')!='1':
        env=dict(os.environ,CRISPASR_OV2_BUNDLE_READY='1',LD_LIBRARY_PATH=str(BUNDLE))
        os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve())],env)
    return kh


def main():
    # Re-exec only after immutable inputs are verified, without repeat cloning.
    if os.environ.get('CRISPASR_OV2_BUNDLE_READY')=='1':
        sys.path.insert(0,str(ROOT/'tools/kaggle'))
        import kaggle_harness as kh
        kh.init_progress(); kh.provenance(VERSION,ROOT)
        RECEIPT['entry_sha256']=sha(Path(__file__))
        RECEIPT.update(base_runtime_source=BASE_SOURCE,base_runtime_sha256=BASE_SHA,
                       ggml=subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip(),
                       hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,compute_cap,memory.total,driver_version','--format=csv,noheader'],text=True).strip())
    else:
        kh=bootstrap()
    original_stderr=os.dup(2)
    native_log=(OUT/'native.log').open('w');os.dup2(native_log.fileno(),2)
    try:
        run([sys.executable,'-m','pip','install','-q','gguf','librosa>=0.10','soundfile','huggingface_hub','resemblyzer','webrtcvad-wheels','--no-deps'],'small-dependencies')
        run([sys.executable,'-m','pip','install','-q','faster-whisper'],'asr-dependencies')
        # Use the prebuilt CUDA ggml. These are ordinary model C++ files only;
        # link our own definitions locally to avoid old SDK symbol interposition.
        common=['c++','-std=c++17','-O2','-fPIC','-fopenmp','-I'+str(ROOT/'src'),'-I'+str(ROOT/'ggml/include'),
                '-I'+str(ROOT/'examples/cli'),'-L'+str(BUNDLE),'-Wl,-rpath,'+str(BUNDLE)]
        run(common+[ROOT/'src/openvoice2.cpp',ROOT/'src/melotts.cpp','-shared','-Wl,-Bsymbolic-functions',
                    '-lcrispasr','-lggml','-lggml-cpu','-lggml-base','-ldl','-pthread','-o',WORK/'libopenvoice-probe.so'],'compile-model-delta')
        run(common+[ROOT/'tools/ci-heavy/openvoice-bake/main.cpp',ROOT/'examples/cli/crispasr_diff.cpp',
                    '-L'+str(WORK),'-Wl,-rpath,'+str(WORK),'-lopenvoice-probe','-lcrispasr',
                    '-lggml','-lggml-cpu','-lggml-base','-ldl','-pthread','-o',WORK/'openvoice-bake'],'compile-diff-driver')
        RECEIPT['native_delta']={name:sha(WORK/name) for name in ['libopenvoice-probe.so','openvoice-bake']}
        import numpy as np
        import soundfile as sf
        import torch
        RECEIPT['torch_version']=torch.__version__
        RECEIPT['torch_reference_device']='cuda' if torch.cuda.get_device_capability()[0]>=7 else 'cpu'
        from huggingface_hub import HfApi,hf_hub_download,snapshot_download
        hub=HfApi()
        model_rev=hub.model_info('myshell-ai/OpenVoiceV2').sha
        model=Path(snapshot_download('myshell-ai/OpenVoiceV2',revision=model_rev,
                    allow_patterns=['converter/*','base_speakers/ses/*.pth','README.md']))
        RECEIPT['original_model']={'repo':'myshell-ai/OpenVoiceV2','revision':model_rev,
            'files':{str(p.relative_to(model)):sha(p) for p in model.rglob('*') if p.is_file()}}
        run([sys.executable,ROOT/'models/convert-openvoice2-to-gguf.py','--ckpt',model/'converter/checkpoint.pth',
             '--config',model/'converter/config.json','--output',WORK/'openvoice-f32sens.gguf'],'convert-f32-sensitive')
        upstream_tar=WORK/'upstream.tar.gz'
        download(f'https://codeload.github.com/myshell-ai/OpenVoice/tar.gz/{UPSTREAM}',upstream_tar)
        with tarfile.open(upstream_tar) as archive:
            archive.extractall(WORK,filter='data')
        upstream=WORK/('OpenVoice-'+UPSTREAM)
        shutil.copy2(upstream/'LICENSE',OUT/'OpenVoice-LICENSE.txt')
        melo_rev=hub.model_info('cstr/melotts-en-v2-GGUF').sha
        melo=hf_hub_download('cstr/melotts-en-v2-GGUF','melotts-en-v2-f16.gguf',revision=melo_rev)
        bert=hf_hub_download('cstr/melotts-en-v2-GGUF','bert-base-uncased-q4k.gguf',revision=melo_rev)
        RECEIPT['melo']={'repo':'cstr/melotts-en-v2-GGUF','revision':melo_rev,'model_sha256':sha(melo),'bert_sha256':sha(bert)}
        class MeloParams(ct.Structure):
            _fields_=[('n_threads',ct.c_int),('verbosity',ct.c_int),('use_gpu',ct.c_bool),
                      ('noise_scale',ct.c_float),('length_scale',ct.c_float),('noise_w',ct.c_float),
                      ('sdp_ratio',ct.c_float),('speaker_id',ct.c_int),('seed',ct.c_uint32)]
        lib=ct.CDLL(str(WORK/'libopenvoice-probe.so'))
        lib.melotts_default_params.restype=MeloParams
        lib.melotts_init_from_file.argtypes=[ct.c_char_p,MeloParams];lib.melotts_init_from_file.restype=ct.c_void_p
        lib.melotts_load_bert.argtypes=[ct.c_void_p,ct.c_char_p];lib.melotts_load_bert.restype=ct.c_bool
        lib.melotts_set_speaker_id.argtypes=[ct.c_void_p,ct.c_int]
        lib.melotts_source_speaker.argtypes=[ct.c_void_p];lib.melotts_source_speaker.restype=ct.c_char_p
        lib.melotts_synthesize.argtypes=[ct.c_void_p,ct.c_char_p,ct.POINTER(ct.POINTER(ct.c_float)),ct.POINTER(ct.c_int)]
        lib.melotts_synthesize.restype=ct.c_int
        lib.melotts_pcm_free.argtypes=[ct.POINTER(ct.c_float)]
        lib.melotts_free.argtypes=[ct.c_void_p]
        params=lib.melotts_default_params();params.n_threads=2;params.use_gpu=False;params.seed=42
        ctx=lib.melotts_init_from_file(os.fsencode(melo),params)
        if not ctx or not lib.melotts_load_bert(ctx,os.fsencode(bert)):raise RuntimeError('Melo/BERT load failed')
        def synth(text,speaker):
            lib.melotts_set_speaker_id(ctx,speaker)
            speaker_key=lib.melotts_source_speaker(ctx).decode()
            ptr=ct.POINTER(ct.c_float)();rate=ct.c_int()
            count=lib.melotts_synthesize(ctx,text.encode(),ct.byref(ptr),ct.byref(rate))
            if count<=0 or not ptr:raise RuntimeError('Empty Melo output')
            try:return np.ctypeslib.as_array(ptr,shape=(count,)).copy(),rate.value,speaker_key
            finally:lib.melotts_pcm_free(ptr)
        texts=['The quick brown fox jumps over the lazy dog.',
               'We enjoy reading books and listening to music in the evening.']
        reference,ref_sr,ref_key=synth('This is a reference voice for a simple speech test.',1)
        sf.write(OUT/'reference.wav',reference,ref_sr,subtype='FLOAT')
        RECEIPT['reference']={'speaker':ref_key,'rate':ref_sr,'sha256':sha(OUT/'reference.wav')}
        sys.path.insert(0,str(ROOT/'tools'))
        from dump_reference import write_gguf_archive
        from reference_backends.openvoice2 import dump,DEFAULT_STAGES
        os.environ.update(CRISPASR_OV2_UPSTREAM=str(upstream),CRISPASR_OV2_REFERENCE_WAV=str(OUT/'reference.wav'),
                          CRISPASR_OV2_DEVICE='cuda' if torch.cuda.get_device_capability()[0]>=7 else 'cpu',
                          CRISPASR_OV2_TAU='0')
        class OpenParams(ct.Structure):
            _fields_=[('n_threads',ct.c_int),('verbosity',ct.c_int),('use_gpu',ct.c_bool),('tau',ct.c_float)]
        lib.openvoice2_context_default_params.restype=OpenParams
        lib.openvoice2_init_from_file.argtypes=[ct.c_char_p,OpenParams];lib.openvoice2_init_from_file.restype=ct.c_void_p
        lib.openvoice2_set_source_speaker.argtypes=[ct.c_void_p,ct.c_char_p];lib.openvoice2_set_source_speaker.restype=ct.c_bool
        floatp=ct.POINTER(ct.c_float)
        lib.openvoice2_convert.argtypes=[ct.c_void_p,floatp,ct.c_int,ct.c_int,floatp,ct.c_int,ct.c_int,ct.POINTER(floatp),ct.POINTER(ct.c_int)]
        lib.openvoice2_convert.restype=ct.c_bool
        lib.openvoice2_extract_speaker_embedding.argtypes=[ct.c_void_p,floatp,ct.c_int,ct.c_int,floatp]
        lib.openvoice2_extract_speaker_embedding.restype=ct.c_bool
        lib.openvoice2_free.argtypes=[ct.c_void_p]
        libc=ct.CDLL(None);libc.free.argtypes=[ct.c_void_p]
        op=lib.openvoice2_context_default_params();op.n_threads=2;op.use_gpu=True;op.tau=0
        ov=lib.openvoice2_init_from_file(os.fsencode(WORK/'openvoice-f32sens.gguf'),op)
        if not ov:raise RuntimeError('Native conversion context load failed')
        for i,text in enumerate(texts):
            baseline,sr,key=synth(text,0)
            sf.write(OUT/f'baseline-{i}.wav',baseline,sr,subtype='FLOAT')
            import librosa
            source16=librosa.resample(baseline,orig_sr=sr,target_sr=16000)
            os.environ['CRISPASR_OV2_SOURCE_SPEAKER']=key
            captures=dump(model,source16,set(DEFAULT_STAGES))
            meta={name:captures.pop(name) for name in list(captures) if isinstance(captures[name],str)}
            meta.update(backend='openvoice2',source_commit=SOURCE,upstream_commit=UPSTREAM,model_revision=model_rev)
            ref=OUT/f'reference-{i}.gguf'
            write_gguf_archive(captures,meta,ref)
            sf.write(OUT/f'original-{i}.wav',captures['output_pcm'].flatten(),22050,subtype='FLOAT')
            kh.step('native-gpu-parity',case=i)
            env=dict(os.environ,CRISPASR_DIFF_USE_GPU='1')
            run([WORK/'openvoice-bake',WORK/'openvoice-f32sens.gguf',ref],f'diff-gpu-{i}',env=env)
            log=(OUT/f'diff-gpu-{i}.log').read_text()
            if log.count('[PASS]')!=27 or '[FAIL]' in log or 'vocoder backend CUDA' not in log:
                raise RuntimeError('GPU stages missing or not actually CUDA')
            # Same public model C ABI without instrumentation, including
            # reference-cache reuse. Original tensors feed the native input.
            source_pcm=np.ascontiguousarray(captures['src_pcm'].flatten(),dtype=np.float32)
            target_pcm=np.ascontiguousarray(captures['ref_pcm'].flatten(),dtype=np.float32)
            if not lib.openvoice2_set_source_speaker(ov,key.encode()):raise RuntimeError('Source identity rejected')
            native=[]
            for repeat in range(2):
                output=floatp();count=ct.c_int()
                ok=lib.openvoice2_convert(ov,source_pcm.ctypes.data_as(floatp),len(source_pcm),22050,
                    target_pcm.ctypes.data_as(floatp),len(target_pcm),22050,ct.byref(output),ct.byref(count))
                if not ok or not output or count.value<=0:raise RuntimeError('Public C conversion failed')
                try:native.append(np.ctypeslib.as_array(output,shape=(count.value,)).copy())
                finally:libc.free(output)
            if not np.array_equal(native[0],native[1]):raise RuntimeError('Cache/reuse changed deterministic output')
            truth=captures['output_pcm'].flatten()
            cosine=float(np.dot(native[0].astype(np.float64),truth)/(np.linalg.norm(native[0].astype(np.float64))*np.linalg.norm(truth)))
            if cosine<.99:raise RuntimeError('Uninstrumented waveform parity failed')
            embedding=np.empty(256,dtype=np.float32)
            if not lib.openvoice2_extract_speaker_embedding(ov,target_pcm.ctypes.data_as(floatp),len(target_pcm),22050,
                                                         embedding.ctypes.data_as(floatp)):raise RuntimeError('Embedding C ABI failed')
            original_embedding=captures['target_se'].flatten()
            emb_cos=float(np.dot(embedding,original_embedding)/(np.linalg.norm(embedding)*np.linalg.norm(original_embedding)))
            if emb_cos<.999:raise RuntimeError('Embedding C ABI parity failed')
            sf.write(OUT/f'clone-{i}.wav',native[0],22050,subtype='FLOAT')
            RECEIPT['cases'].append({'text':text,'speaker':key,'ref_sha256':sha(ref),'gpu_parity':True,
                                     'uninstrumented_waveform_cosine':cosine,'embedding_cosine':emb_cos,'repeat_exact':True})
            save()
            # Checkpoint each completed reference before any later quality run.
            try:
                published=hub.upload_file(path_or_fileobj=str(ref),repo_id='cstr/crispasr-regression-fixtures',
                    repo_type='dataset',path_in_repo=f'openvoice2/bake-20261019/v1/reference-{i}.gguf',
                    commit_message=f'OpenVoice original-module reference case {i}; quality gate separate')
                RECEIPT['cases'][-1]['published_revision']=published.oid
            except Exception as exc:
                RECEIPT['cases'][-1]['publication_error']=type(exc).__name__
            save()
        lib.melotts_free(ctx)
        lib.openvoice2_free(ov)
        RECEIPT['gpu_executed']=True
        # Independent ASR content and speaker encoder, not OpenVoice's own
        # embedding space, judge actual converted waveforms.
        from faster_whisper import WhisperModel
        from resemblyzer import VoiceEncoder,preprocess_wav
        asr_rev=hub.model_info('Systran/faster-whisper-tiny.en').sha
        asr_dir=snapshot_download('Systran/faster-whisper-tiny.en',revision=asr_rev)
        asr=WhisperModel(asr_dir,device='cpu',compute_type='int8',cpu_threads=2)
        encoder=VoiceEncoder(device=RECEIPT['torch_reference_device'])
        ref_vector=encoder.embed_utterance(preprocess_wav(OUT/'reference.wav'))
        def wer(expected,actual):
            import re
            a=re.findall(r"[a-z]+",expected.lower());b=re.findall(r"[a-z]+",actual.lower())
            row=list(range(len(b)+1))
            for i,x in enumerate(a,1):
                nxt=[i]
                for j,y in enumerate(b,1):nxt.append(min(row[j]+1,nxt[-1]+1,row[j-1]+(x!=y)))
                row=nxt
            return row[-1]/max(1,len(a))
        for i,case in enumerate(RECEIPT['cases']):
            quality={}
            for name in ['baseline','original','clone']:
                path=OUT/f'{name}-{i}.wav'
                segments,_=asr.transcribe(str(path),language='en',beam_size=5)
                text=' '.join(segment.text.strip() for segment in segments)
                vector=encoder.embed_utterance(preprocess_wav(path))
                quality[name]={'transcript':text,'wer':wer(case['text'],text),'speaker_cosine':float(np.dot(vector,ref_vector))}
            case['quality']=quality
            case['quality_pass']=all(q['wer']<=.2 for q in quality.values()) and \
                quality['clone']['speaker_cosine']>quality['baseline']['speaker_cosine']+.01
            save()
        RECEIPT['asr']={'repo':'Systran/faster-whisper-tiny.en','revision':asr_rev}
        RECEIPT['references_published']=all('published_revision' in case for case in RECEIPT['cases'])
        RECEIPT['model_acceptance']=all(case['quality_pass'] for case in RECEIPT['cases'])
        RECEIPT['complete']=True
        RECEIPT['artifacts']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.iterdir() if p.is_file() and p.name!='validation.json'}
        save()
        kh.step('parity-complete',cases=len(RECEIPT['cases']),quality=RECEIPT['model_acceptance'])
        if not RECEIPT['model_acceptance']:
            raise RuntimeError('Content or independent speaker-similarity gate failed; preserve original/native controls')
    finally:
        os.dup2(original_stderr,2);os.close(original_stderr);native_log.close()

if __name__=='__main__':
    try:main()
    except Exception:
        RECEIPT['error']=traceback.format_exc();save();raise
