#!/usr/bin/env python3
"""Identify the exact multiply/add recurrence; replay the unchanged original filter."""
import argparse
import ast
import ctypes
import hashlib
import json
from pathlib import Path
import types
from unittest import mock

import numpy as np
import torch


def sha(p):
    with p.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dispatch',type=Path,required=True)
    p.add_argument('--blueprint',type=Path,required=True)
    p.add_argument('--noise',type=Path,required=True)
    p.add_argument('--ggml-library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--terminal',type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(exist_ok=False)
    assert torch.__version__=='2.6.0+cpu'; torch.set_num_threads(4)
    source=args.blueprint/'dia/model.py'
    assert sha(source)=='6bc51e6e5ff58ca9ed66431773df6cda954afee3d4b7ea515c1363b73ca82926'
    tree=ast.parse(source.read_text()); sampler=next(n for n in tree.body if getattr(n,'name',None)=='_sample_next_token')
    cls=next(n for n in tree.body if getattr(n,'name',None)=='Dia')
    step=next(n for n in cls.body if getattr(n,'name',None)=='_decoder_step')
    ns=dict(torch=torch,DecoderInferenceState=object)
    exec(compile(ast.Module(body=[sampler,step],type_ignores=[]),str(source),'exec'),ns)
    assert sha(args.noise)=='043808949bf0fc51206bc154a5755fb80d44127326bc7419c0ca4c635cf77dba'
    noise=np.memmap(args.noise,dtype='<f4',mode='r').reshape(1226,9,1028)[978].copy()
    np.save(args.output/'noise.npy',noise,allow_pickle=False)
    receipt=json.loads((args.dispatch/'receipt.json').read_text()); records={r['path']:r for r in receipt['files']}
    for row in receipt['files']: assert sha(args.dispatch/row['path'])==row['sha256']
    w=np.fromfile(args.dispatch/'inputs/head-weight.f32',dtype='<f4').reshape(2048,9,1028)
    captured={}
    def observe(probabilities,num_samples):
        assert num_samples==1; captured['probabilities']=probabilities.numpy().copy()
        return probabilities.argmax(dim=-1,keepdim=True)
    def draw(raw,label):
        logits=torch.from_numpy(raw.copy()).reshape(2,1,9,1028)
        fake=types.SimpleNamespace(config=types.SimpleNamespace(data=types.SimpleNamespace(audio_eos_value=1024)),
            model=types.SimpleNamespace(decoder=types.SimpleNamespace(decode_step=lambda *_:logits)))
        with mock.patch.object(torch,'multinomial',observe): ns['_decoder_step'](fake,None,None,3.,1.2,.95,45)
        prob=captured['probabilities']; sampled=np.argmax(prob/noise,axis=-1)
        np.save(args.output/(label+'-probabilities.npy'),prob,allow_pickle=False)
        return dict(draws=sampled.tolist(),head2_support=np.flatnonzero(prob[2]).tolist(),head2_draw=int(sampled[2]))
    library_ledger=json.loads((args.ggml_library.parent.parent/'receipt.json').read_text())['actual_library_hashes']
    for name,digest in library_ledger.items(): assert sha(args.ggml_library.parent/name)==digest
    ctypes.CDLL(str(args.ggml_library.with_name('libggml-base.so.0')),mode=ctypes.RTLD_GLOBAL)
    lib=ctypes.CDLL(str(args.ggml_library)); dot=lib.ggml_vec_dot_f32
    fp=ctypes.POINTER(ctypes.c_float)
    dot.argtypes=[ctypes.c_int,fp,ctypes.c_size_t,fp,ctypes.c_size_t,fp,ctypes.c_size_t,ctypes.c_int]
    libm=ctypes.CDLL('libm.so.6'); fma=libm.fmaf
    fma.argtypes=[ctypes.c_float]*3; fma.restype=ctypes.c_float
    # Analytic witness isolates fusion from an already rounded product.
    u=np.float32(1+2**-23); v=np.float32(1-2**-23)
    separate=np.float32(np.float32(u*v)+np.float32(-1))
    fused=fma(float(u),float(v),-1.)
    assert separate==0 and fused==-2**-46
    qweight=np.fromfile(args.dispatch/'inputs/query-weight.f32',dtype='<f4').reshape(2048,16,128)
    qinput=np.load(args.dispatch/'inputs/query-input.npy').reshape(2,2048)
    qserial=np.zeros((2,16,128),dtype='f4')
    for k in range(2048): qserial=np.add(qserial,qinput[:,k,None,None]*qweight[k],dtype='f4')
    assert qserial.tobytes()==np.load(args.dispatch/'inputs/captured-query.npy').tobytes()
    np.save(args.output/'serial-query.npy',qserial,allow_pickle=False)
    terminal=json.loads((args.terminal/'receipt.json').read_text())
    terminal_rows={r['path']:r for r in terminal['payloads']}
    native_logits=[]
    for name in ['uncond','cond']:
        file=args.terminal/('baseline/cpp_step978_'+name+'.f32')
        assert sha(file)==terminal_rows[file.relative_to(args.terminal).as_posix()]['sha256']
        native_logits.append(np.fromfile(file,dtype='<f4').reshape(9,1028))
    native_logits=np.stack(native_logits)
    results=[]
    for name in ['original-hidden','native-hidden']:
        x=np.load(args.dispatch/'inputs'/(name+'.npy')).reshape(2,2048)
        reference=np.load(args.dispatch/'COMPATIBLE'/(name+'.npy')).reshape(2,9,1028)
        serial=np.zeros((2,9,1028),dtype='f4')
        for k in range(2048):
            product=x[:,k,None,None]*w[k]
            serial=np.add(serial,product,dtype='f4')
        assert serial.tobytes()==reference.tobytes()
        np.save(args.output/(name+'-serial.npy'),serial,allow_pickle=False)
        serial_draw=draw(serial,name+'-serial')
        # Literal actual c36 AVX2 dot: four eight-lane FMA accumulators,
        # then (lane0+lane2)+(lane1+lane3), high/low fold and two hadds.
        dots=[]
        for batch in [0,1]:
            for token in [46,904]:
                a=np.ascontiguousarray(w[:,2,token]); b=np.ascontiguousarray(x[batch]); out=ctypes.c_float()
                dot(2048,ctypes.byref(out),0,a.ctypes.data_as(fp),0,b.ctypes.data_as(fp),0,1)
                lanes=np.zeros((4,8),dtype='f4')
                for block in range(0,2048,32):
                    for group in range(4):
                        for lane in range(8):
                            k=block+group*8+lane;lanes[group,lane]=fma(float(a[k]),float(b[k]),float(lanes[group,lane]))
                folded=np.add(np.add(lanes[0],lanes[2]),np.add(lanes[1],lanes[3]))
                four=folded[:4]+folded[4:]; two=four[::2]+four[1::2]; modeled=np.float32(two[0]+two[1])
                assert modeled.tobytes()==np.float32(out.value).tobytes()
                if name=='native-hidden': assert modeled.tobytes()==native_logits[batch,2,token].tobytes()
                dots.append(dict(batch=batch,token=token,ggml=out.value,explicit_fma_recurrence=float(modeled)))
        mode_draws={mode:draw(np.load(args.dispatch/mode/(name+'.npy')).reshape(2,9,1028),name+'-'+mode)
                    for mode in ['AUTO','COMPATIBLE','AVX2','AVX512']}
        results.append(dict(input=name,all_18504_serial_logits_byte_exact=True,serial=serial_draw,
                            mode_draws=mode_draws,actual_ggml_dot_recurrence=dots))
    assert results[1]['serial']['head2_draw']==877
    assert results[1]['mode_draws']['AUTO']['head2_draw']==904
    report=dict(scope=__doc__,results=results,all_4096_query_values_serial_exact=True,
                tiny_fma_witness=dict(separate=float(separate),fused=float(fused),exact=-2**-46),
                native_dot_matches_hosted_logits=True,ggml_library_sha256=sha(args.ggml_library),
                sampler_source_sha256=sha(source),noise_sha256=sha(args.noise),native_speech_acceptance=False)
    (args.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DIA_ORDER_CAUSE_PASS: exact serial logits, actual AVX2/FMA dot recurrence, native-hidden draw904->877',flush=True)

if __name__=='__main__': main()
