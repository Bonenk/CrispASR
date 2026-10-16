#!/usr/bin/env python3
"""Owned-input CPU dispatch experiment; no full model or production recipe change."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ORIGINAL = '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def metric(x, y):
    assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
    a, b = x.astype('f8'), y.astype('f8')
    return dict(byte_exact=x.dtype == y.dtype and x.tobytes() == y.tobytes(),
                relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                max_abs=float(np.max(np.abs(a-b))), changed=int(np.count_nonzero(x != y)),
                mine_norm=float(np.linalg.norm(a)), ref_norm=float(np.linalg.norm(b)))

def margin(x):
    a = x.astype('f8')[:, 2]
    return float(4*(a[1,46]-a[1,904])-3*(a[0,46]-a[0,904]))

def child(args):
    import torch
    assert torch.__version__ == '2.6.0+cpu'
    torch.set_num_threads(4)
    inputs = args.output/'inputs'
    source = inputs/'layers.py'
    nodes = [n for n in ast.parse(source.read_text()).body if getattr(n,'name',None) in {'_normalize_axes','DenseGeneral'}]
    assert len(nodes) == 2
    ns = dict(torch=torch, nn=torch.nn, Tensor=torch.Tensor)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    out = args.output/args.child
    out.mkdir(exist_ok=False)
    with torch.inference_mode():
        for operation, weight_shape, features, input_names in [
            ('query',(2048,16,128),(16,128),['query-input']),
            ('head',(2048,9,1028),(9,1028),['original-hidden','native-hidden'])]:
            weight = np.fromfile(inputs/(operation+'-weight.f32'),dtype='<f4').reshape(weight_shape)
            module = ns['DenseGeneral']((2048,),features,weight_dtype=torch.float32)
            module.weight = torch.nn.Parameter(torch.from_numpy(weight),requires_grad=False)
            for name in input_names:
                a = np.load(inputs/(name+'.npy'),allow_pickle=False)
                x = torch.from_numpy(a).reshape(2,1,2048)
                y = module(x).numpy().copy()
                repeated = module(x).numpy().copy()
                assert y.tobytes() == repeated.tobytes()
                np.save(out/(name+'.npy'),y,allow_pickle=False)
                print('EXACT_REPEAT',operation,name,'input_stride',x.stride(),'weight_stride',module.weight.stride(),flush=True)
    (out/'environment.json').write_text(json.dumps(dict(torch_version=torch.__version__,threads=torch.get_num_threads(),
        cpu_capability=torch.backends.cpu.get_cpu_capability(),mkl_cbwr=os.environ.get('MKL_CBWR'),
        torch_config=torch.__config__.show()),indent=2)+'\n')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['output','blueprint','full-trace','query-proof','head-proof','terminal','repo']:
        p.add_argument('--'+name,type=Path,required=name=='output')
    p.add_argument('--child')
    args = p.parse_args()
    if args.child:
        child(args); return
    args.output.mkdir(exist_ok=False)
    inputs = args.output/'inputs'; inputs.mkdir()
    consumed = []
    def retain(path,name,digest=None):
        if digest: assert sha(path) == digest, path
        dest = inputs/name; shutil.copyfile(path,dest)
        consumed.append(dict(path=str(path),retained=name,bytes=path.stat().st_size,sha256=sha(dest)))
        return dest
    source = args.blueprint/'dia/layers.py'
    assert source.read_bytes() == subprocess.check_output(['git','show',ORIGINAL+':dia/layers.py'],cwd=args.blueprint)
    retain(source,'layers.py')
    full_receipt = json.loads((args.full_trace/'receipt.json').read_text())
    rows = {r['path']:r for r in full_receipt['payloads']}
    storage_path = args.full_trace/'arms/on/array-storage.json'
    retain(storage_path,'full-storage.json',rows['arms/on/array-storage.json']['sha256'])
    storage = {r['name']:r for r in json.loads(storage_path.read_text())}
    def actual(name,output):
        row = storage[name]; assert row['sha256'] == rows[row['storage']]['sha256']
        return retain(args.full_trace/row['storage'],output,row['sha256'])
    actual('decoder_block0_pre_sa_norm','query-input.npy')
    actual('decoder_sa0_pre_q','captured-query.npy')
    actual('decoder_hidden','original-hidden.npy')
    actual('decoder_step_logits','captured-logits.npy')
    query = args.query_proof/'inputs/self_attention.q_proj.weight.original.f32'
    retain(query,'query-weight.f32','63da7f82eeb19a61b192a5ef6ee6c33b36f380583f120a7a7943f1dbc9a1e1e3')
    fetch = json.loads((args.head_proof/'fetch-receipt.json').read_text())
    retain(args.head_proof/'original-heads.f32','head-weight.f32',fetch['ranges'][-1]['sha256'])
    retain(args.head_proof/'fetch-receipt.json','head-physical-fetch.json')
    terminal = json.loads((args.terminal/'receipt.json').read_text())
    ledger = {r['path']:r for r in terminal['payloads']}
    native = args.terminal/'baseline/cpp_final_hidden.f32'
    retain(native,'native-hidden.f32',ledger['baseline/cpp_final_hidden.f32']['sha256'])
    np.save(inputs/'native-hidden.npy',np.fromfile(native,dtype='<f4').reshape(2,1,2048),allow_pickle=False)
    report = dict(scope=__doc__,original_source=ORIGINAL,consumed=consumed,modes=[],production_defaults_changed=False,
                  native_speech_acceptance=False)
    (args.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    for mode in ['AUTO','COMPATIBLE','SSE4_2','AVX','AVX2','AVX512']:
        env = {k:v for k,v in os.environ.items() if not k.startswith(('MKL_','OMP_','ATEN_','DIA_','CRISPASR_'))}
        env.update(MKL_CBWR=mode,MKL_VERBOSE='1',MKL_DYNAMIC='FALSE',OMP_NUM_THREADS='4')
        with (args.output/(mode+'.log')).open('w') as log:
            subprocess.run([sys.executable,__file__,'--output',args.output,'--child',mode],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        q = np.load(args.output/mode/'query-input.npy',allow_pickle=False)
        ref = np.load(inputs/'captured-query.npy',allow_pickle=False)
        heads = {}
        for name in ['original-hidden','native-hidden']:
            a = np.load(args.output/mode/(name+'.npy'),allow_pickle=False).reshape(2,9,1028)
            heads[name] = dict(margin_46_minus_904=margin(a))
            if name == 'original-hidden':
                saved = np.load(inputs/'captured-logits.npy',mmap_mode='r',allow_pickle=False)[978]
                heads[name].update(metric(a,saved))
        row = dict(mode=mode,query=metric(q,ref),heads=heads)
        report['modes'].append(row)
        (args.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    # Each product of two F32 numbers is exactly representable in F64;
    # math.fsum provides an independently rounded high-precision sum.
    weight = np.fromfile(inputs/'head-weight.f32',dtype='<f4').reshape(2048,9,1028)
    exact = {}
    for name in ['original-hidden','native-hidden']:
        hidden = np.load(inputs/(name+'.npy'),allow_pickle=False).reshape(2,2048)
        values = np.zeros((2,9,1028),dtype='f8')
        for batch in [0,1]:
            for token in [46,904]:
                values[batch,2,token] = math.fsum(float(x)*float(y) for x,y in zip(hidden[batch],weight[:,2,token]))
        exact[name] = dict(logits=values[:,2,[46,904]].tolist(),margin_46_minus_904=margin(values))
    report['exact_product_sums'] = exact
    report['files'] = [dict(path=f.relative_to(args.output).as_posix(),bytes=f.stat().st_size,sha256=sha(f))
                       for f in sorted(args.output.rglob('*')) if f.is_file() and f.name != 'receipt.json']
    (args.output/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('DIA_DISPATCH_PROBE_COMPLETE',json.dumps(exact),flush=True)

if __name__ == '__main__': main()
