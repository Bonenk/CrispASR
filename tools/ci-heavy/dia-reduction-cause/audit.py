#!/usr/bin/env python3
"""Independent saved-array audit of CPU dispatch and serial-recurrence evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def sha(p):
    with p.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def serial(x,w):
    # Separate implementation: round products, then cumulative F32 addition
    # along K, one batch at a time. This does not call the producer's loop.
    outputs=[]
    for row in x:
        product=np.multiply(w,row[:,None,None],dtype=np.float32)
        outputs.append(np.cumsum(product,axis=0,dtype=np.float32)[-1])
    return np.stack(outputs)


def same(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def margin(a):
    z=a.astype('f8')[:,2]
    return float((z[1,46]-z[1,904])*4-(z[0,46]-z[0,904])*3)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['dispatch','order','output']: parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args(); assert not a.output.exists()
    receipt=json.loads((a.dispatch/'receipt.json').read_text())
    for row in receipt['files']:
        path=Path(row['path']); assert not path.is_absolute() and '..' not in path.parts
        file=a.dispatch/path; assert file.stat().st_size==row['bytes'] and sha(file)==row['sha256']
    assert receipt['original_source']=='4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'
    fetch=json.loads((a.dispatch/'inputs/head-physical-fetch.json').read_text())
    assert fetch['checkpoint_linked_sha256']=='caba289b60f6d7d1e58fc744f4dc25aae88995fcca46be3d05e220b971486a26'
    w=np.fromfile(a.dispatch/'inputs/head-weight.f32',dtype='<f4').reshape(2048,9,1028)
    for i,row in enumerate(fetch['verified_mapped_heads']):
        mapped=np.ascontiguousarray(w[:,i,:].T)
        assert mapped.nbytes==row['bytes'] and hashlib.sha256(mapped.tobytes()).hexdigest()==row['sha256']
    captured=np.load(a.dispatch/'inputs/captured-logits.npy',mmap_mode='r')[978]
    actual_q=np.load(a.dispatch/'inputs/captured-query.npy')
    modes={r['mode']:r for r in receipt['modes']}; assert len(modes)==6
    metrics=[]
    for mode in modes:
        log=(a.dispatch/(mode+'.log')).read_text()
        assert log.count('EXACT_REPEAT')==3 and log.count('SGEMM(')==6
        actual_mode='SSE4_2' if mode=='AVX' else mode
        assert f'CNR:{actual_mode}' in log and 'NThr:4' in log
        q=np.load(a.dispatch/mode/'query-input.npy'); h=np.load(a.dispatch/mode/'original-hidden.npy').reshape(2,9,1028)
        if mode in ['COMPATIBLE','SSE4_2','AVX']:
            same(q,actual_q); same(h,captured)
        else: assert q.tobytes()!=actual_q.tobytes() and h.tobytes()!=captured.tobytes()
        for label,x,y in [('query',q,actual_q),('heads',h,captured)]:
            xf=x.astype('f8').reshape(-1,x.shape[-1]); yf=y.astype('f8').reshape(xf.shape)
            xn=np.linalg.norm(xf,axis=1); yn=np.linalg.norm(yf,axis=1)
            metrics.append(dict(mode=mode,tensor=label,cosine_min=float(np.min(np.sum(xf*yf,axis=1)/(xn*yn))),
                mine_norm=float(np.linalg.norm(xf)),ref_norm=float(np.linalg.norm(yf)),
                relative_l2=float(np.linalg.norm(xf-yf)/np.linalg.norm(yf)),max_abs=float(np.max(abs(xf-yf)))))
    for name in ['original-hidden','native-hidden']:
        x=np.load(a.dispatch/'inputs'/(name+'.npy')).reshape(2,2048)
        s=serial(x,w); same(s,np.load(a.dispatch/'COMPATIBLE'/(name+'.npy')).reshape(2,9,1028))
        same(s,np.load(a.order/(name+'-serial.npy')))
        assert margin(s)==modes['COMPATIBLE']['heads'][name]['margin_46_minus_904']
    qweight=np.fromfile(a.dispatch/'inputs/query-weight.f32',dtype='<f4').reshape(2048,16,128)
    qx=np.load(a.dispatch/'inputs/query-input.npy').reshape(2,2048)
    same(serial(qx,qweight),actual_q.reshape(2,16,128))
    order=json.loads((a.order/'receipt.json').read_text())
    assert order['all_4096_query_values_serial_exact'] and order['native_dot_matches_hosted_logits']
    assert order['tiny_fma_witness']==dict(separate=0.,fused=-2**-46,exact=-2**-46)
    native=next(r for r in order['results'] if r['input']=='native-hidden')
    assert native['serial']['head2_draw']==877 and native['mode_draws']['AUTO']['head2_draw']==904
    # Scale and reassociation controls must not pass the exact recurrence law.
    assert (2*captured).tobytes()!=captured.tobytes()
    x=np.load(a.dispatch/'inputs/native-hidden.npy').reshape(2,2048)
    f64=np.einsum('bk,khv->bhv',x.astype('f8'),w.astype('f8'),optimize=False).astype('f4')
    assert f64.tobytes()!=np.load(a.order/'native-hidden-serial.npy').tobytes()
    result=dict(independent_saved_array_audit_pass=True,dispatch_modes=6,physical_head_mappings=9,
        serial_query_values=4096,serial_head_values_per_hidden=18504,hidden_inputs=2,
        separate_cumulative_sum_reconstruction=True,metrics=metrics,native_sample_904_to_877=True,
        full_generation_acceptance=False,production_defaults_changed=False)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('DIA_REDUCTION_CAUSE_INDEPENDENT_AUDIT_PASS: six dispatches, full Q/heads, independent cumsum, original/native draw witness')

if __name__=='__main__': main()
