#!/usr/bin/env python3
"""Reject corrupted delayed grids, RNG witnesses, paths and ASR/scale evidence."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('serial_audit',HERE/'audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def reject(call):
    try:call()
    except AssertionError:return
    raise AssertionError('Corrupted evidence accepted')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(exist_ok=False)
    d=np.arange(20*9,dtype='<u4').reshape(20,9)
    # Invalid aligned code is replaced individually, never by deleting its frame.
    d[0,0]=1024
    x=np.vstack([np.full((1,9),1026,dtype='<i4'),d[:-1].astype('<i4')])
    codes=np.stack([d[np.arange(5)+offset,c] for c,offset in enumerate([0,8,9,10,11,12,13,14,15])],axis=1)
    codes=np.where(codes<1024,codes,0).astype('<u4')
    audit.verify_grid(x,d,codes)
    cases=[]
    for slot,index in [(0,(1,0)),(1,(0,1)),(2,(0,1)),(0,(0,1))]:
        wrong=[x.copy(),d.copy(),codes.copy()];wrong[slot][index]+=1
        cases.append(wrong)
    wrong=[x.copy(),d.copy(),codes.copy()];wrong[2][0,0]=1024;cases.append(wrong)
    wrong=[x.copy(),d.copy(),codes[:-1].copy()];cases.append(wrong)
    wrong=[x.astype('f4'),d.copy(),codes.copy()];cases.append(wrong)
    for values in cases:reject(lambda:audit.verify_grid(*values))
    rejected=len(cases)
    cpp=a.output/'rng.cpp';cpp.write_text('#include <random>\n#include <iostream>\nint main(){std::mt19937 r(42);r.discard(1115);std::cout<<r;}\n')
    subprocess.run(['c++','-std=c++17',cpp,'-o',a.output/'rng'],check=True)
    state=subprocess.check_output([a.output/'rng']);(a.output/'rng.txt').write_bytes(state)
    audit.verify_rng(state,state,state)
    for bad in [b'1',state+b'1',b' '.join([b'0']*624+[b'625'])]:
        reject(lambda:audit.verify_rng(state,bad,state));rejected+=1
    for name in ['../escape','/absolute','a/../b','a\\b','.','a//b']:
        reject(lambda:audit.safe(name));rejected+=1
    pcm=np.array([-.25,0,.5,1],dtype='f4');path=a.output/'pcm.npy';np.save(path,pcm,allow_pickle=False)
    row=dict(transcript='We are committed to making it secure.',wer=0,pcm_sha256=audit.sha(path),library_sha256='a'*64,
        asr_sha256='60a2b074cdf46e81d72bc15d3eb79505dafbb697135f558659d9068f651854a2',
        seconds=len(pcm)/44100,rms=float(np.sqrt(np.mean(pcm.astype('f8')**2))))
    audit.check_asr(row,pcm,path,'a'*64)
    for key,value in [('wer',.1),('rms',row['rms']*2),('pcm_sha256','b'*64),('library_sha256','b'*64),('seconds',1)]:
        bad=dict(row);bad[key]=value;reject(lambda:audit.check_asr(bad,pcm,path,'a'*64));rejected+=1
    for value in [np.nan,np.inf]:
        bad=pcm.copy();bad[0]=value;reject(lambda:audit.check_asr(row,bad,path,'a'*64));rejected+=1
    quiet='DIA_CROSS_CACHE: type=F32\nDIA_ROTARY: source=0\n'
    audit.check_native_log(quiet)
    for value in ['max_gen=2048,','DIA_FORCE_TOKENS:','DIA_FORCE_OUTPUT_TOKENS:']:
        reject(lambda:audit.check_native_log(quiet+value));rejected+=1
    result=dict(audit_calibration_pass=True,rejecting_controls=rejected,actual_std_mt19937=True,
        independent_delay_grid_law=True,hosted_artifact_audited=False)
    (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DIA_SERIAL_AUDIT_CALIBRATION_PASS',rejected,'rejects; actual MT state, delay law, paths, ASR and scale')

if __name__=='__main__':main()
