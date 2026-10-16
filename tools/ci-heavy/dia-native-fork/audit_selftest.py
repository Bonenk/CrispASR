"""Independent auditor calibration: controlled prefix, actual MT witness, ASR gates."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('independent_native_fork_audit',HERE/'audit.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def reject(call):
    try: call()
    except AssertionError: return
    raise AssertionError('Corrupted evidence accepted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--scratch',type=Path,required=True)
    args = parser.parse_args(); args.scratch.mkdir(parents=True,exist_ok=False)
    guard = args.scratch/'actual-sampler'
    subprocess.run([sys.executable,HERE/'selftest.py','--repo',args.repo,'--scratch',guard],check=True)
    before = (guard/'rng-before.txt').read_bytes(); after = (guard/'rng-after.txt').read_bytes()
    audit.verify_rng(before,after,before)
    reject(lambda:audit.verify_rng(before,after+b'1',before))
    reject(lambda:audit.verify_rng(before,after,before+b'1'))
    reject(lambda:audit.verify_rng(b'1',b'1',b'1'))
    invalid = b' '.join([b'0']*624+[b'625'])
    reject(lambda:audit.verify_rng(invalid,invalid,invalid))
    rejected = 4
    quiet = 'DIA_CROSS_CACHE: type=F32 bytes=23003136\nDIA_ROTARY: source=0 head_dim=128\n'
    audit.check_native_log(quiet)
    for message in ['max_gen=3072,','max_gen 3072,']:
        audit.check_native_log(quiet+message)
    for message in ['max_gen=2048,','max_gen 2048,','DIA_FORCE_TOKENS:', 'DIA_FORCE_OUTPUT_TOKENS:']:
        reject(lambda:audit.check_native_log(quiet+message)); rejected += 1
    for required in ['DIA_CROSS_CACHE: type=F32','DIA_ROTARY: source=0']:
        reject(lambda:audit.check_native_log(quiet.replace(required,''))); rejected += 1
    base = np.arange(981*9,dtype='i4').reshape(981,9); base[979,2] = 904
    fork = base.copy(); fork[979,2] = 877
    logits = np.arange(981*2*9*3,dtype='f4').reshape(981,2,9,3)
    codes = np.arange(966*9,dtype='u4').reshape(966,9)
    values = [base,fork,logits,logits.copy(),codes,codes.copy()]
    audit.verify_fork(*values)
    for slot,index in [(1,(978,2)),(1,(979,1)),(3,(978,1,2,0)),(5,(962,1))]:
        wrong = [v.copy() for v in values]; wrong[slot][index] += 1
        reject(lambda:audit.verify_fork(*wrong)); rejected += 1
    for token in [904,878]:
        wrong = [v.copy() for v in values]; wrong[1][979,2] = token
        reject(lambda:audit.verify_fork(*wrong)); rejected += 1
    # An actual trajectory may diverge after the single controlled decision.
    later = [v.copy() for v in values]; later[1][980,0] += 1; later[3][979,0,0,0] += 1; later[5][963,0] += 1
    audit.verify_fork(*later)
    pcm = np.array([-.25,0,.5,1],dtype='f4'); path = args.scratch/'pcm.npy'; np.save(path,pcm,allow_pickle=False)
    row = dict(transcript='We are committed to making it secure.',wer=0,pcm_sha256=audit.sha(path),
               library_sha256='a'*64,asr_sha256='60a2b074cdf46e81d72bc15d3eb79505dafbb697135f558659d9068f651854a2',
               seconds=len(pcm)/44100,rms=float(np.sqrt(np.mean(pcm.astype('f8')**2))))
    audit.check_asr(row,pcm,path,'a'*64)
    missing = dict(row,transcript='We are committed to making it.',wer=1/7)
    audit.check_asr(missing,pcm,path,'a'*64)
    for key,value in [('wer',0.1),('rms',row['rms']*2),('pcm_sha256','b'*64),('library_sha256','b'*64),('seconds',1)]:
        wrong = dict(row); wrong[key] = value
        reject(lambda:audit.check_asr(wrong,pcm,path,'a'*64)); rejected += 1
    for value in [np.nan,np.inf]:
        bad = pcm.copy(); bad[0] = value
        reject(lambda:audit.check_asr(row,bad,path,'a'*64)); rejected += 1
    for name in ['../escape','/absolute','a/../b','a\\b','.','a//b']:
        reject(lambda:audit.safe(name)); rejected += 1
    result = dict(audit_contracts_pass=True,rejecting_controls=rejected,actual_pinned_sampler_guard=True,
                  controlled_prefix=True,serialized_mt_state=True,asr_edit_distance_and_scale=True,
                  hosted_artifact_audited=False)
    (args.scratch/'calibration.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DIA_NATIVE_FORK_AUDIT_SELFTEST_PASS:',rejected,'rejecting controls, actual MT witness, controlled prefix and ASR score/scale')


if __name__ == '__main__': main()
