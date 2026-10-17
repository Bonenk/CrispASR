#!/usr/bin/env python3
"""Compile/run the actual graph callback with captured and analytic inputs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

HERE=Path(__file__).resolve().parent
PIN='8d0ad8862d2cad5d305a48a57f2a99b5024bc39a'
GGML='c36dab89b662838f0f5d4826c399198c0b90bbfc'


def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['output','repo','headers','libraries','dispatch','order','terminal','ggml-repo','runtime']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    for name in ['head.h','probe.cpp','patch.py']:shutil.copyfile(HERE/name,a.output/name)
    headers=a.output/'headers';headers.mkdir()
    for file in a.headers.glob('*.h'):
        raw=subprocess.check_output(['git','show',GGML+':include/'+file.name],cwd=a.ggml_repo)
        assert raw==file.read_bytes();shutil.copyfile(file,headers/file.name)
    libs=a.output/'libs';libs.mkdir()
    ledger=json.loads((a.libraries.parent/'receipt.json').read_text())['actual_library_hashes']
    for name,digest in ledger.items():
        assert sha(a.libraries/name)==digest;shutil.copyfile(a.libraries/name,libs/name)
    command=['c++','-std=c++17','-O3','-march=native','-I',str(headers),'-I',str(HERE),str(HERE/'probe.cpp'),
             '-Wl,-rpath,$ORIGIN/libs',str(libs/'libggml-cpu.so.0'),str(libs/'libggml-base.so.0'),'-o',str(a.output/'probe')]
    with (a.output/'build.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    (a.output/'build-command.json').write_text(json.dumps(command,indent=2)+'\n')
    env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','GGML_','CRISPASR_','DIA_'))}
    closure=subprocess.check_output(['ldd',a.output/'probe'],env=env,text=True)
    (a.output/'ldd.log').write_text(closure);assert 'not found' not in closure
    for line in closure.splitlines():
        if 'libggml' in line:assert Path(line.split('=>')[1].split()[0]).resolve().parent==libs.resolve()
    d=json.loads((a.dispatch/'receipt.json').read_text()); records={r['path']:r for r in d['files']}
    for name in ['inputs/head-weight.f32','inputs/original-hidden.npy','inputs/native-hidden.npy']:
        assert sha(a.dispatch/name)==records[name]['sha256']
    w=np.fromfile(a.dispatch/'inputs/head-weight.f32',dtype='<f4').reshape(2048,9,1028)
    native_weight=np.ascontiguousarray(w.transpose(1,2,0));native_weight.tofile(a.output/'native-weight.f32')
    native=json.loads((a.terminal/'receipt.json').read_text()); nr={r['path']:r for r in native['payloads']}
    expected_native=[]
    for part in ['uncond','cond']:
        file=a.terminal/('baseline/cpp_step978_'+part+'.f32')
        assert sha(file)==nr[file.relative_to(a.terminal).as_posix()]['sha256']
        expected_native.append(np.fromfile(file,dtype='<f4').reshape(9,1028))
    expected_native=np.stack(expected_native)
    runs=[]
    def execute(label,weights,hidden,mode,threads,success=True):
        dest=a.output/(label+'.f32')
        result=subprocess.run([a.output/'probe',weights,hidden,mode,str(threads),dest],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (a.output/(label+'.log')).write_bytes(result.stdout)
        assert (result.returncode==0)==success
        runs.append(dict(label=label,exit=result.returncode))
        if success:
            assert b'DIA_SERIAL_HEAD_GRAPH_PASS' in result.stdout
            return np.fromfile(dest,dtype='<f4').reshape(2,9,1028)
    for name in ['original-hidden','native-hidden']:
        hidden=np.load(a.dispatch/'inputs'/(name+'.npy')); hp=a.output/(name+'.f32');hidden.tofile(hp)
        expected=np.load(a.order/(name+'-serial.npy'))
        off=None
        for threads in [1,4]:
            for mode in ['off','identity','serial']:
                label=name+'-'+mode+'-'+str(threads)
                actual=execute(label,a.output/'native-weight.f32',hp,mode,threads)
                if mode=='off':
                    if off is not None:assert actual.tobytes()==off.tobytes()
                    off=actual
                    if name=='native-hidden':assert actual.tobytes()==expected_native.tobytes()
                elif mode=='identity':assert actual.tobytes()==off.tobytes()
                else:assert actual.tobytes()==expected.tobytes()
    # Tiny exact fusion control padded to the actual kernel's K=2048 layout.
    tiny=np.zeros_like(native_weight); x=np.zeros((2,2048),dtype='f4')
    tiny[0,0,0]=-1; tiny[0,0,32]=np.float32(1+2**-23)
    x[:,0]=1; x[:,32]=np.float32(1-2**-23)
    tiny.tofile(a.output/'tiny-weight.f32');x.tofile(a.output/'tiny-hidden.f32')
    for threads in [1,4]:
        old=execute('tiny-off-'+str(threads),a.output/'tiny-weight.f32',a.output/'tiny-hidden.f32','off',threads)
        new=execute('tiny-serial-'+str(threads),a.output/'tiny-weight.f32',a.output/'tiny-hidden.f32','serial',threads)
        assert np.equal(old[:,0,0],-2**-46).all() and np.equal(new,0).all()
    for mode in ['invalid','SERIAL','1']:
        execute('reject-'+mode,a.output/'native-weight.f32',hp,mode,4,False)
    spec=importlib.util.spec_from_file_location('serial_patch',HERE/'patch.py');patcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(patcher)
    cpp=a.output/'patched-dia_tts.cpp';cpp.write_bytes(subprocess.check_output(['git','show',PIN+':src/dia_tts.cpp'],cwd=a.repo))
    patcher.patch(cpp)
    shutil.copyfile(HERE/'head.h',a.output/'dia_serial_head.h')
    with (a.output/'whole-runtime-syntax.log').open('w') as log:
        subprocess.run(['g++','-std=c++17','-fsyntax-only','-I',a.runtime/'src','-I',a.runtime/'include',
                        '-I',a.runtime/'ggml/include','-I',a.runtime/'ggml/src',cpp],check=True,stdout=log,stderr=subprocess.STDOUT)
    try:patcher.patch(cpp)
    except AssertionError:pass
    else:raise AssertionError('Repeated patch accepted')
    report=dict(actual_graph_tests_pass=True,captured_head_values_per_hidden=18504,hidden_cases=2,
        thread_counts=[1,4],whole_pinned_runtime_syntax_pass=True,native_capture_exact=True,identity_exact=True,serial_original_recurrence_exact=True,
        tiny_fusion_witness_pass=True,rejecting_controls=4,runs=runs,full_model_acceptance=False,
        files=[dict(path=f.relative_to(a.output).as_posix(),bytes=f.stat().st_size,sha256=sha(f))
               for f in sorted(a.output.rglob('*')) if f.is_file()])
    (a.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DIA_SERIAL_HEAD_PREFLIGHT_PASS: actual graph, full captured heads, one/four threads, identity, fusion and four rejects')

if __name__=='__main__':main()
