#!/usr/bin/env python3
"""Fresh hosted original COMPATIBLE projection replay; no full model/speech claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
METHOD=Path(__file__).with_name('dia-reduction-cause')
ARCHIVE_SHA='b092ed027e26480ae4ad6c59d30d216662c7c5afe949f5300c9afafe86163397'


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--revision',required=True)
    a=p.parse_args();assert re.fullmatch('[0-9a-f]{40}',a.revision)
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])
    out.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
    if os.environ.get('GITHUB_SHA'): assert source==os.environ['GITHUB_SHA']
    archive=scratch/'proof.zip'
    url='https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/'+a.revision+'/dia-20261016-reduction-cause/complete-reduction-cause-v1.zip'
    with urllib.request.urlopen(url,timeout=300) as response,archive.open('xb') as dest: shutil.copyfileobj(response,dest)
    assert sha(archive)==ARCHIVE_SHA and archive.stat().st_size==170121121
    proof=scratch/'proof';proof.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for item in z.infolist():
            name=Path(item.filename)
            assert not name.is_absolute() and '..' not in name.parts and not stat.S_ISLNK(item.external_attr>>16)
        z.extractall(proof)
    manifest=json.loads((proof/'manifest.json').read_text())
    for row in manifest['files']:
        file=proof/row['path'];assert sha(file)==row['sha256'] and file.stat().st_size==row['bytes']
    for name in ['audit.py','dispatch.py','order.py']:
        assert (METHOD/name).read_bytes()==(proof/'methods'/name).read_bytes()
    with (out/'independent-array-audit.log').open('w') as log:
        subprocess.run([sys.executable,METHOD/'audit.py','--dispatch',proof/'dispatch-v1',
                        '--order',proof/'order-v3','--output',out/'independent-array-audit.json'],check=True,stdout=log,stderr=subprocess.STDOUT)
    env={k:v for k,v in os.environ.items() if not k.startswith(('MKL_','OMP_','ATEN_'))}
    env.update(MKL_CBWR='COMPATIBLE',MKL_VERBOSE='1',MKL_DYNAMIC='FALSE',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    with (out/'fresh-compatible.log').open('w') as log:
        subprocess.run([sys.executable,METHOD/'dispatch.py','--output',proof/'dispatch-v1','--child','fresh-compatible'],
                       env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
    fresh=proof/'dispatch-v1/fresh-compatible'
    for name in ['query-input','original-hidden','native-hidden']:
        actual=np.load(fresh/(name+'.npy'),allow_pickle=False)
        expected=np.load(proof/'dispatch-v1/COMPATIBLE'/(name+'.npy'),allow_pickle=False)
        assert actual.shape==expected.shape and actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes(),name
    log=(out/'fresh-compatible.log').read_text();assert 'CNR:COMPATIBLE' in log and log.count('EXACT_REPEAT')==3
    shutil.copytree(fresh,out/'fresh-compatible');shutil.copytree(METHOD,out/'method')
    shutil.copyfile(proof/'manifest.json',out/'public-proof-manifest.json')
    subprocess.run(['lscpu'],stdout=(out/'lscpu.log').open('w'),check=True)
    receipt=dict(source=source,fixture_revision=a.revision,fixture_sha256=ARCHIVE_SHA,
                 fresh_original_compatible_exact=True,independent_array_audit_pass=True,
                 full_generation_acceptance=False,production_defaults_changed=False)
    receipt['payloads']=[dict(path=f.relative_to(out).as_posix(),bytes=f.stat().st_size,sha256=sha(f))
                         for f in sorted(out.rglob('*')) if f.is_file()]
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('DIA_HOSTED_REDUCTION_CAUSE_PASS: fresh actual original Q and both hidden/head projections match; no model generation',flush=True)

if __name__=='__main__':main()
