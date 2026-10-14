#!/usr/bin/env python3
"""Exercise the complete portable model validator and GPU flag on a CPU host."""
import json
import os
from pathlib import Path
import subprocess
import sys

import cann_model_fixture as fixture


def main():
    fixture.main()
    out = Path(os.environ['HEAVY_OUT'])
    root = Path(__file__).resolve().parents[2]
    build = Path(os.environ['HEAVY_SCRATCH'])/'cann-model-fixture/build'
    run = dict(passed=False, commands=[], source=json.loads((out/'receipt.json').read_text())['source'],
               scope='Full portable validation command exercised on CPU and deliberately unavailable CANN; no NPU acceptance.')
    receipt = out/'validator-receipt.json'
    def save():receipt.write_text(json.dumps(run,indent=2)+'\n')
    def execute(label, command, expected, env=None):
        with (out/(label+'.log')).open('w') as log:
            code = subprocess.run([str(x) for x in command],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        row=dict(label=label,command=[str(x) for x in command],exit=code,expected=expected)
        run['commands'].append(row);save();assert code==expected,row
    save()
    execute('weight-free-controls',[sys.executable,root/'tests/test_cann_model_validator.py','-v'],0)
    from huggingface_hub import hf_hub_download
    archive=Path(hf_hub_download('cstr/crispasr-regression-fixtures',
                 'cann-20261014-model-strict/complete-hosted-strict-fixture.zip',
                 revision='9cf29ec347a91d7a96177cfdc048bb235f8c3289',repo_type='dataset'))
    command=[sys.executable,root/'tools/validate_cann_model.py','--fixture-zip',archive,'--build-dir',build]
    execute('complete-cpu-validation',command+['--backend','cpu','--out',out/'portable-cpu'],0)
    cpu=json.loads((out/'portable-cpu/receipt.json').read_text())
    assert cpu['passed'] and cpu['arms']['cpu']['passed'] and len(cpu['runs'])==7
    execute('complete-cann-rejection',command+['--backend','cann','--out',out/'portable-missing-cann'],1)
    denied=json.loads((out/'portable-missing-cann/receipt.json').read_text())
    assert not denied['passed'] and denied['arms']['cpu']['passed'] and denied['error']=='Supplied build does not enable CANN'
    # Invoke the real public encoder init with GPU=true. CPU-only host must
    # reject the missing registry, not a GPU=false argument wiring mistake.
    lib=build/'src/libcrispasr.so'
    env=dict(os.environ,LD_LIBRARY_PATH=':'.join(sorted({str(p.parent) for p in build.rglob('*.so')})),
             CRISPASR_MOONSHINE_REQUIRE_BACKEND='cann',CRISPASR_MOONSHINE_REQUIRE_DEVICE='0')
    stage=out/'gpu-flag';stage.mkdir()
    execute('gpu-flag-missing-cann',[sys.executable,fixture.__file__,'--native','--encoder-only','--use-gpu',
              '--model',out/'moonshine-tiny.gguf','--lib',lib,'--audio',out/'audio.wav','--output',stage],1,env)
    text=(out/'gpu-flag-missing-cann.log').read_text()
    assert 'required registry=cann ordinal=0 unavailable; refusing fallback' in text
    assert 'GPU disabled' not in text
    run['passed']=True;save()
    (out/'summary.md').write_text('Complete portable Moonshine validation interface passes CPU reference/diff/CLI/auto+explicit Sessions; rejects unavailable CANN after CPU controls. Actual GPU=true public encoder init rejects missing CANN, proving flag forwarding. No physical NPU result.\n')


if __name__=='__main__':main()
