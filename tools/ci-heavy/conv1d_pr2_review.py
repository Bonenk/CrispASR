#!/usr/bin/env python3
"""Build the reviewed Conv1D proposal and test CPU + software Vulkan on GH.

No weights or speed claim. This cannot certify Radeon accumulation or speech.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BASE = 'e2ae679a563014f46c546ce6c6448e8601e4dea6'
PROBE_SHA = '5a6d48da9ddcb5c886a911db0e802d7fbebf2d582bf343d633061bb229aea8be'
PROOF_URL = ('https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/'
             'ad638b4cd525c5633f831da9eb763864e9827ce9/issue-followups-20261020/complete-proof.zip')
PROOF_SHA = 'e446e1a4b2a3c925033598677d3555558f35470d5957d058d0c4c37e9731c70e'
PATCH_URL = 'https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/10385d27b50d21906fff6ee407ac8031c1a08e02/issue-followups-20261020/ggml-pr2-applied.patch'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH'])/'conv1d-pr2'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=False)
    receipt = dict(passed=False, ggml_base=BASE, physical_gpu=False,
                   scope='review candidate scheduler composition: CPU/software-Vulkan numerical, strided-input and BF16-weight controls; no Radeon/speech/performance acceptance',
                   source=subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD'],text=True).strip())

    def run(name, argv, *, env=None, cwd=None):
        with (out/(name+'.log')).open('w') as f:
            subprocess.run(list(map(str,argv)), stdout=f, stderr=subprocess.STDOUT, check=True, env=env, cwd=cwd)

    try:
        archive = scratch/'proof.zip'
        with urllib.request.urlopen(PROOF_URL,timeout=60) as r:archive.write_bytes(r.read())
        assert sha(archive) == PROOF_SHA
        with zipfile.ZipFile(archive) as z:
            identity = json.loads(z.read('controls/conv1d-pr2-review-identity.json'))
        patch = scratch/'candidate.patch'
        with urllib.request.urlopen(PATCH_URL,timeout=30) as response:
            patch.write_bytes(response.read())
        assert sha(patch) == identity['applied_patch_sha256']
        assert sha(ROOT/'tools/ci-heavy/conv1d_pr2_probe.cpp') == PROBE_SHA
        candidate = scratch/'ggml'
        run('worktree',['git','-C',ROOT/'ggml','worktree','add','--detach',candidate,BASE])
        run('apply',['git','apply',patch],cwd=candidate)
        for name, expected in identity['source_hashes'].items():assert sha(candidate/name) == expected
        receipt['candidate_source_hashes'] = identity['source_hashes']
        receipt['original_probe_sha256'] = identity['probe_sha256']
        receipt['probe_sha256'] = PROBE_SHA
        run('dependencies',['sudo','apt-get','update'])
        run('install',['sudo','apt-get','install','-y','ninja-build','libvulkan-dev','vulkan-tools','spirv-headers','glslc','mesa-vulkan-drivers'])
        env = dict(os.environ, GGML_VK_VISIBLE_DEVICES='0')
        run('vulkan-devices',['vulkaninfo','--summary'],env=env)
        build = scratch/'build'
        run('configure',['cmake','-S',candidate,'-B',build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
                         '-DGGML_NATIVE=OFF','-DGGML_VULKAN=ON','-DGGML_METAL=OFF','-DGGML_BLAS=OFF',
                         '-DGGML_BUILD_TESTS=OFF','-DGGML_BUILD_EXAMPLES=OFF','-DGGML_CONV1D_VIA_CONV2D=ON',
                         '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'])
        run('build',['cmake','--build',build,'-j2'])
        commands=(build/'compile_commands.json').read_text()
        assert 'GGML_CONV1D_VIA_CONV2D' in commands
        binary=scratch/'probe'
        run('probe-build',['c++','-std=c++17','-O2','-I'+str(candidate/'include'),
                           ROOT/'tools/ci-heavy/conv1d_pr2_probe.cpp','-L'+str(build/'src'),
                           '-lggml','-lggml-base','-lggml-cpu','-Wl,-rpath,'+str(build/'src'),'-o',binary])
        receipt['binary_sha256']=sha(binary)
        for label,args,count in [('cpu',['CPU'],100),('cpu-fallback',['CPU','--fallback-only'],4),
                                 ('cpu-bf16',['CPU','--bf16-fallback-only'],4),
                                 ('vulkan',['Vulkan0'],100),('vulkan-fallback',['Vulkan0','--fallback-only'],4),
                                 ('vulkan-bf16',['Vulkan0','--bf16-fallback-only'],4)]:
            run(label,[binary]+args,env=env)
            log=(out/(label+'.log')).read_text()
            assert f'cases={count} failed=0' in log
            assert sum(line.startswith('case ') and line.endswith('pass=1') for line in log.splitlines()) == count
            if count == 100:
                direct = [line for line in log.splitlines() if line.startswith('placement op=CONV_2D ')]
                assert len(direct) == 50
                assert all('requested='+args[0]+' selected='+args[0] in line for line in direct)
        receipt['passed']=True
    except BaseException:
        receipt['failure']=traceback.format_exc()
        raise
    finally:
        receipt['files']={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file() and p.name!='receipt.json'}
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        (out/'summary.md').write_text('# Conv1D proposal CPU/software-Vulkan review\n\nPassed: '+str(receipt['passed'])+'\n\n'+receipt['scope']+'\n')


if __name__ == '__main__':
    main()
