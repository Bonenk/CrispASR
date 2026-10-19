#!/usr/bin/env python3
"""Original ReferenceEncoder vs literal native CPU body, without a checkpoint.

Small deterministic synthetic tensors exercise all six convolutions and three
GRU steps. No full SDK/GGML build, model download, pretrained inference or GPU.
The unchanged native math reads F32 vectors; GGUF/dequantization/STFT/cache and
trained-model quality are deliberately outside this fixture.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def block(text, signature):
    start=text.index(signature);opening=text.index('{',start);end=opening+1;depth=1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[start:end]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.nn.utils import weight_norm
    torch.set_num_threads(1);torch.manual_seed(20261019)
    original=(args.upstream/'openvoice/models.py').read_text()
    node=next(n for n in ast.parse(original).body if isinstance(n,ast.ClassDef) and n.name=='ReferenceEncoder')
    namespace=dict(torch=torch,nn=nn,F=F,weight_norm=weight_norm)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'original-ReferenceEncoder','exec'),namespace)
    model=namespace['ReferenceEncoder'](513,gin_channels=256).eval()
    captures={};handles=[]
    def save(name,tensor):captures[name]=tensor.detach().cpu().numpy().copy()
    handles.append(model.layernorm.register_forward_hook(lambda m,a,x:save('ref_enc_layernorm',x)))
    for i,conv in enumerate(model.convs):
        handles.append(conv.register_forward_hook(lambda m,a,x,i=i:save(f'ref_enc_conv{i}',F.relu(x))))
    handles.append(model.gru.register_forward_hook(lambda m,a,x:save('ref_enc_gru_out',x[1])))
    handles.append(model.proj.register_forward_hook(lambda m,a,x:save('target_se',x)))
    spec=torch.rand(1,129,513)*4+.1
    with torch.inference_mode():expected=model(spec)
    for handle in handles:handle.remove()
    # Read each original module's effective (weight-norm fused) coefficients.
    weights=[model.layernorm.weight,model.layernorm.bias]
    for conv in model.convs:weights.extend([conv.weight,conv.bias])
    weights.extend([model.gru.weight_ih_l0,model.gru.weight_hh_l0,model.gru.bias_ih_l0,model.gru.bias_hh_l0,
                    model.proj.weight,model.proj.bias])
    with (out/'weights.bin').open('wb') as stream:
        for weight in weights:
            array=weight.detach().float().cpu().numpy().astype('<f4').ravel()
            stream.write(struct.pack('<Q',array.size));stream.write(array.tobytes())
    spec.numpy().astype('<f4').tofile(out/'spec.bin')
    native=(ROOT/'src/openvoice2.cpp').read_text()
    hp=block(native,'struct openvoice2_hparams {')+';'
    encoder=block(native,'struct ov2_ref_enc {')+';'
    function=block(native,'static bool ref_enc_forward(')
    preamble=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
struct ggml_tensor {std::vector<float> values;};
static std::string output_dir;
'''+hp+'\n'+encoder+r'''
struct openvoice2_context {openvoice2_hparams hp;struct{ov2_ref_enc ref_enc;}w;int verbosity=0;};
static void read_f32(const ggml_tensor* tensor,std::vector<float>& out){out=tensor->values;}
static void dump_stage(const openvoice2_context*,const char* label,const float* data,size_t n){
 FILE* f=fopen((output_dir+"/"+label+".bin").c_str(),"wb");assert(f);assert(fwrite(data,sizeof(float),n,f)==n);fclose(f);
}
'''
    runner=r'''
int main(int argc,char** argv){
 assert(argc==4);output_dir=argv[3];std::ifstream file(argv[1],std::ios::binary);assert(file);
 std::vector<std::unique_ptr<ggml_tensor>> owned;
 auto tensor=[&](){uint64_t n=0;file.read(reinterpret_cast<char*>(&n),8);assert(n && n<10000000);
 auto t=std::make_unique<ggml_tensor>();t->values.resize(n);file.read(reinterpret_cast<char*>(t->values.data()),n*4);assert(file);
 auto* result=t.get();owned.push_back(std::move(t));return result;};
 openvoice2_context ctx;auto& re=ctx.w.ref_enc;re.ln_w=tensor();re.ln_b=tensor();re.convs.resize(6);
 for(auto& conv:re.convs){conv.w=tensor();conv.b=tensor();}
 re.gru_w_ih=tensor();re.gru_w_hh=tensor();re.gru_b_ih=tensor();re.gru_b_hh=tensor();re.proj_w=tensor();re.proj_b=tensor();
 std::vector<float> spec(129*513);std::ifstream input(argv[2],std::ios::binary);input.read(reinterpret_cast<char*>(spec.data()),spec.size()*4);assert(input);
 std::vector<float> embedding;assert(ref_enc_forward(&ctx,spec,129,embedding));assert(embedding.size()==256);
}
'''
    receipt=dict(schema=1,kind='synthetic ReferenceEncoder CPU arithmetic; not pretrained acceptance',seed=20261019,
                 torch=torch.__version__,input_shape=[1,129,513],gru_steps=3,pretrained=False,gpu_executed=False,
                 native_source_sha256=hashlib.sha256((ROOT/'src/openvoice2.cpp').read_bytes()).hexdigest(),
                 original_source_sha256=hashlib.sha256((args.upstream/'openvoice/models.py').read_bytes()).hexdigest(),
                 native_body_sha256=hashlib.sha256(function.encode()).hexdigest(),
                 weights_sha256=hashlib.sha256((out/'weights.bin').read_bytes()).hexdigest(),
                 input_sha256=hashlib.sha256((out/'spec.bin').read_bytes()).hexdigest(),
                 compiler=subprocess.check_output([os.environ.get('CXX','c++'),'--version'],text=True).splitlines()[0],
                 cases={})
    with tempfile.TemporaryDirectory(prefix='openvoice-refenc-toy-') as tmp:
        temp=Path(tmp)
        for variant in ['native','swapped_axes']:
            variant_dir=out/variant;variant_dir.mkdir(exist_ok=True)
            body=function
            if variant=='swapped_axes':
                body=body.replace('int H = T_spec, W = n_bins, C_in = 1;', 'int H = n_bins, W = T_spec, C_in = 1;',1)
                if body==function:raise RuntimeError('Negative control mutation did not apply')
            source=temp/(variant+'.cpp');source.write_text(preamble+body+runner)
            binary=temp/variant
            subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O2',str(source),'-o',str(binary)],check=True)
            subprocess.run([str(binary),str(out/'weights.bin'),str(out/'spec.bin'),str(variant_dir)],check=True)
            stages={}
            for name,truth in captures.items():
                mine=np.fromfile(variant_dir/(name+'.bin'),dtype='<f4').astype(np.float64)
                truth=truth.ravel().astype(np.float64)
                if mine.shape!=truth.shape:raise RuntimeError('Stage count mismatch: '+name)
                a=np.linalg.norm(mine);b=np.linalg.norm(truth);delta=mine-truth
                cosine=float(np.dot(mine,truth)/(a*b));ratio=float(a/b);relative=float(np.linalg.norm(delta)/b)
                stages[name]=dict(elements=len(mine),cosine=cosine,norm_mine=float(a),norm_ref=float(b),
                                  norm_ratio=ratio,relative_l2=relative,max_abs=float(np.max(np.abs(delta))),
                                  passed=bool(np.isfinite(mine).all() and cosine>=.99999 and abs(ratio-1)<1e-4 and relative<1e-3))
            receipt['cases'][variant]=stages
    receipt['passed']=all(row['passed'] for row in receipt['cases']['native'].values()) and \
                       not receipt['cases']['swapped_axes']['target_se']['passed']
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    if not receipt['passed']:raise SystemExit(1)


if __name__=='__main__':main()
