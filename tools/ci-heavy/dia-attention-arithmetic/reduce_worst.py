from pathlib import Path
import argparse,hashlib,json,math,os,subprocess
import numpy as np
import torch
parser=argparse.ArgumentParser(description='Reduce the worst fixed-probability weighted-value difference to one actual ggml dot.');parser.add_argument('--proof',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();proof=args.proof.resolve()
r=json.loads((proof/'receipt.json').read_text());assert r['complete']
def sha(f):
 with f.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
assert sha(proof/'bridge')==r['binary_sha256']
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
case=proof/'original-module-replay'
p=np.fromfile(case/'shadow-probabilities.f32',dtype='<f4').reshape(2,16,1,979)
v=np.fromfile(case/'v.f32',dtype='<f4').reshape(2,979,4,128)
g=np.fromfile(case/'mode-2-threads-4/weighted.f32',dtype='<f4').reshape(2,1,16,128)
t=np.load(case/'shadow-weighted.npy',allow_pickle=False)
b,_,h,d=np.unravel_index(abs(g-t).argmax(),g.shape)
folder=out/'input';folder.mkdir()
np.zeros((1,1,1,1),dtype='<f4').tofile(folder/'q.f32')
np.zeros((1,979,1,1),dtype='<f4').tofile(folder/'k.f32')
values=v[b,:,h//4,d].copy();values.tofile(folder/'v.f32')
probabilities=p[b,h,0].copy();probabilities.tofile(folder/'shadow-probabilities.f32')
env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','GGML_','DIA_','CRISPASR_'))}
run=subprocess.run(list(map(str,[proof/'bridge',folder,out/'native',1,1,1,979,1,2,4])),env=env,capture_output=True,text=True)
(out/'native.log').write_text(run.stdout+run.stderr);assert run.returncode==0
native=float(np.fromfile(out/'native/weighted.f32',dtype='<f4').item())
assert native==float(g[b,0,h,d]),'single ggml dot must reproduce the full attention component'
assert torch.__version__=='2.6.0+cpu';torch.set_num_threads(4)
a=torch.from_numpy(probabilities.copy());z=torch.from_numpy(values.copy())
exact=math.fsum(float(x)*float(y) for x,y in zip(probabilities,values))
row=dict(full_output_index=[int(b),0,int(h),int(d)],kv_head=int(h)//4,terms=979,
 actual_captured_native_component=float(g[b,0,h,d]),actual_original_shadow_component=float(t[b,0,h,d]),
 isolated_ggml_dot=native,isolated_torch_dot=float(torch.dot(a,z)),
 isolated_torch_matmul=float((a.reshape(1,979)@z.reshape(979,1)).item()),
 exact_float64_product_fsum=exact,native_absolute_error=native-exact,
 original_full_shape_absolute_error=float(t[b,0,h,d])-exact,
 native_full_component_reproduced_exactly=True,
 scope='One 979-term dot, one output scalar; no neural weights or full-model run. Scalar Torch kernel shape differs from original SDPA matrix shape.')
(out/'receipt.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row,indent=2))
