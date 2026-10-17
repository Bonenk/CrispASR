"""Independent audit of raw attention arithmetic arrays; imports no producer code."""
from pathlib import Path
import argparse,ast,hashlib,json,os,subprocess,tempfile,types
import numpy as np
import torch
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--proof',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();p=args.proof.resolve();root=args.output.resolve().parent;assert not args.output.exists();root.mkdir(parents=True,exist_ok=True);r=json.loads((p/'receipt.json').read_text())
assert r['complete'] and torch.__version__=='2.6.0+cpu';torch.set_num_threads(4)
def sha(f):
 with f.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
rows={v['path']:v for v in r['payloads']};assert len(rows)==len(r['payloads'])
assert set(rows)=={f.relative_to(p).as_posix() for f in p.rglob('*') if f.is_file()}-{'receipt.json'}
for name,row in rows.items():
 f=p/name;assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256'],name
assert sha(p/'bridge')==r['binary_sha256']
for name,value in r['actual_library_hashes'].items():assert sha(p/'libs'/name)==value
assert sha(p/'inputs/original-layers.py')=='9b77d590101783a7e570bd9889afb09b85912d05072313c32a984dbfe9323024'
# Independently run the actual original Attention.forward SDPA statements.
tree=ast.parse((p/'inputs/original-layers.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Attention');forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
i=next(i for i,n in enumerate(forward.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and getattr(n.value.func,'attr',None)=='scaled_dot_product_attention')
expr=compile(ast.Module(body=forward.body[i:i+2],type_ignores=[]),'independent-actual-sdpa','exec')
metric_count=0

def check_metrics(actual,reference,row):
 global metric_count
 a=actual.astype('f8').reshape(-1,actual.shape[-1]);b=reference.astype('f8').reshape(a.shape)
 an=np.linalg.norm(a,axis=1);bn=np.linalg.norm(b,axis=1);nz=(an>0)&(bn>0)
 expected=dict(byte_exact=actual.tobytes()==reference.tobytes(),mine_norm=float(np.linalg.norm(a)),ref_norm=float(np.linalg.norm(b)),max_abs=float(abs(a-b).max()),relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)) if np.linalg.norm(b) else None,max_row_norm_error=float(abs(an[bn>0]/bn[bn>0]-1).max()) if (bn>0).any() else None,cosine_min=float(np.min(np.sum(a[nz]*b[nz],axis=1)/(an[nz]*bn[nz]))) if nz.any() else None)
 assert set(row)==set(expected)
 for key,value in expected.items():
  if value is None or isinstance(value,bool):assert row[key]==value,(key,row[key],value)
  else:assert abs(value-row[key])<1e-14*max(1,abs(value)),(key,value,row[key])
 metric_count+=1

env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','GGML_','DIA_','CRISPASR_'))};fresh=Path(tempfile.mkdtemp(prefix='independent-attention-',dir=root))
for case in r['cases']:
 folder=p/case['case'];qshape=tuple(case['shape_q']);kvshape=tuple(case['shape_kv']);batch,time,kv,dim=kvshape;heads=qshape[2]
 q=np.fromfile(folder/'q.f32',dtype='<f4').reshape(qshape);k=np.fromfile(folder/'k.f32',dtype='<f4').reshape(kvshape);v=np.fromfile(folder/'v.f32',dtype='<f4').reshape(kvshape)
 qt=torch.from_numpy(q.copy()).transpose(1,2);kc=torch.zeros((batch,kv,3072,dim));vc=torch.zeros_like(kc);kc[:,:,:time]=torch.from_numpy(k.copy()).transpose(1,2);vc[:,:,:time]=torch.from_numpy(v.copy()).transpose(1,2)
 ns=dict(F=torch.nn.functional,self=types.SimpleNamespace(num_gqa_groups=heads//kv),Xq_BxNxTxH=qt,attn_k=kc[:,:,:time],attn_v=vc[:,:,:time],attn_mask=None,is_causal=False)
 with torch.inference_mode():exec(expr,ns)
 actual=ns['attn_output'].numpy();assert actual.tobytes()==np.load(folder/'actual-original-sdpa.npy').tobytes()
 with torch.inference_mode():
  expanded_k=torch.from_numpy(k.copy()).transpose(1,2).repeat_interleave(heads//kv,dim=1);expanded_v=torch.from_numpy(v.copy()).transpose(1,2).repeat_interleave(heads//kv,dim=1)
  score=(qt@expanded_k.transpose(-2,-1)).numpy();probability=torch.softmax(torch.from_numpy(score),dim=-1).numpy();shadow=(torch.from_numpy(probability)@expanded_v).transpose(1,2).contiguous().numpy()
 assert score.tobytes()==(folder/'shadow-scores.f32').read_bytes() and probability.tobytes()==(folder/'shadow-probabilities.f32').read_bytes() and shadow.tobytes()==np.load(folder/'shadow-weighted.npy').tobytes()
 check_metrics(shadow,actual,case['shadow_vs_actual_sdpa'])
 exact_scores=q.transpose(0,2,1,3).astype('f8')@expanded_k.numpy().astype('f8').transpose(0,1,3,2);weighted64=(probability.astype('f8')@expanded_v.numpy().astype('f8')).transpose(0,2,1,3)
 assert np.array_equal(exact_scores,np.load(folder/'scores-float64.npy')) and np.array_equal(weighted64,np.load(folder/'fixed-probability-weighted-float64.npy'))
 check_metrics(score,exact_scores,case['shadow_scores_vs_float64']);check_metrics(shadow,weighted64,case['shadow_fixed_probability_vs_float64'])
 for mode in range(3):
  def load(thread,suffix):
   d=folder/f'mode-{mode}-threads-{thread}{suffix}'
   return {part:np.fromfile(d/(part+'.f32'),dtype='<f4').reshape(qshape if part=='weighted' else score.shape) for part in ['scores','used-scores','probabilities','used-probabilities','weighted']}
  first=load(1,'')
  for thread in [1,4]:
   for suffix in ['','-repeat']:
    second=load(thread,suffix);assert all(first[key].tobytes()==second[key].tobytes() for key in first)
  if mode==1:assert first['used-scores'].tobytes()==score.tobytes()
  if mode==2:assert first['used-probabilities'].tobytes()==probability.tobytes()
  row=case['modes'][mode]
  for key,a,b in [('scores_vs_shadow',first['scores'],score),('probabilities_vs_shadow',first['probabilities'],probability),('weighted_vs_actual_sdpa',first['weighted'],actual),('weighted_vs_shadow',first['weighted'],shadow)]:check_metrics(a,b,row[key])
  if mode==2:check_metrics(first['weighted'],weighted64,row['weighted_vs_fixed_probability_float64'])
  dest=fresh/(case['case']+'-'+str(mode));cmd=[p/'bridge',folder,dest,dim,heads,kv,time,batch,mode,4]
  run=subprocess.run(list(map(str,cmd)),env=env,cwd=fresh,capture_output=True,text=True);(fresh/(dest.name+'.log')).write_text(run.stdout+run.stderr)
  assert run.returncode==0 and 'DIA_ATTENTION_ARITHMETIC_PASS' in run.stdout
  for part in first:assert (dest/(part+'.f32')).read_bytes()==first[part].tobytes()
 if case['case']=='captured-native':assert (folder/'mode-0-threads-4/weighted.f32').read_bytes()==np.load(folder/'retained-attention.npy').tobytes()
 if case['case']=='original-module-replay':assert actual.tobytes()==np.load(folder/'retained-attention.npy').tobytes()
 if case['case']=='tiny-uniform':assert np.array_equal(actual,np.load(folder/'analytic-expected.npy'))
 if case['case']=='tiny-nonuniform':
  assert abs(actual-np.load(folder/'analytic-weighted.npy')).max()<2e-5
  assert abs(probability-np.load(folder/'analytic-probabilities.npy')).max()<1e-7
for negative in r['actual_negative_controls']:
 f=p/('negative-'+negative['control']);bad=np.fromfile(f/'actual/weighted.f32',dtype='<f4').reshape(2,1,4,8);good=np.fromfile(p/'tiny-nonuniform/mode-0-threads-4/weighted.f32',dtype='<f4').reshape(bad.shape)
 check_metrics(bad,good,{k:v for k,v in negative.items() if k!='control'});assert negative['relative_l2']>1e-4
result=dict(independent_audit_pass=True,verified_payloads=len(rows),recomputed_metric_rows=metric_count,actual_original_sdpa_and_shadow_all_cases_exact=True,all_repeats_one_four_threads_exact=True,fresh_actual_elf_all_18_case_modes_exact=True,native_hosted_and_original_module_controls_exact=True,actual_negative_controls=3,scope='Component arithmetic only; original full-model self-Q not retained. No sampling/speech/GPU/default acceptance.')
args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
