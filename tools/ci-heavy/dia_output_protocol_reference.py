#!/usr/bin/env python3
"""Regenerate weight-free Dia protocol witnesses from pinned original methods.

Use the isolated Torch2.6 CPU reference environment. --source-dir points to the
original checkout's dia/ directory; --output-dir must be fresh. No model/DAC
weights are loaded. The neural step supplies synthetic predictions and the DAC
entry captures original codec inputs. No speech acceptance follows.
"""
import argparse
import ast,hashlib,importlib,json,sys,time,types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
assert torch.__version__=='2.6.0+cpu'
torch.set_num_threads(2)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args=parser.parse_args()
root=args.output_dir;root.mkdir(parents=True,exist_ok=False)
source=args.source_dir
expected={'state.py': '83d0ddc651bab76afe91c8851b515f12863d1221778765a6d9d4462581f5d13f', 'model.py': '6bc51e6e5ff58ca9ed66431773df6cda954afee3d4b7ea515c1363b73ca82926', 'audio.py': 'a4a92aba2e0b45a3da2ac8c27c36002501a0500a528eb9e613661f7d9a057875', 'config.py': '6bbdb7c764e11b9b9f4170948cb06da5a4dec00b2c080090ce96975fce731698'}
for name,digest in expected.items():
 assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
package=types.ModuleType('output_original');package.__path__=[str(source)];sys.modules['output_original']=package
audio=importlib.import_module('output_original.audio');state_module=importlib.import_module('output_original.state')
text=(source/'model.py').read_text();lines=text.splitlines(keepends=True);klass=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='Dia');methods=[]
for name in ['_prepare_audio_prompt','generate','_generate_output']:
 node=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name==name);start=min([node.lineno]+[d.lineno for d in node.decorator_list]);methods.append(''.join(lines[start-1:node.end_lineno]))
method_source='class OriginalProtocol:\n'+''.join(methods);(root/'literal-original-methods.py').write_text(method_source)
namespace=dict(torch=torch,np=np,time=time,apply_audio_delay=audio.apply_audio_delay,build_delay_indices=audio.build_delay_indices,build_revert_indices=audio.build_revert_indices,revert_audio_delay=audio.revert_audio_delay)
captured=[]
def decode(_model,codes):captured.append(codes.transpose(1,2)[0].tolist());return torch.zeros(1,1,1)
namespace['decode']=decode;exec(compile(method_source,str(root/'literal-original-methods.py'),'exec'),namespace)
cases=[]
recipes=[('capacity-'+str(cap),cap,-1) for cap in [1,7,14,15,16,17,32,64]]+[(f'eos-{eos}',96,eos) for eos in [0,1,2,20]]+[('eos-before-capacity',40,20),('invalid-code',64,-1)]
for name,cap,eos in recipes:
 dia=namespace['OriginalProtocol']();dia.device=torch.device('cpu');dia.model=SimpleNamespace(eval=lambda:None);dia.dac_model=None
 dia.config=SimpleNamespace(data=SimpleNamespace(channels=9,audio_bos_value=1026,audio_pad_value=1025,audio_eos_value=1024,delay_pattern=[0,8,9,10,11,12,13,14,15],audio_length=3072))
 prefill,start=dia._prepare_audio_prompt(None);state=SimpleNamespace(step=-1);state.prepare_step=lambda step:setattr(state,'step',step);output=state_module.DecoderOutput.new(dia.config,dia.device);output.prefill(prefill,start);dia._prepare_generation=lambda*args:(state,output)
 raw=[[(step*17+c*13+5)%1024 for c in range(9)] for step in range(cap)]
 if eos>=0:raw[eos][0]=1024
 if name=='invalid-code':raw[20][3]=1027;raw[21][5]=1024
 feedback=[];emitted=[];captured.clear()
 def prediction(tokens,state,*args):feedback.append(tokens[0,0].tolist());return torch.tensor(raw[state.step],dtype=torch.int)
 dia._decoder_step=prediction
 original_output=dia._generate_output
 def result(codes):emitted.extend(codes.tolist());return original_output(codes)
 dia._generate_output=result
 dia.generate('protocol-only',max_tokens=cap,cfg_scale=3.,temperature=1.2,top_p=.95,cfg_filter_top_k=45,use_torch_compile=False)
 cases.append(dict(name=name,max_tokens=cap,mode=0,raw=raw,feedback=feedback,emitted=emitted,aligned=captured[0] if captured else []))
# Forced diagnostic modes intentionally bypass original generate's stopping.
# Compare their emission with the actual original DecoderOutput update method.
for mode in [1,2]:
 cap=32;raw=[[(step*17+c*13+5)%1024 for c in range(9)] for step in range(cap)];raw[3][0]=1024
 output=state_module.DecoderOutput.new(dia.config,dia.device);prefill,start=dia._prepare_audio_prompt(None);output.prefill(prefill,start)
 for step,row in enumerate(raw):output.update_one(torch.tensor(row,dtype=torch.int),step+1,max(0,15-step-1)>0)
 emitted=raw if mode==1 else output.generated_tokens[1:cap+1].tolist()
 aligned=[];captured.clear();original_output(torch.tensor(emitted,dtype=torch.int));aligned=captured[0]
 cases.append(dict(name='forced-input' if mode==1 else 'forced-output',max_tokens=cap,mode=mode,raw=raw,feedback=[],emitted=emitted,aligned=aligned))
receipt=dict(original_source='4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8',torch_version=torch.__version__,source_sha256={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.glob('*.py')},literal_methods_sha256=hashlib.sha256(method_source.encode()).hexdigest(),scope='Actual original generation/BOS/stopping/slice/delay output methods; synthetic predictions replace neural step; decode captures codec inputs without DAC/model/speech acceptance.',cases=cases)
(root/'original-fixtures.json').write_text(json.dumps(receipt,separators=(',',':'))+'\n')
for c in cases:print(c['name'],'draws',len(c['feedback']),'emitted',len(c['emitted']),'aligned',len(c['aligned']))
