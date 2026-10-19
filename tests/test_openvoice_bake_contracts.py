#!/usr/bin/env python3
"""Compile the actual diff driver against public headers; inject stage faults.

This is gate/lifecycle evidence, never numerical model acceptance.
"""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OpenVoiceBakeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='openvoice-bake-contract-')
        path = Path(cls.temp.name)
        source = r'''
#include "crispasr_openvoice_diff.h"
#include <cassert>
#include <cstring>
#include <string>
static std::string fault;
static float samples[4]={1,2,3,4};
static int alive=0;
struct openvoice2_context {openvoice2_stage_callback callback=nullptr;void* user=nullptr;bool noisy=false;};
extern "C" openvoice2_context_params openvoice2_context_default_params(){return {2,0,false,0};}
extern "C" openvoice2_context* openvoice2_init_from_file(const char*,openvoice2_context_params p){++alive;auto* c=new openvoice2_context;c->noisy=p.tau>0;return c;}
extern "C" void openvoice2_free(openvoice2_context* c){if(c){--alive;delete c;}}
extern "C" bool openvoice2_set_source_speaker(openvoice2_context*,const char* name){return std::strcmp(name,"en-us")==0;}
extern "C" bool openvoice2_set_posterior_noise(openvoice2_context*,const float*,int){return true;}
extern "C" void openvoice2_set_stage_callback(openvoice2_context* c,openvoice2_stage_callback f,void* u){c->callback=f;c->user=u;}
extern "C" bool openvoice2_convert(openvoice2_context* c,const float*,int,int,const float*,int,int,float** p,int* n){
 const char* names[]={"src_pcm","ref_pcm","src_spec","ref_spec","ref_enc_layernorm",
 "ref_enc_conv0","ref_enc_conv1","ref_enc_conv2","ref_enc_conv3","ref_enc_conv4","ref_enc_conv5",
 "ref_enc_gru_out","target_se","src_se","enc_q_pre","enc_q_wn_out","enc_q_stats","enc_q_z",
 "z_after_flow_fwd","z_after_flow_rev","dec_conv_pre","dec_stage0","dec_stage1","dec_stage2","dec_stage3","dec_output","output_pcm"};
 for(const char* name:names){
  if(fault=="missing_native" && std::strcmp(name,"target_se")==0)continue;
  float data[4]={1,2,3,4};
  if(std::strcmp(name,"target_se")==0){
   if(fault=="scale")for(float& v:data)v*=2;
   if(fault=="nan")data[0]=NAN;
   if(fault=="permutation")std::swap(data[0],data[3]);
  }
  c->callback(name,data,fault=="shape"?3:4,c->user);
 }
 if(c->noisy)c->callback("enc_q_noise",samples,4,c->user);
 *n=4;*p=static_cast<float*>(malloc(4*sizeof(float)));return fault!="conversion";
}
namespace crispasr_diff {
Ref::~Ref(){}
std::pair<const float*,size_t> Ref::get_f32(const std::string& key) const {
 if(fault=="missing_reference" && key=="target_se")return {nullptr,0};
 if(fault=="input" && key=="raw_audio")return {nullptr,0};
 if(fault=="missing_noise" && key=="enc_q_noise")return {nullptr,0};
 return {samples,4};
}
std::string Ref::meta(const std::string& key) const {
 if(key=="ov2_sample_rate")return fault=="rate"?"16000":"22050";
 if(key=="ov2_tau")return fault=="tau"?"0.7":(fault=="noisy" || fault=="missing_noise"?"0.3":"0");
 if(key=="ov2_source_speaker")return fault=="speaker"?"missing":"en-us";
 return "";
}
}
int main(int argc,char** argv){
 fault=argc>1?argv[1]:"";
 crispasr_diff::Ref ref;
 int rc=crispasr_openvoice_diff("model.gguf",ref,false);
 assert(alive==0);
 if(fault.empty() || fault=="noisy")assert(rc==0);else assert(rc!=0);
}
'''
        (path / 'probe.cpp').write_text(source)
        cls.binary = path / 'probe'
        subprocess.run([os.environ.get('CXX','c++'), '-std=c++17', '-I', str(ROOT/'examples/cli'),
                        '-I', str(ROOT/'src'), str(path/'probe.cpp'), '-o', str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_gate_accepts_complete_stages_and_rejects_faults(self):
        for fault in ['', 'scale', 'nan', 'permutation', 'shape', 'missing_native',
                      'missing_reference', 'input', 'rate', 'tau', 'speaker', 'conversion', 'noisy', 'missing_noise']:
            with self.subTest(fault=fault):
                subprocess.run([str(self.binary), fault], check=True, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    unittest.main()
