#!/usr/bin/env python3
"""Compile actual conditioning/averaging bodies with tiny embedding inputs."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def body(source,signature):
    start=source.index(signature);opening=source.index('{',start);end=opening+1;depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start:end]


class ConditioningContracts(unittest.TestCase):
    def test_owned_atomic_conditioning_and_equal_clip_average(self):
        source=(ROOT/'src/openvoice2.cpp').read_text()
        preamble=r'''
#include "openvoice2.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <cassert>
struct openvoice2_context {
 struct {int gin_channels=256;int inter_channels=2;} hp;
 std::vector<float> target_override,posterior_noise;
};
static int calls=0;
extern "C" bool openvoice2_extract_speaker_embedding(openvoice2_context*,const float* pcm,int,int,float* out) {
 ++calls;if(*pcm<0)return false;
 for(int i=0;i<256;++i)out[i]=*pcm;
 return true;
}
'''
        functions='\n'.join(body(source,'extern "C" bool '+name+'(') for name in [
            'openvoice2_set_target_embedding','openvoice2_set_posterior_noise',
            'openvoice2_extract_speaker_embedding_segments'])
        main=r'''
int main(){
 openvoice2_context ctx;float embedding[256];for(float& x:embedding)x=1;
 assert(openvoice2_set_target_embedding(&ctx,embedding));embedding[0]=2;
 assert(ctx.target_override[0]==1);embedding[0]=NAN;
 assert(!openvoice2_set_target_embedding(&ctx,embedding));assert(ctx.target_override[0]==1);
 assert(openvoice2_set_target_embedding(&ctx,nullptr));assert(ctx.target_override.empty());
 float noise[]={1,2,3,4};assert(openvoice2_set_posterior_noise(&ctx,noise,4));noise[0]=5;
 assert(ctx.posterior_noise[0]==1);assert(!openvoice2_set_posterior_noise(&ctx,noise,3));
 noise[0]=INFINITY;assert(!openvoice2_set_posterior_noise(&ctx,noise,4));
 assert(ctx.posterior_noise[0]==1);assert(!openvoice2_set_posterior_noise(&ctx,nullptr,4));
 assert(openvoice2_set_posterior_noise(&ctx,nullptr,0));assert(ctx.posterior_noise.empty());
 float pcm[]={1,1,5,5,5,5};int starts[]={0,2},ends[]={2,6};float out[256];
 for(float& x:out)x=42;
 assert(openvoice2_extract_speaker_embedding_segments(&ctx,pcm,6,22050,starts,ends,2,out));
 for(float x:out)assert(x==3);assert(calls==2);
 starts[1]=1;assert(!openvoice2_extract_speaker_embedding_segments(&ctx,pcm,6,22050,starts,ends,2,out));
 assert(calls==2);starts[1]=2;pcm[2]=-1;
 assert(!openvoice2_extract_speaker_embedding_segments(&ctx,pcm,6,22050,starts,ends,2,out));
 for(float x:out)assert(x==3);
 assert(!openvoice2_extract_speaker_embedding_segments(&ctx,pcm,6,0,starts,ends,2,out));
 assert(!openvoice2_extract_speaker_embedding_segments(&ctx,pcm,6,22050,nullptr,ends,2,out));
 assert(!openvoice2_extract_speaker_embedding_segments(&ctx,pcm,6,22050,starts,ends,0,out));
}
'''
        with tempfile.TemporaryDirectory(prefix='openvoice-conditioning-') as tmp:
            path=Path(tmp);(path/'test.cpp').write_text(preamble+functions+main)
            subprocess.run(['c++','-std=c++17','-I',str(ROOT/'src'),str(path/'test.cpp'),'-o',str(path/'test')],check=True)
            subprocess.run([str(path/'test')],check=True)


if __name__=='__main__':unittest.main()
