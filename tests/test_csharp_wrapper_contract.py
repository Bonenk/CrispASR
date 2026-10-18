"""Model-free C# operations against a compiled public-header ABI probe."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

PROBE = r'''
#include "crispasr_session.h"
static int calls, mode;
static float audio[] = {1,2,3,4};
crispasr_session* crispasr_session_open(const char* p, int threads) { return (crispasr_session*)1; }
void crispasr_session_close(crispasr_session* s) {}
int crispasr_session_convert_content_dim(crispasr_session* s) { return 2; }
int crispasr_session_convert_n_speakers(crispasr_session* s) { return 1; }
int crispasr_session_convert_sample_rate(crispasr_session* s) { return 40000; }
#ifndef OLD_PROBE
int crispasr_session_convert_inter_channels(crispasr_session* s) { return mode ? 2147483647 : 2; }
int crispasr_session_convert_upsample_product(crispasr_session* s) { return 4; }
#endif
int crispasr_session_convert(crispasr_session* s, const float* content, int n, const float* f0,
                             int speaker, const float* zp, const float* sine) {
    ++calls;
    return n == 3 && speaker == 0 ? 4 : -1;
}
const float* crispasr_session_convert_audio(crispasr_session* s, int* n) { *n=4;return audio; }
int probe_calls(void) { return calls; }
void probe_mode(int m) { mode=m; }
void probe_mutate(void) { audio[0]=999; }
'''

PROBE += r"""
#include <stdlib.h>
#include <string.h>
static char token0[3],token1[2];
static int frees,owned;
static crispasr_token_callback token_cb;
int probe_frees(void) { return frees; }
int probe_owned(void) { return owned; }
static crispasr_session_result* result(void) {
    token0[0]=(char)0xe6;token0[1]=(char)0x97;token0[2]=0;
    token1[0]=(char)0xa5;token1[1]=0;
    if(token_cb)token_cb(token0,0,NULL);
    return (crispasr_session_result*)1;
}
crispasr_session_result* crispasr_session_transcribe_lang(crispasr_session* s,const float* p,int n,const char* lang) { return result(); }
crispasr_session_result* crispasr_session_transcribe_chunked_lang(crispasr_session* s,const float* p,int n,int c,int overlap,const char* lang) { return result(); }
crispasr_session_result* crispasr_session_transcribe_vad(crispasr_session* s,const float* p,int n,int sr,const char* model,const crispasr_vad_abi_opts* opts) { return result(); }
int crispasr_session_set_return_logits(crispasr_session* s,int on) { return 0; }
int crispasr_session_result_n_logit_frames(crispasr_session_result* r) { return 0; }
int crispasr_session_result_n_logit_vocab(crispasr_session_result* r) { return 0; }
const float* crispasr_session_result_logits(crispasr_session_result* r) { return NULL; }
int crispasr_session_result_n_segments(crispasr_session_result* r) { return 1; }
const char* crispasr_session_result_segment_text(crispasr_session_result* r,int i) { return "Japanese"; }
int64_t crispasr_session_result_segment_t0(crispasr_session_result* r,int i) { return 123; }
int64_t crispasr_session_result_segment_t1(crispasr_session_result* r,int i) { return 456; }
float crispasr_session_result_segment_no_speech_prob(crispasr_session_result* r,int i) { return .125f; }
const char* crispasr_session_result_segment_speaker(crispasr_session_result* r,int i) { return "speaker"; }
int crispasr_session_result_n_words(crispasr_session_result* r,int i) { return 1; }
const char* crispasr_session_result_word_text(crispasr_session_result* r,int i,int j) { return "word"; }
int64_t crispasr_session_result_word_t0(crispasr_session_result* r,int i,int j) { return -1; }
int64_t crispasr_session_result_word_t1(crispasr_session_result* r,int i,int j) { return -1; }
float crispasr_session_result_word_p(crispasr_session_result* r,int i,int j) { return .75f; }
int crispasr_session_result_word_n_alts(crispasr_session_result* r,int i,int j) { return 1; }
const char* crispasr_session_result_word_alt_text(crispasr_session_result* r,int i,int j,int k) { return "alternative"; }
float crispasr_session_result_word_alt_p(crispasr_session_result* r,int i,int j,int k) { return .25f; }
void crispasr_session_result_free(crispasr_session_result* r) { ++frees;token0[0]='X'; }
#ifndef OLD_PROBE
int crispasr_session_result_n_tokens(crispasr_session_result* r,int i) { return 2; }
const char* crispasr_session_result_token_text(crispasr_session_result* r,int i,int j) { return j ? token1:token0; }
int64_t crispasr_session_result_token_t0(crispasr_session_result* r,int i,int j) { return j ? 123:-1; }
int64_t crispasr_session_result_token_t1(crispasr_session_result* r,int i,int j) { return j ? 124:-1; }
float crispasr_session_result_token_p(crispasr_session_result* r,int i,int j) { return .75f; }
int crispasr_session_set_voice_samples(crispasr_session* s,const float* p,int32_t n,int32_t sr,const char* text) { return n==3 && sr==24000 ? 0:-1; }
int crispasr_session_set_tts_cfg_scale(crispasr_session* s,float scale) { return scale==1.5f ? 0:-2; }
int crispasr_session_tada_set_makeref_models(crispasr_session* s,const char* enc,const char* align) { return enc==NULL && align==NULL ? 0:-1; }
const char* crispasr_session_last_synth_error(crispasr_session* s) { return "probe error"; }
const char* crispasr_session_disclaimer_text(void) { return "AI generated"; }
float* crispasr_session_get_disclaimer_pcm(crispasr_session* s,int* n) { float* p=malloc(8);p[0]=.25f;p[1]=.5f;*n=2;++owned;return p; }
void crispasr_pcm_free(float* p) { if(p){--owned;free(p);} }
int crispasr_session_synthesize_streaming(crispasr_session* s,const char* text,crispasr_pcm_stream_cb cb,void* ud) {
    float pcm[2]={.25f,.5f};cb(pcm,2,0,ud);pcm[0]=.75f;cb(pcm,2,1,ud);return 0;
}
void crispasr_session_set_token_callback(crispasr_session* s,crispasr_token_callback cb,void* ud) { token_cb=cb; }
#endif
"""

PROGRAM = r'''
using System;
using System.Runtime.InteropServices;
using CrispASR;
class Program {
    [DllImport("crispasr",CallingConvention=CallingConvention.Cdecl)] static extern int probe_calls();
    [DllImport("crispasr",CallingConvention=CallingConvention.Cdecl)] static extern void probe_mode(int n);
    [DllImport("crispasr",CallingConvention=CallingConvention.Cdecl)] static extern void probe_mutate();
    [DllImport("crispasr",CallingConvention=CallingConvention.Cdecl)] static extern int probe_frees();
    [DllImport("crispasr",CallingConvention=CallingConvention.Cdecl)] static extern int probe_owned();
    static int assertions;
    static void Equal<T>(T a,T b) { if(!Equals(a,b))throw new Exception($"{a} != {b}");assertions++; }
    static void Reject<T>(Action action) where T:Exception {
        int before=probe_calls();
        try { action(); } catch(T) { Equal(probe_calls(),before);return; }
        throw new Exception("missing "+typeof(T).Name+" before native conversion");
    }
    static void Main(string[] args) {
        using var s=Session.Open("probe");
        var content=new float[6];var f0=new float[3];
        if(args.Length>0 && args[0]=="old") {
            Equal(s.Convert(content,f0).Length,4);
            Reject<NotSupportedException>(()=>s.Convert(content,f0,noiseZp:new float[6]));
            Equal(s.Transcribe(f0)[0].Tokens.Length,0);
        } else {
            Reject<ArgumentException>(()=>s.Convert(content,f0,noiseZp:new float[5]));
            Reject<ArgumentException>(()=>s.Convert(content,f0,noiseSine:new float[11]));
            Reject<ArgumentException>(()=>s.Convert(content,f0,noiseZp:new float[7]));
            Reject<ArgumentOutOfRangeException>(()=>s.Convert(content,f0,speakerId:1));
            var audio=s.Convert(content,f0,noiseZp:new float[6],noiseSine:new float[12]);
            Equal(audio.Length,4);probe_mutate();Equal(audio[0],1f);
            probe_mode(1);
            Reject<OverflowException>(()=>s.Convert(content,f0,noiseZp:new float[6]));
            Equal(s.Convert(content,f0).Length,4); // random draws need no dimension probe
            foreach(var seg in new[]{s.Transcribe(f0)[0],s.TranscribeChunked(f0)[0],
                                      s.TranscribeVad(f0,16000,"vad")[0],s.TranscribeWithLogits(f0).Segments[0]}) {
                Equal(seg.T0,1.23);Equal(seg.Words[0].T0,-1d);Equal(seg.Words[0].HasTimestamps,false);
                Equal(seg.Speaker,"speaker");Equal(seg.Words[0].Alts[0].Text,"alternative");
                Equal(seg.Tokens.Length,2);Equal(seg.Tokens[0].RawText[0],(byte)0xe6);
                Equal(seg.Tokens[0].HasTimestamps,false);Equal(seg.Tokens[1].T0,1.23);
                var bytes=new byte[]{seg.Tokens[0].RawText[0],seg.Tokens[0].RawText[1],seg.Tokens[1].RawText[0]};
                Equal(System.Text.Encoding.UTF8.GetString(bytes),"日");
            }
            s.SetVoiceSamples(f0,24000);s.SetTtsCfgScale(1.5f);s.TadaSetMakerefModels();
            Reject<InvalidOperationException>(()=>s.SetTtsCfgScale(2));
            Equal(Session.DisclaimerText,"AI generated");Equal(s.GetDisclaimerPcm()[0],.25f);Equal(probe_owned(),0);
            var chunks=new System.Collections.Generic.List<float[]>();
            s.SynthesizeStreaming("one. two.",(pcm,final)=>chunks.Add(pcm));
            Equal(chunks.Count,2);Equal(chunks[0][0],.25f);Equal(chunks[1][0],.75f);
            Reject<InvalidOperationException>(()=>s.SynthesizeStreaming("test",(_,__)=>s.Dispose()));
            int before=probe_frees();
            s.SetTokenCallback((_,__)=>throw new ArgumentException("callback"));
            Reject<ArgumentException>(()=>s.Transcribe(f0));
            Equal(probe_frees(),before+1);s.SetTokenCallback(null);
        }
        Console.WriteLine($"PASS C# conversion ABI: {assertions} assertions");
    }
}
'''


@unittest.skipUnless(shutil.which('dotnet') and shutil.which('cc'), 'dotnet SDK and C compiler required')
class CSharpWrapperContract(unittest.TestCase):
    def test_replay_buffers_and_older_library(self):
        with tempfile.TemporaryDirectory(prefix='csharp-contract-') as temporary:
            path = Path(temporary)
            (path/'probe.c').write_text(PROBE)
            (path/'Program.cs').write_text(PROGRAM)
            (path/'Probe.csproj').write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup>
<OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework><Nullable>enable</Nullable>
<AllowUnsafeBlocks>true</AllowUnsafeBlocks><EnableDefaultCompileItems>false</EnableDefaultCompileItems>
</PropertyGroup><ItemGroup><Compile Include="Program.cs"/>
<Compile Include="{ROOT / 'bindings/csharp/CrispASR'}/*.cs"/></ItemGroup></Project>''')
            env=dict(os.environ, DOTNET_CLI_TELEMETRY_OPTOUT='1', DOTNET_NOLOGO='1')
            subprocess.run(['dotnet','build',str(path/'Probe.csproj'),'-c','Release','--nologo','-v','quiet'],
                           env=env,check=True,capture_output=True,text=True)
            for old in (False,True):
                library=path/('old.so' if old else 'probe.so')
                subprocess.run(['cc','-shared','-fPIC','-std=c11','-Werror',
                                '-I'+str(ROOT/'include'), *(['-DOLD_PROBE'] if old else []),
                                str(path/'probe.c'),'-o',str(library)],check=True)
                result=subprocess.run(['dotnet',str(path/'bin/Release/net8.0/Probe.dll'),
                                       *(['old'] if old else [])],
                                      env=dict(env,CRISPASR_LIBRARY_PATH=str(library)),
                                      capture_output=True,text=True,
                                      preexec_fn=(lambda: __import__('resource').setrlimit(__import__('resource').RLIMIT_CORE,(0,0))) if os.name=='posix' else None)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('PASS C# conversion ABI:',result.stdout)
                print(result.stdout.strip())


if __name__=='__main__': unittest.main()
