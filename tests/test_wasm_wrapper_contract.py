"""Actual Embind result/task adapters, compiled into a tiny WASM module."""
import os
import functools
import http.server
import queue
import threading
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
EMXX=os.environ.get('EMXX') or shutil.which('em++')
EMCC=os.environ.get('EMCC') or shutil.which('emcc')
NODE=shutil.which('node')
SHIM=r'''
#include <memory>
#include "crispasr_session.h"
static crispasr_session* g_tts_session=(crispasr_session*)1;
static crispasr_session* g_asr_session=nullptr;
#include "session_results.h"
#include "session_transcription.h"
#include "session_tasks.h"
#include "session_utilities.h"
#include "session_analysis.h"
#include "session_speaker.h"
#include "session_options.h"
#include "session_registry.h"
#include "session_diarization.h"
#include "session_vad.h"
extern "C" void probe_mode(int);
extern "C" int probe_live(void);
extern "C" int probe_calls(void);
extern "C" void probe_poison_drained_tokens(void);
extern "C" void probe_registry_mode(int);
extern "C" void probe_diarize_mode(int);
extern "C" int probe_diarize_calls(void);
extern "C" void probe_vad_expect(int);
extern "C" int probe_vad_calls(void);
extern "C" int probe_result_frees(void);
extern "C" void probe_result_count_mode(int);
extern "C" int probe_result_logits_enabled(void);
extern "C" int probe_analysis_live(void);
extern "C" void probe_analysis_mode(int);
extern "C" void probe_speaker_mode(int);
extern "C" int probe_speaker_live(void);
extern "C" void probe_db_mode(int);
extern "C" int probe_db_live(void);
extern "C" void probe_titanet_mode(int);
extern "C" int probe_titanet_live(void);
extern "C" int probe_invalid_frees(void);
extern "C" void probe_utility_mode(int);
extern "C" int probe_utility_live(void);
static emscripten::val read_result(bool seconds) {
    binding_check_reentry();float p[]={1,3,5};auto r=crispasr_session_transcribe(g_tts_session,p,3);
    auto owned=binding_take_segments(r);
    return binding_encode_segments(owned,seconds?100.0:1.0);
}
static emscripten::val read_logits() {
    binding_check_reentry();
    return binding_encode_transcription(binding_transcribe_logits(g_tts_session, {1,3,5}, "ja"));
}
EMSCRIPTEN_BINDINGS(wrapper_probe) {
    emscripten::function("readLogits",&read_logits);
    emscripten::function("logitsEnabled",&probe_result_logits_enabled);
    emscripten::function("probeBusy",emscripten::optional_override([](bool value){binding_browser_busy=value;}));
    binding_register_session_tasks();
    binding_register_utilities();
    binding_register_analysis();
    binding_register_speaker();
    binding_register_open_options();
    binding_register_registry();
    binding_register_diarization();
    binding_register_vad();
    emscripten::function("speakerMode",&probe_speaker_mode);
    emscripten::function("speakerLive",&probe_speaker_live);
    emscripten::function("dbMode",&probe_db_mode);
    emscripten::function("dbLive",&probe_db_live);
    emscripten::function("titanetMode",&probe_titanet_mode);
    emscripten::function("titanetLive",&probe_titanet_live);
    emscripten::function("analysisLive",&probe_analysis_live);
    emscripten::function("analysisMode",&probe_analysis_mode);
    emscripten::function("utilityMode",&probe_utility_mode);
    emscripten::function("utilityLive",&probe_utility_live);
    emscripten::function("readResult",&read_result);
    emscripten::function("probeMode",&probe_mode);
    emscripten::function("probeLive",&probe_live);
    emscripten::function("probeCalls",&probe_calls);
    emscripten::function("poisonDrainedTokens",&probe_poison_drained_tokens);
    emscripten::function("registryMode",&probe_registry_mode);
    emscripten::function("diarizeMode",&probe_diarize_mode);
    emscripten::function("diarizeCalls",&probe_diarize_calls);
    emscripten::function("vadExpect",&probe_vad_expect);
    emscripten::function("vadCalls",&probe_vad_calls);
    emscripten::function("resultFrees",&probe_result_frees);
    emscripten::function("resultCountMode",&probe_result_count_mode);
    emscripten::function("probeInvalidFrees",&probe_invalid_frees);
}
'''
PROGRAM=r'''
const assert=require('node:assert/strict');
require(process.argv[2])().then(m=>{
 let n=0;const eq=(a,b)=>{assert.deepEqual(a,b);n++};const rejects=fn=>{assert.throws(fn);n++};
 const pcm=new Float32Array([1,3,5]);
 let a=m.readResult(true)[0];eq(a.t0,1.23);eq(a.words[0].alts[0].text,'alternative');eq(a.speaker,'speaker');
 eq(a.tokens[0].t0,-1);eq(a.tokens[0].hasTimestamps,false);eq(Array.from(a.tokens[0].rawText),[0xe6,0x97]);
 eq(new TextDecoder().decode(new Uint8Array([...a.tokens[0].rawText,...a.tokens[1].rawText])),'日');
 eq(m.readResult(false)[0].t0,123);eq(a.tokens[0].rawText[0],0xe6);
 eq(m.sessionBeats(pcm).tempoBpm,123.5);eq(m.sessionTab(pcm,22050).emissions.length,12);
 eq(Array.from(m.sessionConvert(new Float32Array([0,1,2,3,4,5]),pcm,0,null,null)),[0,1,2,3,4,5]);
 rejects(()=>m.sessionConvert(new Float32Array(6),pcm,0,new Float32Array(5),null));
 m.probeMode(1);eq(m.sessionBeats(pcm).events.length,0);eq(m.sessionTab(pcm,22050).emissions.length,0);m.probeMode(0);
 eq(m.sessionSetVoiceSamples(pcm,24000,'hello'),0);eq(m.sessionSetTtsCfgScale(1.5),0);eq(m.sessionSetTtsCfgScale(2),-2);
 eq(m.sessionTadaSetMakerefModels('',''),0);eq(m.sessionGetDisclaimerPcm()[0],0.25);eq(m.probeLive(),0);
 let chunks=[];let result=m.sessionSynthesizeStreaming('a. b.',(p,last)=>chunks.push([p,last]));
 eq(result.failed,false);eq(result.status,0);eq(chunks.length,2);eq(chunks[0][0][0],0.25);eq(chunks[1][0][0],0.75);eq(chunks[1][1],true);
 const error=new Error('sentinel');result=m.sessionSynthesizeStreaming('test',()=>{throw error});eq(result.failed,true);eq(result.error,error);
 result=m.sessionSynthesizeStreaming('test',()=>m.readResult(true));eq(result.failed,true);eq(result.error.message.includes('re-entry'),true);
 eq(m.audioLoad('x',24000).sampleRate,24000);eq(Array.from(m.audioLoad('x',16000).samples),[.25,.5,.75]);
 eq(m.audioLoadStereo('x').sourceChannels,2);m.utilityMode(3);eq(m.audioLoadStereo('x').sourceChannels,1);
 eq(m.audioResample(pcm,16000,24000)[0],.25);m.utilityMode(2);
 rejects(()=>m.audioLoad('x',16000));rejects(()=>m.audioLoadStereo('x'));rejects(()=>m.audioResample(pcm,16000,24000));eq(m.utilityLive(),0);m.utilityMode(0);
 eq(m.backendCapabilities('foo'),'asr,tts');eq(m.allBackendCapabilities().includes('foo'),true);eq(m.cApiVersion(),'probe');
 let processor=new m.TextProcessor('x',1);eq(processor.process('text'),'Grüße');processor.close();rejects(()=>processor.process('text'));processor.delete();
 processor=new m.TextProcessor('x',2);eq(processor.process('text'),'Hello.');processor.delete();
 eq(m.sessionScoreTexts(pcm,['one','two'],'','').tokenCounts[1],3);eq(m.sessionSetPcmSampleRate(16000),0);eq(m.sessionSetGrammarStrict(true),0);
 const stem=m.sessionSeparate(new Float32Array([1,2,3,4,5,6]))[0];eq(stem.name,'voice');eq(stem.samples.length,6);eq(stem.sampleRate,44100);
 eq(m.watermarkDetect(pcm),.75);eq(m.utilityLive(),0);
 const pk=new m.ParakeetContext('probe',4,true);const pkResult=pk.transcribe(pcm,123);eq(pkResult.text,'日本語');eq(pkResult.words[0].t0,223);eq(pkResult.tokens[0].t0,-1);eq(Array.from(pkResult.tokens[0].rawText),[0xe6,0x97]);pk.close();pk.close();rejects(()=>pk.transcribe(pcm,0));pk.delete();
 rejects(()=>new m.SpeakerDB('db','Alice,Bob',false));rejects(()=>new m.SpeakerDB('db','',true));
 const db=new m.SpeakerDB('db','Alice,Bob',true);eq(db.count(),2);eq(m.dbLive(),1);
 const matched=db.match(pcm,.5);eq(matched.name,'アリス');eq(matched.score,.875);
 m.dbMode(1);eq(db.match(pcm,-1).name,null);eq(db.match(pcm,.5).score,-1);eq(matched.name,'アリス');m.dbMode(0);
 rejects(()=>db.match(new Float32Array(0),.5));rejects(()=>db.match(pcm,NaN));rejects(()=>db.enroll('Alice',new Float32Array(0)));
 db.enroll('Alice',pcm);m.dbMode(3);rejects(()=>db.enroll('Alice',pcm));m.dbMode(2);rejects(()=>new m.SpeakerDB('db','Alice,Bob',true));m.dbMode(0);
 db.close();db.close();eq(m.dbLive(),0);rejects(()=>db.count());rejects(()=>db.match(pcm,.5));rejects(()=>db.enroll('Alice',pcm));db.delete();
 const titan=new m.TitaNet('probe',4);eq(titan.embed(pcm).length,192);eq(titan.embed(pcm)[0],1);
 for(const mode of [2,3,4]){m.titanetMode(mode);rejects(()=>titan.embed(pcm));}m.titanetMode(0);titan.close();titan.close();rejects(()=>titan.embed(pcm));titan.delete();eq(m.titanetLive(),0);
 const alignment=m.alignWords(pcm,'model','日本語',123,4);eq(alignment[0].text,'日本語');eq(alignment[0].t0,123);eq(alignment[0].characters[1].text,'本');eq(alignment[0].characters[1].t1,165);eq(m.analysisLive(),0);
 eq(m.vadSpans(pcm,'vad',16000,.5,250,100,4,false)[0].start,1.25);eq(m.vadSlices(pcm,'vad',16000,.5,250,100,0,0,4)[0].end,2.5);eq(m.analysisLive(),0);
 m.analysisMode(1);eq(m.alignWords(pcm,'model','日本語',123,4).length,0);eq(m.vadSlices(pcm,'vad',16000,.5,250,100,0,0,4).length,0);eq(m.analysisLive(),0);
 m.analysisMode(2);rejects(()=>m.alignWords(pcm,'model','日本語',123,4));rejects(()=>m.vadSpans(pcm,'vad',16000,.5,250,100,4,false));rejects(()=>m.vadSlices(pcm,'vad',16000,.5,250,100,0,0,4));eq(m.analysisLive(),0);
 eq(m.probeLive(),0);eq(m.probeInvalidFrees(),0);
 const embedder=new m.SpeakerEmbedder('embed',2,'cache');eq(embedder.dimension(),3);eq(embedder.name(),'generic');
 const embedding=embedder.embed(pcm);eq(Array.from(embedding),[.25,.5,.75]);eq(m.speakerLive(),1);
 rejects(()=>embedder.embed(new Float32Array(0)));rejects(()=>new m.SpeakerEmbedder('embed',0,''));
 for(const mode of [2,4]){m.speakerMode(mode);rejects(()=>embedder.embed(pcm));}m.speakerMode(0);
 embedder.close();embedder.close();eq(m.speakerLive(),0);eq(embedding[2],.75);rejects(()=>embedder.dimension());rejects(()=>embedder.embed(pcm));embedder.delete();
 for(const mode of [1,3]){m.speakerMode(mode);rejects(()=>new m.SpeakerEmbedder('embed',2,'cache'));}m.speakerMode(0);eq(m.speakerLive(),0);
 const cluster=m.speakerCluster([[1,2,3],[4,5,6]],.6,3);eq(cluster.labels,[0,1]);eq(cluster.nClusters,2);
 rejects(()=>m.speakerCluster([],.6,3));rejects(()=>m.speakerCluster([[1],[2,3]],.6,3));rejects(()=>m.speakerCluster([[1],[2]],NaN,3));
 for(const mode of [2,6]){m.speakerMode(mode);rejects(()=>m.speakerCluster([[1,2,3],[4,5,6]],.6,3));}m.speakerMode(0);
 const cache=new m.PyannoteCache(pcm,'seg',2),grid=[{t0:125,t1:175},{t0:200,t1:250}];
 const assigned=cache.apply(grid,125);eq(assigned.map(x=>x.speaker),[2,1]);eq(assigned[0].t0,125);eq(grid[0].speaker,undefined);
 for(const mode of [2,5]){m.speakerMode(mode);rejects(()=>cache.apply(grid,125));}m.speakerMode(0);
 rejects(()=>cache.apply([],125));rejects(()=>cache.apply([{t0:200,t1:100}],125));
 cache.close();cache.close();eq(m.speakerLive(),0);eq(assigned[1].speaker,1);rejects(()=>cache.apply(grid,125));cache.delete();
 m.speakerMode(1);rejects(()=>new m.PyannoteCache(pcm,'seg',2));m.speakerMode(0);rejects(()=>new m.PyannoteCache(new Float32Array(0),'seg',2));
 const load={n_threads:1,use_gpu:false,verbosity:2,flash_attn:false,n_gpu_layers:3};
 eq(m.asrOpenWithOptions('model','foo',load),true);eq(m.ttsOpenWithOptions('model','foo',load),true);
 const before=m.probeCalls();
 for(const bad of [{n_threads:-1},{n_threads:2},{n_threads:1.5},{verbosity:NaN},{verbosity:2147483648},{n_gpu_layers:-2},{use_gpu:1},{flash_attn:'false'},[],false])
   rejects(()=>m.asrOpenWithOptions('model','foo',bad));
 rejects(()=>m.asrOpenWithOptions('','foo',load));rejects(()=>m.ttsOpenWithOptions('model\0path','foo',load));
 eq(m.probeCalls(),before);
 m.probeMode(7);eq(m.asrOpenWithOptions('model','',null),true);eq(m.ttsOpenWithOptions('model','',{n_threads:0}),true);
 m.probeMode(2);eq(m.asrOpenWithOptions('model','foo',load),false);m.probeMode(0);
 const entry={filename:'foo.gguf',url:'https://example.invalid/model',approxSize:'715 MB'};
 eq(m.registryLookup('foo'),entry);eq(m.registryLookupByFilename('foo.gguf'),entry);
 eq(m.registryLookup('unknown'),null);eq(m.registryLookupByFilename('unknown'),null);eq(m.registryBackends(),['foo','empty']);
 const bundle=m.registryDefaultBundle('foo');eq(bundle.backend,'foo');eq(bundle.license,'Apache-2.0');eq(bundle.requiresAcceptance,true);
 eq(bundle.artifacts.map(a=>a.kind),[0,1,2]);eq(bundle.artifacts.map(a=>a.filename),['foo.gguf','codec.gguf','tokens.json']);
 eq(m.registryDefaultBundle('unknown'),null);eq(m.registryDefaultBundle('empty').artifacts,[]);
 eq(m.cacheDirectory(''),'/default-cache');eq(m.cacheDirectory('/cache'),'/cache');
 eq(m.cacheEnsureFile('foo.gguf',entry.url,true,'/cache'),'/cache/foo.gguf');
 rejects(()=>m.cacheEnsureFile('foo.gguf',entry.url,false,'/cache'));rejects(()=>m.registryLookup('foo\0hidden'));
 rejects(()=>m.cacheDirectory('/cache\0hidden'));rejects(()=>m.cacheEnsureFile('foo.gguf','url\0hidden',true,'/cache'));
 m.registryMode(3);eq(m.registryLookup('foo'),entry);eq(m.registryBackends(),['foo','empty']);eq(m.registryDefaultBundle('foo').artifacts.length,3);eq(m.cacheDirectory(''),'/default-cache');
 for(const mode of [2,4,5]){m.registryMode(mode);rejects(()=>m.registryLookup('foo'));rejects(()=>m.registryBackends());rejects(()=>m.registryDefaultBundle('foo'));rejects(()=>m.cacheDirectory(''));}
 for(const mode of [6,7]){m.registryMode(mode);rejects(()=>m.registryDefaultBundle('foo'));}m.registryMode(0);
 eq(bundle.artifacts[1].filename,'codec.gguf');
 const diaOpts={method:4,n_threads:2,slice_t0_cs:125,embedder_path:'embed',min_speakers:2,max_speakers:4,num_speakers:3};
 const gridDia=[{t0:125,t1:175},{t0:200,t1:250}];
 eq(m.diarizeSegments(pcm,null,gridDia,diaOpts).map(x=>x.speaker),[1,0]);eq(gridDia[0].speaker,undefined);
 let turns=m.diarizeSegmentsWithTurns(pcm,null,gridDia,diaOpts);eq(turns.turns,[{t0:125,t1:150,speaker:1},{t0:150,t1:250,speaker:0}]);
 eq(turns.segments[0].t0,125);m.diarizeMode(1);eq(m.diarizeSegmentsWithTurns(pcm,null,gridDia,diaOpts).turns.length,2);
 for(const mode of [2,3,4,5,6,8]){m.diarizeMode(mode);rejects(()=>m.diarizeSegmentsWithTurns(pcm,null,gridDia,diaOpts));}m.diarizeMode(0);
 eq(m.diarizeSegments(pcm,new Float32Array([5,3,1]),gridDia,diaOpts)[1].speaker,0);
 m.diarizeMode(7);eq(m.diarizeSegmentsWithTurns(pcm,null,gridDia,diaOpts).turns,[]);m.diarizeMode(0);
 const diaCalls=m.diarizeCalls();
 rejects(()=>m.diarizeSegments(pcm,new Float32Array(2),gridDia,diaOpts));rejects(()=>m.diarizeSegments(pcm,null,[],diaOpts));
 rejects(()=>m.diarizeSegments(pcm,null,[{t0:1.5,t1:3}],diaOpts));rejects(()=>m.diarizeSegments(pcm,null,[{t0:4,t1:3}],diaOpts));
 for(const patch of [{method:6},{n_threads:0},{max_speakers:1},{num_speakers:5},{embedder_path:''},{embedder_path:'x\0y'},{slice_t0_cs:NaN},{min_speakers:1.5},{method:3,model_path:''}])
   rejects(()=>m.diarizeSegments(pcm,null,gridDia,{...diaOpts,...patch}));
 eq(m.diarizeCalls(),diaCalls);m.diarizeMode(4);rejects(()=>m.diarizeSegments(pcm,null,gridDia,diaOpts));m.diarizeMode(0);
 eq(m.asrOpenWithOptions('model','foo',load),true);
 const vadOpts={threshold:.375,min_speech_duration_ms:123,min_silence_duration_ms:456,speech_pad_ms:78,chunk_seconds:9,n_threads:2};
 let frees=m.resultFrees();m.vadExpect(1);let vad=m.asrTranscribeVad(pcm,16000,'vad','ja',vadOpts);
 eq(vad[0].t0,123);eq(Array.from(vad[0].tokens[0].rawText),[0xe6,0x97]);eq(m.resultFrees(),frees+1);
 m.vadExpect(1);for(const mode of [1,2,3,4]){m.resultCountMode(mode);frees=m.resultFrees();rejects(()=>m.asrTranscribeVad(pcm,16000,'vad','ja',vadOpts));eq(m.resultFrees(),frees+1);}m.resultCountMode(0);
 m.vadExpect(2);eq(m.asrTranscribeVad(pcm,16000,'vad','ja',null).length,1);
 m.vadExpect(0);eq(m.asrTranscribeVad(pcm,16000,'vad','',null).length,1);
 const vadCalls=m.vadCalls();
 for(const patch of [{threshold:NaN},{threshold:2},{threshold:'0.5'},{min_speech_duration_ms:-1},{min_silence_duration_ms:-1},{speech_pad_ms:-1},{chunk_seconds:-1},{n_threads:0}])
   rejects(()=>m.asrTranscribeVad(pcm,16000,'vad','ja',{...vadOpts,...patch}));
 rejects(()=>m.asrTranscribeVad(pcm,16000,'vad','ja',[]));rejects(()=>m.asrTranscribeVad(pcm,0,'vad','ja',vadOpts));
 rejects(()=>m.asrTranscribeVad(pcm,16000,'vad\0x','ja',vadOpts));eq(m.vadCalls(),vadCalls);
 m.vadExpect(3);rejects(()=>m.asrTranscribeVad(pcm,16000,'vad','ja',vadOpts));
 eq(m.asrOpenWithOptions('missing','foo',load),false);rejects(()=>m.asrTranscribeVad(pcm,16000,'vad','ja',vadOpts));
 for(const mode of [1,2,3,4,5,6,7]){m.resultCountMode(mode);frees=m.resultFrees();rejects(()=>m.readLogits());eq(m.resultFrees(),frees+1);eq(m.logitsEnabled(),0);}
 m.resultCountMode(8);const transcription=m.readLogits();eq(transcription.logits.nFrames,1);eq(transcription.logits.nVocab,2);eq(Array.from(transcription.logits.data),[-1,2]);eq(transcription.segments[0].t0,1.23);eq(m.logitsEnabled(),0);
 const U8=global.Uint8Array;try{global.Uint8Array=function(){throw new Error('encoding failure')};frees=m.resultFrees();rejects(()=>m.readLogits());eq(m.resultFrees(),frees+1);eq(m.logitsEnabled(),0);}finally{global.Uint8Array=U8;}
 m.resultCountMode(0);eq(m.readLogits().logits,null);eq(transcription.logits.data[1],2);
 const ExpectedTokens=[[0xe6,0x97],[0xa5],[],[101,110,100]];
 const OriginalU8=global.Uint8Array;
 try {
   global.Uint8Array=class extends OriginalU8 { constructor(...args){super(...args);m.poisonDrainedTokens();} };
   eq(m.drainStreamedTokenBytes().map(x=>Array.from(x)),ExpectedTokens);
 } finally {global.Uint8Array=OriginalU8;}
 for(const mode of [8,9]){m.probeMode(mode);rejects(()=>m.drainStreamedTokenBytes());}
 for(const mode of [10,11]){m.probeMode(mode);eq(m.drainStreamedTokenBytes(),[]);}m.probeMode(0);
 eq(m.getStreamedSegmentCount(),2);eq(m.getStreamedTokenCount(),4);
 m.probeBusy(true);try{rejects(()=>m.getStreamedSegmentCount());rejects(()=>m.getStreamedTokenCount());rejects(()=>m.readResult(false));}finally{m.probeBusy(false);}
 console.log('PASS WASM ABI: ' +n+' assertions');
}).catch(e=>{console.error(e);process.exitCode=1});
'''

@unittest.skipUnless(EMCC and EMXX and NODE,'Emscripten C/C++ compilers and Node required')
class WasmWrapperContract(unittest.TestCase):
    def test_owned_metadata_music_conversion_and_stream_callbacks(self):
        with tempfile.TemporaryDirectory(prefix='wasm-wrapper-abi-') as tmp:
            tmp=Path(tmp);objects=[]
            src=(ROOT/'src/crispasr_c_api.cpp').read_text();layouts=[]
            start=src.index('struct crispasr_open_params_v1 {');(tmp/'runtime-open-layout.h').write_text(src[start:src.index('};',start)+2])
            start=src.index('struct crispasr_vad_abi_opts {');(tmp/'runtime-vad-layout.h').write_text(src[start:src.index('};',start)+2])
            for name in ['crispasr_diarize_seg_abi','crispasr_diarize_turn_abi','crispasr_diarize_opts_abi']:
                start=src.index('struct '+name+' {');layouts.append(src[start:src.index('};',start)+2])
            (tmp/'runtime-abi-layout.h').write_text('\n'.join(layouts))
            for name in ['python-wrapper-abi','wrapper-session-metadata','wrapper-audio-utilities','wasm-wrapper-utilities','wasm-wrapper-analysis','wrapper-parakeet-legacy','wrapper-titanet-legacy','wrapper-speaker-db','wrapper-speaker-primitives','wrapper-registry','wrapper-diarize-arguments']:
                obj=tmp/(name+'.o');objects.append(str(obj))
                subprocess.run([EMCC,'-DCRISPASR_PROBE_WASM_DIA','-DCRISPASR_PROBE_VAD_ARGUMENTS','-DCRISPASR_PROBE_LITERAL_OPEN_LAYOUT','-DCRISPASR_PROBE_WASM_OPEN','-I'+str(tmp),'-I'+str(ROOT/'include'),'-I'+str(ROOT/'ggml/include'),'-c',str(ROOT/'tests/fixtures'/(name+'.c')),'-o',str(obj)],check=True)
            # Unexercised registration dependencies abort if accidentally called.
            # Concrete public-header probes supply every operation asserted below.
            fixture_names=['python-wrapper-abi','wrapper-session-metadata','wrapper-audio-utilities','wasm-wrapper-utilities','wasm-wrapper-analysis','wrapper-parakeet-legacy','wrapper-titanet-legacy','wrapper-speaker-db','wrapper-speaker-primitives','wrapper-registry','wrapper-diarize-arguments']
            concrete='\n'.join((ROOT/'tests/fixtures'/(n+'.c')).read_text() for n in fixture_names)
            adapters='\n'.join((ROOT/'bindings/javascript'/n).read_text() for n in ['session_results.h','session_transcription.h','session_tasks.h','session_utilities.h','session_analysis.h','session_speaker.h','session_options.h','session_registry.h','session_diarization.h','session_vad.h'])
            headers='\n'.join((ROOT/'include'/n).read_text() for n in ['crispasr.h','crispasr_session.h'])
            headers=re.sub(r'/\*.*?\*/|//[^\n]*','',headers,flags=re.S)
            unused=[]
            for name in sorted(set(re.findall(r'\b(crispasr_\w+)\s*\(',adapters))):
                if re.search(r'\b'+name+r'\s*\([^;]*?\)\s*\{',concrete,re.S):
                    continue
                declaration=re.search(r'CRISPASR_(?:SESSION_)?API\s+([\w\s*]+?\b'+name+r'\s*\([^;]*);',headers,re.S)
                if not declaration:raise AssertionError('No public declaration for '+name)
                unused.append(declaration.group(1)+' { abort(); }')
            stubs=tmp/'unused.c';stubs.write_text('#include "crispasr.h"\n#include "crispasr_session.h"\n#include <stdlib.h>\n'+'\n'.join(unused))
            obj=tmp/'unused.o';subprocess.run([EMCC,'-I'+str(ROOT/'include'),'-I'+str(ROOT/'ggml/include'),'-c',str(stubs),'-o',str(obj)],check=True);objects.append(str(obj))
            source=tmp/'probe.cpp';source.write_text(SHIM)
            module=tmp/'probe.js'
            subprocess.run([EMXX,'-std=c++17','-fexceptions','-sDISABLE_EXCEPTION_CATCHING=0','--bind','-sSINGLE_FILE=1',
                '-sMODULARIZE=1','-sENVIRONMENT=node','-sALLOW_MEMORY_GROWTH=1','-I'+str(ROOT/'include'),
                '-I'+str(ROOT/'ggml/include'),'-I'+str(ROOT/'bindings/javascript'),str(source),*objects,'-o',str(module)],check=True)
            program=tmp/'test.cjs';program.write_text(PROGRAM)
            result=subprocess.run([NODE,str(program),str(module)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('PASS WASM ABI:',result.stdout);print(result.stdout.strip())

@unittest.skipUnless(EMCC and EMXX and NODE,'Emscripten C/C++ compilers and Node required')
class WasmChatContract(unittest.TestCase):
    def test_chat_options_owned_bytes_callback_errors_and_lifecycle(self):
        with tempfile.TemporaryDirectory(prefix='wasm-chat-abi-') as tmp:
            tmp=Path(tmp)
            obj=tmp/'chat.o'
            subprocess.run([EMCC,'-Werror','-I'+str(ROOT/'include'),'-c',str(ROOT/'tests/fixtures/wrapper-chat-abi.c'),'-o',str(obj)],check=True)
            source=tmp/'probe.cpp'
            source.write_text(r'''
#include <emscripten.h>
#include <emscripten/bind.h>
EM_JS(void,binding_check_reentry,(),{if(Module['crispasrInCallback'])throw new Error('Session re-entry from callback');});
#define CRISPASR_WASM_CHAT 1
#include "session_chat.h"
extern "C" int probe_chat_sessions(void);
extern "C" int probe_chat_strings(void);
extern "C" int probe_chat_resets(void);
EMSCRIPTEN_BINDINGS(chat_probe){binding_register_chat();emscripten::function("chatLiveSessions",&probe_chat_sessions);emscripten::function("chatLiveStrings",&probe_chat_strings);emscripten::function("chatResets",&probe_chat_resets);}
''')
            module=tmp/'chat.js'
            subprocess.run([EMXX,'-std=c++17','-fexceptions','-sDISABLE_EXCEPTION_CATCHING=0','--bind','-sSINGLE_FILE=1',
                '-sMODULARIZE=1','-sENVIRONMENT=node','-sALLOW_MEMORY_GROWTH=1','-I'+str(ROOT/'include'),
                '-I'+str(ROOT/'bindings/javascript'),str(source),str(obj),'-o',str(module)],check=True)
            result=subprocess.run([NODE,str(ROOT/'tests/fixtures/wasm-chat-program.cjs'),str(module)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('PASS WASM chat:',result.stdout);print(result.stdout.strip())


@unittest.skipUnless(EMXX and NODE,'Emscripten C++ compiler and Node required')
class WasmDispatchContract(unittest.TestCase):
    def test_actual_threaded_dispatch_errors_and_ownership(self):
        with tempfile.TemporaryDirectory(prefix='wasm-dispatch-') as directory:
            tmp=Path(directory);module=tmp/'dispatch.js'
            runtime=(ROOT/'src/crispasr_c_api.cpp').read_text();start=runtime.index('struct crispasr_open_params_v1 {')
            (tmp/'runtime-open-layout.h').write_text(runtime[start:runtime.index('};',start)+2])
            subprocess.run([EMXX,'-std=c++17','-pthread','-fexceptions','-sDISABLE_EXCEPTION_CATCHING=0',
                '--bind','-sPROXY_TO_PTHREAD=1','-sPTHREAD_POOL_SIZE=1','-sEXIT_RUNTIME=0',
                '-sSINGLE_FILE=1','-sMODULARIZE=1','-sENVIRONMENT=web,node,worker','-sINITIAL_MEMORY=32MB',
                '-sEXPORT_NAME=dispatchFactory','-sEXPORTED_RUNTIME_METHODS=count_emval_handles',
                '-I'+str(ROOT/'bindings/javascript'),'-I'+str(ROOT/'include'),'-I'+str(ROOT/'ggml/include'),'-I'+str(tmp),
                str(ROOT/'tests/fixtures/wasm-dispatch-probe.cpp'),str(ROOT/'tests/fixtures/wasm-async-open.cpp'),
                '-o',str(module)],check=True)
            program=tmp/'test.cjs';program.write_text(r'''
const assert=require('node:assert/strict');
require(process.argv[2])().then(async m=>{
  let n=0;const eq=(a,b)=>{assert.deepEqual(a,b);n++};const rejects=f=>{assert.throws(f);n++};
  for(let i=0;!m.computeReady()&&i<200;i++)await new Promise(r=>setTimeout(r,5));
  eq(m.computeReady(),true);
  const handles=m.count_emval_handles();
  const dispatch=mode=>new Promise(resolve=>m.dispatchProbe(mode,resolve));
  for(const [mode,message] of [[1,'work sentinel'],[2,'encoding sentinel'],[3,'JS encoding sentinel'],[4,'Could not dispatch browser compute'],[5,'Proxied compute thread is not ready'],[6,'queue sentinel'],[7,'Unknown native compute exception']]){
    eq((await dispatch(mode)).error,message);eq(m.livePayloads(),0);eq(m.checkIdle(),true);eq(m.count_emval_handles(),handles);
  }
  let beats=0,queries=0;const timer=setInterval(()=>{beats++;m.probeProgress();queries++;},2);
  const success=dispatch(0);rejects(()=>m.checkIdle());rejects(()=>m.dispatchProbe(0,()=>{}));
  eq(await success,42);clearInterval(timer);eq(beats>2,true);eq(queries>2,true);eq(m.livePayloads(),0);
  rejects(()=>m.dispatchProbe(0,null));eq(m.livePayloads(),0);
  const sentinel=new Error('callback sentinel');
  await new Promise(resolve=>m.dispatchProbe(0,()=>{setTimeout(resolve,0);throw sentinel}));
  eq(m.crispasrLastCallbackError===sentinel,true);delete m.crispasrLastCallbackError;
  eq(m.livePayloads(),0);eq(m.checkIdle(),true);eq(m.count_emval_handles(),handles);
  eq(await new Promise(resolve=>m.dispatchProbe(0,first=>{eq(first,42);m.dispatchProbe(0,resolve);rejects(()=>m.checkIdle());})),42);
  eq(m.livePayloads(),0);eq(m.checkIdle(),true);eq(m.count_emval_handles(),handles);
  // A not-ready error owns no busy lease and cannot unlock another queued job.
  const unavailable=dispatch(5),active=dispatch(0);eq((await unavailable).error,'Proxied compute thread is not ready');
  rejects(()=>m.checkIdle());eq(await active,42);eq(m.livePayloads(),0);eq(m.checkIdle(),true);
  const opts={n_threads:7,use_gpu:false,verbosity:2,flash_attn:false,n_gpu_layers:3};
  eq(await new Promise(r=>m.asrOpenWithOptionsAsync('model','foo',opts,r)),true);eq(m.openLive(),1);
  const opens=m.openCalls();rejects(()=>m.asrOpenWithOptionsAsync('model','foo',{n_threads:-1},()=>{}));eq(m.openCalls(),opens);eq(m.openLive(),1);
  const tts=new Promise(r=>m.ttsOpenWithOptionsAsync('model','foo',opts,r));rejects(()=>m.closeAsr());eq(await tts,true);eq(m.openLive(),2);
  eq((await new Promise(r=>m.asrOpenWithOptionsAsync('throw','foo',opts,r))).error,'open sentinel');eq(m.openLive(),1);m.closeAsr();eq(m.openLive(),1);
  m.openDefaults(1);eq(await new Promise(r=>m.asrOpenWithOptionsAsync('model','',null,r)),true);eq(m.openLive(),2);
  const synth=text=>new Promise(r=>m.ttsSynthesizeAsync(text,r));
  const audio=await synth('normal');eq(Array.from(audio),[.25,.5,.75]);eq(m.audioLive(),0);
  for(const text of ['empty','zero-allocated']){eq((await synth(text)).length,0);eq(m.audioLive(),0);}
  for(const text of ['throw','missing','bad']){eq(typeof (await synth(text)).error,'string');eq(m.audioLive(),0);}
  const F32=globalThis.Float32Array;
  try {globalThis.Float32Array=function(){throw new Error('audio encode sentinel')};eq((await synth('normal')).error,'audio encode sentinel');}
  finally {globalThis.Float32Array=F32;}
  eq(m.audioLive(),0);eq(m.count_emval_handles(),handles);
  rejects(()=>m.ttsSynthesizeAsync('bad\0text',()=>{}));eq(m.audioLive(),0);
  m.closeAsr();m.closeTts();eq(m.openLive(),0);eq(m.livePayloads(),0);eq(m.checkIdle(),true);eq(m.count_emval_handles(),handles);
  console.log('PASS WASM threaded dispatch: '+n+' assertions');process.exit(0);
}).catch(e=>{console.error(e);process.exit(1)});
''')
            result=subprocess.run([NODE,str(program),str(module)],capture_output=True,text=True,timeout=25)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('PASS WASM threaded dispatch:',result.stdout);print(result.stdout.strip())
            browser=os.environ.get('CRISPASR_CHROME')
            if browser:
                self._exercise_browser(tmp,program.read_text(),browser)

    def _exercise_browser(self,tmp,node_program,browser):
        # Run the same assertions on the browser servicer with real pthreads.
        body=node_program[node_program.index('  let n=0;'):node_program.index("  console.log('PASS WASM threaded dispatch:")]
        (tmp/'index.html').write_text("""<!doctype html><meta charset="utf-8">
<script src="dispatch.js"></script><script>
const assert={deepEqual(a,b){if(JSON.stringify(a)!==JSON.stringify(b))throw new Error('value mismatch: '+JSON.stringify(a)+' vs '+JSON.stringify(b));},throws(fn){let caught=false;try{fn()}catch(e){caught=true}if(!caught)throw new Error('missing exception')}};
dispatchFactory().then(async m=>{
"""+body+"""
 await fetch('/result',{method:'POST',body:JSON.stringify({ok:true,assertions:n,isolated:crossOriginIsolated})});
}).catch(async e=>{await fetch('/result',{method:'POST',body:JSON.stringify({ok:false,error:String(e),stack:e.stack})})});
</script>""")
        results=queue.Queue()
        class Handler(http.server.SimpleHTTPRequestHandler):
            def end_headers(self):
                self.send_header('Cross-Origin-Opener-Policy','same-origin')
                self.send_header('Cross-Origin-Embedder-Policy','require-corp')
                super().end_headers()
            def log_message(self,*args):pass
            def do_POST(self):
                import json
                try:results.put(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                except Exception as error:results.put({'ok':False,'error':str(error)})
                self.send_response(200);self.end_headers()
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(tmp)))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        log=tmp/'browser.log'
        try:
            with log.open('w') as output:
                process=subprocess.Popen([browser,'--headless=new','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
                    '--no-first-run','--no-default-browser-check','--user-data-dir='+str(tmp/'chrome-profile'),
                    'http://127.0.0.1:'+str(server.server_port)+'/index.html'],stdout=output,stderr=output)
                try:
                    try:result=results.get(timeout=35)
                    except queue.Empty:self.fail('Browser result timeout: '+log.read_text()[-6000:])
                    self.assertTrue(result.get('ok'),result)
                    self.assertTrue(result.get('isolated'),result)
                    print('PASS WASM Chromium pthread dispatch: '+str(result['assertions'])+' assertions')
                finally:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
        finally:
            server.shutdown();server.server_close();thread.join(timeout=5)


if __name__=='__main__':unittest.main()
