"""Actual Embind result/task adapters, compiled into a tiny WASM module."""
import os
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
#include "session_results.h"
#include "session_tasks.h"
#include "session_utilities.h"
#include "session_analysis.h"
extern "C" void probe_mode(int);
extern "C" int probe_live(void);
extern "C" int probe_analysis_live(void);
extern "C" void probe_analysis_mode(int);
extern "C" void probe_db_mode(int);
extern "C" int probe_db_live(void);
extern "C" void probe_titanet_mode(int);
extern "C" int probe_titanet_live(void);
extern "C" int probe_invalid_frees(void);
extern "C" void probe_utility_mode(int);
extern "C" int probe_utility_live(void);
static emscripten::val read_result(bool seconds) {
    binding_check_reentry();float p[]={1,3,5};auto r=crispasr_session_transcribe(g_tts_session,p,3);
    auto owned=binding_copy_segments(r);crispasr_session_result_free(r);
    return binding_encode_segments(owned,seconds?100.0:1.0);
}
EMSCRIPTEN_BINDINGS(wrapper_probe) {
    binding_register_session_tasks();
    binding_register_utilities();
    binding_register_analysis();
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
 console.log('PASS WASM ABI: '+n+' assertions');
}).catch(e=>{console.error(e);process.exitCode=1});
'''

@unittest.skipUnless(EMCC and EMXX and NODE,'Emscripten C/C++ compilers and Node required')
class WasmWrapperContract(unittest.TestCase):
    def test_owned_metadata_music_conversion_and_stream_callbacks(self):
        with tempfile.TemporaryDirectory(prefix='wasm-wrapper-abi-') as tmp:
            tmp=Path(tmp);objects=[]
            for name in ['python-wrapper-abi','wrapper-session-metadata','wrapper-audio-utilities','wasm-wrapper-utilities','wasm-wrapper-analysis','wrapper-parakeet-legacy','wrapper-titanet-legacy','wrapper-speaker-db']:
                obj=tmp/(name+'.o');objects.append(str(obj))
                subprocess.run([EMCC,'-I'+str(ROOT/'include'),'-I'+str(ROOT/'ggml/include'),'-c',str(ROOT/'tests/fixtures'/(name+'.c')),'-o',str(obj)],check=True)
            # Unexercised registration dependencies abort if accidentally called.
            # Concrete public-header probes supply every operation asserted below.
            fixture_names=['python-wrapper-abi','wrapper-session-metadata','wrapper-audio-utilities','wasm-wrapper-utilities','wasm-wrapper-analysis','wrapper-parakeet-legacy','wrapper-titanet-legacy','wrapper-speaker-db']
            concrete='\n'.join((ROOT/'tests/fixtures'/(n+'.c')).read_text() for n in fixture_names)
            adapters='\n'.join((ROOT/'bindings/javascript'/n).read_text() for n in ['session_results.h','session_tasks.h','session_utilities.h','session_analysis.h'])
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

if __name__=='__main__':unittest.main()
