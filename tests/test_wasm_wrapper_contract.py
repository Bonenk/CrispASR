"""Actual Embind result/task adapters, compiled into a tiny WASM module."""
import os
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
extern "C" void probe_mode(int);
extern "C" int probe_live(void);
extern "C" int probe_invalid_frees(void);
static emscripten::val read_result(bool seconds) {
    binding_check_reentry();float p[]={1,3,5};auto r=crispasr_session_transcribe(g_tts_session,p,3);
    auto owned=binding_copy_segments(r);crispasr_session_result_free(r);
    return binding_encode_segments(owned,seconds?100.0:1.0);
}
EMSCRIPTEN_BINDINGS(wrapper_probe) {
    binding_register_session_tasks();
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
 eq(m.probeLive(),0);eq(m.probeInvalidFrees(),0);
 console.log('PASS WASM ABI: '+n+' assertions');
}).catch(e=>{console.error(e);process.exitCode=1});
'''

@unittest.skipUnless(EMCC and EMXX and NODE,'Emscripten C/C++ compilers and Node required')
class WasmWrapperContract(unittest.TestCase):
    def test_owned_metadata_music_conversion_and_stream_callbacks(self):
        with tempfile.TemporaryDirectory(prefix='wasm-wrapper-abi-') as tmp:
            tmp=Path(tmp);objects=[]
            for name in ['python-wrapper-abi','wrapper-session-metadata']:
                obj=tmp/(name+'.o');objects.append(str(obj))
                subprocess.run([EMCC,'-I'+str(ROOT/'include'),'-c',str(ROOT/'tests/fixtures'/(name+'.c')),'-o',str(obj)],check=True)
            source=tmp/'probe.cpp';source.write_text(SHIM)
            module=tmp/'probe.js'
            subprocess.run([EMXX,'-std=c++17','-fexceptions','-sDISABLE_EXCEPTION_CATCHING=0','--bind','-sSINGLE_FILE=1',
                '-sMODULARIZE=1','-sENVIRONMENT=node','-sALLOW_MEMORY_GROWTH=1','-I'+str(ROOT/'include'),
                '-I'+str(ROOT/'bindings/javascript'),str(source),*objects,'-o',str(module)],check=True)
            program=tmp/'test.cjs';program.write_text(PROGRAM)
            result=subprocess.run([NODE,str(program),str(module)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('PASS WASM ABI:',result.stdout);print(result.stdout.strip())

if __name__=='__main__':unittest.main()
