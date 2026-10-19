"""Compile isolated native probes and exercise the actual Go/Rust wrappers."""
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from build_legacy_wrapper_probe import extract_function

ROOT=Path(__file__).resolve().parents[1]
TOOLS=all(shutil.which(name) for name in ('cc','c++','ar','go','cargo'))
SYSTEM_LIBS={'stdc++','c++','m','c','dl','pthread','rt','atomic','gcc','gcc_s'}


@unittest.skipUnless(TOOLS and sys.platform.startswith('linux'),'Linux C/Go/Rust toolchains required')
class GoRustWrapperContract(unittest.TestCase):
    def test_native_metadata_tasks_ownership_and_callbacks(self):
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        with tempfile.TemporaryDirectory(prefix='go-rust-abi-') as directory:
            tmp=Path(directory);lib=tmp/'src';lib.mkdir()
            runtime=(ROOT/'src/crispasr_c_api.cpp').read_text()
            start=runtime.index('struct crispasr_open_params_v1 {')
            (tmp/'runtime-open-layout.h').write_text(runtime[start:runtime.index('};',start)+2])
            layouts=[]
            for name in ['crispasr_diarize_seg_abi','crispasr_diarize_turn_abi','crispasr_diarize_opts_abi']:
                start=runtime.index('struct '+name+' {');end=runtime.index('};',start)+2;layouts.append(runtime[start:end])
            (tmp/'runtime-abi-layout.h').write_text('\n'.join(layouts))
            obj=tmp/'probe.o'
            subprocess.run(['cc','-std=c11','-Werror','-fPIC','-I'+str(tmp),'-I'+str(ROOT/'include'),'-I'+str(ROOT/'ggml/include'),
                '-c',str(ROOT/'tests/fixtures/go-rust-wrapper-abi.c'),'-o',str(obj)],check=True)
            # Compile the literal scalar setters as C++, not mirrored C bodies.
            setters=sorted(set(re.findall(r'CA_EXPORT void (crispasr_params_set_\w+)\(',runtime)))
            # Completeness tripwire, in addition to the compiled field oracles.
            rust_public=(ROOT/'crispasr/src/lib.rs').read_text()+(ROOT/'crispasr/src/whisper_controls.rs').read_text()
            for name in setters:
                self.assertIn(name,rust_public,'Missing public Rust control route: '+name)

            adapters=tmp/'literal-whisper-setters.cpp'
            adapters.write_text('#include "crispasr.h"\n#include "crispasr_session.h"\n#define CA_EXPORT extern "C"\n'+
                '\n'.join(extract_function(runtime,name) for name in setters))
            setter_obj=tmp/'setters.o'
            subprocess.run(['c++','-std=c++17','-Werror','-fPIC','-I'+str(ROOT/'include'),'-I'+str(ROOT/'ggml/include'),
                '-c',str(adapters),'-o',str(setter_obj)],check=True)
            subprocess.run(['c++','-shared',str(obj),str(setter_obj),'-o',str(lib/'libcrispasr.so')],check=True)
            subprocess.run(['ar','rcs',str(lib/'libcrispasr.a'),str(obj),str(setter_obj)],check=True)
            print(f'Compiled {len(setters)} literal native Whisper scalar setters')
            # Read generated cgo dependencies instead of maintaining another list.
            flags=(ROOT/'bindings/go/whisper.go').read_text()
            names=set(re.findall(r'-l([A-Za-z0-9_+.-]+)',flags))-SYSTEM_LIBS-{'crispasr'}
            for name in sorted(names):
                subprocess.run(['ar','rcs',str(lib/('lib'+name+'.a'))],check=True)
            env=os.environ.copy()
            env['CGO_LDFLAGS']=' '.join([shlex.quote('-L'+str(lib)),shlex.quote('-Wl,-rpath,'+str(lib)),env.get('CGO_LDFLAGS','')]).strip()
            env['LD_LIBRARY_PATH']=str(lib)+os.pathsep+env.get('LD_LIBRARY_PATH','')
            # Every run gets a fresh linked binary. No fallback to a discovered build.
            go_test=['go','test']
            if env.get('CRISPASR_GO_RACE')=='1':
                go_test.append('-race');print('Go race detector enabled')
            result=subprocess.run(go_test+['-tags','probe','-run','TestProbe','-count=1','-timeout=60s','-v'],
                cwd=ROOT/'bindings/go',env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('PASS',result.stdout);print(result.stdout)
            # Compile the higher-level public interface against the same isolated
            # library; do not run its model-dependent tests in this contract.
            result=subprocess.run(go_test+['./pkg/whisper','-run','^$'],
                cwd=ROOT/'bindings/go',env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            print('PASS Go high-level package compilation (no model tests run)')
            result=subprocess.run(go_test+['-tags','probe','./pkg/whisper','-run','TestProbe','-count=1','-timeout=60s','-v'],
                cwd=ROOT/'bindings/go',env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('PASS',result.stdout);print(result.stdout)

            env['CRISPASR_SYS_LIB_DIR']=str(tmp)
            env.pop('CRISPASR_LIB_DIR',None)
            env.pop('CRISPASR_LIB_NAME',None)
            env.setdefault('CARGO_TARGET_DIR',str(tmp/'cargo-target'))
            # The fixture's mode/buffers are process-global; serialize its tests.
            result=subprocess.run(['cargo','test','--features','abi-probe','--test','probe','--','--nocapture','--test-threads=1'],
                cwd=ROOT/'crispasr',env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertRegex(result.stdout,r'test result: ok\. [1-9][0-9]* passed; 0 failed')
            print(result.stdout)
            # Includes compile-fail ownership examples; no model inference occurs.
            result=subprocess.run(['cargo','test','--doc'],cwd=ROOT/'crispasr',env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('test result: ok.',result.stdout);print(result.stdout)


if __name__=='__main__':
    unittest.main()
