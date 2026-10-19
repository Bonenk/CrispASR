"""Build a model-free legacy Whisper probe using literal native C ABI adapters."""
from pathlib import Path
import re
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def extract_function(source,name):
    declaration=re.search(r'CA_EXPORT\s+[^;{}]*\b'+re.escape(name)+r'\s*\([^;{}]*\)\s*\{',source)
    if not declaration:raise ValueError('Native function not found: '+name)
    start=declaration.start();brace=declaration.end()-1;depth=1;end=brace+1
    while depth:
        if source[end]=='{':depth+=1
        elif source[end]=='}':depth-=1
        end+=1
    return source[start:end]


def build(output):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    source=(ROOT/'src/crispasr_c_api.cpp').read_text()
    names=sorted(set(re.findall(r'CA_EXPORT void (crispasr_params_set_\w+)\(',source)))
    names+=['crispasr_ctx_params_set_dtw','crispasr_token_t0','crispasr_token_t1','crispasr_token_p',
        'crispasr_token_dtw_t','crispasr_token_n_alts','crispasr_token_alt_id','crispasr_token_alt_p','crispasr_token_alt_text',
        'crispasr_transcribe_parallel','crispasr_transcribe_parallel_by_ref','crispasr_get_progress']
    cpp=output.parent/'legacy-native-adapters.cpp'
    cpp.write_text('#include "crispasr.h"\n#include "crispasr_session.h"\n#include <atomic>\n#include <cstring>\n#define CA_EXPORT extern "C"\nstatic std::atomic<int> g_progress{-1};\n'+
        '\n'.join(extract_function(source,name) for name in names))
    subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-pthread','-shared','-fPIC','-Werror','-I'+str(ROOT/'include'),'-I'+str(ROOT/'ggml/include'),
        str(cpp),str(ROOT/'tests/fixtures/wrapper-whisper-legacy.cpp'),'-o',str(output)],check=True)
    return output


if __name__=='__main__':build(sys.argv[1])
