#!/usr/bin/env python3
"""Literal-body allocator regression for omnivoice output tails.

Extracts BOTH whole public functions from src/omnivoice.cpp via balanced
braces (no hand-copy), compiles those exact strings with tiny upstream stubs
and deterministic failing allocator. Permanent CI: current bodies must pass
normal+OOM; guard-removal mutation of the same bodies must crash on OOM.
Optional historical 5243 baseline (fixture or --baseline-source / env) is
verified the same way when present — proof-only, not required for CI.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/omnivoice-alloc-baseline.json"

def block(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index('{', start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]

PREAMBLE = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
static bool ov_bench_enabled(){ return false; }
struct ov_bench_stage{
    const char* name; std::chrono::steady_clock::time_point t0;
    explicit ov_bench_stage(const char* n):name(n),t0(std::chrono::steady_clock::now()){}
    ~ov_bench_stage(){}
};
struct ov_hp{ uint32_t n_codebooks=8; };
struct ov_higgs_tokenizer{ bool loaded=false; };
struct omnivoice_context{
    ov_hp hp;
    ov_higgs_tokenizer tokenizer;
    int verbosity=0;
    float ref_rms=0.0f;
};
struct ov_gen_result{ std::vector<int32_t> codes; int T=0; int n_codebooks=0; };
static std::vector<int32_t> STUB_CODES = {1,2,3,4,5,6,7,8,  10,20,30,40,50,60,70,80};
static std::vector<float>   STUB_PCM   = []{
    std::vector<float> v(240); for(size_t i=0;i<v.size();++i) v[i]=float(i)*0.001f; return v;
}();
static ov_gen_result generate_iterative(omnivoice_context* ctx, const std::string& text){
    (void)ctx; (void)text;
    ov_gen_result r; r.codes = STUB_CODES; r.T = 2; r.n_codebooks=8; return r;
}
static std::vector<float> higgs_decode(omnivoice_context* ctx, const int32_t* codes, int n_cb, int T_frames){
    (void)ctx; (void)codes; (void)n_cb; (void)T_frames;
    std::vector<float> pcm = STUB_PCM;
    if(pcm.empty()) pcm.push_back(0.0f);
    return pcm;
}
static bool g_fail_next = false;
static void* test_malloc(size_t n){
    if(g_fail_next){ g_fail_next=false; return nullptr; }
    return std::malloc(n);
}
static void test_set_fail_next(bool v){ g_fail_next=v; }
'''

HARNESS_TAIL = r'''
int main(int argc,char**argv){
    if(argc<2){ std::cerr<<"usage: prog <mode>\n"; return 2; }
    std::string mode=argv[1];
    omnivoice_context ctx;
    ctx.hp.n_codebooks=8;
    ctx.tokenizer.loaded=true;
    ctx.verbosity=0;
    ctx.ref_rms=0.0f;
    if(mode=="normal-codes"){
        int n=-1;
        int32_t* p = omnivoice_synthesize_codes(&ctx, "hello", &n);
        if(!p || n!= (int)STUB_CODES.size()){ std::cerr<<"normal codes: null or n="<<n<<"\n"; return 1; }
        if(std::memcmp(p, STUB_CODES.data(), n*sizeof(int32_t))!=0){ std::cerr<<"normal codes mismatch\n"; free(p); return 1; }
        free(p);
        std::cout<<"PASS normal-codes n="<<n<<"\n"; return 0;
    }
    if(mode=="normal-pcm"){
        int32_t dummy[16]={0};
        int m=-1;
        float* q = omnivoice_decode_codes(&ctx, dummy, 16, &m);
        if(!q || m!=(int)STUB_PCM.size()){ std::cerr<<"normal pcm: null or m="<<m<<"\n"; return 1; }
        if(std::memcmp(q, STUB_PCM.data(), m*sizeof(float))!=0){ std::cerr<<"normal pcm mismatch\n"; free(q); return 1; }
        free(q);
        std::cout<<"PASS normal-pcm m="<<m<<"\n"; return 0;
    }
    if(mode=="oom-codes"){
        test_set_fail_next(true);
        int n=999;
        int32_t* p = omnivoice_synthesize_codes(&ctx, "hello", &n);
        if(p!=nullptr || n!=0){ std::cerr<<"oom codes: expected null+0 got "<<(void*)p<<" n="<<n<<"\n"; if(p) free(p); return 1; }
        std::cout<<"PASS oom-codes null+0\n"; return 0;
    }
    if(mode=="oom-pcm"){
        int32_t dummy[16]={0};
        test_set_fail_next(true);
        int m=999;
        float* q = omnivoice_decode_codes(&ctx, dummy, 16, &m);
        if(q!=nullptr || m!=0){ std::cerr<<"oom pcm: expected null+0 got "<<(void*)q<<" m="<<m<<"\n"; if(q) free(q); return 1; }
        std::cout<<"PASS oom-pcm null+0\n"; return 0;
    }
    if(mode=="quiet-gain"){
        omnivoice_context ctx2=ctx; ctx2.ref_rms=0.05f;
        int32_t dummy[16]={0};
        int m=-1;
        float* q = omnivoice_decode_codes(&ctx2, dummy, 16, &m);
        if(!q){ std::cerr<<"quiet gain: null\n"; return 1; }
        bool ok=true;
        for(int i=0;i<m;++i){ float exp = STUB_PCM[i]*0.5f; if(q[i]!=exp){ ok=false; break; } }
        free(q);
        if(!ok){ std::cerr<<"quiet gain mismatch\n"; return 1; }
        std::cout<<"PASS quiet-gain\n"; return 0;
    }
    std::cerr<<"unknown mode "<<mode<<"\n"; return 2;
}
'''

def make_executable(source_text: str, directory: Path, label: str, mutated=False):
    bodies = '\n'.join(block(source_text, sig) for sig in [
        'int32_t* omnivoice_synthesize_codes(',
        'float* omnivoice_decode_codes(',
    ])
    if mutated:
        bodies = bodies.replace('    if (!out) {\n        *out_n_codes = 0;\n        return nullptr;\n    }\n', '')
        bodies = bodies.replace('    if (!out) {\n        *out_n_samples = 0;\n        return nullptr;\n    }\n', '')
    compiler = shutil.which('c++') or shutil.which('g++')
    if not compiler:
        raise RuntimeError('c++ required for omnivoice alloc regression')
    harness = PREAMBLE + '\n#define malloc test_malloc\n' + bodies + '\n' + HARNESS_TAIL
    cpp = directory / f'{label}.cpp'
    exe = directory / label
    cpp.write_text(harness)
    cmd = [compiler, '-std=c++17', '-O2', str(cpp), '-o', str(exe)]
    build = subprocess.run(cmd, capture_output=True, text=True)
    if build.returncode != 0:
        raise RuntimeError(f"compile {label} failed: {build.stderr}\n{cmd}")
    return exe, harness, bodies, cmd, build

def run_case(exe: Path, mode: str):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        pass  # Non-Linux or restricted; harmless — core dumps are best-effort disabled.
    result = subprocess.run([str(exe), mode], capture_output=True, text=True)
    return result

def load_baseline_source(baseline_arg=None):
    """Resolve optional historical baseline source.

    Priority: CLI --baseline-source > env CRISPASR_OMNIVOICE_ALLOC_BASELINE >
    repo fixture tests/fixtures/omnivoice-alloc-baseline.json (functions glued).
    Returns (source_text, pin) or (None, None) if not available — caller skips.
    """
    path = None
    if baseline_arg:
        path = Path(baseline_arg)
    elif os.environ.get('CRISPASR_OMNIVOICE_ALLOC_BASELINE'):
        path = Path(os.environ['CRISPASR_OMNIVOICE_ALLOC_BASELINE'])
    elif FIXTURE.exists():
        fixture = json.loads(FIXTURE.read_text())
        for name, function in fixture['functions'].items():
            actual = hashlib.sha256(function['body'].encode()).hexdigest()
            if actual != function['sha256']:
                raise ValueError(f'Historical function checksum mismatch: {name}')
        # Glue the checked historical functions into the standalone harness.
        text = '\n'.join(f["body"] for f in fixture["functions"].values())
        return text, fixture.get("pin", "fixture")
    else:
        return None, None
    return path.read_text(), str(path)

def run_harness(current_source_text: str, baseline_source_text=None, proof_dir=None, verbose=True):
    """Core harness: compile+run fixed, mutated, and optional baseline.

    Returns dict with results, recorded argv/codes, and proof retention if asked.
    No absolute maintainer paths; TMPDIR caller-owned (storage/scratch-tmp).
    """
    env_tmp = os.environ.get('TMPDIR')
    tmpdir_base = Path(env_tmp) if env_tmp else None
    if tmpdir_base is not None:
        tmpdir_base.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='omnivoice-alloc-', dir=tmpdir_base) as tmp:
        tmpdir = Path(tmp)
        # Build fixed (current).
        exe_fixed, harness_fixed, _, cmd_fixed, build_fixed = make_executable(current_source_text, tmpdir, 'fixed')
        # Build mutated (guard removal) from same current.
        exe_mutated, harness_mutated, _, cmd_mutated, build_mutated = make_executable(current_source_text, tmpdir, 'mutated', mutated=True)
        # Optional historical baseline (frozen 5243) — only if provided.
        exe_baseline = harness_baseline = cmd_baseline = None
        if baseline_source_text:
            exe_baseline, harness_baseline, _, cmd_baseline, build_baseline = make_executable(baseline_source_text, tmpdir, 'baseline')

        modes = ['normal-codes', 'normal-pcm', 'oom-codes', 'oom-pcm', 'quiet-gain']
        results = {'fixed': {}, 'mutated': {}, 'baseline': {}}

        for mode in modes:
            r = run_case(exe_fixed, mode)
            results['fixed'][mode] = {'rc': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr, 'argv': [str(exe_fixed), mode]}
            if r.returncode != 0:
                raise AssertionError(f"fixed {mode} failed: rc={r.returncode} stdout={r.stdout!r} stderr={r.stderr!r}")
            if verbose:
                print(f"PASS fixed {mode}: {r.stdout.strip()}")

        for mode in ['oom-codes', 'oom-pcm']:
            r = run_case(exe_mutated, mode)
            results['mutated'][mode] = {'rc': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr, 'argv': [str(exe_mutated), mode]}
            if r.returncode == 0:
                raise AssertionError(f"mutated {mode} unexpectedly passed — guards not load-bearing")
            if verbose:
                print(f"CONFIRMED mutated {mode} fails as expected (rc={r.returncode}, signal {-r.returncode if r.returncode<0 else 'n/a'})")

        if exe_baseline:
            for mode in ['oom-codes', 'oom-pcm']:
                r = run_case(exe_baseline, mode)
                results['baseline'][mode] = {'rc': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr, 'argv': [str(exe_baseline), mode]}
                if r.returncode == 0:
                    raise AssertionError(f"baseline {mode} unexpectedly passed — frozen 5243 should crash on OOM")
                if verbose:
                    print(f"CONFIRMED baseline {mode} fails as expected (rc={r.returncode})")

        if proof_dir:
            proof_dir = Path(proof_dir)
            proof_dir.mkdir(parents=True, exist_ok=False)
            current_sha = hashlib.sha256(current_source_text.encode()).hexdigest()
            baseline_sha = hashlib.sha256(baseline_source_text.encode()).hexdigest() if baseline_source_text else None
            meta = {
                'current_source_sha256': current_sha,
                'baseline_source_sha256': baseline_sha,
                'scope': 'Literal extracted omnivoice_synthesize_codes + omnivoice_decode_codes bodies with failing-allocator guard; no graph/math/default changed',
                'modes_passed_fixed': modes,
                'mutated_oom_must_crash': ['oom-codes', 'oom-pcm'],
                'baseline_oom_must_crash': ['oom-codes', 'oom-pcm'] if baseline_source_text else 'skipped (no baseline provided)',
                'compiler_argv': {'fixed': cmd_fixed, 'mutated': cmd_mutated, 'baseline': cmd_baseline if baseline_source_text else None},
                'child_exit_codes': {
                    'fixed': {k: v['rc'] for k, v in results['fixed'].items()},
                    'mutated': {k: v['rc'] for k, v in results['mutated'].items()},
                    'baseline': {k: v['rc'] for k, v in results['baseline'].items()} if baseline_source_text else None,
                },
                'build': {
                    'fixed': {'stdout': build_fixed.stdout, 'stderr': build_fixed.stderr},
                    'mutated': {'stdout': build_mutated.stdout, 'stderr': build_mutated.stderr},
                },
            }
            (proof_dir / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
            for name, exe, harness in [('fixed', exe_fixed, harness_fixed), ('mutated', exe_mutated, harness_mutated)]:
                shutil.copy2(exe, proof_dir / name)
                (proof_dir / f'{name}.cpp').write_text(harness)
            if exe_baseline:
                shutil.copy2(exe_baseline, proof_dir / 'baseline')
                (proof_dir / 'baseline.cpp').write_text(harness_baseline)
            for mode in modes:
                (proof_dir / f'fixed-{mode}.log').write_text(results['fixed'][mode]['stdout'] + results['fixed'][mode]['stderr'])
            for mode in ['oom-codes', 'oom-pcm']:
                for label in ['mutated', 'baseline']:
                    if label in results and mode in results[label]:
                        r = results[label][mode]
                        (proof_dir / f'{label}-{mode}.log').write_text(f"rc={r['rc']}\nargv={r['argv']}\nstdout={r['stdout']}\nstderr={r['stderr']}\n")
            if verbose:
                print(f"Proof retained under {proof_dir}")

        return results

# ---- unittest discovery (CI) ------------------------------------------

class OmnivoiceAllocTest(unittest.TestCase):
    """CI-discoverable wrapper: exercises actual source via literal extraction."""

    def test_normal_and_oom_via_extraction(self):
        current = (ROOT / 'src/omnivoice.cpp').read_text()
        baseline_text, _ = load_baseline_source()
        run_harness(current, baseline_text, verbose=False)

def main():
    parser = argparse.ArgumentParser(description='Omnivoice allocator literal-body regression')
    parser.add_argument('--baseline-source', help='Optional historical source file for frozen pin comparison')
    parser.add_argument('--proof-dir', help='Retain proof under this directory')
    args, _ = parser.parse_known_args()
    current = (ROOT / 'src/omnivoice.cpp').read_text()
    baseline_text = None
    if args.baseline_source:
        baseline_text = Path(args.baseline_source).read_text()
    else:
        baseline_text, _ = load_baseline_source()
    run_harness(current, baseline_text, proof_dir=args.proof_dir)

if __name__ == '__main__':
    if any(a.startswith('--proof-dir') or a.startswith('--baseline-source') for a in sys.argv[1:]):
        main()
    elif len(sys.argv) > 1 and sys.argv[1].startswith('-'):
        unittest.main()
    else:
        main()
