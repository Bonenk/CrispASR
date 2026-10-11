#!/usr/bin/env python3
"""Actual ctype baseline/candidate and reporter WAVs through the Qwen3 C ABI.

GH CPU on Linux, and CPU plus available physical Metal on macOS. Preserve raw
assembled token bytes separately from result getters; no CPU/Metal logit claim.
"""
import argparse
import ctypes as ct
import hashlib
import json
import locale
import os
from pathlib import Path
import platform
import subprocess
import sys
from urllib.request import urlretrieve
import wave

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'qwen3-reporter'
CLIPS = (
    ('fail_chunk_01.wav', '33299295', 'b1e7cbf4abb432c2f84409ea00db0c45d94f466017f27dc4f23d357313db8e5a'),
    ('fail_chunk_16.wav', '33299296', '9080677ff23e65fe4ffeba70a7e504505869d020cfd64d9941c03f0421143d4a'),
)
MODEL = ('cstr/qwen3-asr-0.6b-GGUF', '58bd3202f835f46b24b17142cb503b0860c737a5',
         'qwen3-asr-0.6b-q4_k.gguf')


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


BASELINE_SOURCE = '7e7094320f1d9f765379a7fcf5f84a03283fb21b'
BASELINE_HEADER_SHA256 = '5f83c2c972c3fd66eb0c4c3497a274d868ab11d1e08c7c340f97a84026d939c3'


def baseline_header():
    # Actions checks out only HEAD. Fetch the immutable baseline if absent;
    # never substitute a moving branch or reconstruct the old implementation.
    if subprocess.run(['git', 'cat-file', '-e', BASELINE_SOURCE + '^{commit}'],
                      cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        subprocess.run(['git', 'fetch', '--depth=1',
                        'https://github.com/CrispStrobe/CrispASR.git', BASELINE_SOURCE],
                       cwd=ROOT, check=True, timeout=300)
    raw = subprocess.check_output(['git', 'show', BASELINE_SOURCE + ':src/core/ngram_loop_fix.h'], cwd=ROOT)
    assert hashlib.sha256(raw).hexdigest() == BASELINE_HEADER_SHA256, 'Baseline header provenance mismatch'
    return raw.decode('utf8')


def run(cmd, name, timeout=7200):
    with (OUT / (name + '.log')).open('w') as log:
        p = subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, timeout=timeout)
    print(name, p.returncode, (OUT / (name + '.log')).read_text(errors='replace')[-2000:], flush=True)
    assert p.returncode == 0, name


def infer(library, model, gpu):
    import numpy as np
    lib = ct.CDLL(str(library))
    def signature(name, args, result):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
        return fn
    class Params(ct.Structure):
        _fields_ = [(n, ct.c_int) for n in ('version', 'threads', 'gpu', 'verbosity', 'flash', 'layers')] + [
            ('reserved', ct.c_int * 6)]
    callback_type = ct.CFUNCTYPE(None, ct.c_char_p, ct.c_int, ct.c_void_p)
    opening = signature('crispasr_session_open_with_params', [ct.c_char_p, ct.c_char_p, ct.POINTER(Params)], ct.c_void_p)
    closing = signature('crispasr_session_close', [ct.c_void_p], None)
    language = signature('crispasr_session_set_source_language', [ct.c_void_p, ct.c_char_p], ct.c_int)
    maximum = signature('crispasr_session_set_max_new_tokens', [ct.c_void_p, ct.c_int], ct.c_int)
    token_cb = signature('crispasr_session_set_token_callback', [ct.c_void_p, callback_type, ct.c_void_p], None)
    transcribe = signature('crispasr_session_transcribe_lang', [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_int, ct.c_char_p], ct.c_void_p)
    nseg = signature('crispasr_session_result_n_segments', [ct.c_void_p], ct.c_int)
    segtext = signature('crispasr_session_result_segment_text', [ct.c_void_p, ct.c_int], ct.c_char_p)
    nwords = signature('crispasr_session_result_n_words', [ct.c_void_p, ct.c_int], ct.c_int)
    wordtext = signature('crispasr_session_result_word_text', [ct.c_void_p, ct.c_int, ct.c_int], ct.c_char_p)
    free = signature('crispasr_session_result_free', [ct.c_void_p], None)
    params = Params(2, 3, int(gpu), 1, 1, -1)
    session = opening(str(model).encode(), b'qwen3', ct.byref(params))
    assert session, 'Actual Qwen3 session open failed'
    rows = []
    pieces = []
    callback = callback_type(lambda data, index, user: pieces.append(data or b''))
    try:
        language(session, b'zh')
        maximum(session, 1024)
        token_cb(session, callback, None)
        for name, _, sha in CLIPS:
            path = OUT / name
            with wave.open(str(path)) as wav:
                assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
            pieces.clear()
            result = transcribe(session, pcm.ctypes.data_as(ct.POINTER(ct.c_float)), len(pcm), b'zh')
            assert result, name
            try:
                raw = b''.join(pieces)
                raw_text = raw.decode('utf8', 'strict')
                texts = [(segtext(result, i) or b'').decode('utf8', 'strict') for i in range(nseg(result))]
                words = [(wordtext(result, i, j) or b'').decode('utf8', 'strict')
                         for i in range(nseg(result)) for j in range(nwords(result, i))]
                text = ''.join(texts)
                assert text.strip() and '\ufffd' not in text and all('\ufffd' not in w for w in words), name
                assert sum('\u4e00' <= c <= '\u9fff' for c in text) >= 5, name
                for c in ('造', '加'):
                    assert text.count(c) == raw_text.count(c), (name, c, raw_text, text)
                (OUT / (name + f'.gpu{int(gpu)}.tokens.bin')).write_bytes(raw)
                rows.append(dict(clip=name, clip_sha256=sha, samples=len(pcm), gpu_requested=gpu,
                                 raw_tokens=raw_text, segments=texts, words=words,
                                 continuation_a0_bytes=raw.count(b'\xa0'),
                                 zao_count=text.count('造'), jia_count=text.count('加')))
            finally:
                free(result)
    finally:
        closing(session)
    (OUT / f'inference-{int(gpu)}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')


def main():
    from huggingface_hub import hf_hub_download
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    receipt = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   platform=platform.platform(), machine=platform.machine(), locale=locale.setlocale(locale.LC_CTYPE),
                   reporter_m4_macos26_tested=False, passed=False)
    try:
        # These libraries see the actual Python process ctype locale, with no classifier injection.
        baseline = baseline_header()
        receipt['baseline_source'] = BASELINE_SOURCE
        receipt['baseline_header_sha256'] = BASELINE_HEADER_SHA256
        for label, header in [('baseline', baseline), ('candidate', (ROOT / 'src/core/ngram_loop_fix.h').read_text())]:
            (SCRATCH / (label + '.h')).write_text(header)
            cpp = SCRATCH / (label + '.cpp')
            cpp.write_text('#include <cctype>\n#define core_ngram ' + label + '_ngram\n#include "' + label + '.h"\n'
                           'extern "C" int classify(int c) { return std::isspace((unsigned char)c) != 0; }\n'
                           'extern "C" const char* clean(const char* s) { static std::string out; '
                           'out = ' + label + '_ngram::fix_loops(s); return out.c_str(); }\n')
            shared = SCRATCH / (label + ('.dylib' if sys.platform == 'darwin' else '.so'))
            run(['c++', '-std=c++17', '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                 '-fPIC', cpp, '-o', shared], label + '-build')
            probe = ct.CDLL(str(shared))
            probe.classify.argtypes, probe.classify.restype = [ct.c_int], ct.c_int
            probe.clean.argtypes, probe.clean.restype = [ct.c_char_p], ct.c_char_p
            cases = {}
            for text in ('系统改造，建设70兆瓦。', '电加热管击穿、电加热元件、电加热管束、电加热器。'):
                raw = probe.clean(text.encode())
                cases[text] = raw.hex()
                if label == 'candidate':
                    assert raw == text.encode(), (label, text, raw)
            receipt[label] = dict(a0_is_space=bool(probe.classify(0xA0)),
                                  byte85_is_space=bool(probe.classify(0x85)), cases=cases)
        build = SCRATCH / 'build'
        run(['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
             '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF',
             '-DGGML_METAL=' + ('ON' if sys.platform == 'darwin' else 'OFF'),
             '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=OFF'], 'configure')
        run(['cmake', '--build', build, '--target', 'crispasr-lib', '-j3'], 'build')
        libraries = list(build.rglob('libcrispasr.dylib' if sys.platform == 'darwin' else 'libcrispasr.so'))
        assert len(libraries) == 1, libraries
        receipt['library_sha256'] = digest(libraries[0])
        for name, number, sha in CLIPS:
            p = OUT / name
            urlretrieve(f'https://github.com/user-attachments/files/{number}/{name}', p)
            assert digest(p) == sha, name
        model = Path(hf_hub_download(MODEL[0], MODEL[2], revision=MODEL[1]))
        receipt['model'] = dict(repo=MODEL[0], revision=MODEL[1], filename=MODEL[2], sha256=digest(model))
        run([sys.executable, __file__, '--infer', libraries[0], model, '0'], 'inference-cpu')
        receipt['cpu'] = json.loads((OUT / 'inference-0.json').read_text())
        if sys.platform == 'darwin':
            gpu = subprocess.check_output(['swift', '-e',
                'import Metal; print(MTLCreateSystemDefaultDevice()?.name ?? "NONE")'], text=True).strip()
            receipt['metal_device'] = gpu
            if gpu != 'NONE':
                run([sys.executable, __file__, '--infer', libraries[0], model, '1'], 'inference-metal')
                log = (OUT / 'inference-metal.log').read_text()
                assert 'ggml_metal' in log and gpu in log, 'Requested Metal must actually initialize'
                receipt['metal'] = json.loads((OUT / 'inference-1.json').read_text())
        receipt['passed'] = True
    except BaseException as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (OUT / 'validation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
        (OUT / 'summary.md').write_text(f"Qwen3 reporter clip/ctype check: passed={receipt['passed']}. "
                                       'M4/macOS26 and CPU/Metal logit parity remain separate.\n')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--infer':
        infer(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4] == '1')
    else:
        main()
