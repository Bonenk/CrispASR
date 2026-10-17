#!/usr/bin/env python3
"""Independent pinned upstream waveform/text oracle, no weights or torch.

Executes the actual upstream utility ASTs with numpy/pydub, not a Python
translation of the native implementation. All numerical calls are archived.
"""
import ast
import ctypes as C
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import shutil
import urllib.request
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, split_on_silence

ROOT = Path(__file__).resolve().parents[2]
PIN = '08be0b4ccbac3e13e374e86fbfead4b4cac343e2'
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'omnivoice-audio'
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
ARRAYS = OUT / 'arrays'
ARRAYS.mkdir(exist_ok=True)
SOURCES = OUT / 'native-sources'
SOURCES.mkdir(exist_ok=True)
for name in ('src/core/omnivoice_audio.h', 'tools/ci-heavy/omnivoice_audio_probe.cpp',
             'tools/ci-heavy/omnivoice_audio_parity.py', 'LICENSE'):
    shutil.copyfile(ROOT / name, SOURCES / Path(name).name)
namespace = dict(np=np, AudioSegment=AudioSegment, detect_leading_silence=detect_leading_silence,
                 split_on_silence=split_on_silence)
receipt = dict(upstream_revision=PIN, source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
               sources={}, waveforms=[], texts=[], runtime_acceptance=False, component_pass=False,
               original_pr=521, original_commit='a1ae00dff82185c3ec8a44cef3054a09d3ab70b1',
               arrays=[], inputs=[],
               numpy_version=np.__version__, pydub_version=importlib.metadata.version('pydub'),
               native_sources={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                   ('src/core/omnivoice_audio.h','tools/ci-heavy/omnivoice_audio_probe.cpp')})
def checkpoint():
    (OUT/'omnivoice-audio-parity.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n')


def retain(value, name):
    target = ARRAYS / (name + '.npy')
    np.save(target, value)
    row = dict(path=target.relative_to(OUT).as_posix(), bytes=target.stat().st_size,
               sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
               shape=list(value.shape), dtype=str(value.dtype))
    receipt['arrays'].append(row)
    return row['path']


checkpoint()
for name, selected in [('audio', {'numpy_to_audiosegment', 'audiosegment_to_numpy', 'remove_silence',
                                 'remove_silence_edges', 'fade_and_pad_audio'}),
                       ('text', {'add_punctuation'})]:
    url = f'https://raw.githubusercontent.com/k2-fsa/OmniVoice/{PIN}/omnivoice/utils/{name}.py'
    source = urllib.request.urlopen(url, timeout=60).read()
    (OUT / (name + '-upstream.py')).write_bytes(source)
    expected_source = {'audio': '6e2154cb8936138ee6c417c253dc9b222500ed322e6fbb2c0377a0b6850b8b48',
                       'text': 'ec833278ea7dcb22dfd3d27c3a932f8fb64948826242abb721a63b33b1c0a45a'}
    assert hashlib.sha256(source).hexdigest() == expected_source[name]

    receipt['sources'][name] = dict(url=url, sha256=hashlib.sha256(source).hexdigest())
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if
                 isinstance(node, ast.FunctionDef) and node.name in selected or
                 isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'END_PUNCTUATION' for t in node.targets)]
    exec(compile(tree, url, 'exec'), namespace)
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT), 'Oracle requires a clean committed checkout'
exe = SCRATCH / 'probe.so'
command = ['g++','-std=c++17','-O2','-shared','-fPIC','-Isrc',
           'tools/ci-heavy/omnivoice_audio_probe.cpp','-o',str(exe)]
receipt['compiler_command'] = command
with (OUT/'native-build.log').open('w') as log:
    subprocess.run(['g++', '--version'], stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
shutil.copyfile(exe, OUT/'actual-probe.so')
receipt['native_binary_sha256'] = hashlib.sha256(exe.read_bytes()).hexdigest()
with (OUT/'native-ldd.log').open('w') as log:
    subprocess.run(['ldd', exe], stdout=log, stderr=subprocess.STDOUT, check=True)
receipt['actual_linked_libraries'] = []
for line in (OUT/'native-ldd.log').read_text().splitlines():
    for token in line.split():
        library = Path(token)
        if token.startswith('/') and library.is_file():
            receipt['actual_linked_libraries'].append(dict(path=token,
                sha256=hashlib.sha256(library.read_bytes()).hexdigest()))

checkpoint()
lib = C.CDLL(str(exe))
ptr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
lib.clean.argtypes = [ptr, C.c_int, C.c_int, C.c_int, C.c_int, ptr, C.c_int]
lib.fade.argtypes = [ptr, C.c_int, C.c_float, C.c_float, ptr, C.c_int]
lib.punctuate.argtypes = [C.c_char_p, C.c_void_p, C.c_int]
rng = np.random.default_rng(518)
fixtures = [np.empty(0,dtype=np.float32), np.zeros(24000,dtype=np.float32)]
for n in (1,2,3,11,12,13,23,24,25,239,240,241,4799,4800,4801,11999,12000,12001,24013,96001):
    for amp in (.0001,.00314,.00318,.1,1.2):
        fixtures.append((rng.standard_normal(n)*amp).astype(np.float32))
for _ in range(20):
    blocks = []
    for _ in range(12):
        n = int(rng.integers(1, 18000))
        amplitude = rng.choice([0.,103/32768,104/32768,.2])
        blocks.append((rng.choice([-1.,1.],n)*amplitude).astype(np.float32))
    fixtures.append(np.concatenate(blocks))
fixtures += [np.concatenate([np.zeros(24000),rng.uniform(-.3,.3,24000),np.zeros(48000),
                             rng.uniform(-.3,.3,24000),np.zeros(24000)]).astype(np.float32)]
negative_control = False
for i, audio in enumerate(fixtures):
    receipt['inputs'].append(dict(fixture=i, path=retain(audio, f'input-{i:03d}')))
    for mid, lead, trail in ((200,100,200),(500,100,100),(0,100,100)):
        expected = namespace['remove_silence'](audio[None,:],24000,mid,lead,trail).ravel()
        output = np.empty(len(audio)+48000,dtype=np.float32)
        n = lib.clean(audio,len(audio),mid,lead,trail,output,len(output))
        actual = output[:max(0,n)]
        expected_path = retain(expected, f'clean-{i:03d}-{mid}-expected')
        actual_path = retain(actual.copy(), f'clean-{i:03d}-{mid}-actual')
        checkpoint()
        assert n >= 0 and np.array_equal(expected,actual), (i,mid,expected.shape,actual.shape)
        negative_control |= not np.array_equal(audio,expected)
        receipt['waveforms'].append(dict(fixture=i,operation='clean',mid=mid,lead=lead,trail=trail,samples=n,
            expected=expected_path, actual=actual_path, exact=True,sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
    for pad, duration in ((.1,.1),(0.,0.),(0.,.0001)):
        expected = namespace['fade_and_pad_audio'](audio[None,:],pad,duration).ravel()
        n = lib.fade(audio,len(audio),pad,duration,output,len(output))
        actual = output[:max(0,n)]
        expected_path = retain(expected, f'fade-{i:03d}-{pad}-{duration}-expected')
        actual_path = retain(actual.copy(), f'fade-{i:03d}-{pad}-{duration}-actual')
        checkpoint()
        assert n == len(expected), (i,'fade length',n,len(expected))
        error = float(np.max(np.abs(expected-actual))) if n else 0.
        assert error <= 2e-7, (i,'fade error',error)
        receipt['waveforms'].append(dict(fixture=i,operation='fade',pad=pad,duration=duration,
                                        expected=expected_path,actual=actual_path,samples=n,max_abs=error))
assert negative_control, 'Raw/unpatched output must fail the independent oracle'
for text in ('','   ','hello',' hello! ','你好','こんにちは','привет','مرحبا','हिन्दी',
             '\u3000你好\u00a0','end”','end…','end;','汉字 english','かな','end）','end】','end、'):
    expected = namespace['add_punctuation'](text)
    buf = C.create_string_buffer(4096)
    n = lib.punctuate(text.encode(),buf,len(buf))
    actual = buf.value.decode()
    assert n >= 0 and actual == expected, (text,expected,actual)
    receipt['texts'].append(dict(input=text,output=actual,exact=True))
# Retain a deterministic nonzero fixture as scale/order rejection witnesses.
witness = next(row for row in receipt['waveforms'] if row['operation'] == 'clean'
               and np.load(OUT/row['expected']).size > 1000
               and np.max(np.abs(np.load(OUT/row['expected']))) > .01)
reference = np.load(OUT/witness['expected'])
assert not np.array_equal(reference*2, reference)
assert not np.array_equal(reference[::-1], reference)
receipt['rejecting_controls'] = dict(fixture=witness['fixture'], expected=witness['expected'],
    scaled=retain(reference*2, 'negative-scaled'), reversed=retain(reference[::-1].copy(), 'negative-reversed'))
receipt['component_pass'] = True
receipt['unpatched_negative_control_rejected'] = negative_control
(OUT/'omnivoice-audio-parity.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
print('OMNIVOICE_AUDIO_PARITY_PASS',len(receipt['waveforms']),len(receipt['texts']))
