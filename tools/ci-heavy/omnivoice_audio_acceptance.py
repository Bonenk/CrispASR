#!/usr/bin/env python3
"""Pinned utility oracle, real CLI/session synthesis, and TTS->ASR acceptance.

References must pass their own transcript with at least one of the two pinned
ASR judges. Every eligible reference is scored on both cross-sentence targets;
clone outcomes must never determine reference eligibility. ASR disagreement is
evidence about this benchmark, not proof of what a speaker actually pronounced.
"""
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
import re
import subprocess
import sys
import wave
import numpy as np
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
TEMP = Path(os.environ['HEAVY_SCRATCH']) / 'omnivoice-518'
OUT.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(TEMP)
MODEL_PIN = '92266739c4496b04a497e164cf8c2801a792551e'
FIXTURE_PIN = 'd0a7d7a8be318a5841dfdbe6ad37d3acf75523e3'
TEXT = 'The quick brown fox jumps over the lazy dog.'
TEXT2 = 'We should bring enough water and food for the long walk home.'
WER_GATE = 0.2
# Two independent architectures judge the reference: an RNNT and a CTC model.
# A single engine can share a bad prior with the generation control.
ASR_MODELS = {
    'nemotron': ('cstr/nemotron-3.5-asr-streaming-GGUF',
                 'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf',
                 'bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2'),
    'parakeet': ('cstr/parakeet-ctc-0.6b-GGUF',
                 'parakeet-ctc-0.6b-q4_k.gguf',
                 '159612ab8741e595e95e992298acfc400dc502b7'),
}


def run(command, tag, env=None):
    with (OUT / (tag + '.log')).open('w') as log:
        result = subprocess.run(list(map(str,command)), cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=3600)
    print(tag, result.returncode, (OUT/(tag+'.log')).read_text()[-2000:], flush=True)
    assert result.returncode == 0, tag


def words(text):
    return re.findall('[a-z]+', re.sub(r'<[^>]*>', '', text).lower())


def word_wer(expected, actual):
    ref, hyp = words(expected), words(actual)
    row = list(range(len(hyp) + 1))
    for i, w in enumerate(ref, 1):
        new = [i]
        for j, v in enumerate(hyp, 1):
            new.append(min(new[-1]+1, row[j]+1, row[j-1]+(w != v)))
        row = new
    return row[-1] / max(1, len(ref))


def read_pcm16(path):
    with wave.open(str(path)) as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768


run([sys.executable, ROOT/'tools/ci-heavy/omnivoice_audio_parity.py'], 'utility-oracle')
build = TEMP/'build'
run(['cmake','-S',ROOT,'-B',build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
     '-DBUILD_SHARED_LIBS=ON','-DGGML_NATIVE=OFF','-DGGML_CUDA=OFF',
     '-DCRISPASR_BUILD_TESTS=OFF','-DCRISPASR_BUILD_SERVER=OFF'], 'configure')
run(['cmake','--build',build,'--target','crispasr-cli','crispasr-lib','-j4'], 'build')
model_dir = TEMP/'models'
paths = {name: hf_hub_download('cstr/omnivoice-GGUF',name,revision=MODEL_PIN,local_dir=model_dir)
         for name in ('omnivoice-q8_0.gguf','omnivoice-tokenizer-f16.gguf')}
manifest_path = hf_hub_download('cstr/crispasr-regression-fixtures',
    'index-echo-9b/roundtrip-piper/roundtrip-audio.json', revision=FIXTURE_PIN)
manifest = json.loads(Path(manifest_path).read_text())
fixtures = {}
for name, item in manifest['cases'].items():
    path = hf_hub_download('cstr/crispasr-regression-fixtures',
        'index-echo-9b/roundtrip-piper/' + item['audio'], revision=FIXTURE_PIN)
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == item['sha256'], name
    fixtures[name] = dict(item, path=path)

sys.path.insert(0,str(ROOT/'python'))
from crispasr import Session
lib = next(p for p in build.rglob('libcrispasr.*') if p.suffix in ('.so', '.dylib'))

# --- reference integrity gate: check ASR readability before any synthesis.
# This evaluates benchmark suitability, without assigning error causality.
asr_paths = {}
for engine, (repo, name, rev) in ASR_MODELS.items():
    asr_paths[engine] = hf_hub_download(repo, name, revision=rev, local_dir=TEMP/'asr'/engine)
integrity = {}
with ExitStack() as stack:
    judges = {engine: stack.enter_context(Session(model, lib_path=str(lib), n_threads=4))
              for engine, model in asr_paths.items()}
    for name, item in fixtures.items():
        pcm = read_pcm16(item['path'])
        row = dict(text=item['text'], seconds=item['seconds'], engines={})
        for engine, judge in judges.items():
            actual = ' '.join(segment.text for segment in
                              judge.transcribe(pcm, sample_rate=16000, language='en'))
            row['engines'][engine] = dict(transcript=actual, wer=word_wer(item['text'], actual))
        row['min_wer'] = min(v['wer'] for v in row['engines'].values())
        row['max_wer'] = max(v['wer'] for v in row['engines'].values())
        integrity[name] = row
        print('reference', name, row['min_wer'], row['engines'], flush=True)
(OUT/'reference-integrity.json').write_text(json.dumps(integrity, indent=2) + '\n')

# Apply the declared integrity rule before observing any clone output.
# Every eligible reference participates; a failed synthesis is a failed gate,
# never a reason to remove that reference from this campaign.
CLONE_REFS = tuple(name for name in fixtures if integrity[name]['min_wer'] <= WER_GATE)
assert CLONE_REFS, 'No reference passes the integrity gate'
assert all(fixtures[name]['text'].rstrip('.') != target.rstrip('.')
           for name in CLONE_REFS for target in (TEXT, TEXT2)), 'clone target repeats the reference text'
os.environ['CRISPASR_OMNIVOICE_AUDIO_LEGACY'] = '0'
outputs = {}

# A no-reference raw/full pair holds generated codes fixed, directly checking
# the actual decode funnel against the independent upstream utilities.
for legacy in (True,False):
    label = 'session-raw' if legacy else 'session-full'
    os.environ['CRISPASR_OMNIVOICE_AUDIO_LEGACY'] = '1' if legacy else '0'
    os.environ['CRISPASR_OMNIVOICE_DUMP_CODES'] = str(OUT/(label+'.codes'))
    with Session(paths['omnivoice-q8_0.gguf'],lib_path=str(lib),backend='omnivoice',n_threads=4) as s:
        s.set_codec_path(paths['omnivoice-tokenizer-f16.gguf'])
        s.set_tts_seed(42)
        assert s._lib.crispasr_session_set_tts_seed(s._handle,42) == 0
        s.set_tts_steps(32)
        assert s._lib.crispasr_session_set_tts_steps(s._handle,32) == 0
        # Compare native DSP before the default API adds its watermark.
        s.accept_marking_responsibility('Independent upstream waveform acceptance fixture')
        pcm = s.synthesize_raw(TEXT)
        assert s.output_sample_rate() == 24000
        assert np.isfinite(pcm).all() and len(pcm) > 24000
        outputs[label] = np.asarray(pcm,dtype=np.float32)
        np.save(OUT/(label+'.npy'),pcm)
        if not legacy:
            s.set_tts_seed(42)
            os.environ['CRISPASR_OMNIVOICE_DUMP_CODES'] = str(OUT/'session-marked.codes')
            outputs['session-marked'] = np.asarray(s.synthesize(TEXT),dtype=np.float32)
            np.save(OUT/'session-marked.npy',outputs['session-marked'])
assert (OUT/'session-raw.codes').read_bytes() == (OUT/'session-full.codes').read_bytes() == (OUT/'session-marked.codes').read_bytes(), 'Generation changed in decode/marking-only A/B'
# Load actual upstream utility AST functions through the independent module.
import runpy
oracle = runpy.run_path(str(ROOT/'tools/ci-heavy/omnivoice_audio_parity.py'))['namespace']
expected = oracle['remove_silence'](outputs['session-raw'][None,:],24000,500,100,100)
peak = np.max(np.abs(expected)) if expected.size else 0
if peak > 1e-6:
    expected *= .5/peak
expected = oracle['fade_and_pad_audio'](expected).ravel()
assert expected.shape == outputs['session-full'].shape
error = float(np.max(np.abs(expected-outputs['session-full'])))
(OUT/'waveform-oracle.json').write_text(json.dumps(dict(max_abs=error,
    compared_surface='attested raw C ABI; upstream DSP before automatic watermark',
    marked_output_differs=bool(not np.array_equal(outputs['session-marked'],outputs['session-full']))),indent=2)+'\n')
assert error <= 2e-6, error
assert not np.array_equal(outputs['session-marked'],outputs['session-full']), 'Default watermark not exercised'

# Cloning runs through both real user surfaces, cross-sentence, on every vetted
# reference and both targets. The marked arm keeps the default watermark path.
os.environ.pop('CRISPASR_OMNIVOICE_DUMP_CODES',None)
for ref_name in CLONE_REFS:
    ref = fixtures[ref_name]
    ref_text = ref['text'].rstrip('.')
    for target_name, target in (('primary', TEXT), ('second', TEXT2)):
        label = f'session-clone-{ref_name}-{target_name}'
        os.environ['CRISPASR_OMNIVOICE_DUMP_CODES'] = str(OUT/(label+'.codes'))
        with Session(paths['omnivoice-q8_0.gguf'],lib_path=str(lib),backend='omnivoice',n_threads=4) as s:
            s.set_codec_path(paths['omnivoice-tokenizer-f16.gguf'])
            s.set_tts_seed(42)
            s.set_tts_steps(32)
            s.set_voice(ref['path'],ref_text=ref_text)
            s.accept_marking_responsibility('Reference DSP acceptance fixture')
            outputs[label] = np.asarray(s.synthesize_raw(target),dtype=np.float32)
            np.save(OUT/(label+'.npy'),outputs[label])
            if ref_name == CLONE_REFS[0] and target_name == 'primary':
                s.set_tts_seed(42)
                os.environ['CRISPASR_OMNIVOICE_DUMP_CODES'] = str(OUT/'session-marked-clone.codes')
                outputs['session-marked-clone'] = np.asarray(s.synthesize(target),dtype=np.float32)
                np.save(OUT/'session-marked-clone.npy',outputs['session-marked-clone'])
                assert (OUT/(label+'.codes')).read_bytes() == (OUT/'session-marked-clone.codes').read_bytes(), 'Marking changed clone generation'
primary = fixtures[CLONE_REFS[0]]
cli_wav = OUT/'cli-clone.wav'
os.environ['CRISPASR_OMNIVOICE_DUMP_CODES'] = str(OUT/'cli-clone.codes')
run([build/'bin/crispasr','--backend','omnivoice','-m',paths['omnivoice-q8_0.gguf'],
     '--codec-model',paths['omnivoice-tokenizer-f16.gguf'],'--no-gpu','-t','4',
     '--seed','42','--tts-steps','32','--i-have-rights','--voice',primary['path'],
     '--ref-text',primary['text'].rstrip('.'),
     '--tts',TEXT,'--tts-output',cli_wav,'--no-watermark','--no-spoken-disclaimer',
     '--accept-marking-responsibility'], 'cli-clone')
os.environ.pop('CRISPASR_OMNIVOICE_DUMP_CODES', None)
with wave.open(str(cli_wav)) as wav:
    assert wav.getframerate() == 24000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
    outputs['cli-clone'] = np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768
expected_outputs = {'session-raw', 'session-full', 'session-marked',
                    'session-marked-clone', 'cli-clone'}
expected_outputs.update(f'session-clone-{ref}-{target}'
                        for ref in CLONE_REFS for target in ('primary', 'second'))
assert set(outputs) == expected_outputs, sorted(outputs)
# Keep generated codes for every surface so platform failures can distinguish
# generation drift from decoder/DSP or ASR-judge differences.
code_arrays = {}
for label in sorted(expected_outputs):
    payload = (OUT/(label+'.codes')).read_bytes()
    assert payload and len(payload) % (8 * 4) == 0, label
    code_arrays[label] = dict(file=label+'.codes', bytes=len(payload),
                              sha256=hashlib.sha256(payload).hexdigest())

# Every clone arm is scored against its own target; the no-reference arms keep
# the original control text.
gate_targets = {'session-raw': TEXT, 'session-full': TEXT, 'session-marked': TEXT,
                'session-marked-clone': TEXT, 'cli-clone': TEXT}
gate_targets.update({name: TEXT2 if name.endswith('-second') else TEXT
                     for name in outputs if name.startswith('session-clone-')})
results = {}
with Session(asr_paths['nemotron'],lib_path=str(lib),backend='nemotron',n_threads=4) as s:
    for label,pcm in outputs.items():
        assert np.isfinite(pcm).all() and len(pcm) > 24000, label
        if label not in ('session-raw','session-marked','session-marked-clone'):
            assert np.max(np.abs(pcm[:2400])) <= 1/32768 and np.max(np.abs(pcm[-2400:])) <= 1/32768, label
        actual = ' '.join(segment.text for segment in s.transcribe(pcm,sample_rate=24000,language='en'))
        results[label] = dict(transcript=actual, expected=gate_targets[label],
                              wer=word_wer(gate_targets[label], actual), samples=len(pcm),
                              sha256=hashlib.sha256(pcm.tobytes()).hexdigest())
        (OUT/'roundtrips.json').write_text(json.dumps(results,indent=2)+'\n')
passed = set(results) == expected_outputs and all(case['wer'] <= WER_GATE for case in results.values())
receipt = dict(passed=passed,model_revision=MODEL_PIN,fixture_revision=FIXTURE_PIN,
    asr_models={engine:dict(repo=repo,name=name,revision=rev)
                for engine,(repo,name,rev) in ASR_MODELS.items()},
    wer_gate=WER_GATE,clone_references=list(CLONE_REFS),reference_integrity=integrity,
    roundtrips=results,code_arrays=code_arrays,decode_waveform_max_abs=error,decode_codes_exact=True,
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
(OUT/'omnivoice-518-acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
(OUT/'summary.md').write_text('OmniVoice reference-integrity, utility/decode oracle, CLI/session '
    'cross-sentence cloning and ASR roundtrips ' + ('PASS' if passed else 'FAIL') + '.\n')
# Persist a failed aggregate receipt before making the workflow red.
assert passed, results
