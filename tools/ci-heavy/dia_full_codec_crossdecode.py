#!/usr/bin/env python3
"""Decode complete retained native codes with the actual original Dia DAC.

No neural generation or runtime changes. An original-code/PCM/ASR positive
control validates the reference path. Native silence or a generation slicing
difference remains a diagnostic result, never whole-port acceptance.
"""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT_URL = ('https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/'
             'e22cb7451fa0b073e3b497273266338913243831/'
             'dia-20261015-full-codec/retained-inputs-v1.zip')
INPUT_SHA = 'b711e33d4bb60f6c421397a3ca2eaf8600facd196a23c03e2bce20bafbee6126'
CHECKPOINT_URL = 'https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth'
CHECKPOINT_SHA = 'a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa'
DIA_PIN = '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'
TEXT = 'We are committed to making it secure.'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def metric(actual, expected):
    assert actual.shape == expected.shape and actual.size, 'matching nonempty shape'
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), 'finite arrays'
    a, b = actual.astype('f8').ravel(), expected.astype('f8').ravel()
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    assert an > 0 and bn > 0, 'nonzero norms'
    return dict(cosine=float(a @ b / (an*bn)), mine_norm=an, ref_norm=bn,
                relative_l2=float(np.linalg.norm(a-b)/bn), norm_error=abs(an/bn-1),
                max_abs=float(np.max(np.abs(a-b))), byte_exact=actual.tobytes() == expected.tobytes())


def positive_gate(row):
    return row['cosine'] >= .999999 and row['relative_l2'] <= 1e-4 and row['norm_error'] <= 1e-4


def self_test():
    x = np.linspace(-1, 1, 512, dtype=np.float32)
    assert positive_gate(metric(x.copy(), x))
    assert not positive_gate(metric(2*x, x))
    assert not positive_gate(metric(x[::-1].copy(), x))
    for bad in [x[:-1], x*np.nan, np.zeros_like(x)]:
        try:
            metric(bad, x)
        except AssertionError:
            continue
        raise AssertionError('invalid control accepted')
    return dict(passed=True, rejected=['scale', 'order', 'shape', 'nonfinite', 'zero-norm'])


def main():
    import requests
    import torch
    import dac
    import dac.model.dac as dac_module
    import dac.nn.layers as dac_layers
    import dac.nn.quantize as dac_quantize
    import soundfile as sf
    import whisper

    assert torch.__version__ == '2.6.0+cpu', torch.__version__
    torch.set_num_threads(4)
    out = Path(os.environ['HEAVY_OUT'])/'dia-full-codec-v1'
    scratch = Path(os.environ['HEAVY_SCRATCH'])/'dia-full-codec-v1'
    out.mkdir()
    scratch.mkdir()
    os.environ['TMPDIR'] = str(scratch)
    report = dict(scope=__doc__, source=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        torch_version=torch.__version__, controls=self_test(), diagnosis_complete=False,
        model_speech_acceptance=False, production_changes=False, runs=[])

    def checkpoint():
        (out/'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    def run(cmd, label, expected_success=True):
        with (out/(label+'.log')).open('wb') as log:
            result = subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=1800)
        report['runs'].append(dict(label=label, returncode=result.returncode))
        checkpoint()
        assert (result.returncode == 0) == expected_success, label

    def download(url, path, expected):
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with path.open('xb') as stream:
                for chunk in response.iter_content(4*1024*1024):
                    stream.write(chunk)
        assert digest(path) == expected, path.name

    checkpoint()
    # Whisper's reference load_audio calls the real ffmpeg executable. Hosted
    # images do not guarantee it is installed; check before expensive decoding.
    if not shutil.which('ffmpeg'):
        assert sys.platform == 'linux' and os.environ.get('GITHUB_ACTIONS') == 'true', (
            'ffmpeg is required; install it before running this local reference')
        run(['sudo', 'bash', ROOT/'tools/ci-apt.sh', 'update'], 'ffmpeg-apt-update')
        run(['sudo', 'bash', ROOT/'tools/ci-apt.sh', 'install', '-y', 'ffmpeg'], 'ffmpeg-apt-install')
    assert shutil.which('ffmpeg'), 'ffmpeg unavailable after prerequisite setup'
    report['ffmpeg_executable'] = shutil.which('ffmpeg')
    run(['ffmpeg', '-version'], 'ffmpeg-version')
    archive = out/'retained-inputs-v1.zip'
    download(INPUT_URL, archive, INPUT_SHA)
    inputs = out/'inputs'
    inputs.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
        assert len(z.infolist()) < 100 and sum(i.file_size for i in z.infolist()) < 32*1024*1024
        for info in z.infolist():
            name = PurePosixPath(info.filename)
            assert not name.is_absolute() and '..' not in name.parts and '\\' not in info.filename
            assert not stat.S_ISLNK(info.external_attr >> 16)
        manifest = json.loads(z.read('manifest.json'))
        assert set(z.namelist()) == {'manifest.json'} | {row['path'] for row in manifest['files']}
        for row in manifest['files']:
            data = z.read(row['path'])
            assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
        z.extractall(inputs)
    report['input_archive_sha256'] = digest(archive)
    # Recompile literal producer stopping/BOS/reversal code on the hosted CPU.
    replay = out/'hosted-literal-native-replay'
    run(['c++', '-std=c++17', '-O2', inputs/'literal-native-replay.cpp', '-o', replay], 'compile-replay')
    for surface in ['cabi', 'cli']:
        dest = out/(surface+'-hosted-replay')
        run([replay, inputs/(surface+'-sampled.i32'), inputs/(surface+'-inputs.i32'), dest], surface+'-replay')
        for name in ['emitted.i32', 'aligned.i32', 'final-emitted-row.i32']:
            assert (dest/name).read_bytes() == (inputs/(surface+'-replay')/name).read_bytes()
    run([replay, inputs/'corrupt-sampled.i32', inputs/'cabi-inputs.i32', out/'corrupt-replay'],
        'corrupt-replay', expected_success=False)
    report['hosted_replay_sha256'] = digest(replay)
    blueprint = scratch/'dia-source'
    run(['git', 'clone', '--filter=blob:none', '--no-checkout',
         'https://github.com/nari-labs/dia.git', blueprint], 'clone-original')
    run(['git', '-C', blueprint, 'checkout', '--detach', DIA_PIN], 'pin-original')
    assert subprocess.check_output(['git', '-C', blueprint, 'rev-parse', 'HEAD'], text=True).strip() == DIA_PIN
    provenance = json.loads((inputs/'original-provenance.json').read_text())
    expected_sources = json.loads(provenance['dia_source_hashes'])
    for name, expected in expected_sources.items():
        assert digest(blueprint/name) == expected
        target = out/'actual-original-source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((blueprint/name).read_bytes())
    for module, prior in [(dac_module, 'dac'), (dac_layers, 'layers'), (dac_quantize, 'quantize')]:
        file = Path(inspect.getsourcefile(module))
        assert file.read_bytes() == (inputs/('original-dac-'+prior+'.py')).read_bytes()
        (out/('actual-dac-'+prior+'.py')).write_bytes(file.read_bytes())
    sys.path.insert(0, str(blueprint))
    from dia.model import Dia
    from dia.state import DecoderOutput
    from dia.audio import decode, build_revert_indices, revert_audio_delay
    from types import SimpleNamespace

    # Replay the ORIGINAL generation protocol on the captured predictions.
    # Only neural step computation is replaced; no new neural-model inference.
    raw = np.fromfile(inputs/'cabi-sampled.i32', dtype='<i4').reshape(-1, 9)
    actual_inputs = np.fromfile(inputs/'cabi-inputs.i32', dtype='<i4').reshape(-1, 9)
    dia = Dia.__new__(Dia)
    dia.device = torch.device('cpu')
    dia.config = SimpleNamespace(data=SimpleNamespace(channels=9, audio_bos_value=1026,
        audio_pad_value=1025, audio_eos_value=1024, delay_pattern=[0, 8, 9, 10, 11, 12, 13, 14, 15],
        audio_length=3072))
    dia.model = SimpleNamespace(eval=lambda: None)
    prefill, start = dia._prepare_audio_prompt(None)
    state = SimpleNamespace(step=-1)
    state.prepare_step = lambda step: setattr(state, 'step', step)
    output = DecoderOutput.new(dia.config, dia.device)
    output.prefill(prefill, start)
    dia._prepare_generation = lambda *args: (state, output)
    seen = []

    def prediction(tokens, state, *args):
        assert state.step == len(seen) and state.step < len(raw)
        assert np.array_equal(tokens[0, 0].numpy(), actual_inputs[state.step])
        assert torch.equal(tokens[0], tokens[1])
        seen.append(tokens[0, 0].clone())
        return torch.from_numpy(raw[state.step].copy())

    dia._decoder_step = prediction
    dia._generate_output = lambda codes: codes.clone()
    original_emitted = dia.generate(TEXT, max_tokens=3072, cfg_scale=3., temperature=1.2,
                                   top_p=.95, cfg_filter_top_k=45, use_torch_compile=False)
    assert len(seen) == len(raw)
    native_emitted = np.fromfile(inputs/'cabi-replay/emitted.i32', dtype='<i4').reshape(-1, 9)
    assert np.array_equal(original_emitted.numpy(), native_emitted[:len(original_emitted)])
    np.save(out/'original-protocol-emitted.npy', original_emitted.numpy())
    np.save(out/'original-protocol-feedback.npy', torch.stack(seen).numpy())
    report['protocol'] = dict(all_feedback_rows_exact=len(seen), native_rows=len(native_emitted),
                             original_returned_rows=len(original_emitted),
                             native_extra_trailing_rows=len(native_emitted)-len(original_emitted))

    def align(rows):
        values = torch.from_numpy(rows.copy()).unsqueeze(0)
        indices = build_revert_indices(1, len(rows), 9, dia.config.data.delay_pattern)
        codes = revert_audio_delay(values, 1025, indices, len(rows))[:, :-15, :]
        codes[(codes < 0) | (codes > 1023)] = 0
        return codes[0].numpy()

    aligned = align(native_emitted)
    literal = np.fromfile(inputs/'cabi-replay/aligned.i32', dtype='<i4').reshape(-1, 9)
    assert aligned.tobytes() == literal.tobytes()
    for offset in [-1, 1]:
        assert not np.array_equal(np.roll(aligned, offset, axis=0), aligned)
    checkpoint()
    weights = scratch/'original-dac-0.0.1.pth'
    download(CHECKPOINT_URL, weights, CHECKPOINT_SHA)
    codec = dac.DAC.load(str(weights)).cpu().eval()
    assert all(parameter.dtype == torch.float32 for parameter in codec.parameters())
    report['checkpoint_sha256'] = digest(weights)
    report['pcm'] = {}
    decoded = {}
    for arm, codes in [('original-codes', np.load(inputs/'original-aligned-codes.npy')),
                       ('native-complete-codes', aligned),
                       ('native-original-return-slice', align(original_emitted.numpy()))]:
        with torch.inference_mode():
            pcm = decode(codec, torch.from_numpy(codes.copy()).unsqueeze(0).transpose(1, 2)).squeeze().numpy()
        assert pcm.dtype == np.float32 and pcm.ndim == 1 and len(pcm) == len(codes)*512
        assert np.isfinite(pcm).all()
        np.save(out/(arm+'.npy'), pcm)
        sf.write(out/(arm+'.wav'), pcm, 44100, subtype='FLOAT')
        decoded[arm] = pcm
        report['pcm'][arm] = dict(samples=len(pcm), rms=float(np.sqrt(np.mean(pcm.astype('f8')**2))),
                                  peak=float(np.max(np.abs(pcm))), sha256=digest(out/(arm+'.npy')))
        checkpoint()
    positive = metric(decoded['original-codes'], np.load(inputs/'original-pcm.npy'))
    report['original_positive_pcm'] = positive
    checkpoint()
    assert positive_gate(positive), positive
    native_pcm = np.load(inputs/'cabi-logging.npy')
    assert len(native_pcm) == len(decoded['native-complete-codes'])
    report['same_native_codes_original_vs_native_pcm'] = metric(decoded['native-complete-codes'], native_pcm)
    report['native_vs_positive_original_dac_rms_ratio'] = (
        report['pcm']['native-complete-codes']['rms']/report['pcm']['original-codes']['rms'])
    report['scale_negative_rejected'] = not positive_gate(metric(2*decoded['original-codes'], decoded['original-codes']))
    assert report['scale_negative_rejected']
    checkpoint()
    # Same fixed independent ASR recipe for every waveform, with a positive control.
    asr = whisper.load_model('tiny', device='cpu', download_root=str(scratch/'whisper'))
    url = whisper._MODELS['tiny']
    report['asr_model'] = dict(url=url, sha256=digest(scratch/'whisper/tiny.pt'),
                              recipe=dict(language='en', task='transcribe', temperature=0.,
                                          beam_size=5, fp16=False, condition_on_previous_text=False))
    assert report['asr_model']['sha256'] == url.split('/')[-2]
    report['asr'] = {}
    for arm in decoded:
        audio = whisper.load_audio(str(out/(arm+'.wav')))
        np.save(out/(arm+'-asr-input.npy'), audio)
        result = asr.transcribe(audio, language='en', task='transcribe', temperature=0., beam_size=5,
                                fp16=False, condition_on_previous_text=False)
        text = result['text'].strip()
        expected, actual = re.findall('[a-z]+', TEXT.lower()), re.findall('[a-z]+', text.lower())
        distance = list(range(len(actual)+1))
        for i, token in enumerate(expected, 1):
            next_row = [i]
            for j, other in enumerate(actual, 1):
                next_row.append(min(distance[j]+1, next_row[j-1]+1, distance[j-1]+(token != other)))
            distance = next_row
        report['asr'][arm] = dict(text=text, wer=distance[-1]/len(expected), segments=result['segments'])
        checkpoint()
    assert report['asr']['original-codes']['wer'] == 0, report['asr']['original-codes']
    report['diagnosis_complete'] = True
    report['original_dac_rescues_native_speech'] = report['asr']['native-complete-codes']['wer'] == 0
    report['files'] = [dict(path=f.relative_to(out).as_posix(), bytes=f.stat().st_size, sha256=digest(f))
                       for f in sorted(out.rglob('*')) if f.is_file() and f.name != 'receipt.json']
    checkpoint()
    print(json.dumps({k: v for k, v in report.items() if k not in ['files', 'asr']}, indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        print(json.dumps(self_test()))
    else:
        main()
