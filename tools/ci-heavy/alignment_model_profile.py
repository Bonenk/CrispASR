#!/usr/bin/env python3
"""Hosted actual-model CPU alignment parity/profiling, no production/default change.

Retain PCM, native encoder logits, vocab, source/ELFs and OFF/ON/OFF controls.
FLEURS has no word boundaries: this is NOT a boundary-accuracy benchmark.
"""
import argparse
import ctypes
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHOD = Path(__file__).resolve().with_name('alignment-model-profile')
GATE = 'CRISPASR_SUBWORD_CTC_ALIGN_OPT'
FP = ctypes.POINTER(ctypes.c_float)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def verify_file(path, sha256, size=None):
    if size is not None and Path(path).stat().st_size != size:
        raise ValueError('Pinned file size mismatch: ' + Path(path).name)
    if digest(path) != sha256:
        raise ValueError('Pinned file SHA256 mismatch: ' + Path(path).name)


def same_array(expected, actual):
    # A close cosine cannot replace exact owned-input equality for this change.
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError('Nonfinite native tensor')
    if expected.dtype != actual.dtype or expected.shape != actual.shape or expected.tobytes() != actual.tobytes():
        raise ValueError('Actual native tensor identity failed')


def centiseconds(value):
    value = float(value) * 100
    if not math.isfinite(value) or abs(value - round(value)) > 1e-7:
        raise ValueError('Non-centisecond or nonfinite output')
    return round(value)


def validate_words(words, texts, duration_cs, offset=0):
    if not texts or len(words) != len(texts):
        raise ValueError('Dropped/extra alignment units')
    previous_end = offset
    for word, text in zip(words, texts):
        if word['word'] != text:
            raise ValueError('Alignment text/order changed')
        start, end = word['t0'], word['t1']
        if type(start) is not int or type(end) is not int:
            raise ValueError('Native centisecond integers required')
        if not offset <= start < end <= offset + duration_cs or start < previous_end:
            raise ValueError('Empty, overlapping, unordered or out-of-range native word')
        previous_end = end


def select_recording(rows, pin):
    # FLEURS sentence IDs repeat across recordings. Filename is mandatory.
    matches = [row for row in rows if Path(row['audio']['path']).name == pin['filename']]
    if len(matches) != 1:
        raise ValueError('Pinned recording is absent or duplicated')
    row = matches[0]
    for key in ('id', 'num_samples', 'raw_transcription', 'transcription'):
        if row[key] != pin[key]:
            raise ValueError('Pinned recording metadata mismatch: ' + key)
    return row


def timing(values):
    return {'samples_ms': values, 'median_ms': statistics.median(values), 'min_ms': min(values), 'max_ms': max(values)}


class Params(ctypes.Structure):
    _fields_ = [('n_threads', ctypes.c_int), ('verbosity', ctypes.c_int)]


class Word(ctypes.Structure):
    _fields_ = [('text', ctypes.c_char * 64), ('t0', ctypes.c_int64), ('t1', ctypes.c_int64)]


class Native:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(library))
        signatures = {
            'canary_ctc_init_from_file': ([ctypes.c_char_p, Params], ctypes.c_void_p),
            'canary_ctc_free': ([ctypes.c_void_p], None),
            'canary_ctc_compute_logits': ([ctypes.c_void_p, FP, ctypes.c_int, ctypes.POINTER(FP),
                                          ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            'canary_ctc_align_words': ([ctypes.c_void_p, FP, ctypes.c_int, ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.POINTER(Word)], ctypes.c_int),
            'canary_ctc_token_text': ([ctypes.c_void_p, ctypes.c_int], ctypes.c_char_p),
            'canary_ctc_greedy_decode': ([ctypes.c_void_p, FP, ctypes.c_int, ctypes.c_int], ctypes.c_void_p),
            'crispasr_audio_load': ([ctypes.c_char_p, ctypes.POINTER(FP), ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            'crispasr_audio_free': ([FP], None),
        }
        for name in ('n_vocab', 'blank_id', 'frame_dur_cs', 'sample_rate'):
            signatures['canary_ctc_' + name] = ([ctypes.c_void_p], ctypes.c_int)
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result
        self.free = ctypes.CDLL(None).free
        self.free.argtypes, self.free.restype = [ctypes.c_void_p], None

    def audio(self, path):
        data, count, rate = FP(), ctypes.c_int(), ctypes.c_int()
        rc = self.lib.crispasr_audio_load(str(path).encode(), ctypes.byref(data), ctypes.byref(count), ctypes.byref(rate))
        try:
            if rc or not data or rate.value != 16000 or count.value <= 0:
                raise ValueError('Actual C ABI audio decoder failed')
            return np.ctypeslib.as_array(data, shape=(count.value,)).copy()
        finally:
            if data:
                self.lib.crispasr_audio_free(data)

    def logits(self, ctx, pcm):
        data, frames, vocab = FP(), ctypes.c_int(), ctypes.c_int()
        start = time.perf_counter_ns()
        rc = self.lib.canary_ctc_compute_logits(ctx, pcm.ctypes.data_as(FP), len(pcm),
                                                ctypes.byref(data), ctypes.byref(frames), ctypes.byref(vocab))
        elapsed = (time.perf_counter_ns() - start) / 1e6
        try:
            if rc or not data or frames.value <= 0 or vocab.value <= 0:
                raise ValueError(f'Actual encoder failed: rc={rc}')
            scores = np.ctypeslib.as_array(data, shape=(frames.value * vocab.value,)).copy()
            scores = scores.reshape(frames.value, vocab.value)
            if not np.isfinite(scores).all():
                raise ValueError('Nonfinite native encoder logits')
            return scores, elapsed
        finally:
            if data:
                self.free(data)

    def align(self, ctx, scores, labels):
        pointers = (ctypes.c_char_p * len(labels))(*(label.encode() for label in labels))
        output = (Word * len(labels))()
        start = time.perf_counter_ns()
        rc = self.lib.canary_ctc_align_words(ctx, scores.ctypes.data_as(FP), *scores.shape, pointers, len(labels), output)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        stamps = [{'word': w.text.decode(), 't0': w.t0, 't1': w.t1} for w in output]
        return rc, stamps, elapsed


def run_hosted(args):
    if os.environ.get('GITHUB_ACTIONS') != 'true' or platform.system() != 'Linux':
        raise RuntimeError('Model downloads/build/inference require a hosted Linux Actions worker')
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    import soundfile as sf

    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'alignment-model'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    for key in list(os.environ):
        if key.startswith('CRISPASR_'):
            del os.environ[key]
    os.environ.update(HF_HOME=str(scratch / 'hf'), TMPDIR=str(scratch), OMP_NUM_THREADS=str(args.threads))
    pins = json.loads((METHOD / 'pins.json').read_text())
    receipt = {'schema': 1, 'status': 'running', 'scope': 'actual Japanese F16 CPU alignment parity and stage timing',
               'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'pins': pins, 'threads': args.threads, 'backend': 'CPU only; CUDA/Metal/Vulkan/BLAS disabled',
               'limitations': ['no original-Python or new encoder mathematical parity claim',
                               'no manually labeled word boundaries; no boundary accuracy acceptance',
                               'no GPU/VRAM, long-file or Subtimer comparison; no default promotion'], 'phases': {}}

    def save():
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')

    def command(argv, tag, timeout=3600, env=None):
        with (out / f'{tag}.log').open('w') as log:
            result = subprocess.run(list(map(str, argv)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout, env=env)
        if result.returncode:
            raise RuntimeError(f'{tag} failed: exit {result.returncode}; retained log')
        print(tag + ' PASS', flush=True)

    save()
    try:
        # Never download large files on the maintainer machine.
        files = {}
        for key, repo_type in (('model', 'model'), ('dataset', 'dataset')):
            pin = pins[key]
            files[key] = Path(hf_hub_download(pin['repo'], pin['file'], revision=pin['revision'], repo_type=repo_type,
                                             cache_dir=scratch / 'hf'))
            verify_file(files[key], pin['sha256'], pin['bytes'])
        tsv = Path(hf_hub_download(pins['dataset']['repo'], pins['dataset']['tsv'],
                                  revision=pins['dataset']['revision'], repo_type='dataset', cache_dir=scratch / 'hf'))
        verify_file(tsv, pins['dataset']['tsv_sha256'])
        shutil.copyfile(tsv, out / 'fleurs-ja-test.tsv')
        card = hf_hub_download(pins['dataset']['repo'], 'README.md', revision=pins['dataset']['revision'],
                               repo_type='dataset', cache_dir=scratch / 'hf')
        shutil.copyfile(card, out / 'fleurs-dataset-card.md')
        (out / 'ATTRIBUTION.md').write_text(pins['dataset']['attribution'] + '\n\n' + pins['dataset']['source_url']
                                          + '\n\nLicense: ' + pins['dataset']['license_url']
                                          + '\n\nRetained recording decoded to float32 WAV/PCM; no cuts or resampling.\n')
        matches = []
        for batch in pq.ParquetFile(files['dataset']).iter_batches(batch_size=8,
                columns=['id', 'audio', 'num_samples', 'raw_transcription', 'transcription']):
            matches.extend(row for row in batch.to_pylist()
                           if Path(row['audio']['path']).name == pins['recording']['filename'])
        row = select_recording(matches, pins['recording'])
        encoded = row['audio']['bytes']
        (out / 'source-audio.wav').write_bytes(encoded)
        pcm, rate = sf.read(io.BytesIO(encoded), dtype='float32')
        if rate != 16000 or pcm.ndim != 1 or len(pcm) != pins['recording']['num_samples'] or not np.isfinite(pcm).all():
            raise ValueError('Pinned PCM shape/rate/finite contract failed')
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        sf.write(out / 'fixture.wav', pcm, rate, subtype='FLOAT')
        np.save(out / 'pcm.npy', pcm)
        transcript = row['transcription']
        (out / 'transcript.txt').write_text(transcript + '\n')
        receipt['phases']['fixture'] = {'samples': len(pcm), 'seconds': len(pcm) / rate,
                                        'encoded_sha256': hashlib.sha256(encoded).hexdigest()}
        save()

        build = scratch / 'build'
        command(['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
                 '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_METAL=OFF',
                 '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=OFF'], 'configure')
        command(['cmake', '--build', build, '--target', 'crispasr-lib', 'crispasr-cli', '-j2'], 'build', timeout=5400)
        libraries = list(build.rglob('libcrispasr.so'))
        if len(libraries) != 1:
            raise ValueError('Expected exactly one actual CrispASR shared library')
        library, cli = libraries[0], build / 'bin/crispasr'
        native_dir = out / 'native'
        native_dir.mkdir()
        for binary in [cli, *build.rglob('*.so*')]:
            if binary.is_file():
                target = native_dir / binary.name
                if target.exists() and digest(target) != digest(binary):
                    raise ValueError('Ambiguous native library basename')
                shutil.copyfile(binary, target)
        for name in ('CMakeCache.txt', 'compile_commands.json'):
            shutil.copyfile(build / name, out / name)
        command(['ldd', library], 'library-dependencies')
        command(['c++', '--version'], 'compiler')
        command(['git', 'submodule', 'status', '--recursive'], 'submodules')
        source_dir = out / 'source'
        source_dir.mkdir()
        for name in ('src/canary_ctc.cpp', 'src/canary_ctc.h', 'src/crispasr_aligner.cpp',
                     'src/core/ctc_windows.h', 'src/crispasr_aligner.h',
                     'src/crispasr_c_api.cpp', 'examples/cli/crispasr_run.cpp', 'examples/common-crispasr.cpp',
                     'src/crispasr_audio.cpp', 'include/crispasr_session.h',
                     'python/crispasr/_binding.py', 'tools/ci-heavy/alignment_model_profile.py',
                     'tools/ci-heavy/alignment-model-profile/units.cpp', 'tools/ci-heavy/alignment-model-profile/pins.json',
                     'tests/subword_align_harness.py', 'tests/subword_align_probe.cpp', 'tests/ctc_alloc_tracker.h',
                     'tests/fixtures/canary-align-baseline-deaf9c010.inc'):
            target = source_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
            raise ValueError('Tracked source changed during build')
        unit_program = scratch / 'units'
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        command(compiler + ['-std=c++17', '-O2', '-I', ROOT / 'src', '-I', ROOT / 'examples', METHOD / 'units.cpp',
                           '-L', library.parent, '-lcrispasr', '-Wl,-rpath,' + str(library.parent), '-o', unit_program], 'build-units')
        units = json.loads(subprocess.check_output([str(unit_program), transcript], text=True))
        (out / 'units.json').write_text(json.dumps(units, ensure_ascii=False, indent=2) + '\n')
        shutil.copyfile(unit_program, native_dir / 'units')
        labels, display = units['labels'], units['display']
        if not labels or len(labels) != len(display):
            raise ValueError('Actual native label/display cardinality mismatch')
        receipt['phases']['build'] = {'library_sha256': digest(library), 'cli_sha256': digest(cli), 'units': len(labels)}
        save()

        native = Native(library)
        same_array(pcm, native.audio(out / 'fixture.wav'))
        start = time.perf_counter_ns()
        ctx = native.lib.canary_ctc_init_from_file(str(files['model']).encode(), Params(args.threads, 1))
        receipt['phases']['load'] = {'ms': (time.perf_counter_ns() - start) / 1e6,
                                        'scope': 'first context load; OS file cache warmed by hashing, not cold disk'}
        if not ctx:
            raise ValueError('Japanese F16 model failed to load')
        try:
            nv = native.lib.canary_ctc_n_vocab(ctx)
            vocab = [native.lib.canary_ctc_token_text(ctx, i).decode() for i in range(nv)]
            blank = native.lib.canary_ctc_blank_id(ctx)
            frame_cs = native.lib.canary_ctc_frame_dur_cs(ctx)
            if native.lib.canary_ctc_sample_rate(ctx) != 16000 or not all(vocab):
                raise ValueError('Model sample rate or vocabulary getter contract failed')
            (out / 'vocab.json').write_text(json.dumps(vocab, ensure_ascii=False) + '\n')
            receipt['model_hparams'] = {'vocabulary_pieces': nv, 'blank': blank, 'frame_cs': frame_cs}
            os.environ[GATE] = '0'
            warm, warm_ms = native.logits(ctx, pcm)
            encoder_times = []
            for i, gate in enumerate((0, 1, 0)):
                os.environ[GATE] = str(gate)
                scores, elapsed = native.logits(ctx, pcm)
                np.save(out / f'logits-{i}-gate{gate}.npy', scores)
                same_array(warm, scores)
                encoder_times.append(elapsed)
            receipt['phases']['encoder'] = {'warmup_ms': warm_ms, **timing(encoder_times),
                                             'logits_shape': list(scores.shape), 'off_on_off': 'byte-exact'}
            if scores.shape[1] != nv + 1 or not 0 <= blank < scores.shape[1] or frame_cs <= 0:
                raise ValueError('Model vocabulary/logit/frame-duration contract failed')
            duration_cs = math.ceil(len(pcm) / 160) + frame_cs
            expected = None
            samples = {0: [], 1: []}
            raw = []
            for index, gate in enumerate([0, 1] + [0, 1, 1, 0] * args.repeats):
                os.environ[GATE] = str(gate)
                rc, stamps, elapsed = native.align(ctx, scores, labels)
                raw.append({'gate': gate, 'rc': rc, 'ms': elapsed, 'words': stamps, 'warmup': index < 2})
                (out / 'native-align-calls.json').write_text(json.dumps(raw, ensure_ascii=False, indent=2) + '\n')
                if rc:
                    raise ValueError(f'Native alignment failed on actual Japanese logits: rc={rc}')
                validate_words(stamps, labels, duration_cs)
                if expected is None:
                    expected = stamps
                if stamps != expected:
                    raise ValueError('Actual native gate changed word text/times')
                if index >= 2:
                    samples[gate].append(elapsed)
            receipt['phases']['postprocessor'] = {'gate_off': timing(samples[0]), 'gate_on': timing(samples[1]),
                                                   'schedule': 'one warm-up per gate, then repeated OFF/ON/ON/OFF',
                                                   'scope': 'native tokenizer/DP call incl ctypes entry; excludes encoder and output allocation'}
            receipt['phases']['postprocessor']['median_ratio'] = statistics.median(samples[0]) / statistics.median(samples[1])
            ptr = native.lib.canary_ctc_greedy_decode(ctx, scores.ctypes.data_as(FP), *scores.shape)
            if not ptr:
                raise ValueError('Actual model greedy decode failed')
            try:
                greedy = ctypes.string_at(ptr).decode()
            finally:
                native.free(ptr)
            (out / 'greedy.txt').write_text(greedy + '\n')
            if not greedy.strip():
                raise ValueError('Model produced empty speech control')
            save()
            # Literal own DP replay and immutable own reference on the same physical logits/vocab.
            sys.path.insert(0, str(ROOT / 'tests'))
            from subword_align_harness import Probe
            probe = Probe(build_dir=scratch)
            try:
                replay = []
                for ref, gate in ((True, 0), (False, 0), (False, 1)):
                    result = probe.run(scores, labels, vocab, blank=blank, duration=frame_cs, reference=ref, gate=gate)
                    if result['rc'] != 0 or [(w['word'].encode(), w['t0'], w['t1']) for w in expected] != result['stamps']:
                        raise ValueError('Frozen/literal DP versus actual library parity failed')
                    replay.append({'reference': ref, 'gate': gate, 'peak_heap_bytes': result['peak_heap_bytes'],
                                   'allocations': result['allocations'], 'kernel_ns': result['kernel_ns']})
                if replay[2]['peak_heap_bytes'] >= replay[1]['peak_heap_bytes'] or replay[2]['allocations'] >= replay[1]['allocations']:
                    raise ValueError('Actual-logit workspace experiment did not reduce tracked heap/allocations')
                receipt['phases']['owned_input_replay'] = replay
            finally:
                probe.close()
        finally:
            native.lib.canary_ctc_free(ctx)
        save()

        # The shipped Python wrapper invokes the actual exported C ABI.
        sys.path.insert(0, str(ROOT / 'python'))
        from crispasr import align_words, free_aligner_cache
        free_aligner_cache(lib_path=str(library))
        expected_display = [dict(word, word=text) for word, text in zip(expected, display)]
        public_calls = []
        try:
            for index, (gate, offset) in enumerate(((0, 0), (0, 0), (1, 0), (0, 0), (1, 1234))):
                os.environ[GATE] = str(gate)
                start = time.perf_counter_ns()
                aligned = align_words(str(files['model']), transcript, pcm, t_offset=offset / 100,
                                      n_threads=args.threads, lib_path=str(library))
                elapsed = (time.perf_counter_ns() - start) / 1e6
                stamps = [{'word': w.text, 't0': centiseconds(w.start), 't1': centiseconds(w.end)} for w in aligned]
                public_calls.append({'gate': gate, 'offset_cs': offset, 'ms': elapsed,
                                     'cold_model_cache': index == 0, 'words': stamps})
                (out / 'python-cabi-calls.json').write_text(json.dumps(public_calls, ensure_ascii=False, indent=2) + '\n')
                validate_words(stamps, display, duration_cs, offset)
                if stamps != [dict(w, t0=w['t0'] + offset, t1=w['t1'] + offset) for w in expected_display]:
                    raise ValueError('Python/exported C ABI changed native times/text/absolute offset')
                if any(w.characters for w in aligned):
                    raise ValueError('Subword path unexpectedly fabricated character timings')
        finally:
            free_aligner_cache(lib_path=str(library))
        receipt['phases']['python_cabi'] = {'calls': len(public_calls), 'off_on_off_and_offset': 'exact'}
        save()
        for index, gate in enumerate((0, 1, 0)):
            env = dict(os.environ, **{GATE: str(gate)})
            target = out / f'cli-{index}-gate{gate}.json'
            command([cli, '--align-only', '-am', files['model'], '-f', out / 'fixture.wav',
                     '--text-file', out / 'transcript.txt', '--align-format', 'json', '--align-granularity', 'word',
                     '--align-output', target, '--no-gpu', '-t', args.threads, '-l', 'ja'], f'cli-{index}-gate{gate}', env=env)
            actual = json.loads(target.read_text())
            stamps = [{'word': w['word'], 't0': centiseconds(w['start']), 't1': centiseconds(w['end'])} for w in actual]
            if stamps != expected_display:
                raise ValueError('Actual CLI versus C ABI/native word parity failed')
        receipt['phases']['cli'] = {'off_on_off': 'exact native/C ABI word text and centisecond bounds'}
        if args.long_windows:
            # Repeated licensed speech tests real window seams and global label
            # ownership. It supplies no manually measured boundary ground truth.
            long_pcm = np.tile(pcm, 7)
            long_text = ' '.join([transcript] * 7)
            long_units = json.loads(subprocess.check_output([str(unit_program), long_text], text=True))
            long_display = long_units['display']
            if len(long_display) != len(display) * 7:
                raise ValueError('Repeated transcript changed native alignment unit count')
            (out / 'long-units.json').write_text(json.dumps(long_units, ensure_ascii=False, indent=2))
            sf.write(out / 'long-fixture.wav', long_pcm, rate, subtype='FLOAT')
            (out / 'long-transcript.txt').write_text(long_text)
            long_calls = []
            for chunk in (0, 30, 10, 30):
                index = len(long_calls)
                target = out / f'long-cli-{index}-chunk{chunk}.json'
                started = time.perf_counter_ns()
                command([cli, '--align-only', '-am', files['model'], '-f', out / 'long-fixture.wav',
                         '--text-file', out / 'long-transcript.txt', '--align-format', 'json',
                         '--align-granularity', 'word', '--align-output', target, '--chunk-seconds', chunk,
                         '--no-gpu', '-t', args.threads, '-l', 'ja'], f'long-cli-{index}-chunk{chunk}')
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                actual = json.loads(target.read_text())
                stamps = [{'word': w['word'], 't0': centiseconds(w['start']), 't1': centiseconds(w['end'])}
                          for w in actual]
                validate_words(stamps, long_display, math.ceil(len(long_pcm) * 100 / rate) + frame_cs)
                long_calls.append({'chunk_seconds': chunk, 'elapsed_ms': elapsed_ms, 'words': stamps})
                (out / 'long-calls.json').write_text(json.dumps(long_calls, ensure_ascii=False, indent=2))
            if long_calls[1]['words'] != long_calls[3]['words']:
                raise ValueError('Repeated 30-second window inference changed alignment')
            free_aligner_cache(lib_path=str(library))
            try:
                result = align_words(str(files['model']), long_text, long_pcm, t_offset=12.34,
                                     n_threads=args.threads, lib_path=str(library))
                public = [{'word': w.text, 't0': centiseconds(w.start), 't1': centiseconds(w.end)} for w in result]
                expected_long = [dict(w, t0=w['t0'] + 1234, t1=w['t1'] + 1234) for w in long_calls[1]['words']]
                if public != expected_long:
                    raise ValueError('Long default C ABI/Python versus CLI window/offset parity failed')
                (out / 'long-python-cabi.json').write_text(json.dumps(public, ensure_ascii=False, indent=2))
            finally:
                free_aligner_cache(lib_path=str(library))
            baseline = long_calls[0]['words']
            drift = [abs(w[k] - b[k]) for w, b in zip(long_calls[1]['words'], baseline) for k in ('t0', 't1')]
            receipt['phases']['long_windows'] = {
                'samples': len(long_pcm), 'seconds': len(long_pcm) / rate, 'units': len(long_display),
                'max_boundary_difference_cs_vs_full_native': max(drift),
                'mean_boundary_difference_cs_vs_full_native': sum(drift) / len(drift),
                'scope': 'full-native comparison, exact repeat/default public parity; no labeled boundary accuracy or GPU claim'}
        receipt['status'] = 'passed'
    except BaseException as exc:
        receipt['status'] = 'failed'
        receipt['error'] = str(exc)
        (out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        receipt['artifacts'] = {str(p.relative_to(out)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
                                 for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('receipt.json', 'summary.md')}
        save()
        (out / 'summary.md').write_text('# Japanese alignment CPU profile\n\nStatus: ' + receipt['status']
                + '\n\nExact-source acceptance only. No word-boundary ground truth, GPU/VRAM, long-file, '
                  'original-Python, Subtimer ranking or default promotion.\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--long-windows', action='store_true', help='Also validate 44.52s repeated speech through bounded CLI and public ABI')
    parser.add_argument('--repeats', type=int, default=5, help='Measured OFF/ON/ON/OFF cycles on frozen logits')
    args = parser.parse_args()
    if not 1 <= args.threads <= 8 or not 3 <= args.repeats <= 15:
        parser.error('threads must be 1..8 and repeats 3..15')
    run_hosted(args)


if __name__ == '__main__':
    main()
