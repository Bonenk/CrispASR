#!/usr/bin/env python3
"""Issue550: pinned real CUDA alignment, 44.52s -> 2703s -> 44.52s.

No build or Torch. A failure remains a failure; no CPU fallback acceptance,
whole-file OOM baseline, acoustic-accuracy or exact allocator-peak claim.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request

VERSION = '2026-10-20.alignment-cuda-memory.3'
SOURCE = '191815511b85a526aace111b1bda2e28ecb7faad'
REPO_URL = 'https://github.com/CrispStrobe/CrispASR.git'
MIRROR_REV = '425c555beb63de4665cba604c47ea38232f5d080'
MIRROR_ROOT = 'https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/'
BUNDLE_PATH = 'alignment-long-cuda-20261019/v2/runtime.tar.gz'
BUNDLE_SHA = '63474b5f034ebac850b7b87ef76eecec12d009092d4c72370aeeb6be341365c5'
FIXTURE_PATH = 'alignment-long-cuda-20261019/v2/fixture.tar.gz'
FIXTURE_SHA = 'c46cdb1044d6837714ab7572a2edc319fc15ccf3e3645fb8991b93503995c4c8'
OUT = Path('/kaggle/working')
SCRATCH = Path('/kaggle/temp/alignment-cuda-memory')
ROOT = SCRATCH / 'sdk'
BUNDLE = SCRATCH / 'bundle'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(command, **kwargs):
    return subprocess.check_output(list(map(str, command)), text=True, **kwargs)


def download_public(name, expected_sha, target):
    # Anonymous reads cannot accidentally send another service's credentials.
    url = f'{MIRROR_ROOT}{MIRROR_REV}/{name}'
    with urllib.request.urlopen(url, timeout=180) as response, target.open('wb') as output:
        shutil.copyfileobj(response, output, 4 * 1024 * 1024)
    if sha(target) != expected_sha:
        raise ValueError('Immutable public mirror checksum mismatch')


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def load_bundle_backends(bundle):
    # This frozen runtime predates adjacent-library plugin discovery. Python's
    # executable and cwd are outside the bundle; LD_LIBRARY_PATH resolves ELF
    # dependencies but does not register ggml's dynamically loaded backends.
    ggml = ctypes.CDLL(str(bundle / 'libggml.so'))
    ggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
    ggml.ggml_backend_load_all_from_path.restype = None
    ggml.ggml_backend_load_all_from_path(os.fsencode(bundle))
    ggml.ggml_backend_dev_by_name.argtypes = [ctypes.c_char_p]
    ggml.ggml_backend_dev_by_name.restype = ctypes.c_void_p
    for name in (b'CPU', b'CUDA0'):
        if not ggml.ggml_backend_dev_by_name(name):
            raise RuntimeError('Frozen bundle backend unavailable: ' + name.decode())
    return ggml  # retain the exact registry library for the public calls


def bootstrap():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(SCRATCH), HF_HOME=str(SCRATCH / 'hf'),
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
    hardware = checked(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total,driver_version',
                        '--format=csv,noheader']).strip()
    if not hardware or any(float(row.split(',')[1]) not in (6.0, 7.5) for row in hardware.splitlines()):
        raise RuntimeError('This frozen bundle supports only actual sm60/sm75 hardware')
    if not (ROOT / '.git').exists():
        for command in (['git', 'init', ROOT], ['git', '-C', ROOT, 'remote', 'add', 'origin', REPO_URL],
                        ['git', '-C', ROOT, 'fetch', '--depth=1', 'origin', SOURCE],
                        ['git', '-C', ROOT, 'checkout', 'FETCH_HEAD']):
            subprocess.run(list(map(str, command)), check=True)
    if checked(['git', '-C', ROOT, 'rev-parse', 'HEAD']).strip() != SOURCE:
        raise ValueError('Wrong frozen runtime/harness source')
    sys.path.insert(0, str(ROOT / 'tools/kaggle'))
    import kaggle_harness as kh
    kh.init_progress()
    kh.provenance(VERSION, ROOT)
    os.environ['HF_TOKEN'] = kh.resolve_hf_token(require=True)
    if not BUNDLE.exists():
        package = SCRATCH / 'runtime.tar.gz'
        download_public(BUNDLE_PATH, BUNDLE_SHA, package)
        with tarfile.open(package) as archive:
            archive.extractall(SCRATCH, filter='data')
        fixture_package = SCRATCH / 'fixture.tar.gz'
        download_public(FIXTURE_PATH, FIXTURE_SHA, fixture_package)
        with tarfile.open(fixture_package) as archive:
            archive.extractall(SCRATCH, filter='data')
        fixture_package.unlink()
        package.unlink()
    provenance = json.loads((BUNDLE / 'provenance.json').read_text())
    if provenance['source'] != SOURCE or not provenance['build_complete'] or provenance['gpu_executed']:
        raise ValueError('Invalid frozen build provenance')
    for name, row in provenance['files'].items():
        path = BUNDLE / name
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Runtime member checksum mismatch: ' + name)
    if list(BUNDLE.glob('libcuda.so*')):
        raise ValueError('Driver or link stub must never be bundled')
    if os.environ.get('CRISPASR_ALIGNMENT_BUNDLED_RUNTIME') != '1':
        env = dict(os.environ, CRISPASR_ALIGNMENT_BUNDLED_RUNTIME='1')
        env['LD_LIBRARY_PATH'] = str(BUNDLE) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
        # Every graph placement is retained. The short CLI control must show
        # CUDA splits as well as nonzero device utilization during inference.
        env['GGML_SCHED_DEBUG'] = '1'
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve())], env)
    return kh, hardware, provenance


def main():
    kh, hardware, provenance = bootstrap()
    # Fail before downloading weights when package discovery is broken.
    backend_registry = load_bundle_backends(BUNDLE)
    kh.step('bundle-backends', cpu=True, cuda=True)
    import numpy as np
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, str(ROOT / 'tools/ci-heavy'))
    from alignment_model_profile import centiseconds, validate_words
    sys.path.insert(0, str(BUNDLE / 'python'))
    from crispasr import align_words, free_aligner_cache
    fixture = SCRATCH / 'fixture'
    cpu = json.loads((fixture / 'receipt.json').read_text())
    if cpu['source_commit'] != '62e598ca01471ad9e8c6ee3c4f7bb77e1cef62f1' or cpu['status'] != 'passed':
        raise ValueError('Wrong accepted CPU fixture receipt')
    for name in ('pcm.npy', 'units.json', 'transcript.txt', 'ATTRIBUTION.md'):
        if sha(fixture / name) != cpu['artifacts'][name]['sha256']:
            raise ValueError('CPU fixture checksum mismatch: ' + name)
    pcm = np.load(fixture / 'pcm.npy', allow_pickle=False)
    if pcm.dtype != np.float32 or pcm.shape != (101760,) or not np.isfinite(pcm).all():
        raise ValueError('Wrong pinned recording')
    display = json.loads((fixture / 'units.json').read_text())['display']
    text = (fixture / 'transcript.txt').read_text().strip()
    shutil.copy2(fixture / 'ATTRIBUTION.md', OUT / 'ATTRIBUTION.md')
    with (OUT / 'ATTRIBUTION.md').open('a') as stream:
        stream.write('\nTest transformation: exact repetition of decoded PCM/transcript, 7 and 425 times.\n')
    model_pin = cpu['pins']['model']
    model = Path(hf_hub_download(model_pin['repo'], model_pin['file'], revision=model_pin['revision']))
    if model.stat().st_size != model_pin['bytes'] or sha(model) != model_pin['sha256']:
        raise ValueError('Model checksum mismatch')
    library = str(BUNDLE / 'libcrispasr.so')
    receipt = dict(schema=1, source=SOURCE, version=VERSION, entry_sha256=sha(Path(__file__)),
                   hardware=hardware, bundle_sha256=BUNDLE_SHA, mirror_revision=MIRROR_REV,
                   fixture_archive_sha256=FIXTURE_SHA, model=model_pin,
                   fixture_pins=cpu['pins'], complete=False, gpu_executed=False, cases=[],
                   limitations=['repeated speech, not natural conversational 45-minute audio',
                                'no boundary ground truth or CPU/GPU byte-exact claim',
                                'sampled device memory, not exact allocator peak',
                                'host logits/traceback memory remains duration dependent'])
    save_json(OUT / 'validation.json', receipt)
    stop = threading.Event()
    phase = ['idle']

    def monitor():
        heartbeat = time.monotonic()
        with (OUT / 'gpu-samples.jsonl').open('w', buffering=1) as stream:
            while not stop.is_set():
                try:
                    rows = checked(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], timeout=15).strip()
                    stream.write(json.dumps(dict(time=time.time(), phase=phase[0], samples=rows)) + '\n')
                except Exception as exc:
                    stream.write(json.dumps(dict(time=time.time(), error=type(exc).__name__)) + '\n')
                if time.monotonic() - heartbeat >= 30:
                    kh.step('gpu-heartbeat', phase=phase[0])
                    heartbeat = time.monotonic()
                stop.wait(.5)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    time.sleep(2)
    original_stderr = os.dup(2)
    native_log = (OUT / 'public-native.log').open('w')
    os.dup2(native_log.fileno(), 2)
    try:
        free_aligner_cache(lib_path=library)
        results = []
        for label, repeats, offset in [('short-before', 7, 0), ('long', 425, 1234), ('short-after', 7, 1234)]:
            phase[0] = label
            kh.step(label, repeats=repeats, seconds=repeats * 6.36)
            samples = np.tile(pcm, repeats)
            transcript = ' '.join([text] * repeats)
            # Native split_alignment_words keeps one separator on the first
            # unit of each later sentence; preserve that real display contract.
            units = display + [word for _ in range(repeats - 1) for word in [' ' + display[0]] + display[1:]]
            started = time.monotonic()
            aligned = align_words(str(model), transcript, samples, t_offset=offset / 100,
                                  n_threads=4, lib_path=library)
            words = [dict(word=w.text, t0=centiseconds(w.start), t1=centiseconds(w.end)) for w in aligned]
            save_json(OUT / (label + '.json'), words)
            validate_words(words, units, len(samples) // 160 + 8, offset)
            if any(w.characters for w in aligned):
                raise ValueError('Unexpected fabricated character output')
            maps = Path('/proc/self/maps').read_text()
            for name in ('libggml-cuda.so', 'libcudart.so.12', 'libcublas.so.12', 'libcublasLt.so.12'):
                if str(BUNDLE / name) not in maps:
                    raise ValueError('Packaged CUDA runtime was not loaded: ' + name)
            receipt['cases'].append(dict(label=label, repeats=repeats, samples=len(samples),
                                         pcm_sha256=hashlib.sha256(samples.tobytes()).hexdigest(),
                                         transcript_sha256=hashlib.sha256(transcript.encode()).hexdigest(),
                                         offset_cs=offset, units=len(words), seconds=time.monotonic() - started,
                                         host_process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            save_json(OUT / 'validation.json', receipt)
            results.append(words)
            del samples, aligned
        shifted = [dict(w, t0=w['t0'] - 1234, t1=w['t1'] - 1234) for w in results[2]]
        if results[0] != shifted:
            raise ValueError('Short result changed after long-input cache/scheduler reuse')
        # Reuse acceptance is already complete; release its GPU arenas before
        # an independent CLI process loads another copy of the model.
        free_aligner_cache(lib_path=library)
        # Fresh CLI must match cached public ABI on the same 44.52s samples.
        phase[0] = 'cli'
        import soundfile as sf
        sf.write(SCRATCH / 'short.wav', np.tile(pcm, 7), 16000, subtype='FLOAT')
        (SCRATCH / 'short.txt').write_text(' '.join([text] * 7))
        with (OUT / 'cli.log').open('w') as log:
            subprocess.run(list(map(str, [BUNDLE / 'crispasr', '--align-only', '-am', model,
                           '-f', SCRATCH / 'short.wav', '--text-file', SCRATCH / 'short.txt',
                           '--align-format', 'json', '--align-granularity', 'word',
                           '--align-output', OUT / 'cli.json', '--chunk-seconds', '30',
                           '--gpu-backend', 'cuda', '-t', '4', '-l', 'ja'])),
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        cli = json.loads((OUT / 'cli.json').read_text())
        cli_words = [dict(word=w['word'], t0=centiseconds(w['start']), t1=centiseconds(w['end'])) for w in cli]
        if cli_words != results[0]:
            raise ValueError('CLI versus public C ABI mismatch')
        cli_log = (OUT / 'cli.log').read_text()
        if not re.search(r'## SPLIT #\d+: CUDA', cli_log):
            raise ValueError('Missing actual CUDA graph placement evidence')
        native_log.flush()
        if not re.search(r'## SPLIT #\d+: CUDA', (OUT / 'public-native.log').read_text()):
            raise ValueError('Public C ABI lacks CUDA graph placement evidence')
        samples = [json.loads(line) for line in (OUT / 'gpu-samples.jsonl').read_text().splitlines()]
        summaries = {}
        for label in ('idle', 'short-before', 'long', 'short-after', 'cli'):
            rows = [row for sample in samples if sample.get('phase') == label
                    for row in sample.get('samples', '').splitlines() if row.strip()]
            if not rows:
                raise ValueError('Missing device samples for ' + label)
            summaries[label] = dict(sampled_device_peak_mib=max(int(row.split(',')[1]) for row in rows),
                                    sampled_utilization_max=max(int(row.split(',')[2]) for row in rows))
        if summaries['long']['sampled_utilization_max'] <= 0:
            raise ValueError('No device execution utilization observed during long inference')
        receipt['device_samples_summary'] = summaries
        receipt['gpu_executed'] = True
        receipt['complete'] = True
    except BaseException:
        receipt['failure'] = traceback.format_exc()
        raise
    finally:
        phase[0] = 'cleanup'
        free_aligner_cache(lib_path=library)
        os.dup2(original_stderr, 2)
        os.close(original_stderr)
        native_log.close()
        stop.set()
        watcher.join(timeout=20)
        receipt['artifacts'] = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
                                for p in OUT.iterdir() if p.is_file() and p.name != 'validation.json'}
        save_json(OUT / 'validation.json', receipt)
        kh.step('terminal', complete=receipt['complete'], cases=len(receipt['cases']))


if __name__ == '__main__':
    main()
