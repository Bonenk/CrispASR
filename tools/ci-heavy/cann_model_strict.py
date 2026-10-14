#!/usr/bin/env python3
"""Prove strict Moonshine routing and diff failure controls on a CPU host."""
import json
import os
from pathlib import Path
import subprocess
import sys

import cann_model_fixture as fixture


def main():
    # Run the already-proved complete original/native fixture on this source.
    fixture.main()
    import numpy as np
    from gguf import GGUFReader, GGUFWriter
    out = Path(os.environ['HEAVY_OUT'])
    build = Path(os.environ['HEAVY_SCRATCH'])/'cann-model-fixture/build'
    model = out/'moonshine-tiny.gguf'
    lib = next(build.rglob('libcrispasr.so'))
    cli = build/'bin/crispasr'
    diff = build/'bin/crispasr-diff'
    env = dict(os.environ, LD_LIBRARY_PATH=':'.join(sorted({str(p.parent) for p in build.rglob('*.so')})))
    env.pop('CRISPASR_MOONSHINE_REQUIRE_BACKEND', None)
    env.pop('CRISPASR_MOONSHINE_REQUIRE_DEVICE', None)
    result = dict(passed=False, source=json.loads((out/'receipt.json').read_text())['source'], producer_sha256=fixture.digest(Path(__file__)), controls=[], scope='CPU routing/CLI/C ABI/diff controls only; no physical NPU/ACL/quant/performance acceptance.')
    receipt = out/'strict-receipt.json'

    def save():
        receipt.write_text(json.dumps(result, indent=2)+'\n')

    def run(label, command, environment, expected, contains=()):
        with (out/(label+'.log')).open('w') as log:
            completed = subprocess.run([str(x) for x in command], env=environment, stdout=log, stderr=subprocess.STDOUT)
        row = dict(label=label, command=[str(x) for x in command], exit=completed.returncode, expected=expected, passed=False)
        result['controls'].append(row); save()
        assert completed.returncode == expected, row
        text = (out/(label+'.log')).read_text()
        assert all(value in text for value in contains), label
        row['passed'] = True; save()

    save()
    strict = dict(env, CRISPASR_MOONSHINE_REQUIRE_BACKEND='cpu', CRISPASR_MOONSHINE_REQUIRE_DEVICE='0')
    placement = ('required_matmuls=39 unassigned=0 compute_success=1',)
    run('strict-diff', [diff, 'moonshine', model, out/'reference.gguf', out/'audio.wav'], strict, 0, placement+('magnitude=PASS', '1 pass, 0 fail, 0 skip'))
    run('strict-cli', [cli, '--backend', 'moonshine', '-m', model, '-f', out/'audio.wav', '--no-gpu', '-t', '4', '-l', 'en', '--no-punctuation', '-osrt', '-of', out/'strict-cli'], strict, 0, placement)
    assert fixture.words((out/'strict-cli.srt').read_text().split('\n', 2)[2]) == fixture.words(json.loads((out/'receipt.json').read_text())['original_transcript'])
    session_out = out/'strict-session'; session_out.mkdir(exist_ok=True)
    run('strict-session', [sys.executable, fixture.__file__, '--native', '--model', model, '--lib', lib, '--audio', out/'audio.wav', '--output', session_out], strict, 0, placement)
    assert json.loads((session_out/'session-outputs.json').read_text()) == json.loads((out/'session-outputs.json').read_text())
    for i in range(2):
        assert np.array_equal(np.load(session_out/f'encoder-native-{i}.npy'), np.load(out/f'encoder-native-{i}.npy'))
    # Exact PCM, encoder shape, finiteness and magnitude are independent gates.
    reference = {t.name: t.data.copy() for t in GGUFReader(str(out/'reference.gguf')).tensors}
    for label in ('scale', 'nonfinite', 'wrong-shape', 'changed-pcm', 'missing-stage'):
        path = out/(label+'-reference.gguf')
        writer = GGUFWriter(str(path), 'reference')
        pcm = reference['raw_audio'].copy()
        encoder = reference['encoder_output'].copy()
        if label == 'scale': encoder *= 2
        if label == 'nonfinite': encoder.flat[0] = np.nan
        if label == 'wrong-shape': encoder = encoder.reshape(288, 457)
        if label == 'changed-pcm': pcm.flat[0] += .01
        writer.add_tensor('raw_audio', pcm)
        if label != 'missing-stage': writer.add_tensor('encoder_output', encoder)
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
        run('reject-'+label, [diff, 'moonshine', model, path, out/'audio.wav'], strict, 4 if label in ('changed-pcm', 'missing-stage') else 6)
    for label, backend, ordinal in [('missing-cann', 'cann', '0'), ('invalid-ordinal', 'cpu', '-1'), ('unavailable-ordinal', 'cpu', '1')]:
        denied = dict(env, CRISPASR_MOONSHINE_REQUIRE_BACKEND=backend, CRISPASR_MOONSHINE_REQUIRE_DEVICE=ordinal)
        run('diff-'+label, [diff, 'moonshine', model, out/'reference.gguf', out/'audio.wav'], denied, 4, ('refusing fallback',))
    # Exact init rejection through both CLI and the shared-library Session path.
    # These processes must fail, never produce a successful CPU transcription.
    denied = dict(env, CRISPASR_MOONSHINE_REQUIRE_BACKEND='cann', CRISPASR_MOONSHINE_REQUIRE_DEVICE='0')
    for label, command in [
        ('cli-missing-cann', [cli, '--backend', 'moonshine', '-m', model, '-f', out/'audio.wav', '--gpu-backend', 'cann', '-l', 'en', '--no-punctuation']),
        ('session-missing-cann', [sys.executable, fixture.__file__, '--native', '--model', model, '--lib', lib, '--audio', out/'audio.wav', '--output', session_out]),
    ]:
        with (out/(label+'.log')).open('w') as log:
            completed = subprocess.run([str(x) for x in command], env=denied, stdout=log, stderr=subprocess.STDOUT)
        text = (out/(label+'.log')).read_text()
        row = dict(label=label, exit=completed.returncode, passed=completed.returncode != 0 and 'refusing fallback' in text)
        result['controls'].append(row); save(); assert row['passed'], row
    result['passed'] = True; save()
    (out/'summary.md').write_text('Full Moonshine CPU fixture and strict CPU diff/CLI/auto+explicit C ABI Sessions pass; bad reference/PCM/device controls rejected. No physical CANN/NPU claim.\n')


if __name__ == '__main__':
    main()
