"""Weight-free retained-input integrity and lossless capture controls."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import dia_suffix_replay as worker


def main():
    with tempfile.TemporaryDirectory(prefix='dia-suffix-') as tmp:
        tmp = Path(tmp)
        baseline = tmp/'baseline'; baseline.mkdir()
        dest = tmp/'arm'; dest.mkdir()
        x = np.arange(2*979*512, dtype='<f4').reshape(2, 979, 512)
        name = 'cpp_decoder_sa0_k.f32'
        x.tofile(baseline/name)
        changed = x.copy(); changed[:, -1] *= -1
        changed.tofile(dest/name)
        expected = (dest/name).read_bytes()
        np.arange(8, dtype='<f4').tofile(baseline/'cpp_final_hidden.f32')
        (dest/'cpp_final_hidden.f32').write_bytes((baseline/'cpp_final_hidden.f32').read_bytes())
        rows = worker.compact_captures(dest, baseline)
        assert len(rows) == 2 and not (dest/name).exists()
        rebuilt = x.copy()
        rebuilt[:, -1] = np.fromfile(dest/'cpp_decoder_sa0_k.current.f32', dtype='<f4').reshape(2, 512)
        assert rebuilt.tobytes() == expected
        assert {r['kind'] for r in rows} == {'current-row', 'baseline'}
        changed[:, 0] = -1
        changed.tofile(dest/name)
        try:
            worker.compact_captures(dest, baseline)
        except AssertionError:
            pass
        else:
            raise AssertionError('Changed past cache accepted')
        patched = tmp/'dia_tts.cpp'
        patched.write_bytes(subprocess.check_output(['git', 'show', worker.patcher.SOURCE+':src/dia_tts.cpp'], cwd=worker.ROOT))
        worker.patcher.patch(patched)
        text = patched.read_text()
        assert 'step = suffix_dir ? 978 : 0' in text
        assert 'ggml_build_forward_expand(gf, suffix_computed)' in text
        try:
            worker.patcher.patch(patched)
        except AssertionError:
            pass
        else:
            raise AssertionError('Patch accepted twice')
        original_sha = worker.PRIOR_SHA
        for malformed in [False, True]:
            archive = tmp/f'archive-{malformed}.zip'
            payload = b'held-input'
            name = '../escape' if malformed else 'inputs/tokens.i32'
            receipt = dict(runtime_source=worker.patcher.SOURCE, ggml_source=worker.GGML_SOURCE,
                           conditional_diagnostic_complete=True, step=978,
                           payloads=[dict(path=name, bytes=len(payload), sha256=worker.hashlib.sha256(payload).hexdigest())])
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr(name, payload)
                z.writestr('receipt.json', json.dumps(receipt))
            worker.PRIOR_SHA = worker.digest(archive)
            try:
                worker.unpack_prior(archive, tmp/f'extracted-{malformed}')
            except AssertionError:
                assert malformed
            else:
                assert not malformed
        worker.PRIOR_SHA = original_sha
        # Exercise the actual shared finite/exact-size file loader for activation inputs.
        reader = tmp/'reader.cpp'
        reader.write_text('''#include "key_input.h"
int main(int argc, char** argv) {
    std::vector<float> values;
    return argc == 2 && dia_read_intervention_keys(argv[1], 4096, values) ? 0 : 1;
}
''')
        subprocess.run(['c++', '-std=c++17', '-I', str(worker.BASE_METHOD), str(reader), '-o', str(tmp/'reader')], check=True)
        file = tmp/'activation.f32'
        for values, accepted in [(np.arange(4096, dtype='<f4'), True), (np.zeros(4095, dtype='<f4'), False),
                                 (np.zeros(4097, dtype='<f4'), False), (np.full(4096, np.nan, dtype='<f4'), False),
                                 (np.full(4096, np.inf, dtype='<f4'), False)]:
            values.tofile(file)
            assert (subprocess.run([str(tmp/'reader'), str(file)]).returncode == 0) == accepted
        assert subprocess.run([str(tmp/'reader'), str(tmp/'missing')]).returncode != 0
    print('DIA_SUFFIX_SELFTEST_PASS: lossless current-row reconstruction, changed-prefix rejection, pinned single-use patch, ZIP path/hash, actual finite activation loader')


if __name__ == '__main__':
    main()
