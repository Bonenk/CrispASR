"""Actual paired-file loader, factorial ownership and scratch-patch controls."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('joint_worker', HERE.parent/'dia_query_key_intervention.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def main():
    native = np.arange(2*979*4*128, dtype='<f4').reshape(2, 979, 4, 128)
    original = -native-1
    native_q = np.arange(2*16*128, dtype='<f4').reshape(2, 1, 16, 128)
    original_q = -native_q-2
    pairs = worker.arms(native, original, native_q, original_q)
    for name, k, q in [('native-control', native, native_q), ('original-query', native, original_q),
                       ('original-keys', original, native_q), ('original-query-and-keys', original, original_q)]:
        assert pairs[name][0].tobytes() == k.tobytes()
        assert pairs[name][1].tobytes() == q.tobytes()
        assert not np.shares_memory(pairs[name][0], k) and not np.shares_memory(pairs[name][1], q)
    rejected = 0
    for index, wrong in [(0, original[:, :-1]), (1, original.astype('f8')),
                         (2, native_q[:, :, :-1]), (3, original_q.astype('f8')),
                         (3, np.full(original_q.shape, np.nan, dtype='<f4')),
                         (0, np.full(native.shape, np.inf, dtype='<f4'))]:
        inputs = [native, original, native_q, original_q]; inputs[index] = wrong
        try: worker.arms(*inputs)
        except AssertionError: rejected += 1
        else: raise AssertionError('Malformed factorial input accepted')
    with tempfile.TemporaryDirectory(prefix='dia-qk-loader-') as tmp:
        tmp = Path(tmp)
        (tmp/'reader.cpp').write_text('''#include "key_input.h"
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::vector<float> values;
    return dia_read_intervention_keys(argv[1], (size_t)std::strtoul(argv[2], nullptr, 10), values) ? 0 : 1;
}
''')
        subprocess.run(['c++', '-std=c++17', '-include', 'cstdlib', '-I', str(worker.BASE_METHOD),
                        str(tmp/'reader.cpp'), '-o', str(tmp/'reader')], check=True)
        for count in [4096, 2*979*4*128]:
            valid = np.arange(count, dtype='<f4'); path = tmp/'input.f32'
            for values, accepted in [(valid, True), (valid[:-1], False), (np.append(valid, np.float32(1)), False),
                                     (np.full(count, np.nan, dtype='<f4'), False),
                                     (np.full(count, np.inf, dtype='<f4'), False)]:
                values.astype('<f4').tofile(path)
                assert (subprocess.run([str(tmp/'reader'), str(path), str(count)]).returncode == 0) == accepted
            assert subprocess.run([str(tmp/'reader'), str(tmp/'missing'), str(count)]).returncode == 1
    print(f'DIA_QUERY_KEY_SELFTEST_PASS:4 factorial arms, {rejected} malformed arrays, actual Q/K C++ loaders and rejecting files')


if __name__ == '__main__':
    main()
