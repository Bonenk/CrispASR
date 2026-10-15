"""Small guards for the actual patch, axis ownership and compiled file reader."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('worker', HERE.parent / 'dia_cache_intervention.py')
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def main():
    native = np.arange(2*979*4*128, dtype='<f4').reshape(2, 979, 4, 128)
    original = -native-1
    values = worker.arms(native, original)
    np.testing.assert_array_equal(values['native-control'], native)
    np.testing.assert_array_equal(values['original-prefix'][:, :978], original[:, :978])
    np.testing.assert_array_equal(values['original-prefix'][:, 978:], native[:, 978:])
    np.testing.assert_array_equal(values['original-current'][:, :978], native[:, :978])
    np.testing.assert_array_equal(values['original-current'][:, 978:], original[:, 978:])
    np.testing.assert_array_equal(values['original-both'], original)
    for wrong in [original[:, :-1], original.astype('f8'), np.full(original.shape, np.nan, dtype='<f4')]:
        try: worker.arms(native, wrong)
        except AssertionError: pass
        else: raise AssertionError('Malformed replacement accepted')
    with tempfile.TemporaryDirectory(prefix='dia-key-selftest-') as tmp:
        tmp = Path(tmp)
        (tmp / 'reader.cpp').write_text('''#include "key_input.h"
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::vector<float> values;
    return dia_read_intervention_keys(argv[1], 4, values) ? 0 : 1;
}
''')
        subprocess.run(['c++', '-std=c++17', '-I', str(HERE), str(tmp/'reader.cpp'), '-o', str(tmp/'reader')], check=True)
        for name, array, accepted in [('valid', [1,2,3,4], True), ('short', [1,2,3], False),
                                      ('extra', [1,2,3,4,5], False), ('nan', [1,2,3,np.nan], False),
                                      ('inf', [1,2,3,np.inf], False)]:
            p = tmp/name; np.asarray(array, dtype='<f4').tofile(p)
            assert (subprocess.run([str(tmp/'reader'), str(p)]).returncode == 0) == accepted, name
        assert subprocess.run([str(tmp/'reader'), str(tmp/'missing')]).returncode == 1
    print('DIA_KEY_INTERVENTION_SELFTEST_PASS: axis ownership, three malformed arrays, actual C++ reader valid plus five rejecting files')


if __name__ == '__main__': main()
