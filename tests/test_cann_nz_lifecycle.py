#!/usr/bin/env python3
"""Exercise production CANN cache state with mocked SDK memory calls, never NPU arithmetic."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def extract(source, marker, semicolon=False):
    start = source.index(marker)
    brace = source.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end] + (';' if semicolon else '') + '\n'


class Lifecycle(unittest.TestCase):
    def test_production_cache_lifecycle(self):
        source = (ROOT/'ggml/src/ggml-cann/ggml-cann.cpp').read_text()
        declarations = [extract(source, 'struct '+name+' {', True) for name in (
            'TensorSetTracker', 'ggml_cann_nz_cache', 'ggml_backend_cann_buffer_context')]
        declarations += [extract(source, marker) for marker in (
            'static bool ggml_cann_set_cached_weight(', 'void* ggml_cann_cached_nz_weight(',
            'static size_t ggml_cann_nz_cache_size(', 'static uint64_t ggml_cann_nz_cache_selections(',
            'static void ggml_cann_invalidate_nz(')]
        compiler = os.environ.get('CXX') or shutil.which('c++')
        self.assertTrue(compiler, 'C++ compiler required')
        with tempfile.TemporaryDirectory(prefix='cann-nz-lifecycle-') as directory:
            path = Path(directory)
            (path/'cann-cache-production.inc').write_text('\n'.join(declarations))
            result = subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                                     '-I', str(ROOT/'ggml/include'), '-I', str(ROOT/'ggml/src/ggml-cann'),
                                     '-I', directory, str(ROOT/'tests/cann-nz-cache-host.cpp'), '-o', str(path/'probe')],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            result = subprocess.run([str(path/'probe')], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('PASS host lifecycle checks=', result.stdout)
            print(result.stdout.strip())


if __name__ == '__main__':
    unittest.main()
