"""Frozen package loader regression: no model download or GPU is needed."""
import ctypes
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'alignment_cuda_memory', ROOT / 'tools/kaggle/alignment-cuda-memory/alignment_cuda_memory.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class FrozenLoaderTest(unittest.TestCase):
    def library(self, missing=None):
        library = Mock()
        library.ggml_backend_dev_by_name.side_effect = lambda name: None if name == missing else 1
        return library

    def test_explicit_bundle_path_registers_backends_from_foreign_cwd(self):
        bundle = Path('/frozen bundle')
        cwd = Path.cwd()
        library = self.library()
        with patch.object(worker.ctypes, 'CDLL', return_value=library) as load:
            self.assertIs(worker.load_bundle_backends(bundle), library)
        load.assert_called_once_with(str(bundle / 'libggml.so'))
        library.ggml_backend_load_all_from_path.assert_called_once_with(bytes(str(bundle), 'utf8'))
        self.assertEqual(library.ggml_backend_load_all_from_path.argtypes, [ctypes.c_char_p])
        self.assertEqual([c.args[0] for c in library.ggml_backend_dev_by_name.call_args_list], [b'CPU', b'CUDA0'])
        self.assertEqual(Path.cwd(), cwd)

    def test_missing_cpu_is_a_loader_failure(self):
        with patch.object(worker.ctypes, 'CDLL', return_value=self.library(b'CPU')):
            with self.assertRaisesRegex(RuntimeError, 'backend unavailable: CPU'):
                worker.load_bundle_backends(Path('/bundle'))

    def test_missing_cuda_cannot_pass_as_cpu(self):
        with patch.object(worker.ctypes, 'CDLL', return_value=self.library(b'CUDA0')):
            with self.assertRaisesRegex(RuntimeError, 'backend unavailable: CUDA0'):
                worker.load_bundle_backends(Path('/bundle'))


if __name__ == '__main__':
    unittest.main()
