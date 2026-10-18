"""Run the Dart wrapper's compiled public-header FFI probes."""
import os
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
DART = os.environ.get('DART') or shutil.which('dart')


@unittest.skipUnless(DART and shutil.which('cc'), 'Dart SDK and C compiler required')
class DartWrapperContract(unittest.TestCase):
    def test_compiled_native_ffi_operations(self):
        package = ROOT/'flutter/crispasr'
        subprocess.run([DART, 'pub', 'get'], cwd=package, check=True)
        subprocess.run([DART, 'analyze', '--fatal-infos'], cwd=package, check=True)
        subprocess.run([DART, 'test', 'test/ffi_behavior_test.dart'], cwd=package, check=True)


if __name__ == '__main__':
    unittest.main()
