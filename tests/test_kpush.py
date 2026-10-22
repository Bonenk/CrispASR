"""Run the launcher with a CLI-only Kaggle package, without credentials/network."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class KpushTest(unittest.TestCase):
    def test_cli_without_package_main_renders_and_forwards(self):
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            package = tmp / 'kaggle'
            package.mkdir()
            (package / '__init__.py').write_text('')
            (package / 'cli.py').write_text('''
import json, os, sys
from pathlib import Path
def main():
    assert sys.argv[1:4] == ['kernels', 'push', '-p']
    assert os.environ['KAGGLE_API_TOKEN'] == 'dummy-token'
    root = Path(sys.argv[4])
    print(json.dumps({'metadata': json.loads((root/'kernel-metadata.json').read_text()),
                      'code': (root/'worker.py').read_text(), 'args': sys.argv[5:]}))
''')
            kernel = tmp / 'kernel'
            kernel.mkdir()
            (kernel / 'kernel-metadata.json').write_text(json.dumps({
                'id': '${KAGGLE_ACCOUNT}/test', 'code_file': 'worker.py',
                'dataset_sources': ['${KAGGLE_ACCOUNT}/fixture']}))
            (kernel / 'worker.py').write_text("OWNER='${KAGGLE_ACCOUNT}'")
            env = dict(os.environ, KAGGLE_ACCOUNT='test-owner', KAGGLE_TOKEN='dummy-token', PYTHONPATH=str(tmp))
            r = subprocess.run([sys.executable, str(ROOT/'tools/kaggle/kpush.py'), str(kernel), '--accelerator', 'test'],
                               env=env, capture_output=True, text=True, check=True)
            row = json.loads(r.stdout)
            self.assertEqual(row['metadata']['id'], 'test-owner/test')
            self.assertEqual(row['metadata']['dataset_sources'], ['test-owner/fixture'])
            self.assertEqual(row['code'], "OWNER='test-owner'")
            self.assertEqual(row['args'], ['--accelerator', 'test'])
            self.assertIn('${KAGGLE_ACCOUNT}', (kernel/'worker.py').read_text())


if __name__ == '__main__':
    unittest.main()
