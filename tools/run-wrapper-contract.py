#!/usr/bin/env python3
"""Run model-free ABI contracts; missing toolchains are failures in this runner."""
import argparse
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
TESTS = {'go-rust': 'test_go_rust_wrapper_contract.py', 'dart': 'test_dart_wrapper_contract.py', 'csharp': 'test_csharp_wrapper_contract.py', 'java': 'test_java_wrapper_contract.py',
         'ruby': 'test_ruby_wrapper_contract.py', 'wasm': 'test_wasm_wrapper_contract.py'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('language', choices=TESTS)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT/'tests'))
    spec = importlib.util.spec_from_file_location('wrapper_contract', ROOT/'tests'/TESTS[args.language])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if result.skipped:
        print('Required ABI contracts skipped; install the documented toolchain.', file=sys.stderr)
    return 0 if result.wasSuccessful() and not result.skipped and result.testsRun else 1


if __name__ == '__main__':
    sys.exit(main())
