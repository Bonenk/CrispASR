#!/usr/bin/env python3
"""Bounded loader/environment checks; never start Kaggle or import Torch."""
import ast
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/'tools/kaggle/openvoice-bake/openvoice_bake.py'


class WorkerContracts(unittest.TestCase):
    def test_host_loader_path_survives_native_bundle_prefix(self):
        spec=importlib.util.spec_from_file_location('openvoice_worker',WORKER)
        worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
        with patch.dict(os.environ,{'LD_LIBRARY_PATH':'/bundle:/driver',
                                   'CRISPASR_OV2_HOST_LD_LIBRARY_PATH':'/driver'},clear=True):
            self.assertEqual(worker.host_environment()['LD_LIBRARY_PATH'],'/driver')
        with patch.dict(os.environ,{'LD_LIBRARY_PATH':'/driver'},clear=True):
            self.assertEqual(worker.host_environment()['LD_LIBRARY_PATH'],'/driver')

    def test_embedded_python_phases_compile(self):
        tree=ast.parse(WORKER.read_text())
        phases=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.startswith('import sys,'):
                compile(node.value,'embedded-phase','exec');phases.append(node.value)
        self.assertEqual(len(phases),2)
        outer_imports=[alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names]
        self.assertNotIn('torch',outer_imports)
        # Modules may load Torch in these isolated child processes; never put
        # CUDA12.4's LD_LIBRARY_PATH into the child environment.


if __name__=='__main__':unittest.main()
