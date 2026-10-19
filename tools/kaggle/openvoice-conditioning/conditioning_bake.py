#!/usr/bin/env python3
"""Immutable shared worker bootstrap. Only this script is uploaded by Kaggle."""
import hashlib
import os
from pathlib import Path
import runpy
import urllib.request

WORKER_SOURCE = "faa6cfa4fbf0e1b24d39a82b500cd8825f193af2"
WORKER_SHA256 = "61e4cf492d2d355779cb3fa53e15c9de9d03b61912552959430906312e118a84"
path = Path('/kaggle/temp/openvoice-conditioning-worker.py')
path.parent.mkdir(parents=True, exist_ok=True)
url = ('https://raw.githubusercontent.com/CrispStrobe/CrispASR/' + WORKER_SOURCE +
       '/tools/kaggle/openvoice-bake/openvoice_bake.py')
with urllib.request.urlopen(url, timeout=60) as response:
    code = response.read()
if hashlib.sha256(code).hexdigest() != WORKER_SHA256:
    raise RuntimeError('Pinned shared worker checksum mismatch')
path.write_bytes(code)
os.environ['CRISPASR_OV2_CONDITIONING'] = '1'
runpy.run_path(str(path), run_name='__main__')
