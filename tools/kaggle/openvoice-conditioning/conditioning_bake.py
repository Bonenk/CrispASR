#!/usr/bin/env python3
"""Immutable shared worker bootstrap. Only this script is uploaded by Kaggle."""
import hashlib
import os
from pathlib import Path
import runpy
import urllib.request

WORKER_SOURCE = "5f9ad8b36be773b1483b71a01ce0ef5a06ca3cc6"
WORKER_SHA256 = "59b3ccc7129c64dbfb0dfc32fe27d6a655f9e4f14db04f580daeed9f7ac8b74c"
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
