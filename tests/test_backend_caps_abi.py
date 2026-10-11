#!/usr/bin/env python3
"""Compare every actually compiled CLI capability row with the native C ABI.

Requires real built artifacts. Missing binaries/libraries fail rather than
silently skipping the drift check. No model or GPU inference is claimed.
"""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess


def check(cli, library):
    payload = json.loads(subprocess.check_output([str(cli.resolve()), '--list-backends-json'], text=True))
    rows = payload['backends'] if isinstance(payload, dict) else payload
    assert rows, 'CLI returned no backend rows'
    lib = ctypes.CDLL(str(library.resolve()))
    lookup = lib.crispasr_backend_caps_abi
    lookup.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int32]
    lookup.restype = ctypes.c_int
    listing = lib.crispasr_backend_caps_list_abi
    listing.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    listing.restype = ctypes.c_int
    required = -listing(None, 0)
    assert required > 0
    buffer = ctypes.create_string_buffer(required)
    count = listing(buffer, required)
    assert count == len(buffer.value) and count + 1 == required
    listed = dict(line.split('\t', 1) for line in buffer.value.decode().splitlines())
    failures = []
    for row in rows:
        expected = ','.join(row['caps'])
        name = row['name']
        output = ctypes.create_string_buffer(4096)
        rc = lookup(name.encode(), output, len(output))
        if rc != len(expected) or output.value.decode() != expected or listed.get(name) != expected:
            failures.append(f'{name}: CLI={expected!r}, ABI rc={rc} csv={output.value!r}, list={listed.get(name)!r}')
        if expected and rc >= 0:
            small_rc = lookup(name.encode(), output, 1)
            if small_rc != -2:
                failures.append(f'{name}: small-buffer rc={small_rc}, expected -2')
    assert lookup(None, buffer, required) == -1
    assert lookup(b'__unknown_backend__', buffer, required) == -3
    assert listing(buffer, 1) == -required
    assert not failures, '\n'.join(failures)
    print(f'Actual CLI/C ABI capabilities: {len(rows)} compiled rows PASS; size/error contracts PASS.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--library', type=Path)
    group.add_argument('--build-dir', type=Path)
    args = parser.parse_args()
    library = args.library
    if library is None:
        candidates = sorted(args.build_dir.rglob('libcrispasr.so'))
        if len(candidates) != 1:
            raise RuntimeError(f'Expected one built libcrispasr.so, found {candidates}')
        library = candidates[0]
    check(args.cli, library)
