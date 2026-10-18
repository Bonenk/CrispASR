#!/usr/bin/env python3
"""Inventory the complete public session header, including generated bindings.

This is a drift guard, not behavioral proof. ABI fixtures and real-library CI
separately check signatures, ownership and results. Comments/docstrings alone
cannot satisfy this census. Retired ABI stubs remain explicitly excluded.
"""
import ast
import argparse
import ctypes
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RETIRED = {
    'crispasr_speaker_db_load': 'removed native stub; use consent-aware SpeakerDB/open (#266)',
    'crispasr_speaker_db_enroll': 'removed native stub; use consent-aware SpeakerDB/enroll2 (#266)',
}
EQUIVALENT = {
    'crispasr_session_pitch_frame': 'pitch() copies the entire pitch_frames buffer',
    'crispasr_session_transcribe_chunked': 'transcribe_chunked() uses the language-aware superset',
    'crispasr_transcribe_parallel': 'compiled _helpers.c adapter; struct-by-value ABI is unsafe in ctypes',
}


def header_symbols(text):
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    return set(re.findall(r'CRISPASR_(?:SESSION_)?API[^;]+?\b(crispasr_\w+)\s*\(', text))


def python_symbols(paths):
    symbols = set()
    for path in paths:
        tree = ast.parse(path.read_text())
        # Remove docstrings so documentation does not count as implementation.
        for node in ast.walk(tree):
            body = getattr(node, 'body', None)
            if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
                value = body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    body.pop(0)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                symbols.add(node.attr)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                symbols.add(node.value)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ('_extra', '_query', '_run_pcm', '_view') and node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        symbols.add('crispasr_session_' + arg.value)
                if node.func.attr == '_set_callback' and node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        symbols.add('crispasr_session_set_' + arg.value + '_callback')
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_prefix' for t in node.targets):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    symbols.update(node.value.value + suffix for suffix in ('_init', '_process', '_free', '_free_text'))
    return symbols


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, help='also resolve all declared exports in a complete native build')
    args = parser.parse_args(argv)
    declared = set().union(*(header_symbols((ROOT / 'include' / name).read_text())
                             for name in ('crispasr_session.h', 'crispasr.h')))
    bound = python_symbols((ROOT / 'python/crispasr').glob('*.py'))
    exclusions = RETIRED | EQUIVALENT
    stale = set(exclusions) - declared
    missing = declared - bound - set(exclusions)
    if missing or stale:
        raise SystemExit(f'Unwrapped public symbols: {sorted(missing)}; stale exclusions: {sorted(stale)}')
    if args.library is not None:
        library = ctypes.CDLL(str(args.library.resolve()))
        unresolved = sorted(name for name in declared if not hasattr(library, name))
        if unresolved:
            raise SystemExit(f'Public header exports missing from {args.library.name}: {unresolved}')
        print(f'{len(declared)} header exports resolved in {args.library.name}')
    print(f'{len(declared)} unique header functions: {len(declared - set(exclusions))} bound, '
          f'{len(EQUIVALENT)} equivalent/compiled adapters, {len(RETIRED)} retired stubs')


if __name__ == '__main__':
    main()
