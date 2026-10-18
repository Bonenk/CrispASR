#!/usr/bin/env python3
"""Keep context guards on Ruby Session/Stream/Chat operations, preserving method arities."""
import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT/'bindings/ruby/ext/ruby_crispasr_aux.h', ROOT/'bindings/ruby/ext/ruby_crispasr_session.c']
BEGIN = '// BEGIN GENERATED CONTEXT GUARDS\n'
END = '// END GENERATED CONTEXT GUARDS\n'
REGISTRATION = re.compile(r'(rb_define_singleton_method\((?:mSession|mStream|mChat),\s*"[^"]+",\s*)(\w+)(,\s*(-?\d+)\))')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    known = {}
    stale = []
    for path in TARGETS:
        original = path.read_text()
        source = re.sub(re.escape(BEGIN)+r'.*?'+re.escape(END), '', original, flags=re.S)
        methods = []
        def rewrite(match):
            name = match[2].removeprefix('rb_context_guard_')
            arity = int(match[4])
            if arity not in range(7) and arity != -1:
                raise ValueError(f'unsupported arity {arity}: {name}')
            if name in known and known[name] != arity:
                raise ValueError(f'inconsistent arities for {name}')
            if name not in known:
                known[name] = arity
                methods.append((name, arity))
            return match[1]+'rb_context_guard_'+name+match[3]
        source = REGISTRATION.sub(rewrite, source)
        declarations = ''.join(f'RB_CONTEXT_WRAPPER_{"VAR" if arity == -1 else arity}({name});\n' for name, arity in methods)
        marker = '/* Registration helper called from Init */' if path.suffix == '.h' else 'void init_ruby_crispasr_session('
        index = source.index(marker)
        result = source[:index]+BEGIN+declarations+END+source[index:]
        if result != original:
            stale.append(str(path.relative_to(ROOT)))
            if not args.check:
                path.write_text(result)
    if args.check and stale:
        parser.exit(1, 'Run python3 tools/sync_ruby_context_guards.py: '+', '.join(stale)+'\n')
    print(f'Ruby context guards {"checked" if args.check else "synchronized"}: {len(known)} callable operations')


if __name__ == '__main__':
    main()
