"""Accepted loader/ownership preparations plus single-use terminal scratch patch."""
from pathlib import Path
import importlib.util
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('terminal_patch', HERE/'patch.py')
patcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(patcher)


def main():
    subprocess.run([sys.executable, HERE.parent/'dia-suffix-replay/selftest.py'], check=True)
    with tempfile.TemporaryDirectory(prefix='dia-terminal-patch-') as tmp:
        p = Path(tmp)/'dia_tts.cpp'
        p.write_bytes(subprocess.check_output(['git', 'show', patcher.SOURCE+':src/dia_tts.cpp']))
        patcher.patch(p)
        source = p.read_text()
        assert source.count('ggml_build_forward_expand(gf, terminal_computed)') == 1
        assert source.index('terminal_computed = cur;') < source.index('// Project to logits for each codebook head')
        assert '(!suffix_dir || suffix_file || suffix_layer_text ||' in source
        assert 'ggml_backend_tensor_set(terminal_input, terminal_values.data()' in source
        try:
            patcher.patch(p)
        except AssertionError:
            pass
        else:
            raise AssertionError('Terminal patch accepted twice')
    print('DIA_TERMINAL_SELFTEST_PASS: accepted finite loader/prefix controls, terminal-only placement and single-use pinned patch')


if __name__ == '__main__':
    main()
