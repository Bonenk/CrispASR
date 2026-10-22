"""Exercise the literal portability formatter, including the signed minimum."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Int128Formatting(unittest.TestCase):
    def test_extreme_signed_values_under_ubsan(self):
        compiler = shutil.which(os.environ.get('CXX', 'c++'))
        if not compiler:
            self.skipTest('C++ compiler required')
        body = (ROOT / 'src/core/tts_voice_policy.h').read_text().split('namespace core_tts_voice')[0]
        body = body.replace('#pragma once', '')
        probe = r'''
#include <sstream>
#include <iostream>
int main() {
#ifdef __SIZEOF_INT128__
    __int128 lo = -((__int128)1 << 126);
    lo += lo;
    __int128 hi = -(lo + 1);
    for (__int128 value : {(__int128)0, (__int128)1, (__int128)-1, lo, hi}) {
        std::ostringstream out;
        operator<<(out, value);
        std::cout << out.str() << "\n";
    }
#else
    return 77;
#endif
}
'''
        with tempfile.TemporaryDirectory(prefix='voice-int128-') as tmp:
            source, binary = Path(tmp) / 'probe.cpp', Path(tmp) / 'probe'
            source.write_text(body + probe)
            subprocess.run([compiler, '-std=c++17', '-O1', '-fsanitize=undefined',
                            '-fno-sanitize-recover=all', str(source), '-o', str(binary)], check=True)
            result = subprocess.run([str(binary)], text=True, capture_output=True)
            if result.returncode == 77:
                self.skipTest('Compiler has no signed128 type')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), [str(n) for n in (0, 1, -1, -(1 << 127), (1 << 127) - 1)])


if __name__ == '__main__':
    unittest.main()
