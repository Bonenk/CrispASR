#!/usr/bin/env python3
"""Production UTF-8 loop cleanup must preserve bytes under any ctype locale (#533).

Inject the classifier at its two production dependency calls, without rewriting
word splitting or loop collapse. Some platform locales classify raw A0/85 as
whitespace, although these bytes can be part of an unrelated UTF-8 character.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NgramLocaleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ngram-locale-')
        directory = Path(cls.temp.name)
        header = (ROOT / 'src/core/ngram_loop_fix.h').read_text()
        header = header.replace('std::isspace(', 'fixture_isspace(')
        (directory / 'production.h').write_text(header)
        source = directory / 'fixture.cpp'
        source.write_text('''#include <cctype>
#include <iostream>
#include <iterator>
#include <string>
int fixture_isspace(int c) { return c == 0xA0 || c == 0x85 || std::isspace(c); }
#include "production.h"
int main() {
    const std::string text((std::istreambuf_iterator<char>(std::cin)), {});
    std::cout << core_ngram::fix_loops(text);
}
''')
        cls.binary = directory / 'fixture'
        compiler = shutil.which('c++')
        if not compiler:
            raise RuntimeError('C++ compiler required; this regression must not skip')
        subprocess.run([compiler, '-std=c++17', str(source), '-o', str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def clean(self, text):
        env = os.environ.copy()
        env.pop('CRISPASR_NGRAM_LOOPFIX_OFF', None)
        raw = subprocess.check_output([str(self.binary)], input=text.encode('utf8'), env=env)
        return raw.decode('utf8', 'strict')

    def test_reporter_zao_preserved(self):
        text = '熔盐电蓄热系统改造，建设70兆瓦。'
        self.assertEqual(self.clean(text), text)

    def test_reporter_four_jia_preserved(self):
        text = '导致电加热管击穿、电加热元件受损、电加热管束泄漏、电加热器故障。'
        self.assertEqual(self.clean(text), text)
        self.assertEqual(self.clean(text).count('加'), 4)

    def test_continuation_bytes_and_real_nbsp_preserved(self):
        text = ''.join(chr(0x4E00 + b) for b in range(0x80, 0xC0)) + ' 加造\u00a0₠😠'
        self.assertEqual(self.clean(text), text)

    def test_ascii_whitespace_still_normalized(self):
        self.assertEqual(self.clean(' \talpha\n\rbeta\v\fgamma '), 'alpha beta gamma')

    def test_loop_collapse_still_runs(self):
        self.assertEqual(self.clean('start ' + 'echo ' * 30 + 'end'), 'start echo echo echo end')


if __name__ == '__main__':
    unittest.main()
