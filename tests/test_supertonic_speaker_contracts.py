#!/usr/bin/env python3
"""Exercise production speaker getters without loading model weights."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_tts_session_contracts import block

ROOT = Path(__file__).resolve().parents[1]


class SupertonicSpeakerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ.get('CRISPASR_SUPERTONIC_SOURCE',
                                     ROOT / 'src/supertonic_tts.cpp')).read_text()
        getters = block(source, 'int supertonic_n_speakers(')
        getters += '\n' + block(source, 'const char* supertonic_get_speaker_name(')
        text = r'''
#include <cassert>
#include <cstring>
#include <iterator>
#include <map>
#include <string>
struct supertonic_context { std::map<std::string, int> voices; };
''' + getters + r'''
int main(int argc, char** argv) {
    assert(argc == 2);
    std::string scenario = argv[1];
    supertonic_context ctx;
    if (scenario == "null") {
        assert(supertonic_n_speakers(nullptr) == 0);
        assert(supertonic_get_speaker_name(nullptr, 0) == nullptr);
    } else if (scenario == "bounds") {
        assert(supertonic_n_speakers(&ctx) == 0);
        assert(supertonic_get_speaker_name(&ctx, 0) == nullptr);
        ctx.voices = {{"F1", 1}, {"M1", 2}};
        assert(supertonic_get_speaker_name(&ctx, -1) == nullptr);
        assert(supertonic_get_speaker_name(&ctx, 2) == nullptr);
        assert(supertonic_get_speaker_name(&ctx, 2147483647) == nullptr);
    } else if (scenario == "valid") {
        ctx.voices = {{"M1", 2}, {"F1", 1}};
        assert(supertonic_n_speakers(&ctx) == 2);
        assert(std::strcmp(supertonic_get_speaker_name(&ctx, 0), "F1") == 0);
        assert(std::strcmp(supertonic_get_speaker_name(&ctx, 1), "M1") == 0);
    } else {
        return 2;
    }
}
'''
        cls.tmp = tempfile.TemporaryDirectory(prefix='crispasr-speaker-contract-')
        path = Path(cls.tmp.name)
        (path / 'contract.cpp').write_text(text)
        cls.binary = path / 'contract'
        subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                        str(path / 'contract.cpp'), '-o', str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_null_context(self):
        subprocess.run([str(self.binary), 'null'], check=True, timeout=10)

    def test_empty_negative_and_out_of_range(self):
        subprocess.run([str(self.binary), 'bounds'], check=True, timeout=10)

    def test_valid_names_keep_map_order(self):
        subprocess.run([str(self.binary), 'valid'], check=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
