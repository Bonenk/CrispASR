#!/usr/bin/env python3
"""Compare literal native output protocol against pinned original Python witnesses.

Compile production stopping/BOS/emission/reversal bodies without model weights.
The fixture records the original generate/_generate_output/DecoderOutput methods
on synthetic predictions; this tests codec inputs, not neural or speech parity.
CRISPASR_DIA_PROTOCOL_SOURCE can select the source before a proposed repair.
"""
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def block(source, signature):
    start = source.index(signature)
    opening = source.index('{', start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


PREAMBLE = r'''
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
struct dia_model {
    uint32_t eos_token_id=1024, pad_token_id=1025, bos_token_id=1026;
    uint32_t audio_vocab_size=1024, n_output_heads=9, max_delay=15;
    std::vector<uint32_t> delay_pattern={0,8,9,10,11,12,13,14,15};
};
struct dia_tts_context {
    dia_model model;
    std::vector<uint32_t> current_audio_tokens=std::vector<uint32_t>(9);
    int delay_steps=-1;
    uint32_t current_position=0;
    std::vector<uint32_t> output_tokens;
};
static void print(const std::vector<uint32_t>& values) {
    std::cout << '[';
    for (size_t i=0;i<values.size();++i) {
        if(i) std::cout << ',';
        std::cout << values[i];
    }
    std::cout << ']';
}
'''


class DiaOutputProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('c++')
        if not compiler:
            raise RuntimeError('c++ required for native output protocol regression')
        cls.directory = tempfile.TemporaryDirectory(prefix='dia-output-protocol-')
        cls.addClassCleanup(cls.directory.cleanup)
        directory = Path(cls.directory.name)
        source_path = Path(os.environ.get('CRISPASR_DIA_PROTOCOL_SOURCE', ROOT / 'src/dia_tts.cpp'))
        source = source_path.read_text()
        functions = '\n'.join(block(source, name) for name in [
            'static bool dia_check_stopping(', 'static void dia_revert_delay('])
        start = source.index('    const int dia_max_delay = (int)m.max_delay;')
        end = source.index('    int bos_countdown = dia_max_delay;', start)
        end += len('    int bos_countdown = dia_max_delay;')
        initialization = source[start:end]
        start = source.index('            bool stop = !dia_force && !dia_force_outputs')
        end = source.index('\n            if (p.verbosity >= 2', start)
        emission_and_stop = source[start:end]
        harness = PREAMBLE + functions + r'''
int main(int argc,char**argv) {
    if(argc!=3) return 2;
    const uint32_t max_gen=std::stoul(argv[1]);
    const int mode=std::stoi(argv[2]);
    const bool dia_force=mode==1, dia_force_outputs=mode==2;
    struct {int verbosity=0;} p;
    dia_tts_context context;
    auto* ctx=&context;
    auto& m=ctx->model;
    std::vector<uint32_t> feedback;
''' + initialization + r'''
    for(uint32_t step=0;step<max_gen;++step) {
        ctx->current_position=step;
        for(uint32_t c=0;c<9;++c) {
            feedback.push_back(gen[step][c]>=0 ? (uint32_t)gen[step][c] : m.bos_token_id);
            if(!(std::cin >> ctx->current_audio_tokens[c])) return 3;
        }
''' + emission_and_stop + r'''
    }
    std::vector<uint32_t> aligned;
    dia_revert_delay(ctx->output_tokens,aligned,m);
    std::cout << "{\"feedback\":"; print(feedback);
    std::cout << ",\"emitted\":"; print(ctx->output_tokens);
    std::cout << ",\"aligned\":"; print(aligned);
    std::cout << "}\n";
}
'''
        (directory / 'protocol.cpp').write_text(harness)
        cls.executable = directory / 'protocol'
        command = [compiler, '-std=c++17', '-O2', str(directory / 'protocol.cpp'), '-o', str(cls.executable)]
        build = subprocess.run(command, check=True, capture_output=True, text=True)
        cls.proof = None
        if os.environ.get('CRISPASR_DIA_PROTOCOL_PROOF_DIR'):
            cls.proof = Path(os.environ['CRISPASR_DIA_PROTOCOL_PROOF_DIR'])
            cls.proof.mkdir(parents=True, exist_ok=False)
            shutil.copy2(directory / 'protocol.cpp', cls.proof / 'protocol.cpp')
            shutil.copy2(cls.executable, cls.proof / 'protocol')
            metadata = dict(
                source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                executable_sha256=hashlib.sha256(cls.executable.read_bytes()).hexdigest(),
                compiler=subprocess.check_output([compiler, '--version'], text=True),
                command=command, build_stdout=build.stdout, build_stderr=build.stderr,
                scope='Actual extracted production protocol; no neural, DAC or speech execution')
            (cls.proof / 'build.json').write_text(json.dumps(metadata, indent=2) + '\n')
        cls.fixture = json.loads((ROOT / 'tests/fixtures/dia-output-protocol.json').read_text())

    def test_original_protocol_codec_inputs(self):
        self.assertEqual(self.fixture['original_source'], '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8')
        flatten = lambda rows: [value for row in rows for value in row]
        for case in self.fixture['cases']:
            with self.subTest(case=case['name']):
                raw = ' '.join(map(str, flatten(case['raw'])))
                result = subprocess.run([str(self.executable), str(case['max_tokens']), str(case['mode'])],
                                        input=raw, text=True, capture_output=True, check=True)
                native = json.loads(result.stdout)
                if self.proof:
                    (self.proof / (case['name'] + '.json')).write_text(result.stdout)
                if case['mode'] == 0:
                    self.assertEqual(native['feedback'], flatten(case['feedback']), 'all decoder feedback rows')
                self.assertEqual(native['emitted'], flatten(case['emitted']), 'original return slice')
                self.assertEqual(native['aligned'], flatten(case['aligned']), 'complete original codec input')


if __name__ == '__main__':
    unittest.main()
