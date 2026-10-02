#!/usr/bin/env python3
"""Compare the production Dia sampler to the pinned official PyTorch sampler."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
PIN = '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'
source = urllib.request.urlopen(f'https://raw.githubusercontent.com/nari-labs/dia/{PIN}/dia/model.py').read().decode()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_sample_next_token')
namespace = {'torch': torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), 'official-dia-sampler', 'exec'), namespace)
reference = namespace['_sample_next_token']
production = (ROOT / 'src/dia_sampling.h').read_text()
function = production[production.index('static uint32_t dia_sample_token('):]
# Compile the actual production function; no copied filter implementation.
cpp = '#include <algorithm>\n#include <cmath>\n#include <cstdint>\n#include <iostream>\n#include <random>\n#include <vector>\n' + function + '''
int main() {
    int n, k, draws; float temperature, p;
    std::cin >> n >> k >> draws >> temperature >> p;
    std::vector<float> x(n); for (auto &v : x) std::cin >> v;
    std::mt19937 rng(42); std::vector<int> counts(n);
    for (int i=0; i<draws; ++i) ++counts[dia_sample_token(x.data(), n, temperature, p, k, rng)];
    for (int c : counts) std::cout << c << ' ';
}
'''
(SCRATCH / 'sampling.cpp').write_text(cpp)
subprocess.run(['g++', '-std=c++17', '-O2', SCRATCH / 'sampling.cpp', '-o', SCRATCH / 'sampling'], check=True)
fixtures = [([20, 0, -1], 1.2, .95, 0), ([2, 1, 0, -1], 1., .5, 0),
            ([2, 1, 0, -1], 1., .95, 2), ([2, 1, 0, -1], 0., .95, 2)]
for seed in range(8):
    x = np.random.default_rng(seed).normal(0, 3, 32).astype(np.float32).tolist()
    fixtures.append((x, 1.2, .95, 7))
receipt = {'official_pin': PIN, 'official_sha256': hashlib.sha256(source.encode()).hexdigest(),
           'native_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'cases': []}
original = torch.multinomial
for x, temp, top_p, k in fixtures:
    captured = []
    def capture(probabilities, num_samples):
        captured.append(probabilities.detach().numpy().reshape(-1))
        return probabilities.argmax(dim=-1, keepdim=True)
    if temp:
        torch.multinomial = capture
        try:
            reference(torch.tensor([x]), temp, top_p, k or None)
        finally:
            torch.multinomial = original
        expected = captured[0]
    else:
        expected = np.zeros(len(x)); expected[int(np.argmax(x))] = 1
    draws = 50000
    raw = subprocess.check_output([SCRATCH / 'sampling'], input=' '.join(map(str,
        [len(x), k, draws, temp, top_p] + x)), text=True)
    observed = np.array([int(n) for n in raw.split()]) / draws
    tolerance = 6 * np.sqrt(expected * (1 - expected) / draws) + .003
    case = {'logits': x, 'temperature': temp, 'top_p': top_p, 'top_k': k,
            'expected': expected.tolist(), 'observed': observed.tolist()}
    receipt['cases'].append(case)
    (OUT / 'sampling-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    assert np.all(abs(observed - expected) <= tolerance), case
    assert np.all(observed[expected == 0] == 0), case
print('DIA_OFFICIAL_SAMPLER_PARITY_PASS', len(fixtures), flush=True)
