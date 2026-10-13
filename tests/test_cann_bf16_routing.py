"""Compile the actual CANN support predicate for bounded float routing controls.

Uses real ggml enums/tensors. Unexercised helper calls abort, so this is routing
coverage only; it does not simulate or accept device execution.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CannBf16RoutingTest(unittest.TestCase):
    def test_actual_predicate_on_310p_and_910b(self):
        source = Path(os.environ.get('CANN_ROUTING_SOURCE', ROOT / 'ggml/src/ggml-cann/ggml-cann.cpp'))
        text = source.read_text()
        start = text.index('static bool ggml_backend_cann_supports_op(')
        end = text.index('\n/**', start)
        predicate = text[start:end]
        headers = ROOT / 'ggml/include'
        preamble = r'''
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
extern "C" enum ggml_unary_op ggml_get_unary_op(const ggml_tensor *) { std::abort(); }
extern "C" enum ggml_glu_op ggml_get_glu_op(const ggml_tensor *) { std::abort(); }
extern "C" bool ggml_is_contiguous(const ggml_tensor *) { std::abort(); }
extern "C" bool ggml_is_contiguous_rows(const ggml_tensor *) { std::abort(); }
static int32_t ggml_get_op_params_i32(const ggml_tensor *, uint32_t) { std::abort(); }
'''
        driver = r'''
int main() {
#ifdef ASCEND_310P
    constexpr bool bf16_supported = false;
#else
    constexpr bool bf16_supported = true;
#endif
    struct Case { ggml_op op; ggml_type a, b, out; bool pass; };
    const Case cases[] = {
        {GGML_OP_ADD, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, true},
        {GGML_OP_ADD, GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F16, true},
        {GGML_OP_ADD, GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_BF16, bf16_supported},
        {GGML_OP_ADD, GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_F32, bf16_supported},
        {GGML_OP_ADD, GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_F32, bf16_supported},
        {GGML_OP_ADD, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_BF16, bf16_supported},
        {GGML_OP_MUL_MAT, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, true},
        {GGML_OP_MUL_MAT, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_F32, true},
        {GGML_OP_MUL_MAT, GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F32, true},
        {GGML_OP_MUL_MAT, GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_F32, bf16_supported},
        {GGML_OP_MUL_MAT, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F32, bf16_supported},
        {GGML_OP_MUL_MAT, GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_F32, bf16_supported},
        {GGML_OP_MUL_MAT, GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_F32, bf16_supported},
    };
    int failures = 0;
    for (unsigned i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
        ggml_tensor a = {}, b = {}, out = {};
        a.type = cases[i].a; b.type = cases[i].b; out.type = cases[i].out;
        out.op = cases[i].op; out.src[0] = &a; out.src[1] = &b;
        if (ggml_backend_cann_supports_op(nullptr, &out) != cases[i].pass) {
            std::fprintf(stderr, "routing mismatch case %u\n", i);
            ++failures;
        }
    }
    return failures;
}
'''
        with tempfile.TemporaryDirectory(prefix='cann-routing-') as temp:
            path = Path(temp); cpp = path / 'predicate.cpp'
            cpp.write_text(preamble + predicate + driver)
            for soc in ['ASCEND_310P', 'ASCEND_910B']:
                with self.subTest(soc=soc):
                    binary = path / soc
                    subprocess.run(['c++', '-std=c++17', '-O0', '-I', str(headers),
                                    '-D' + soc, str(cpp), '-o', str(binary)], check=True)
                    subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
