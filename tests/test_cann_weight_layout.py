#!/usr/bin/env python3
"""Compile the production SDK-free CANN layout policy. No simulated NPU claim."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CannWeightLayout(unittest.TestCase):
    def test_production_policy_and_chunk_coverage(self):
        compiler = os.environ.get("CXX") or shutil.which("c++")
        self.assertTrue(compiler, "C++ compiler required")
        source = r'''
#include "weight_layout.h"
#include <cstdio>
#include <cstdlib>
extern "C" const char* ggml_get_name(const ggml_tensor* t) { return t->name; }
extern "C" bool ggml_is_contiguous(const ggml_tensor* t) {
    const size_t elem = t->type == GGML_TYPE_F32 ? 4 : 2;
    return t->nb[0] == elem && t->nb[1] == elem * size_t(t->ne[0]);
}
static int checks = 0;
static void check(bool ok) { ++checks; if (!ok) std::exit(1); }
int main() {
    ggml_tensor t{};
    t.type = GGML_TYPE_F16;
    t.ne[0] = 32; t.ne[1] = 48; t.ne[2] = t.ne[3] = 1;
    t.nb[0] = 2; t.nb[1] = 64;
    for (auto name : {"blk.0.attn_q.weight", "audio.blk.0.attn.q.weight", "model.layers.0.attn.qkv.weight",
                      "blk.0.attn_qkv.weight", "llm.lm_head.weight", "ffn.down.weight"}) {
        std::strcpy(t.name, name); check(ggml_cann_nz_candidate(&t));
    }
    for (auto name : {"notoutput.weight", "output.weight.bias", "attn.q.weight.extra", "embedding.weight"}) {
        std::strcpy(t.name, name); check(!ggml_cann_nz_candidate(&t));
    }
    std::strcpy(t.name, "arbitrary");
    t.flags = GGML_TENSOR_FLAG_WEIGHT_MATMUL_ONLY;
    check(ggml_cann_nz_candidate(&t));
    t.flags |= GGML_TENSOR_FLAG_WEIGHT_GENERAL; check(!ggml_cann_nz_candidate(&t));
    t.flags = GGML_TENSOR_FLAG_WEIGHT_MATMUL_ONLY;
    t.type = GGML_TYPE_BF16; check(!ggml_cann_nz_candidate(&t));
    t.type = GGML_TYPE_Q4_K; check(!ggml_cann_nz_candidate(&t));
    t.type = GGML_TYPE_F16; t.ne[2] = 2; check(!ggml_cann_nz_candidate(&t));
    t.ne[2] = 1; t.view_src = &t; check(!ggml_cann_nz_candidate(&t));
    t.view_src = nullptr; t.nb[0] = 4; check(!ggml_cann_nz_candidate(&t));
    t.nb[0] = 2; t.op = GGML_OP_MUL_MAT; check(!ggml_cann_nz_candidate(&t));
    ggml_cann_upload_coverage c;
    check(!c.add(50, 50, 100)); check(!c.add(50, 50, 100));
    check(!c.add(0, 25, 100)); check(!c.add(20, 10, 100)); check(c.add(30, 20, 100));
    c.clear(); check(!c.add(0, 99, 100)); check(!c.add(99, 2, 100)); check(c.add(99, 1, 100));
    std::printf("PASS production policy checks=%d\n", checks);
}
'''
        with tempfile.TemporaryDirectory(prefix="cann-layout-policy-") as directory:
            path = Path(directory)
            (path / "probe.cpp").write_text(source)
            compile_result = subprocess.run(
                [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "ggml/include"),
                 "-I", str(ROOT / "ggml/src/ggml-cann"), str(path / "probe.cpp"), "-o", str(path / "probe")],
                text=True, capture_output=True)
            self.assertEqual(compile_result.returncode, 0, compile_result.stdout + compile_result.stderr)
            result = subprocess.run([str(path / "probe")], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS production policy checks=26", result.stdout)


if __name__ == "__main__":
    unittest.main()
