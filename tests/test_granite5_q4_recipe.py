"""Small GGUF controls for quantizer recipe evidence; no model execution."""
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/ci-heavy'))
recipe = importlib.import_module('granite5_q4_recipe')


class GraniteQ4Inventory(unittest.TestCase):
    def write(self, path, *, quantized=False, damaged_frontend=False):
        from gguf import GGMLQuantizationType as Q, GGUFWriter
        writer = GGUFWriter(path, 'granite_speech')
        writer.add_tensor('enc.input.weight', np.full((32, 32),
                          2 if damaged_frontend else 1, dtype=np.float32))
        writer.add_tensor('token_embd.weight', np.ones((32, 256), dtype=np.float16))
        writer.add_tensor('blk.0.attn_norm.weight', np.ones(256, dtype=np.float32))
        writer.add_tensor('output.weight', np.ones((32, 256), dtype=np.float16))
        if quantized:
            # Format-valid synthetic Q4_K blocks; never used for inference.
            writer.add_tensor('blk.0.ffn_gate.weight', np.zeros((32, 144), dtype=np.uint8), raw_dtype=Q.Q4_K)
        else:
            writer.add_tensor('blk.0.ffn_gate.weight', np.ones((32, 256), dtype=np.float16))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_actual_q4_payload_and_protected_weights_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            original, candidate = (Path(directory) / name for name in ('original.gguf', 'candidate.gguf'))
            self.write(original)
            self.write(candidate, quantized=True)
            report = recipe.tensor_inventory(original, candidate, [recipe.HEAD])
            self.assertEqual(report['elements_by_type']['Q4_K'], 32 * 256)
            self.write(candidate, quantized=True, damaged_frontend=True)
            with self.assertRaisesRegex(ValueError, 'Protected source payload changed'):
                recipe.tensor_inventory(original, candidate, [recipe.HEAD])

    def test_f16_relabel_and_ignored_override_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            original, candidate = (Path(directory) / name for name in ('original.gguf', 'candidate.gguf'))
            self.write(original)
            self.write(candidate)
            with self.assertRaisesRegex(ValueError, 'real Q4_K'):
                recipe.tensor_inventory(original, candidate, [recipe.HEAD])
            self.write(candidate, quantized=True)
            with self.assertRaisesRegex(ValueError, 'override not applied'):
                recipe.tensor_inventory(original, candidate, [r'^blk\.0\.ffn_gate\.weight$=f16'])


if __name__ == '__main__':
    unittest.main()
