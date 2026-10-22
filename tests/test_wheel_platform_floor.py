"""Reject compatibility labels below actual native requirements."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wheel_floor", ROOT / "python/tools/check_wheel_platform.py")
FLOOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FLOOR)

MACHO = """Load command 4
      cmd LC_BUILD_VERSION
 platform 1
    minos 11.0
      sdk 26.0
    ntools 1
      tool LD
   version 1115.7.3
"""


class WheelPlatformFloorTest(unittest.TestCase):
    def policy(self, name, arch="x86_64"):
        return json.loads(FLOOR.POLICY_FILE.read_text())["policies"][name][arch]

    def test_original_linux_label_rejects_actual_release_requirements(self):
        required = {"GLIBC": ["2.34"], "GLIBCXX": ["3.4.30"], "CXXABI": ["1.3.13"]}
        bad = FLOOR.linux_violations(required, self.policy("manylinux_2_28"))
        self.assertIn("GLIBC_2.34", bad)
        self.assertIn("GLIBCXX_3.4.30", bad)
        self.assertEqual(FLOOR.linux_violations(required, self.policy("manylinux_2_35")), [])

    def test_newer_cpp_runtime_is_checked_even_with_older_glibc(self):
        required = {"GLIBC": ["2.17"], "GLIBCXX": ["3.4.30"]}
        self.assertEqual(FLOOR.linux_violations(required, self.policy("manylinux_2_28")), ["GLIBCXX_3.4.30"])

    def test_aarch64_uses_its_pinned_policy(self):
        self.assertEqual(FLOOR.linux_violations({"GLIBC": ["2.34"]}, self.policy("manylinux_2_35", "aarch64")), [])

    def test_unknown_host_symbol_version_is_rejected(self):
        self.assertIn("GLIBC_PRIVATE", FLOOR.linux_violations({"GLIBC": ["PRIVATE"]}, self.policy("manylinux_2_35")))

    def test_readelf_ignores_exported_definitions(self):
        text = "Version definition section:\n Name: GLIBCXX_99.0\nVersion needs section:\n Name: GLIBC_2.34\n Name: CXXABI_1.3.13\n"
        result = FLOOR.readelf_requirements(text)
        self.assertEqual(result["GLIBCXX"], [])
        self.assertEqual(result["GLIBC"], ["2.34"])

    def test_llvm_uses_only_undefined_symbols(self):
        result = FLOOR.objdump_requirements("0 DF *UND* 0 (GLIBCXX_3.4.30) wait\n1 DF .text 0 (GLIBC_99.0) local\n")
        self.assertEqual(result["GLIBCXX"], ["3.4.30"])
        self.assertEqual(result["GLIBC"], [])

    def test_macos_ignores_sdk_and_linker_versions(self):
        self.assertEqual(FLOOR.macos_minima(MACHO), ["11.0"])
        self.assertEqual(FLOOR.macos_violations(["11.0.0"], "11.0"), [])

    def test_macos_helper_regression_and_repaired_target(self):
        self.assertEqual(FLOOR.macos_violations(["11.0", "14.0"], "11.0"), ["14.0"])
        self.assertEqual(FLOOR.macos_violations(["11.0", "11.0"], "11.0"), [])

    def test_macos_all_fat_slices_are_checked(self):
        values = FLOOR.macos_minima(MACHO + MACHO.replace("11.0", "14.0"))
        self.assertEqual(FLOOR.macos_violations(values, "11.0"), ["14.0"])

    def test_legacy_macos_command(self):
        self.assertEqual(FLOOR.macos_minima("Load command 2\n cmd LC_VERSION_MIN_MACOSX\n version 11.0\n sdk 14.0\n"), ["11.0"])

    def test_wrong_platform_or_missing_minimum_is_not_a_pass(self):
        for text in ("", MACHO.replace("platform 1", "platform 2"), MACHO.replace("minos 11.0", "missing 11.0")):
            with self.assertRaises(ValueError):
                FLOOR.macos_minima(text)

    def test_every_library_including_helper_is_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "libcore.dylib").write_bytes(b"core")
            (path / "libhelpers.dylib").write_bytes(b"helper")
            with patch.object(FLOOR, "command", side_effect=[MACHO, MACHO.replace("11.0", "14.0")]):
                result = FLOOR.audit(path, "macosx_11_0_arm64")
            self.assertFalse(result["passed"])
            self.assertIn("libhelpers.dylib", result["violations"][0])

    def test_empty_package_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "No staged"):
                FLOOR.audit(Path(folder), "manylinux_2_35_x86_64")

    def test_missing_core_abi_imports_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "libcrispasr.so").write_bytes(b"fixture")
            with patch.object(FLOOR, "command", return_value="unrecognized output"):
                with self.assertRaisesRegex(ValueError, "no GLIBC imports"):
                    FLOOR.audit(path, "manylinux_2_35_x86_64")

    def test_wheel_metadata_must_match_requested_platform(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wheel = root / "fixture.whl"
            with zipfile.ZipFile(wheel, "w") as z:
                z.writestr("fixture.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-manylinux_2_28_x86_64\n")
            receipt = root / "assessment.json"
            argv = ["check", "--wheel", str(wheel), "--platform-tag", "manylinux_2_35_x86_64", "--receipt", str(receipt)]
            with patch("sys.argv", argv), patch.object(FLOOR, "audit") as audit:
                with patch("sys.stderr"):
                    self.assertEqual(FLOOR.main(), 1)
                audit.assert_not_called()
            self.assertFalse(json.loads(receipt.read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
