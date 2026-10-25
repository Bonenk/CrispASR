"""Publication failures must be detected before the first distribution uploads."""

import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pypi_distributions", ROOT / "python/tools/check_pypi_distributions.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class PublicationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.metadata = b"Metadata-Version: 2.1\nName: crispasr\nVersion: 0.8.44\n"
        for platform in CHECK.PLATFORMS:
            self.wheel(platform, self.metadata)
        with tarfile.open(self.folder / "crispasr-0.8.44.tar.gz", "w:gz") as archive:
            member = tarfile.TarInfo("crispasr-0.8.44/PKG-INFO")
            member.size = len(self.metadata)
            archive.addfile(member, io.BytesIO(self.metadata))

    def wheel(self, platform, metadata):
        path = self.folder / f"crispasr-0.8.44-py3-none-{platform}.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("crispasr-0.8.44.dist-info/METADATA", metadata)
        return path

    def test_complete_set_has_hashes(self):
        result = CHECK.validate(self.folder, "0.8.44")
        self.assertEqual(len(result), 5)
        self.assertTrue(all(len(item["sha256"]) == 64 and item["bytes"] > 0 for item in result))

    def test_hashing_does_not_require_python311_file_digest(self):
        # The CTest unit runner uses Python3.10, which lacks this API.
        with patch.object(CHECK.hashlib, "file_digest", None, create=True):
            self.assertEqual(len(CHECK.validate(self.folder, "0.8.44")), 5)

    def test_original_receipt_contamination_is_rejected(self):
        (self.folder / "wheel-abi.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "unexpected=.*wheel-abi.json"):
            CHECK.validate(self.folder, "0.8.44")

    def test_missing_platform_is_rejected(self):
        (self.folder / "crispasr-0.8.44-py3-none-win_amd64.whl").unlink()
        with self.assertRaisesRegex(ValueError, "missing=.*win_amd64"):
            CHECK.validate(self.folder, "0.8.44")

    def test_filename_cannot_hide_wrong_version(self):
        self.wheel("win_amd64", self.metadata.replace(b"0.8.44", b"0.8.43"))
        with self.assertRaisesRegex(ValueError, "metadata does not match"):
            CHECK.validate(self.folder, "0.8.44")

    def test_wrong_project_is_rejected(self):
        self.wheel("win_amd64", self.metadata.replace(b"Name: crispasr", b"Name: other"))
        with self.assertRaisesRegex(ValueError, "metadata does not match"):
            CHECK.validate(self.folder, "0.8.44")

    def test_gpu_local_version_is_not_cpu_publication(self):
        path = self.folder / "crispasr-0.8.44-py3-none-win_amd64.whl"
        path.rename(path.with_name(path.name.replace("0.8.44", "0.8.44+cuda")))
        with self.assertRaisesRegex(ValueError, "Distribution set mismatch"):
            CHECK.validate(self.folder, "0.8.44")


if __name__ == "__main__":
    unittest.main()
