"""Validate the complete CPU publication set before any PyPI upload."""

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile


PLATFORMS = (
    "manylinux_2_35_x86_64",
    "manylinux_2_35_aarch64",
    "macosx_11_0_arm64",
    "win_amd64",
)


def validate(folder, version):
    folder = Path(folder)
    expected = {f"crispasr-{version}-py3-none-{platform}.whl" for platform in PLATFORMS}
    expected.add(f"crispasr-{version}.tar.gz")
    actual = {p.name for p in folder.iterdir()}
    if actual != expected:
        raise ValueError(f"Distribution set mismatch: missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}")
    receipt = []
    for name in sorted(expected):
        path = folder / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Not a regular distribution: {name}")
        if name.endswith(".whl"):
            with zipfile.ZipFile(path) as archive:
                corrupt = archive.testzip()
                if corrupt:
                    raise ValueError(f"Corrupt wheel member: {corrupt}")
                names = archive.namelist()
                metadata = [n for n in names if n.endswith(".dist-info/METADATA")]
                if len(metadata) != 1:
                    raise ValueError(f"Expected one wheel METADATA: {name}")
                raw = archive.read(metadata[0])
        else:
            with tarfile.open(path, "r:gz") as archive:
                metadata = [m for m in archive.getmembers() if m.name == f"crispasr-{version}/PKG-INFO"]
                if len(metadata) != 1 or not metadata[0].isfile():
                    raise ValueError(f"Expected root sdist PKG-INFO: {name}")
                raw = archive.extractfile(metadata[0]).read()
        fields = BytesParser().parsebytes(raw)
        if fields.get_all("Name") != ["crispasr"] or fields.get_all("Version") != [version]:
            raise ValueError(f"Distribution metadata does not match requested release: {name}")
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        receipt.append({"filename": name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    print(json.dumps({"version": args.version, "distributions": validate(args.folder, args.version)}, indent=2))


if __name__ == "__main__":
    main()
