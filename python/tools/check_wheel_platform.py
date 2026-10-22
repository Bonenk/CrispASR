#!/usr/bin/env python3
"""Check staged Linux host ABI symbols or macOS deployment minima against a tag.

This checks OS/standard-library floors, not CPU ISA or model correctness.
Linux policy data is pinned from auditwheel with its MIT notice alongside it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

NAMESPACES = ("GLIBC", "GLIBCXX", "CXXABI", "GCC")
POLICY_FILE = Path(__file__).with_name("wheel_abi_policies.json")


def symbol_versions(text):
    result = {name: set() for name in NAMESPACES}
    for name, value in re.findall(r"\b(GLIBCXX|GLIBC|CXXABI|GCC)_([A-Za-z0-9_.]+)", text):
        result[name].add(value)
    return {name: sorted(values) for name, values in result.items()}


def readelf_requirements(text):
    # Definitions describe what a library supplies, not what its host must supply.
    return symbol_versions(text.partition("Version needs section")[2])


def objdump_requirements(text):
    # Cross-inspection on Mac uses LLVM instead of GNU readelf.
    return symbol_versions("\n".join(line for line in text.splitlines() if "*UND*" in line))


def macos_minima(text):
    versions = []
    for block in re.split(r"Load command \d+", text):
        if re.search(r"\bcmd LC_BUILD_VERSION\b", block):
            platform = re.search(r"^\s*platform (\S+)", block, re.M)
            if not platform or platform.group(1) not in ("1", "MACOS"):
                raise ValueError("Mach-O slice is not a macOS library")
            match = re.search(r"^\s*minos ([0-9.]+)", block, re.M)
        elif re.search(r"\bcmd LC_VERSION_MIN_MACOSX\b", block):
            match = re.search(r"^\s*version ([0-9.]+)", block, re.M)
        else:
            continue
        if not match:
            raise ValueError("Missing Mach-O deployment minimum")
        versions.append(match.group(1))
    if not versions:
        raise ValueError("No Mach-O macOS deployment command found")
    return versions


def version_tuple(value):
    parts = tuple(map(int, value.split(".")))
    return parts + (0,) * max(0, 3 - len(parts))


def linux_violations(required, allowed):
    return [f"{name}_{value}" for name in NAMESPACES
            for value in sorted(set(required.get(name, ())) - set(allowed[name]))]


def macos_violations(required, target):
    return [value for value in required if version_tuple(value) > version_tuple(target)]


def command(args):
    return subprocess.check_output(args, text=True, env=dict(os.environ, LC_ALL="C"))


def audit(package, tag):
    policy_data = json.loads(POLICY_FILE.read_text())
    linux = re.fullmatch(r"(manylinux_\d+_\d+)_(x86_64|aarch64)", tag)
    mac = re.fullmatch(r"macosx_(\d+)_(\d+)_(arm64|x86_64|universal2)", tag)
    if linux:
        policy, arch = linux.groups()
        try:
            allowed = policy_data["policies"][policy][arch]
        except KeyError:
            raise ValueError(f"No pinned ABI policy for {tag}") from None
    elif mac:
        target = ".".join(mac.groups()[:2])
    else:
        raise ValueError(f"Unsupported OS-floor tag: {tag}")
    rows, cache, failures = {}, {}, []
    for path in sorted(package.rglob("*")):
        if not path.is_file() or not (path.name.endswith(".dylib") or ".so" in path.name):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in cache:
            if linux:
                if sys.platform == "darwin":
                    required = objdump_requirements(command(["xcrun", "llvm-objdump", "-T", str(path)]))
                else:
                    required = readelf_requirements(command(["readelf", "--version-info", "--wide", str(path)]))
                if path.name.startswith("libcrispasr.so") and not required["GLIBC"]:
                    raise ValueError("Core library selected but no GLIBC imports found")
                errors = linux_violations(required, allowed)
            else:
                required = macos_minima(command(["otool", "-l", str(path)]))
                errors = macos_violations(required, target)
            cache[digest] = (required, errors)
        required, errors = cache[digest]
        name = path.relative_to(package).as_posix()
        rows[name] = {"sha256": digest, "requirements": required}
        failures.extend(f"{name}: {value} exceeds {tag}" for value in errors)
    if not rows:
        raise ValueError("No staged native libraries selected")
    return {"platform_tag": tag, "passed": not failures, "violations": failures,
            "libraries": rows, "policy_source": policy_data["source"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--package", type=Path)
    source.add_argument("--wheel", type=Path)
    parser.add_argument("--platform-tag", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.wheel:
            with tempfile.TemporaryDirectory() as folder, zipfile.ZipFile(args.wheel) as wheel:
                metadata = [n for n in wheel.namelist() if n.endswith(".dist-info/WHEEL")]
                if len(metadata) != 1:
                    raise ValueError("Expected one WHEEL metadata file")
                tags = re.findall(r"^Tag:\s*\S+-\S+-(\S+)\s*$", wheel.read(metadata[0]).decode(), re.M)
                if tags != [args.platform_tag]:
                    raise ValueError(f"WHEEL platform {tags} differs from {args.platform_tag}")
                wheel.extractall(folder)
                result = audit(Path(folder), args.platform_tag)
        else:
            result = audit(args.package, args.platform_tag)
    except (ValueError, OSError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        result = {"platform_tag": args.platform_tag, "passed": False, "error": str(exc)}
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        print(json.dumps(result.get("violations", result.get("error")), indent=2), file=sys.stderr)
        return 1
    print(f"OS/stdlib floor check passed: {len(result['libraries'])} libraries, {args.platform_tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
