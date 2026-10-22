#!/usr/bin/env python3
"""#551: real CLI rejects foreign Parakeet schemas without a model download."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile


def string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def fixture(path, architecture="parakeet", head="rnnt"):
    # Format from transcribe.cpp: same architecture, distinct metadata and
    # predictor naming. No upstream weights; a single zero F32 is enough.
    metadata = [("general.architecture", architecture),
                ("general.name", "parakeet-unified-en-0.6b"),
                ("stt.parakeet.head_kind", head)]
    raw = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata))
    for key, value in metadata:
        raw += string(key) + struct.pack("<I", 8) + string(value)
    raw += string("pred.embed.weight") + struct.pack("<IQQIQ", 2, 1, 1, 0, 0)
    path.write_bytes(raw + bytes((-len(raw)) % 32) + struct.pack("<f", 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    args = parser.parse_args()
    cli = args.cli.resolve(strict=True)
    audio = args.audio.resolve(strict=True)
    count = 0
    with tempfile.TemporaryDirectory(prefix="crispasr-551-") as folder:
        root = Path(folder)
        for name in ("parakeet-unified-en-0.6b-Q8_0.gguf", "renamed.gguf", "parakeet-ctc-external.gguf"):
            model = root / name
            fixture(model)
            for backend in (None, "parakeet", "fastconformer-ctc", "canary-ctc",
                            "fastconformer_ctc", "canary_ctc", "quds-fa", "whisper"):
                output = root / "transcript"
                command = [str(cli), "-m", str(model), "-f", str(audio), "--no-gpu",
                           "-l", "en", "-oj", "-of", str(output)]
                if backend:
                    command += ["--backend", backend]
                result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                log = result.stdout + result.stderr
                assert result.returncode > 0, (command, result.returncode, log)
                assert "unsupported Parakeet GGUF architecture/layout" in log, log
                assert "parakeet-unified-en-0.6b" in log, log
                assert "auto-routing to --backend fastconformer-ctc" not in log, log
                assert "required tensor" not in log, log
                assert not output.with_suffix(".json").exists(), log
                count += 1
    print(f"PASS: {count} real CLI unsupported-layout checks; nonzero exit, no wrong loader/output")


if __name__ == "__main__":
    main()
