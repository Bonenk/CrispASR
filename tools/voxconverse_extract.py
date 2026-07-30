#!/usr/bin/env python3
"""Extract a VoxConverse dev shard (HF parquet) into wav/ + ref.json.

The diarization evaluation corpus, made reproducible. One dev shard is 44 files
/ ~4.2 h with 1-17 speakers each, which is a far better spread than the 8-file
subset the #324/#326 work was steered by.

    python tools/voxconverse_extract.py \\
        --parquet .../voxconverse/data/dev-00000-of-00005.parquet \\
        --out /path/to/vox_dev44

Files are named by SHA1 of their audio bytes, so names are stable across runs
and independent of row order — which matters because tools/diarize_eval.py
derives its tune/holdout split from the name.

Output layout matches what diarize_eval.py expects:
    <out>/wav/<name>.wav
    <out>/ref.json      {name: [{start, end, speaker}, ...]}
"""

import argparse, hashlib, json, os, wave


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    os.makedirs(os.path.join(args.out, "wav"), exist_ok=True)
    t = pq.read_table(args.parquet)
    need = {"audio", "timestamps_start", "timestamps_end", "speakers"}
    missing = need - set(t.schema.names)
    if missing:
        raise SystemExit(f"parquet is missing columns: {sorted(missing)}")

    ref = {}
    for i in range(t.num_rows):
        raw = t.column("audio")[i].as_py()["bytes"]
        name = hashlib.sha1(raw).hexdigest()[:8]
        path = os.path.join(args.out, "wav", name + ".wav")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(raw)
        ref[name] = [
            {"start": s, "end": e, "speaker": k}
            for s, e, k in zip(
                t.column("timestamps_start")[i].as_py(),
                t.column("timestamps_end")[i].as_py(),
                t.column("speakers")[i].as_py(),
            )
        ]

    with open(os.path.join(args.out, "ref.json"), "w") as f:
        json.dump(ref, f, indent=1)

    total = 0.0
    for n in ref:
        w = wave.open(os.path.join(args.out, "wav", n + ".wav"))
        total += w.getnframes() / w.getframerate()
        w.close()
    counts = sorted(len({s["speaker"] for s in v}) for v in ref.values())
    print(f"{len(ref)} files, {total/3600:.2f} h, speakers/file min {counts[0]} max {counts[-1]}")
    print("⚠ files above --diarize-max-speakers cannot be counted correctly; "
          f"{sum(1 for c in counts if c > 8)} of these exceed the default cap of 8")


if __name__ == "__main__":
    main()
