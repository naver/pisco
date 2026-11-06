"""Fetch the LVBench videos that ExtremeWhenBench uses (lmms-eval/LVBench, ungated, CC-BY-NC-SA-4.0; 14 zip chunks,
62 GB for 103 videos): download one chunk at a time, extract only the needed `<youtube id>.mp4`, delete the chunk.

  python scripts/video/fetch_lvbench.py --ewb expQ/video/ewb_test.json --out data/video/lvbench
"""

import argparse
import json
import os
import pathlib
import sys
import zipfile

REPO = "lmms-eval/LVBench"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ewb", required=True, help="ExtremeWhenBench test rows (JSON list)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    need = {r["video_id"] for r in json.load(open(args.ewb)) if r["source_corpus"] == "LVBench"}
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in out.glob("*.mp4")}
    print(f"[lvbench] {len(need)} videos needed, {len(need & have)} already here", flush=True)
    chunks = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".zip"))
    tmp = out / "_zips"
    for c in chunks:
        if need <= have:
            break
        z = hf_hub_download(REPO, c, repo_type="dataset", local_dir=str(tmp))
        with zipfile.ZipFile(z) as zf:
            for m in zf.infolist():
                stem = pathlib.Path(m.filename).stem
                if m.filename.endswith(".mp4") and stem in need and stem not in have:
                    with zf.open(m) as src, open(out / f"{stem}.mp4", "wb") as dst:
                        while True:
                            b = src.read(1 << 24)
                            if not b:
                                break
                            dst.write(b)
                    have.add(stem)
        os.remove(z)
        print(f"[lvbench] {c}: {len(need & have)}/{len(need)} videos", flush=True)
    missing = sorted(need - have)
    json.dump({v: str(out / f"{v}.mp4") for v in sorted(need & have)}, open(out / "videos.json", "w"), indent=0)
    print(f"[lvbench] done: {len(need & have)} videos in {out}; missing {missing}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
