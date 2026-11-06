"""Decode videos to JPEG frames at a fixed rate (default 1 frame / 5 s): frame k (file k+1) is the frame at t = k / fps
seconds. ffmpeg comes from `pixi exec -s ffmpeg` (not in the project env); pass its path with --ffmpeg.
Writes <out>/<video id>/%05d.jpg and <out>/frames.json {video id: n_frames}. Resumable (a video with a done-marker
is skipped).

  python scripts/video/extract_frames.py --videos data/video/lvbench/videos.json --out data/video/lvbench_frames \
      --ffmpeg $(pixi exec -s ffmpeg -- which ffmpeg)
"""

import argparse
import json
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="JSON {video id: mp4 path}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--fps", default="1/5")
    ap.add_argument("--max_width", type=int, default=896)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    videos = json.load(open(args.videos))
    out = pathlib.Path(args.out)

    def one(item):
        vid, path = item
        d = out / vid
        if (d / "_done").exists():
            return vid, len(list(d.glob("*.jpg"))), None
        d.mkdir(parents=True, exist_ok=True)
        cmd = [args.ffmpeg, "-nostdin", "-loglevel", "error", "-threads", "2", "-i", path,
               "-vf", f"fps={args.fps},scale='min({args.max_width},iw)':-2", "-q:v", "3", str(d / "%05d.jpg")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            return vid, 0, r.stderr[-300:]
        (d / "_done").touch()
        return vid, len(list(d.glob("*.jpg"))), None

    counts, errs = {}, {}
    with ThreadPoolExecutor(args.workers) as ex:
        for i, (vid, n, err) in enumerate(ex.map(one, sorted(videos.items()))):
            if err:
                errs[vid] = err
            else:
                counts[vid] = n
            print(f"[frames] {i + 1}/{len(videos)} {vid}: {n} frames{' ERROR ' + err if err else ''}", flush=True)
    json.dump(counts, open(out / "frames.json", "w"), indent=0)
    print(f"[frames] {len(counts)} videos, {sum(counts.values())} frames; {len(errs)} errors")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
