"""Candidate windows for the search-only video test (ExtremeWhenBench step V1): per query, the one-minute window that
holds the event's midpoint plus `--windows - 1` other minutes of the same video, shuffled. Minutes within 60 s of the
event are not used as distractors, so no distractor shows part of it. A minute is a window when all its frames exist
(frame k = t 5k s, scripts/video/extract_frames.py at fps 1/5 -> 12 frames per minute). Stdlib only.

  python3 scripts/video/build_ewb_windows.py --ewb expQ/video/ewb_test.json \
      --frames data/video/lvbench_frames/frames.json --out expQ/video/ewb_lvbench_w10.json --n 500
"""

import argparse
import json
import random
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ewb", required=True)
    ap.add_argument("--frames", required=True, help="frames.json {video id: n_frames}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="LVBench")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--window_s", type=int, default=60)
    ap.add_argument("--frame_s", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [r for r in json.load(open(args.ewb)) if r["source_corpus"] == args.source]
    n_frames = json.load(open(args.frames))
    fpw = args.window_s // args.frame_s
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    out, skipped = [], Counter()
    for r in rows:
        if len(out) == args.n:
            break
        if r["video_id"] not in n_frames:
            skipped["no frames"] += 1
            continue
        s, e = r["correct_interval"]
        gold_m = int((s + e) / 2 // args.window_s)
        n_min = n_frames[r["video_id"]] // fpw
        near = {m for m in range(n_min) if m * args.window_s < e + args.window_s and (m + 1) * args.window_s > s - args.window_s}
        pool = [m for m in range(n_min) if m not in near]
        if gold_m >= n_min or len(pool) < args.windows - 1:
            skipped["video too short"] += 1
            continue
        wins = rng.sample(pool, args.windows - 1) + [gold_m]
        rng.shuffle(wins)
        out.append({"qid": r["qid"], "video_id": r["video_id"], "question": r["question"], "category": r["category"],
                    "interval": [s, e], "windows": wins, "gold": [wins.index(gold_m)],
                    "event_inside_window": gold_m * args.window_s <= s and e <= (gold_m + 1) * args.window_s,
                    "frames_per_window": fpw})
    json.dump(out, open(args.out, "w"))
    inside = sum(x["event_inside_window"] for x in out)
    print(f"{len(out)} queries over {len({x['video_id'] for x in out})} videos; event fully inside its window "
          f"{inside}/{len(out)}; skipped {dict(skipped)}; categories {dict(Counter(x['category'] for x in out))}")
    return 0 if len(out) == args.n else 1


if __name__ == "__main__":
    sys.exit(main())
