"""Speech transcripts for the LVBench videos (the transcript layer of the hierarchy test): ffmpeg extracts 16 kHz
mono audio, Whisper (transformers ASR pipeline, chunked long-form with timestamps) transcribes it in its original
language. One JSON per video in --out: {"video_id", "duration_s", "has_audio", "segments": [{"start", "end", "text"}]}.
Shard by video with --shard/--nshards. Videos without an audio stream get has_audio=false and no segments.

  python scripts/video/transcribe.py --videos data/video/lvbench/videos.json --out data/video/lvbench_asr \
      --ffmpeg "$(pixi exec -s ffmpeg sh -c 'command -v ffmpeg')" --shard 0 --nshards 4
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time
import wave

import numpy as np

SR = 16000


def extract(ffmpeg, mp4, wav):
    r = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", mp4, "-vn", "-ac", "1", "-ar", str(SR),
                        "-c:a", "pcm_s16le", wav], capture_output=True, text=True)
    if r.returncode != 0:
        if "does not contain any stream" in r.stderr or "Output file is empty" in r.stderr or "no audio" in r.stderr.lower():
            return False
        raise RuntimeError(f"ffmpeg failed on {mp4}: {r.stderr.strip()[:400]}")
    return pathlib.Path(wav).stat().st_size > 1000


def load_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return a.astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data/video/lvbench/videos.json")
    ap.add_argument("--out", default="data/video/lvbench_asr")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--chunk_s", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    vids = sorted(json.load(open(args.videos)).items())[args.shard::args.nshards][: args.limit]
    out = pathlib.Path(args.out)
    (out / "_wav").mkdir(parents=True, exist_ok=True)
    todo = [(v, p) for v, p in vids if not (out / f"{v}.json").exists()]
    print(f"[asr] shard {args.shard}/{args.nshards}: {len(vids)} videos, {len(todo)} to do", flush=True)
    if not todo:
        return 0
    import torch
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=args.model, dtype=torch.float16, device="cuda")
    for v, mp4 in todo:
        t = time.perf_counter()
        wav = str(out / "_wav" / f"{v}.wav")
        has_audio = extract(args.ffmpeg, mp4, wav)
        rec = {"video_id": v, "has_audio": has_audio, "segments": [], "model": args.model}
        if has_audio:
            audio = load_wav(wav)
            rec["duration_s"] = len(audio) / SR
            res = asr({"raw": audio, "sampling_rate": SR}, chunk_length_s=args.chunk_s, batch_size=args.batch,
                      return_timestamps=True, generate_kwargs={"task": "transcribe"})
            last = 0.0
            for c in res.get("chunks", []):
                s, e = c.get("timestamp", (None, None))
                s = last if s is None else float(s)
                e = s if e is None else float(e)
                text = (c.get("text") or "").strip()
                if text:
                    rec["segments"].append({"start": s, "end": e, "text": text})
                last = e
            pathlib.Path(wav).unlink(missing_ok=True)
        json.dump(rec, open(out / f"{v}.json", "w"), ensure_ascii=False)
        print(f"[asr] {v}: audio={has_audio} {rec.get('duration_s', 0):.0f}s {len(rec['segments'])} segments "
              f"{sum(len(s['text']) for s in rec['segments'])} chars  {time.perf_counter() - t:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
