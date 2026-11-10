#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/data_check_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/data_check_%j.out
# Long-chunk data check: are the stage-1 pretraining texts (EleutherAI/SmolLM2-135M-10B, first 5M rows as the
# 5M stage-1 run used them) long enough to fill compressor chunks of 1,024 / 2,048 tokens?
# Usage: data_check.sbatch [ROWS_TOTAL=5000000] [SAMPLE_EVERY=50]
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=true
ROWS=${1:-5000000}; EVERY=${2:-50}
echo "FACTORS kind=data_check dataset=smollm2_10b rows=$ROWS every=$EVERY tokenizer=qwen3.5-4b"
OUT=/beegfs/scratch/user/hdejean/pisco/expQ/longchunk/data_check_${SLURM_JOB_ID}.json
env CONDA_OVERRIDE_CUDA=12.7 pixi run python - "$ROWS" "$EVERY" "$OUT" <<'PY'
import glob, json, sys, time, collections
import numpy as np, pyarrow.parquet as pq
from transformers import AutoTokenizer
rows_total, every, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
snap = glob.glob("/beegfs/scratch/user/hdejean/Reranking/cache/hub/datasets--EleutherAI--SmolLM2-135M-10B/snapshots/*/data/train-*.parquet")
snap.sort()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
texts, sources, seen = [], [], 0
t0 = time.time()
for f in snap:                      # shards are in row order: the first ROWS rows = what data.samples=ROWS selects
    t = pq.read_table(f, columns=["text", "source"])
    n = t.num_rows
    idx = [i for i in range(0, n) if (seen + i) % every == 0 and seen + i < rows_total]
    if idx:
        texts += t.column("text").take(idx).to_pylist(); sources += t.column("source").take(idx).to_pylist()
    seen += n
    if seen >= rows_total: break
print(f"read {len(texts)} sampled rows out of the first {min(seen, rows_total)} ({time.time()-t0:.0f}s, {len(snap)} shards available)", flush=True)
lens = []
B = 2000
for i in range(0, len(texts), B):
    enc = tok(texts[i:i+B], add_special_tokens=False, return_length=True)
    lens += list(enc["length"])
lens = np.array(lens); src = np.array(sources)
print(f"tokenized ({time.time()-t0:.0f}s)", flush=True)
def stats(a):
    a = np.asarray(a); tot = a.sum()
    d = {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
         "p10": float(np.percentile(a, 10)), "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90))}
    for L in (256, 512, 1024, 2048, 4096):
        d[f"frac_docs_ge_{L}"] = float((a >= L).mean())
        d[f"frac_tokens_in_docs_ge_{L}"] = float(a[a >= L].sum() / tot)
        d[f"mean_fill_first_chunk_{L}"] = float(np.minimum(a, L).mean() / L)   # how full one chunk of L is, on average
        d[f"mean_tokens_kept_cap_{L}"] = float(np.minimum(a, L).mean())         # tokens per doc after truncation to L
    return d
res = {"rows_total": rows_total, "every": every, "all": stats(lens), "by_source": {}}
for s, c in collections.Counter(sources).most_common():
    res["by_source"][s] = dict(share=c / len(sources), **stats(lens[src == s]))
json.dump(res, open(out, "w"), indent=1)
a = res["all"]
print("\n=== LONG-CHUNK DATA CHECK (Qwen3.5-4B tokens per document) ===")
print(f"docs sampled {a['n']}  mean {a['mean']:.0f}  median {a['median']:.0f}  p25 {a['p25']:.0f}  p75 {a['p75']:.0f}  p90 {a['p90']:.0f}")
print("chunk L | docs >= L | tokens in docs >= L | mean fill of one L-chunk | tokens kept per doc at cap L")
for L in (256, 512, 1024, 2048, 4096):
    print(f"{L:7d} | {a[f'frac_docs_ge_{L}']:.3f} | {a[f'frac_tokens_in_docs_ge_{L}']:.3f} | {a[f'mean_fill_first_chunk_{L}']:.3f} | {a[f'mean_tokens_kept_cap_{L}']:.0f}")
print("\nby source (share, median tokens, docs>=1024, docs>=2048):")
for s, d in res["by_source"].items():
    print(f"  {s:28s} {d['share']:.3f}  {d['median']:6.0f}  {d['frac_docs_ge_1024']:.3f}  {d['frac_docs_ge_2048']:.3f}")
print("RESULT", out)
PY
echo "=== EXIT CODE $? ==="
