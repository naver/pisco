#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu,gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/build_msmarco_docs_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/build_msmarco_docs_%j.out
# Campaign longchunk, data: MS MARCO document collection (3.2M web pages, TREC format) -> an HF `text` dataset for
# stage 1 (data.training_dataset=<OUT>), shuffled once (seed 42), plus the long-chunk data check on a 100k sample
# (Qwen3.5-4B tokens per document: share of documents >= 256 / 512 / 1,024 / 2,048 / 4,096 tokens, chunk fill).
# text = title + body (the URL line is dropped). Usage: build_msmarco_docs.sbatch [OUT=data/msmarco_docs] [SAMPLE_EVERY=30]
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=true
OUT=${1:-/beegfs/scratch/user/hdejean/pisco/data/msmarco_docs}; EVERY=${2:-30}
SRC=/beegfs/scratch/user/hdejean/.ir_datasets/msmarco-document/collection.tsv.gz
echo "FACTORS kind=data_build dataset=msmarco_docs every=$EVERY tokenizer=qwen3.5-4b"
env CONDA_OVERRIDE_CUDA=12.7 pixi run python - "$SRC" "$OUT" "$EVERY" "/beegfs/scratch/user/hdejean/pisco/expQ/longchunk/data_check_msmarco_${SLURM_JOB_ID}.json" <<'PY'
import gzip, json, sys, time, collections
import numpy as np
import datasets
from transformers import AutoTokenizer
src, out, every, stats_out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
t0 = time.time()

def docs():
    """TREC format: <DOC><DOCNO>id</DOCNO><TEXT>url\ntitle\nbody...</TEXT></DOC>."""
    with gzip.open(src, "rt", encoding="utf-8", errors="replace") as fh:
        docno, intext, lines = None, False, []
        for line in fh:
            if line.startswith("<DOCNO>"):
                docno = line[7:line.find("</DOCNO>")]
            elif line.startswith("<TEXT>"):
                intext, lines = True, []
            elif line.startswith("</TEXT>"):
                intext = False
                body = lines[1:] if lines and lines[0].startswith("http") else lines   # drop the URL line
                text = "".join(body).strip()
                if text:
                    yield {"id": docno, "text": text}
            elif intext:
                lines.append(line)

ds = datasets.Dataset.from_generator(docs, cache_dir="/beegfs/scratch/user/hdejean/pisco/data/.hf_cache")
print(f"parsed {len(ds)} documents ({time.time()-t0:.0f}s)", flush=True)
ds = ds.shuffle(seed=42).flatten_indices()
ds.save_to_disk(out)
print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
idx = list(range(0, len(ds), every))
texts = ds.select(idx)["text"]
lens = []
B = 1000
for i in range(0, len(texts), B):
    enc = tok(texts[i:i+B], add_special_tokens=False, truncation=True, max_length=16384, return_length=True)
    lens += list(enc["length"])
lens = np.array(lens)
print(f"tokenized {len(lens)} sampled documents ({time.time()-t0:.0f}s)", flush=True)
def stats(a):
    a = np.asarray(a); tot = a.sum()
    d = {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
         "p10": float(np.percentile(a, 10)), "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90)),
         "note": "lengths capped at 16,384 tokens"}
    for L in (256, 512, 1024, 2048, 4096):
        d[f"frac_docs_ge_{L}"] = float((a >= L).mean())
        d[f"frac_tokens_in_docs_ge_{L}"] = float(a[a >= L].sum() / tot)
        d[f"mean_fill_first_chunk_{L}"] = float(np.minimum(a, L).mean() / L)
        d[f"mean_tokens_kept_cap_{L}"] = float(np.minimum(a, L).mean())
    return d
res = {"dataset": out, "n_docs_total": len(ds), "every": every, "all": stats(lens)}
json.dump(res, open(stats_out, "w"), indent=1)
a = res["all"]
print("\n=== LONG-CHUNK DATA CHECK, MS MARCO documents (Qwen3.5-4B tokens per document) ===")
print(f"documents in the set {len(ds)}; sampled {a['n']}  mean {a['mean']:.0f}  median {a['median']:.0f}  p25 {a['p25']:.0f}  p75 {a['p75']:.0f}  p90 {a['p90']:.0f}")
print("chunk L | docs >= L | tokens in docs >= L | mean fill of one L-chunk | tokens kept per doc at cap L")
for L in (256, 512, 1024, 2048, 4096):
    print(f"{L:7d} | {a[f'frac_docs_ge_{L}']:.3f} | {a[f'frac_tokens_in_docs_ge_{L}']:.3f} | {a[f'mean_fill_first_chunk_{L}']:.3f} | {a[f'mean_tokens_kept_cap_{L}']:.0f}")
print("RESULT", stats_out)
PY
echo "=== EXIT CODE $? ==="
