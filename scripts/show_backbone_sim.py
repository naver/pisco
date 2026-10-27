"""PISCO speed-up on two backbones from simulated-compression runs (scripts/bench_vllm_sim.py --lens, same lengths).

Each json holds raw and x8 prompt lengths for 8 and 16 doc pages (labels raw8 / x8p8 / raw16 / x8p16). Prints the
speed-up (raw wall time / PISCO wall time) per (pages, concurrent requests, generated tokens) for each backbone, then
requests/s. Stdlib only.

  python3 scripts/show_backbone_sim.py expQ/bench/vllm/lens_sim_archq_Qwen35-4B.json expQ/bench/vllm/lens_sim_archq_Qwen3-4B.json
"""
import json
import sys

runs = [json.load(open(f)) for f in sys.argv[1:]]
tabs = [{(r["arm"], r["N"], r["G"]): r for r in d["results"]} for d in runs]
names = [d["model"].split("/")[-1] for d in runs]
Gs = sorted({k[2] for t in tabs for k in t})
Ns = sorted({k[1] for t in tabs for k in t})
for title, key in (("speed-up PISCO x8 vs raw", "speed"), ("requests/s raw -> PISCO", "rps")):
    print(f"\n{title} ({' / '.join(names)})")
    print("| pages | requests | " + " | ".join(f"{g} tokens" for g in Gs) + " |")
    print("|---|---|" + "---|" * len(Gs))
    for raw, pis, pages in (("raw8", "x8p8", 8), ("raw16", "x8p16", 16)):
        for n in Ns:
            cells = []
            for g in Gs:
                vals = []
                for t in tabs:
                    a, b = t.get((raw, n, g)), t.get((pis, n, g))
                    if not (a and b):
                        vals.append("-")
                    elif key == "speed":
                        vals.append(f"{a['wall_s'] / b['wall_s']:.1f}x")
                    else:
                        vals.append(f"{a['req_per_s']:.1f}->{b['req_per_s']:.1f}")
                cells.append(" / ".join(vals))
            print(f"| {pages} | {n} | " + " | ".join(cells) + " |")
