"""When does compression stop paying? Reads the vLLM generated-length sweep (scripts/bench_vllm_sim.py, SUF=_glen) and,
per (K images, N concurrent requests), fits wall time T(G) = a + b*G for the raw-length and the PISCO-length prompt:
  a = prompt reading (prefill) + fixed overhead, b = cost per generated token for the whole batch.
Speedup S(G) = T_raw(G) / T_pisco(G) falls from a_raw/a_pisco (short answers) toward b_raw/b_pisco (long answers).
Reports measured S at each G, the fitted floor, and the answer length where S drops below 1.5x and 1.2x.

  python3 scripts/show_vllm_glen.py expQ/bench/vllm/nav_sim_glen.json [expQ/bench/vllm/doc_sim_glen.json]
"""
import json
import os
import sys

GMIN = int(os.environ.get("GMIN", "8"))   # fit only on G >= GMIN (G=1 is mostly fixed overhead)


def fit(points):
    n = len(points)
    mx = sum(g for g, _ in points) / n
    my = sum(t for _, t in points) / n
    sxx = sum((g - mx) ** 2 for g, _ in points)
    b = sum((g - mx) * (t - my) for g, t in points) / sxx
    return my - b * mx, b


def crossing(ar, br, ap, bp, thr):
    """smallest G >= 0 with (ar + br G) / (ap + bp G) <= thr, or None if the floor stays above thr."""
    if br / bp > thr:
        return None
    if ar / ap <= thr:
        return 0.0
    return (ar - thr * ap) / (thr * bp - br)


def main():
    for path in sys.argv[1:]:
        d = json.load(open(path))
        rows = {(r["arm"], r["K"], r["N"], r["G"]): r for r in d["results"]}
        Ks = sorted({k for _, k, _, _ in rows}); Ns = sorted({n for _, _, n, _ in rows}); Gs = sorted({g for *_, g in rows})
        print(f"\n=== {path}   prompt tokens raw {d['lengths']['raw']}  pisco {d['lengths']['pisco']}")
        print("speedup raw/pisco at G generated tokens  | fitted: floor (long answers), G where speedup < 1.5x, < 1.2x;"
              " per-token ms/request raw vs pisco")
        print("%3s %5s | %s | %6s %8s %8s | %s" % ("K", "N", " ".join("G=%-5d" % g for g in Gs), "floor", "G@1.5x", "G@1.2x",
                                                   "step ms raw/pisco"))
        for K in Ks:
            for N in Ns:
                S = []
                pts = {"raw": [], "pisco": []}
                for G in Gs:
                    r, p = rows.get(("raw", K, N, G)), rows.get(("pisco", K, N, G))
                    S.append(r["wall_s"] / p["wall_s"] if r and p else None)
                    for arm, x in (("raw", r), ("pisco", p)):
                        if x and G >= GMIN:
                            pts[arm].append((G, x["wall_s"]))
                if len(pts["raw"]) < 2 or len(pts["pisco"]) < 2:
                    continue
                ar, br = fit(pts["raw"]); ap, bp = fit(pts["pisco"])
                c15, c12 = crossing(ar, br, ap, bp, 1.5), crossing(ar, br, ap, bp, 1.2)
                fmt = lambda c: "never" if c is None else "%d" % round(c)  # noqa: E731
                print("%3d %5d | %s | %5.2fx %8s %8s | %.1f / %.1f" % (
                    K, N, " ".join("%-7s" % ("%.1fx" % s if s else "-") for s in S), br / bp, fmt(c15), fmt(c12),
                    1000 * br / N, 1000 * bp / N))


if __name__ == "__main__":
    main()
