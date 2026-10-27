"""When does compressing images pay off? Slide-ready figures from the vLLM answer-length sweeps (scripts/bench_vllm_sim.py).

Every point is measured (plain Qwen3.5-4B in vLLM, random prompts at the raw and at the compressed length; each
request has its own prompt, nothing cached). Speed-up = raw wall time / compressed wall time.
  default   8 pages: A = one request, one line per compression rate; B = many requests at once, x8
  --single  panel A only
  --pages   grid: columns 1 / 8 / 16 pages, rows = A and B
Data: LENS_JSON (bench_vllm_sim_4b.sh with LENS=..., median of 3 except 16 pages batched) and, for 8 pages batched,
the glen sweep (DOC).

  python scripts/plot_compression_payoff.py [--single | --pages] [--out docs/figures/compression_payoff]
"""
import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

V = "expQ/bench/vllm/"
DOC = V + "doc_sim_glen_partial.json"                    # arms raw/pisco (x8), K = 4/8/16 pages, N = 1/16/64/256
LENS_JSON = {1: [V + "lens_sim_doc1_rates.json"],        # arms raw, x4, x8, x16, x32
             8: [V + "lens_sim_doc8_rates.json"],
             16: [V + "lens_sim_doc16_rates.json", V + "lens_sim_doc16_rates_batch.json"]}
IMG_TOK = 434
RATES = [4, 8, 16, 32]
NREQ = [1, 16, 64, 256]
GS = [1, 8, 32, 128, 512, 2048]
GWORDS = ["yes/no", "a few words", "a sentence", "a paragraph", "a page", "4 pages"]
GSHORT = ["yes/no", "words", "sentence", "paragraph", "page", "4 pages"]
BLUE = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]       # ordinal ramp, validated (dataviz validator --ordinal)
ORANGE = ["#ef9a74", "#eb6834", "#b4471b", "#7a2c0c"]
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
THR = 1.2
LABEL_NUDGE = {32: 0.17, 16: -0.17}                       # 8 pages: x16 and x32 start 0.1x apart


def rows(path):
    return {(r["arm"], r["K"], r["N"], r["G"]): r["wall_s"] for r in json.load(open(path))["results"]}


def lens(pages):
    m = {}
    for p in LENS_JSON[pages]:
        m.update(rows(p))
    return m


def raw_tokens(pages):
    return json.load(open(LENS_JSON[pages][0]))["lengths"]["raw"][str(pages)]


def by_rate(pages):
    """{rate: speed-up at each G}, one request."""
    m = lens(pages)
    return {r: [m[("raw", pages, 1, G)] / m[(f"x{r}", pages, 1, G)] for G in GS] for r in RATES}


def by_requests(pages):
    """{N: speed-up at each G} at x8; from the LENS runs when they have N, else from the glen sweep."""
    m, d = lens(pages), rows(DOC)
    out = {}
    for n in NREQ:
        if ("raw", pages, n, GS[0]) in m:
            out[n] = [m[("raw", pages, n, G)] / m[("x8", pages, n, G)] for G in GS]
        else:
            out[n] = [d[("raw", pages, n, G)] / d[("pisco", pages, n, G)] for G in GS]
    return out


def crossing(S):
    """answer length where the speed-up falls under THR (log-linear between measured points), or None."""
    for (g0, s0), (g1, s1) in zip(zip(GS, S), zip(GS[1:], S[1:])):
        if s0 >= THR > s1:
            return 2 ** (math.log2(g0) + (s0 - THR) / (s0 - s1) * (math.log2(g1) - math.log2(g0)))
    return None


def style(ax, title, yticks, words=GWORDS, band_text=True, xlabel=True, fs=1.0):
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.5, 2900)
    ax.set_ylim(0.8, yticks[-1] + 0.2)
    ax.set_xticks(GS)
    ax.set_xticklabels([f"{g}\n{w}" for g, w in zip(GS, words)] if words else [str(g) for g in GS],
                       fontsize=8.5 * fs, color=INK2)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y}×" for y in yticks], fontsize=9 * fs, color=INK2)
    ax.minorticks_off()
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, pad=6)
    ax.axhspan(0.8, THR, color="#f0efec", zorder=0)
    ax.axhline(1, color=AXIS, lw=1, zorder=1)
    if band_text:
        ax.text(0.56, 1.0, f"under {THR}×: not worth it", ha="left", va="center", fontsize=8.5 * fs, color=MUTED)
    if xlabel:
        ax.set_xlabel("Answer length (generated tokens)", fontsize=10 * fs, color=INK2, labelpad=6)
    if title:
        ax.set_title(title, loc="left", fontsize=11.5 * fs, color=INK, pad=10)


def plot_lines(ax, series, colors, legend=True, fs=1.0):
    for (lab, S), c in zip(series, colors):
        ax.plot(GS, S, color=c, lw=2, solid_capstyle="round", solid_joinstyle="round", label=lab, zorder=3)
        ax.scatter(GS, S, s=36, color=c, edgecolors=SURF, linewidths=2, zorder=4)
    if legend:
        ax.legend(loc="upper right", frameon=False, fontsize=8.5 * fs, labelcolor=INK2, handlelength=1.6)


def rate_series(pages):
    S = by_rate(pages)
    for r in RATES:
        c = crossing(S[r])
        print(f"{pages:2d} pages, 1 request, x{r}: " + "  ".join(f"G{G} {s:.2f}x" for G, s in zip(GS, S[r]))
              + (f"  | under {THR}x past {c:.0f} tokens" if c else ""))
    order = RATES[::-1]                                     # legend top-down = lines top-down
    return S, [(f"×{r} compression", S[r]) for r in order], [BLUE[RATES.index(r)] for r in order]


def request_series(pages):
    S = by_requests(pages)
    order = NREQ[::-1]
    return [("1 request" if n == 1 else f"{n} requests at once", S[n]) for n in order], \
        [ORANGE[NREQ.index(n)] for n in order]


def panel_a(ax, title):
    style(ax, title, list(range(1, 8)))
    S, series, colors = rate_series(8)
    plot_lines(ax, series, colors)
    for r in RATES:
        ax.text(0.9, S[r][0] + LABEL_NUDGE.get(r, 0), f"×{r}", ha="right", va="center", fontsize=9, color=INK2)
    ax.set_ylabel("Speed-up vs. reading the raw images", fontsize=10, color=INK2, labelpad=6)
    ax.annotate("Short answers: up to 4.5× faster.\nCompressing more than ×16 adds nothing", xy=(1.1, 4.5),
                xytext=(2.2, 6.3), fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.annotate(f"Between 32 and 128 tokens the gain falls\nunder {THR}×, whatever the compression rate",
                xy=(64, THR), xytext=(40, 3.1), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))


def panel_b(ax):
    style(ax, "B. Many requests at once (×8 compression)", list(range(1, 8)))
    plot_lines(ax, *request_series(8))
    ax.annotate("Many users at once:\nthe gain lasts even\nfor long answers", xy=(2048, 1.95), xytext=(300, 3.6),
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))


def grid(axes):
    yt = [1, 2, 4, 6, 8, 10]
    for j, pages in enumerate((1, 8, 16)):
        top, bot = axes[0][j], axes[1][j]
        style(top, f"{pages} page{'s' if pages > 1 else ''}  ·  {raw_tokens(pages):,} tokens uncompressed", yt,
              words=None, band_text=False, xlabel=False, fs=0.95)
        plot_lines(top, *rate_series(pages)[1:], legend=j == 2, fs=0.95)
        style(bot, None, yt, words=GSHORT, band_text=False, fs=0.95)
        plot_lines(bot, *request_series(pages), legend=j == 2, fs=0.95)
    axes[0][0].set_ylabel("One request at a time\n(speed-up)", fontsize=10.5, color=INK, labelpad=8)
    axes[1][0].set_ylabel("Many requests at once, ×8\n(speed-up)", fontsize=10.5, color=INK, labelpad=8)
    axes[0][0].text(8, 3.2, "1 page: no gain", fontsize=9, color=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/compression_payoff")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--single", action="store_true", help="8 pages, one request only")
    mode.add_argument("--pages", action="store_true", help="grid over 1 / 8 / 16 pages")
    a = ap.parse_args()

    if a.pages:
        fig, axes = plt.subplots(2, 3, figsize=(13.3, 8.0), facecolor=SURF)
    else:
        n = 1 if a.single else 2
        fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 4.9), facecolor=SURF, squeeze=False)
    for ax in fig.axes:
        ax.set_facecolor(SURF)
    if a.pages:
        grid(axes)
        sub = "Compression pays when the input is long and the answer short, or when many requests share the GPU."
        foot = (f"Grey band: under {THR}× faster, not worth it. Compression ×r keeps {IMG_TOK}/r + 1 vectors per page; "
                "the text part (63 tokens) is never compressed. Compression runs once, offline (not counted).")
    else:
        panel_a(axes[0][0], "One request at a time" if a.single else "A. One request at a time")
        if not a.single:
            panel_b(axes[0][1])
        sub = ("Compression pays when the answer is short compared to the input"
               + ("" if a.single else ", or when many requests share the GPU") + ".")
        foot = (f"Input: 8 document pages, {raw_tokens(8):,} prompt tokens uncompressed; compression ×r keeps "
                f"{IMG_TOK}/r + 1 vectors per page. Compression runs once, offline (not counted).")
    top = 0.985
    fig.suptitle("When does compressing images pay off?", x=0.01, y=top, ha="left", va="top", fontsize=14,
                 color=INK, fontweight="bold")
    fig.text(0.01, top - (0.035 if a.pages else 0.06), sub, ha="left", va="top", fontsize=10.5, color=INK2)
    fig.text(0.01, 0.01, "Measured: Qwen3.5-4B in vLLM on one A100; every request has its own prompt, nothing is "
             "cached. " + foot, fontsize=7.5, color=MUTED, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.04 if a.pages else (0.09 if a.single else 0.05), 1, 0.93 if a.pages else 0.91))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{a.out}.{ext}", dpi=200, facecolor=SURF)
    print("wrote", ", ".join(f"{a.out}.{e}" for e in ("png", "pdf", "svg")))


if __name__ == "__main__":
    main()
