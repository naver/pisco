"""Length curve for campaign longdoc (arm E), stdlib only (no plotting library in the pixi env): writes an SVG.

x = `<MEM>` slots per question (median of `mems_used` in the R file, log scale), y = judge accuracy (rubric, higher is
better). One colour per modality; solid = compressed R (mean of seeds 42 and 43, each seed as a dot), dashed =
uncompressed on the same questions. Text buckets come from the `source` field of the whole-set files.
Usage: python3 scripts/longdoc/plot_length.py [OUT.svg]
"""
import collections, json, math, statistics, sys

P = "mm_docqa_pisco_k1000_px1003520_cml1024_mnt512_longdoc_"
B = "expQ/base_qwen35_4b/eval/mm_docqa_base_k1000_px1003520_cml1024_mnt512_longdoc_"
SETS = {  # modality -> list of (file suffix, source/bucket, label)
    "page images": [("pages_1500_p10", None, "10 p"), ("pages_1500_p20", None, "20 p"), ("pages_1500_p40", None, "40 p")],
    "mixed (image or text layer)": [("mixed_1500_m10", None, "10"), ("mixed_1500_m20", None, "20"), ("mixed_1500_m40", None, "40")],
    "text layer": [("text_1500", f"longdocurl_{b}", b) for b in ("t4k", "t8k", "t16k", "t32k")],
}
COL = {"page images": "#3b6fb6", "mixed (image or text layer)": "#c0612b", "text layer": "#3a8f5c"}


def score(path, src):
    r = json.load(open(path))["records"]
    if src:
        r = [x for x in r if x["source"] == src]
    return statistics.mean((x.get("judge") or 0) for x in r), r


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "expQ/longdoc/figures/length_curve.svg"
    pts = collections.defaultdict(list)  # modality -> (x, r42, r43, u, label)
    for mod, cells in SETS.items():
        for suf, src, lab in cells:
            r42, rows = score(f"expQ/longdoc/e_r42/eval/{P}{suf}.json", src)
            r43, _ = score(f"expQ/longdoc/e_r43/eval/{P}{suf}.json", src)
            u, _ = score(f"{B}{suf}.json", src)
            pts[mod].append((statistics.median(x["mems_used"] for x in rows), r42, r43, u, lab))
    W, H, L, R, T, Bm = 760, 470, 70, 230, 50, 60
    xs = [p[0] for v in pts.values() for p in v]
    lx0, lx1 = math.log2(min(xs) / 1.3), math.log2(max(xs) * 1.3)
    y0, y1 = 0.30, 0.72
    X = lambda x: L + (math.log2(x) - lx0) / (lx1 - lx0) * (W - L - R)
    Y = lambda y: T + (y1 - y) / (y1 - y0) * (H - T - Bm)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{L}" y="24" font-size="15">Long documents (LongDocURL, arm E): judge accuracy vs slots per question</text>']
    for k in range(7):
        y = y0 + k * 0.07
        s.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(y):.1f}" y2="{Y(y):.1f}" stroke="#e5e5e5"/>'
                 f'<text x="{L-8}" y="{Y(y)+4:.1f}" text-anchor="end">{y:.2f}</text>')
    for x in (512, 1024, 2048, 4096):
        s.append(f'<line x1="{X(x):.1f}" x2="{X(x):.1f}" y1="{T}" y2="{H-Bm}" stroke="#e5e5e5"/>'
                 f'<text x="{X(x):.1f}" y="{H-Bm+16}" text-anchor="middle">{x}</text>')
    s.append(f'<text x="{(L+W-R)/2}" y="{H-18}" text-anchor="middle">&lt;MEM&gt; slots per question (median, log scale; 8 tokens per slot)</text>')
    s.append(f'<text x="18" y="{(T+H-Bm)/2}" transform="rotate(-90 18 {(T+H-Bm)/2})" text-anchor="middle">judge accuracy (higher is better)</text>')
    ly = T + 10
    for mod, v in pts.items():
        c = COL[mod]
        for dash, f in (("", lambda p: (p[1] + p[2]) / 2), ("6,4", lambda p: p[3])):
            s.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.2" stroke-dasharray="{dash}" points="'
                     + " ".join(f"{X(p[0]):.1f},{Y(f(p)):.1f}" for p in v) + '"/>')
        for p in v:
            for yy in (p[1], p[2]):
                s.append(f'<circle cx="{X(p[0]):.1f}" cy="{Y(yy):.1f}" r="2.6" fill="{c}"/>')
            s.append(f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[3]):.1f}" r="3" fill="white" stroke="{c}" stroke-width="1.5"/>')
            s.append(f'<text x="{X(p[0]):.1f}" y="{Y(min(p[1], p[2]))+16:.1f}" text-anchor="middle" fill="{c}" font-size="10">{p[4]}</text>')
        s.append(f'<line x1="{W-R+15}" x2="{W-R+45}" y1="{ly}" y2="{ly}" stroke="{c}" stroke-width="2.2"/>'
                 f'<text x="{W-R+52}" y="{ly+4}">{mod}</text>')
        ly += 20
    s.append(f'<line x1="{W-R+15}" x2="{W-R+45}" y1="{ly+6}" y2="{ly+6}" stroke="#555" stroke-width="2.2"/>'
             f'<text x="{W-R+52}" y="{ly+10}">compressed R (dots: s42, s43)</text>')
    s.append(f'<line x1="{W-R+15}" x2="{W-R+45}" y1="{ly+26}" y2="{ly+26}" stroke="#555" stroke-width="2.2" stroke-dasharray="6,4"/>'
             f'<text x="{W-R+52}" y="{ly+30}">uncompressed, same questions</text>')
    s.append("</svg>")
    open(out, "w").write("\n".join(s))
    for mod, v in pts.items():
        print(mod, [(int(p[0]), round(p[1], 3), round(p[2], 3), round(p[3], 3)) for p in v])
    print("wrote", out)


if __name__ == "__main__":
    main()
