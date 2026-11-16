"""Markdown data-audit table (docs/longdoc_diagnosis.md part 3) from an arm-A audit.json. Stdlib only.
  python scripts/longdoc/audit_table.py expQ/longdoc/audit_<job>/audit.json
"""
import json, sys

a = json.load(open(sys.argv[1]))
m = a["meta"]
f = lambda d: f"{d['p50']} / {d['p99']} / {d['max']}"
print(f"Audit {sys.argv[1]}: R = ft_mm_4b_184778, PX {m['PX']}, rate 8, stride {m['stride']}; finetune max "
      f"{m['finetune_max_mem']} `<MEM>`s, position span {m['finetune_max_span']}; {m['n_errors']} row errors; "
      f"health: 4-page documents total {a['health']['docs_p4_mems_total']:,} `<MEM>`s (expected {a['health']['expected']:,}).\n")
print("| input | n | gating | `<MEM>`s per question p50 / p99 / max | tokens per chunk p50 / max | decoder prompt tokens p50 / p99 / max "
      "| decoder position span p50 / p99 / max | uncompressed prompt tokens p99 / max | cut rate ↓ (count) | gold page past TOPK | "
      "parity ↑ | beyond finetune range ↓ |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for v, s in a["variants"].items():
    print(f"| {v} ({s['set']}, TOPK {s['topk']}, XD {s['xd']}, CML {s['cml']}{', gold only' if s['gold'] else ''}) | {s['n']} "
          f"| {'yes' if s['gating'] else 'earlier claim'} | {f(s['mems'])} | {s['chunk_tokens']['p50']} / {s['chunk_tokens']['max']} "
          f"| {f(s['prompt'])} | {f(s['span'])} | {s['base_prompt']['p99']} / {s['base_prompt']['max']} "
          f"| {s['cut_rate']:.4f} ({s['cut_pisco']}) | {s['gold_dropped']} | {s['parity']:.4f} | {s.get('out_of_range_share', float('nan')):.4f} |")
for v, s in a["stages"].items():
    print(f"| training: {v} | {s['n']} | — | {f(s['mems'])} | {s['chunk_tokens']['p50']} / {s['chunk_tokens']['max']} "
          f"| {f(s['prompt'])} (incl. target) | {f(s['span'])} | — | — | — | — | — |")
