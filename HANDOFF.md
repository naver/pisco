# HANDOFF — PISCO

_Updated 2026-07-17 by NCP (merging NLE's 07-14 handoff — their sections kept below)._
_Full dated science record: `EXPERIMENTS.md` (`expmon ledger`)._

## Goal
Map PISCO's quality levers (both clusters), characterize inference speed (NLE), and close the
remaining factorial cells. The lever question is ANSWERED; two open items remain (one per cluster).

## Shared conclusions (both clusters converge)
- **Finetuning diet is THE lever.** Multitask mix beats KILT-only by +3.4–3.5 LongBench
  (NLE: 35.7; NCP replication on its own backbones: 35.5). Costs on NCP's other axes:
  −1 RAG mean M, agent-QA collapse .64→.31 (~60% = dropped numeric IDs, abstractive style).
  cml128 > cml256 even for doc-256 backbones.
- **Both data axes saturate.** Pretrain volume null (NLE: 500K≈1M≈2M; NCP: 1M≈2M≈5M at rate8
  doc256 — RAG .573/.577/.573, LB 32.0/32.0/31.7). Finetune 50K≈100K.
- **Pretrain objective null too (NCP, new).** `MultiTaskPretrainingCollator`
  (cloze/multidoc/midmem/noisyAE, ef40457) ≈ baseline after identical kilt-100K ft
  (RAG .5731, LB 31.9, agentQA .468). The ~100K ft erases pretrain differences.
- **Verbosity, not incorrectness (NLE):** LLM-judge (gpt-4.1) — compressed PISCO ties
  uncompressed semantically (~0.58); F1/EM gap is verbosity. r8-mt ties uncompressed-9B on
  LongBench (p=.89). Framing: ~10–14× compression, quality preserved.
- **Benchmark fix (NCP):** `my_qa.json` had 9 stale-state golds → use **`my_qa_fixed.json`**
  (gitignored, exists on NCP only). All agent-QA numbers before 2026-07-15 are on buggy labels
  (incl. the 0.479 ministral ref).

## Open item 1 — NLE: evaluate the two done finetunes
- `167200` = 4B→9B ft on multitask_v1 (decoder-size test) — `expQ/167200/model` saved, NOT evaluated.
- `167198` = cml=512 rate-8 ft on multitask_50k (compression-span test) — saved, NOT evaluated.
```
sbatch launchers/eval_longbench.sh 167200    # vs 158433 (4B→4B mt r16 = 33.9)
sbatch launchers/eval_longbench.sh 167198    # vs 161788 (4B→4B mt r8  = 35.7)
```

## Open item 2 — NCP: finish the mtPT × mt-50K factorial cell
- FULLY QUEUED (campaign `mtPTxmtFT`, all afterok-chained): `145643` ft (mtPT-1M × multitask-50K
  cml128) → `145644` LongBench + `145680` agent-QA (my_qa_fixed) + `145681` bergen.
- NEXT STEP on NCP: when all four are done, read the results and fill the last master-grid row —
  LB `expQ/ft_mt_4b_50k_mt1M_cml128_145643/eval/results_lb_*.json`,
  agentQA `outputs/ft_mt_4b_50k_mt1M_cml128_145643_fixedqa.json`,
  bergen `~/bergen_eval/expPISCO/ftmtpt_1M_*/eval_dev_metrics.json`.
  If the cell is flat vs `1M/mt-50K` (LB 34.2), mtPT is conclusively closed.
- **Queued idea (user-endorsed):** rebalanced ft mix on the 2M backbone
  (`kilt:60,…,wikisum:6,dialogsum:2,samsum:2`, total 50K) — target LB ≥34 AND agentQA ≥0.5;
  build with `scripts/build_arc_mix.py`, launch `qwen35_4b_ft_mt.sh <2M abs path> <mix> 50000 128 rebal_2M_cml128`.

## Key checkpoints
- NLE: `161788` (4B→4B mt r8, LB 35.7 best), `158433` (mt r16 33.9), `149907`/`167200` (4B→9B),
  `149314` (4B→4B bidi 500K pretrain).
- NCP: `expQ/qwen35_4b_pt_{1M_143229,2M_143230,5M_143228,mt_1M_144649}/model` (pretrains);
  best per-axis fts: `ft_100K_4b_5M_144096` (agentQA .638), `ft_mt_4b_50k_5M_cml128_144664`
  (LB 35.5), `ft_100K_4b_2M_144097` (RAG .5773). Recipe going forward: 2M pretrain is the
  cost/quality sweet spot for future retrains.

## Key files & commands
- Science record: `EXPERIMENTS.md` (`expmon note`/`ledger`). Recipe: `scripts/specs/multitask_100k.txt`.
- NCP eval launchers (repo root, gitignored): `eval_longbench_4b.sh <expQ-run-id>`,
  `eval_agent_4b.sh` (env `DATA`/`SUFFIX`), `eval_bergen_4b.sh <abs ckpt> <prefix>`;
  ft: `qwen35_4b_ft.sh` (kilt-100K), `qwen35_4b_ft_mt.sh <backbone> <data> <samples> <cml> <tag>`.
  **Backbone paths must be ABSOLUTE** (hydra chdir).
- NCP bergen: `~/bergen_eval` (naver/bergen `pisco-eval`); data `/nfs/data/calmar/rv2/rag/*`;
  6/7 datasets (no 2wikimultihopqa); top_200 rerank except kilt_nq top_50.
- NCP multitask data: `/beegfs/scratch/user/hdejean/arc_ft_data/*` (md5-verified).
- NLE speed tooling: `scripts/bench_throughput*.py`, `scripts/profile_decode.py`,
  `scripts/llm_judge.py` (`$OPENAI_API_KEY`), `scripts/sig_test.py`.
- Results tables: `expmon results` (LongBench grid); raw: `expQ/*/eval/results_lb_*.json`,
  `outputs/*_fixedqa.json`, `bergen_eval/expPISCO/*/eval_dev_metrics.json` (NCP paths).

## Don't-break list
- `*.sh` launchers are GITIGNORED; durable configs → `scripts/` or `EXPERIMENTS.md`.
- `mask_before_mem` is a no-op in base `PretrainingCollator` — do NOT "fix" mid-campaign
  (only `MultiTaskPretrainingCollator` uses the corrected order).
- `regen/*.jsonl` are stochastic & local — exact files from private HF `Herve/pisco-multitask-regen`.
- NCP cluster (2026-07-17): `sbatch --constraint=...` BROKEN mid-upgrade — constraint lines
  commented out of NCP launchers; don't re-add yet.
- NLE expmon daemon 163132 may still be running on NLE (`expmon kill 163132` when done).
- Decoder LoRA unmerged at eval by default — `--merge_lora` for deployable-speed numbers.
- Agent-QA numbers pre-2026-07-15 are on buggy labels — don't mix with `my_qa_fixed.json` numbers.
