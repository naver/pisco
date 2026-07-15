# HANDOFF — PISCO

_Updated 2026-07-14._ (Full dated science record is in `EXPERIMENTS.md`.)

## Goal
Map PISCO's quality levers, characterize its inference-speed behavior, and support NCP in
reproducing the multitask finetune recipe.

## Status

### A. Quality campaign — COMPLETE (numbers in EXPERIMENTS.md)
- **Finetuning recipe is THE lever.** Multitask mix beats KILT-only by **+3.4 LongBench**.
  `rate-8` helps LongBench (+1.5–1.8, but sig **only on KILT**, p=0.03; multitask +1.66 is
  n.s. p=0.07), ~0 on RAG. **Both data axes saturate:** pretrain 500K≈1M≈2M; finetune
  50K≈100K (33.9), 200K=33.0.
- **LLM-judge (gpt-4.1, `scripts/llm_judge.py`):** compressed PISCO **ties uncompressed**
  semantically (~0.58); the big F1/EM gap is *verbosity*, not incorrectness.
- **Significance (`scripts/sig_test.py`, paired):** PISCO r8-mt **ties uncompressed-9B** on
  LongBench (p=0.89) and **ties uncompressed-4B** on RAG-M (p=0.52); loses RAG-F1 (p<1e-4).
- **Framing (agreed):** PISCO's achievement = **~10–14× compression, quality preserved**
  (semantic-judge + LongBench). "Any-length" is a *separate, optional* extension, not a flaw.

### B. Two finetunes DONE but NOT EVALUATED  ← main open item
- **`167200`** = 4B→9B ft on multitask_v1 (decoder-size test; matches 158433 except 4B→9B).
- **`167198`** = cml=512 rate-8 ft on multitask_50k (compression-span test; matches 161788
  except cml 128→512).
- Both have `expQ/<id>/model` saved. **Neither is evaluated yet.**

### C. NCP multitask recipe — DONE (committed + synced)
- Recovered spec + ft hyperparams from ground truth → **pisco `arc` 896e392**:
  `scripts/specs/multitask_100k.txt` + `build_arc_mix.py --zip/--regen_dir`.
- 193 MB `regen/*.jsonl` uploaded to **private HF `Herve/pisco-multitask-regen`**.
- Replied to NCP on `~/.claude` SYNC.md `main` (c3df327 / 5c14d5b).

### D. Speed investigation — COMPLETE (PISCO is NOT slow)
- Efficiency = **throughput + max-batch**, ~1× at batch=1, growing with batch/context/compression.
- Reproduced the paper's regime: 9B/rate16 → **3.66× @ bs256** (synthetic 5×128) and **2.59×
  @ bs256** (real RAG), and PISCO fits **≥2–4× the batch** (uncompressed OOMs).
- **Decode profile (`167291`):** attention (the only part PISCO cuts) is **~12%**; ~66% is
  framework overhead at bs=1; LM-head/vocab is **2–7% (negligible)**. Ratio levers = **long
  context + long output + high batch**.

## Next step (do first)
Evaluate the two done finetunes, then compare:
```
sbatch launchers/eval_longbench.sh 167200    # 4B→9B multitask  → vs 158433 (4B→4B mt = 33.9)
sbatch launchers/eval_longbench.sh 167198    # cml=512 rate8    → vs 161788 (cml=128 r8 = 35.7)
```
Read qa f1 from `expQ/<id>/eval/results_lb_*.json` (`metrics.f1`×100). Answers: does a bigger
decoder help at the best recipe (167200)? does a bigger compression span help at rate-8
(167198)? Optionally RAG-eval both via bergen.

## Key files & commands
- **Science record:** `EXPERIMENTS.md` (`expmon ledger` / `expmon note`).
- **Recipe (committed):** `scripts/specs/multitask_100k.txt`; `scripts/build_arc_mix.py`.
- **Quality eval:** `launchers/eval_longbench.sh <jobid>`; `scripts/eval_agent_qa.py`
  (now has timing, `--merge_lora`, `--min_new_tokens`). Uncompressed baseline needs a large
  `--decoder_max_length` (20000); PISCO uses 2048.
- **Speed tooling:** `scripts/bench_throughput.py`, `scripts/bench_throughput_rag.py`,
  `scripts/profile_decode.py` (+ their `launchers/*.sh`).
- **LLM-judge / sig:** `scripts/llm_judge.py` (OpenAI, `$OPENAI_API_KEY`, proxy OK),
  `scripts/sig_test.py`.
- **Monitoring daemon:** `expmon-pisco` job **163132** (30-min scans, emails). Still running —
  `expmon kill 163132` when done.
- **Key checkpoints:** `161788` (4B→4B mt rate8, best LB **35.7**), `158433` (4B→4B mt rate16
  33.9), `149907`/`167200` (4B→9B), `149314` (4B→4B bidi 500K pretrain, ft source).

## Open questions / decisions pending
- **Evaluate 167200 / 167198** (decoder-size, cml-512) — not done.
- **"Any-length" extension** (fixed-MEM-budget / query-dependent OSCAR / hierarchical) —
  proposed, not started. Only relevant if pushing PISCO beyond normal-length RAG.
- **Speed crossover** — extrapolated ~25–30k input tokens where PISCO decode beats
  uncompressed even at bs=1; not measured (would need RULER/NIAH long inputs).

## Don't-break list
- **`*.sh` launchers are GITIGNORED** — durable configs live in `scripts/` or `EXPERIMENTS.md`.
- **In-session watchers/daemons don't survive session end** *except* the SLURM daemon 163132;
  always re-check `squeue`.
- **Decoder is LoRA, unmerged at eval by default** — use `--merge_lora` for deployable-speed
  measurements (adapter overhead otherwise dominates bs=1 decode).
- **`mask_before_mem` is a no-op in the base `PretrainingCollator`** (NCP finding) — do NOT
  "fix" it mid-campaign; the new `MultiTaskPretrainingCollator` handles it internally.
- **`regen/*.jsonl` are stochastic & local** — bit-for-bit rebuild needs the exact files
  (private HF `Herve/pisco-multitask-regen`), not regeneration.
