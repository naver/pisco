# STATE — campaign probes   (rewritten WHOLESALE by every LLM wake; never append)
last wake: 2026-11-11 03:30-03:50 KST (woken because 7 queued cells were flagged "cancelled": a gpu-be preemption)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- The instrument is reproducible on the plain document cell: re-running the reference model 184778 (1,500 questions,
  PX=1003520) gives judge accuracy .610 = the recorded .610 (job 184781), paired difference +.000, CI [+.000, +.000],
  0 discordant pairs, n=1,500; all 1,500 answers byte-identical, mems_used 646,042 both. Job 188419, verdict closed (PASS).
- provisional: the eval is deterministic for every probe cell, so a paired difference comes from the probe. Shown only
  for the plain brief-answer document cell (188419); not yet for evidence-then-answer (256 new tokens), picked-page-raw
  (6,144-token prompts), text sets or another GPU. The int8 cell (188461) and the 300-question regressions (188456,
  188458) will show it on more paths.
- plumbing only: the page-picking fusion script reproduces the saved route numbers exactly — route .8607, attention
  .8853, n=1,500 rows, model 183814, 10-page document set (selftest job 188427 vs probe 185577's summary.json).
- plumbing only: the eval patch of this wake passes the invariant unit tests, 45/45 (test_positions 12/12,
  test_mm_collators 33/33), job 188455, verdict closed; and leaves the uncompressed reader's default path unchanged:
  300/300 predictions and prompt token counts identical to job 185542's first 300 MRQA rows (EM .367 both), job 188456,
  verdict closed.

**Open, and running** (15 cells; none has a result yet)
- arm A (use-recovery on documents), 1,500 questions each:
  seed 42, model 184778, baseline .610: evidence-then-answer (188439), gold page only (188440), picked page raw =
  route-then-expand (188441), slots stored in int8 (188461, expected null).
  seed 43, model 184780, baseline .615 (184784): gold page only (188459), picked page raw (188460).
- arm C (reasoning vs reading on text), MRQA out-of-domain 1,500 questions: compressed model 184778 plain one-chunk
  (188445) and evidence-then-answer (188446); uncompressed reader with evidence-then-answer (188457, chained after the
  patch checks 188455 + 188456).
- arm D (page picking without training), model 183814: attention-route re-runs saving the full digit scores, 10-page
  documents (188442) and HotpotQA 10 passages (188443), then fusion chained afterok on both (188444, CPU).
- patch check left: reference model plain, 300 document questions (188458), EM within ±.01 of the first 300 rows of
  184781 (the law's third check; the patch touches the compressed path only through one assert).

## What happened this wake
- 03:12 seven of the eight 03:05 cells started on homer-008; 03:22:06 all seven were PREEMPTED (gpu-be lends idle GPUs
  and takes them back). sacct: PREEMPTED, Restarts=1; SLURM requeued them, now PENDING (est. start 13:15-18:00).
  No result file was written (each `eval/` dir is empty), so they restart clean. Diagnosis: benign (known_issues row
  added). The MRQA plain cell had reached 810/1,500 rows in 7 min, so a text cell takes about 13 min.
- Found and fixed a silent no-op: `STYLE=expand` was ignored in `base` mode (the uncompressed reader): `gen_base` took
  no system prompt, so arm C's raw-reader side would have rerun the plain prompt with 256 new tokens.
  Patch `scripts/eval_mm_docqa.py` (13 lines, 9+/4-): `gen_base(..., system=None)` passes the prompt to
  `_build_chat_prompt`; base mode applies `final_answer` and keeps `prediction_raw` under expand; an assert refuses
  `--answer_style expand` with `--raw_policy` (still not wired). Default path unchanged: `system=None` falls back to
  the same default system prompt. Queued cells run the working-tree script, and none of them uses base mode or
  expand+raw, so they are unaffected.
  Invariant tests (law, "## Patchable"), job 188455, `run_tests_cpu_188455.out`:
  ```
  === test_positions.py ===   Ran 12 tests in 0.156s   OK
  === test_mm_collators.py === Ran 33 tests in 24.990s  OK
  === EXIT CODE 0 ===
  ```
  Direct check of the changed path, job 188456 (base plain, 300 MRQA rows): 300/300 predictions identical to 185542.
  The law's third check (a LIMIT=300 plain cell of the reference model, EM within ±.01) is 188458, still queued.
- 7 submissions (cap 8): 188455 tests, 188456 base plain 300 Q, 188457 base expand 1,500 Q (afterok 188455:188456),
  188458 reference plain 300 Q, 188459 gold s43, 188460 route1 s43, 188461 int8 s42.
- Symlinks: `expQ/probes/{a_gold_s43,a_route1_s43}/model → expQ/ft_mm_4b_184780/model`,
  `expQ/probes/{a_mq8_s42,reg_184778}/model → expQ/ft_mm_4b_184778/model`.
- Base-mode cells write to `expQ/base_qwen35_4b/eval/` (the launcher fixes that dir); the new names
  `mm_docqa_base_px1003520_text_mrqa_ood_1500_lim300.json` and `..._expand_text_mrqa_ood_1500.json` did not exist
  (checked). CAMPAIGN.md now states this exception and the "read the dispatch first" rule.
- Gripes: check misclassifies a preempted+requeued job; the brief told this wake its own ✗ lines belonged to others;
  runs.jsonl does not record the launch env (STYLE/GOLD/RAW/CML), so the grid cannot show a pending cell's probe.

## Owned jobs (jobid | arm | state at 03:50 | rows | note)
| 188439 | A expand s42 | PENDING (requeued after preemption) | 0/1500 | `expQ/probes/a_expand_s42/eval/`; --time 5:12:00 |
| 188440 | A gold s42 | PENDING (requeued) | 0/1500 | `a_gold_s42`; slot total smaller by design |
| 188441 | A route1 s42 | PENDING (requeued) | 0/1500 | `a_route1_s42`; DML 6144 |
| 188461 | A int8 s42 | PENDING | 0/1500 | `a_mq8_s42`; mems_used must equal .610 cell's 646,042 |
| 188459 | A gold s43 | PENDING | 0/1500 | `a_gold_s43`; baseline `expQ/ft_mm_4b_184780/eval/…1500_1m.json` .615 |
| 188460 | A route1 s43 | PENDING | 0/1500 | `a_route1_s43` |
| 188442 | D attn route retr10 | PENDING (requeued; was at 200/1500) | 0/1500 | → `expQ/probes/attn_route_188442/`; control digit .861 / attention .885 |
| 188443 | D attn route hotpot | PENDING (requeued) | 0/1500 | → `attn_route_188443/`; control digit .922 / attention .873 |
| 188444 | D fusion (CPU) | PENDING (afterok 188442:188443) | — | `fusion.json` + `fused_<picker>.json` per dir. NB `route_fusion.sbatch` always exits 0 (ends on `echo`), so a later afterok on it would not catch a crash |
| 188445 | C plain s42 | PENDING (requeued; was at 810/1500) | 0/1500 | `c_plain_s42`; CML=4096 |
| 188446 | C expand s42 | PENDING (requeued) | 0/1500 | `c_expand_s42`; CML=4096 |
| 188457 | C base expand | PENDING (afterok 188455:188456) | 0/1500 | `expQ/base_qwen35_4b/eval/mm_docqa_base_px1003520_expand_text_mrqa_ood_1500.json`; partition gpu,gpu-be |
| 188456 | patch check, base plain 300 Q | finishing its judge pass; verdict closed PASS | 300/300 | 300/300 predictions identical to 185542 |
| 188458 | patch check, reference plain 300 Q | PENDING | 0/300 | `expQ/probes/reg_184778/eval/…_lim300.json` vs the first 300 rows of 184781's file, EM ±.01 |

## Adopted jobs
(none. The sibling `longchunk` campaign's 188421 (running, 4 GPUs on `gpu`) and 188422 (pending) are not ours.)

## Open arms and the next decision for each
| arm | status | next action |
|---|---|---|
| A. use-recovery | s42 ×4 and s43 ×2 queued | per cell: health (unparseable 0, 1,500 rows, mems_used ratio; reported, not gated, for gold/route1) → `scripts/paired_judge.py` vs the same seed's plain file. Launch s43 expand only if s42 expand is a tie or a gain (a clear loss needs no repeat) |
| B. long-chunk mechanism | not started | write `scripts/probes/build_groups.py` (g3/g10 were built ad hoc on 2026-11-06): gold-passages-joined and distractors-joined sets from `text_hotpotqa_1500`, built through SLURM (gpu-be, gres:0), first check it reproduces g10 |
| C. reasoning vs reading | 3 cells queued | after 188445/188446/188457: per-source judge gap compressed−raw on DROP / DuoRC / RACE, plain vs expand; gate ≥ 5 pt shrink. Thinking mode still needs a flag (`enable_thinking=False` hard-coded in `_build_chat_prompt`, line 80) |
| D. page picking | re-runs + fusion queued | read `fusion.json`: controls recovered? gate pick@1 ≥ +2 pt on retr10 AND no loss on hotpot; then `RAW=route1 RANK=fused_<picker>.json` |
| E. attention-guided pruning | not started | `scripts/probes/prune_by_attention.py` not written |
| F. photo text vs pixels | blocked on data | build `eval_data/probes_textvqa_ocr` (500 TextVQA rows + `ocr_tokens` from lmms-lab/textvqa) via a SLURM job |
| patch checks | 188456 PASS, 188458 queued | if 188458's EM is off by > .01 from the first 300 rows of 184781's file: revert the patch, `expmon kill` the cells launched after it, BLOCKED.md |

## Health and budget
- gpu-be preempts: 7 of 8 cells lost 10 min at 03:22. Cells with no dependents go on `gpu,gpu-be` only when at most
  one can sit on `gpu` (owner's 8-GPU cap; longchunk holds 4). This wake: only the base chain (188456 → 188457).
- budget: 17 of 60 cells committed (gate, selftest, 8 + 7 queued, tests); GPU-hours spent ≈ 0.7 of 40; the 15 queued
  cells add ≈ 11 from their measured walls.

## Next wake
`wake-when` on the first of three groups leaving the queue: arm C's three text cells (188445, 188446, 188457), arm A's
seed-42 trio (188439-188441), arm D's re-runs + fusion (188442-188444); heartbeat at 6 h.

## Blockers / needs the owner
- none blocking. FYI: `eval_mm_4b.sh`, `probe_attn_route.sh` and `run_tests_cpu.sh` are NOT under git (`.gitignore`
  has `*.sh`); force-add them when convenient. The ALERT about the 01:00/01:30 failed wakes is yours to clear (cause:
  the account's 5-hour usage limit).
