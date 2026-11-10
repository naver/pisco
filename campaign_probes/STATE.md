# STATE — campaign probes   (rewritten WHOLESALE by every LLM wake; never append)
last wake: 2026-11-11 04:00-04:30 KST (woken by the verifier's ALERT line on the last STATE.md)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- The instrument reproduces on the SAME node: re-running the reference model 184778 on documents (1,500 questions,
  PX=1003520) on homer-011, the node of the recorded run 184781, gives judge .610 = .610, 1,500/1,500 answers
  identical, 0 discordant pairs. Job 188419, verdict closed (PASS).
- On ANOTHER node it does not reproduce for image inputs: the same cell on homer-008 (older system image, kernel
  el9_7), first 300 questions: judge .623 vs .610 recorded, 14/300 discordant (9 up, 5 down), 243/300 answers
  identical, EM .490 vs .477, mems_used 118,147 both. Job 188458, verdict closed. The eval fix of the last wake is not
  the cause: its diff touches only the uncompressed reader + one assert, and that cell runs the compressed path.
- Data for arm B exists and is checked: `scripts/probes/build_groups.py` rebuilds the old one-chunk HotpotQA set
  (`text_hotpotqa_1500_g10`) on 1,500/1,500 rows identical, then writes three grouped sets of 1,500 rows. Job 188465,
  verdict closed. (First try 188464 stopped itself on a missing separator; nothing written; closed.)
- plumbing only: page-picking fusion reproduces the saved route numbers exactly — route .8607, attention .8853,
  n=1,500 rows, model 183814 (selftest 188427 vs 185577). Eval-fix unit tests 45/45 (188455) and the uncompressed
  reader's default path unchanged, 300/300 answers identical to 185542 (188456); both verdict closed.
- provisional: the compressed reference model on MRQA text, one chunk per passage: judge .718, n=1,500, unparseable 0,
  mems_used 80,003 (DROP .580, DuoRC .600, RACE .528, 250 each) vs the uncompressed reader .807 (185542; DROP .784,
  DuoRC .684, RACE .680). Job 188445 attempt 2 (homer-016); SLURM requeued the job after its batch step had
  completed, so the grid shows it PENDING; copy kept as `expQ/probes/c_plain_s42/eval/attempt2_0357_*.json`;
  `expmon probe 188445 judge=0.718` recorded.
- provisional: text inputs do not move with the node. Uncompressed: 300/300 identical across homer-008 / homer-011
  (188456 vs 185542). Compressed: 188445's re-run on homer-008 printed the same running EM / F1 / match as attempt 2
  (homer-016) at rows 710 and 1,110, then was preempted; no full-file comparison yet.

**Open, and running** (17 cells; 7 running, 10 queued at 04:25; none of them has a result file yet)
- determinism checks, reference model plain, 300 document questions each: homer-016 (188476, same image as the
  recorded run) and homer-007 (188477, older image; waits for that node). They decide image vs single node.
- arm A (use-recovery on documents), 1,500 questions each. Seed 42, model 184778, baseline .610 (184781, homer-011):
  evidence-then-answer (188439, running on homer-016), gold page only (188440, requeued), picked page raw (188441,
  running on homer-016), slots in int8 (188478, queued off the older-image nodes; replaces 188461).
  Seed 43, model 184780, baseline .615 (184784, written on homer-016): gold page only (188459), picked page raw (188460),
  both queued with no node constraint.
- arm B (long-chunk mechanism), model 183814, HotpotQA 10 passages, 1,500 questions: golds joined (188471, requeued),
  distractors joined (188472, running on homer-020), all joined in a shuffled order (188473, queued). Baselines g1 .579
  (185505, file written on homer-006, el9_7) and g10 .617 (185601, homer-017, el9_8) — text, so provisional-safe.
- arm C (reasoning vs reading on text), MRQA 1,500 questions: compressed plain (188445, requeued; attempt 2 above),
  compressed evidence-then-answer (188446, running on homer-016), uncompressed evidence-then-answer (188457, running
  on homer-011).
- arm D (page picking), model 183814: attention-route re-runs on 10-page documents (188442, running on homer-016) and
  HotpotQA (188443, running on homer-013), then fusion afterok on both (188444, CPU).

## What happened this wake
- Verifier fix: the last STATE.md said "15 cells, none has a result"; it was 13 open cells, and 188455/188456 had
  results. Counted from squeue this time.
- Arm B started: wrote `scripts/probes/build_groups.py` + `build_groups.sbatch` (free path). Sets
  `eval_data/probes_hotpot_gold_joined` (9 chunks on 1,487/1,500 rows), `_distr_joined` (3 chunks on 1,495/1,500;
  7 rows have ≤ 1 distractor and stay unjoined), `_g10_shuf` (1 chunk). Ledger `resource:` note written. Three cells
  launched on symlinks `expQ/probes/b_{gold_joined,distr_joined,g10_shuf}/model → ft_mm_4b_183814/model`,
  CML=4096 TOPK=10, `--time 1:01:00` (slowest completed grouping cell 185600, 20:02, × 3). Gates in the ledger.
- The law's fix check failed its letter (EM off by .013 > .01) and the cause is the node, not the fix (see Settled).
  The fix stays. Two 300-question checks pinned to homer-016 and homer-007 split "image" from "homer-008 alone".
- CAMPAIGN.md tightened: the feasibility gate's "the pipeline is deterministic" is marked as a wrong generalisation
  (kept), and a "Same node rule" is added: image cells pair only within one node image; image cells launch with
  `--exclude=homer-[005-009]` until the checks say otherwise.
- Node of every baseline (attempt that wrote the file): 184781 homer-011, 184784 homer-016, 185542 homer-008,
  185505 homer-006, 185601 homer-017, 185775 / 185577 / 185580 homer-014. Images: el9_7 = homer-005..009,
  el9_8 = homer-010..020. NB the long-chunk +.037 (g10 vs g1) crosses images; it is text, so probably safe.
- gpu-be preemptions: 6 cells at 04:11 (all on homer-008) and 188440 by 04:25. 188445 has now been preempted 3 times.
- 8 submissions (cap 8): 188464 / 188465 builds, 188471-188473 arm B, 188476 / 188477 checks, 188478 int8 relaunch.
  `expmon kill` 188461 (pending, no file). Every launch needed `--force`: expmon's dead-arm check matches any launch to
  a closed run with empty pre-launch factors (gripe filed).
- Gripes: the dead-arm false match; no node column in runs.jsonl / grid.

## Owned jobs (jobid | arm | state at 04:25 | node | note)
| 188439 | A expand s42 | RUNNING (370/1500 at 04:06) | homer-016 | `expQ/probes/a_expand_s42/eval/`; --time 5:12:00 |
| 188440 | A gold s42 | PENDING (preempted) | — | `a_gold_s42`; image cell: check its node before pairing |
| 188441 | A route1 s42 | RUNNING | homer-016 | `a_route1_s42`; DML 6144 |
| 188478 | A int8 s42 | PENDING | excl. 005-009 | `a_mq8_s42`; mems_used must equal 646,042 |
| 188459 | A gold s43 | PENDING (preempted) | — | `a_gold_s43`; baseline 184784 (homer-016) |
| 188460 | A route1 s43 | PENDING (preempted) | — | `a_route1_s43` |
| 188471 | B golds joined | PENDING (preempted) | — | `b_gold_joined`; vs g1 185505 / g10 185601 |
| 188472 | B distractors joined | RUNNING | homer-020 | `b_distr_joined` |
| 188473 | B g10 shuffled | PENDING | — | `b_g10_shuf`; vs g10 185601 |
| 188445 | C plain s42 | PENDING (3rd requeue) | — | attempt-2 file is the result; the re-run overwrites it with (expected) the same bytes — compare to the copy |
| 188446 | C expand s42 | RUNNING | homer-016 | `c_expand_s42`; CML=4096 (file suffix `_cml4096`) |
| 188457 | C base expand | RUNNING (550/1500 at 04:06) | homer-011 | `expQ/base_qwen35_4b/eval/mm_docqa_base_px1003520_expand_text_mrqa_ood_1500.json` |
| 188442 | D attn route retr10 | RUNNING | homer-016 | → `expQ/probes/attn_route_188442/`; control digit .861 / attention .885 |
| 188443 | D attn route hotpot | RUNNING | homer-013 | → `attn_route_188443/`; control digit .922 / attention .873 |
| 188444 | D fusion (CPU) | PENDING (afterok 188442:188443) | — | `route_fusion.sbatch` always exits 0 |
| 188476 | node check | PENDING | homer-016 pinned | `expQ/probes/reg_h016/eval/…_lim300.json` vs first 300 of 184781 |
| 188477 | node check | PENDING (homer-007 busy) | homer-007 pinned | `expQ/probes/reg_h007/eval/…_lim300.json` vs 184781 AND vs 188458 |

## Adopted jobs
(none. 188421/188422 and 188466-188470 belong to the sibling `longchunk` campaign.)

## Open arms and the next decision for each
| arm | status | next action |
|---|---|---|
| node checks | 2 queued | 188476 identical to 184781 → el9_8 nodes agree, the arm A s42 cells stand. 188477 identical to 188458 → the image is the cause (keep the exclude); identical to 184781 → homer-008 alone (exclude only it). Then rewrite the Same node rule with the result |
| A. use-recovery | s42 ×4, s43 ×2 open | per cell: node (sacct -D), health, `scripts/paired_judge.py` vs the same seed's plain file, only if same image. A cell that lands on el9_7: pair it with a plain baseline run on el9_7 (one cell per seed, which also measures the node effect at n=1,500) or relaunch it with the exclude. s43 expand only if s42 expand is a tie or a gain |
| B. long-chunk mechanism | 3 cells open | paired vs g1 and g10 (gates in the ledger, 188471-188473); then the tail control 183538 only for the grouping that decides |
| C. reasoning vs reading | plain done once, 2 open | after 188446 / 188457: per-source gap compressed − raw on DROP / DuoRC / RACE, plain vs expand; gate ≥ 5 pt shrink. Compare 188445's re-run file to the attempt-2 copy (text determinism, full file). Thinking mode: no flag yet; decide after expand |
| D. page picking | 2 re-runs + fusion open | read `fusion.json`: controls recovered? gate pick@1 ≥ +2 pt on retr10 AND no loss on hotpot; then `RAW=route1 RANK=fused_<picker>.json` (an image cell: same-image rule) |
| E. attention-guided pruning | not started | `scripts/probes/prune_by_attention.py` not written |
| F. photo text vs pixels | blocked on data | build `eval_data/probes_textvqa_ocr` (500 TextVQA rows + `ocr_tokens` from lmms-lab/textvqa) via a SLURM job |

## Health and budget
- budget: 25 of 60 cells committed (17 before this wake + 8 submissions); GPU-hours of 40: the tick's sensor reports
  0.5 GPU-h in the last 24 h; the 17 open cells add ≈ 12 from their measured walls.
- gpu-be preempts often: 7 cells at 03:22, 6 at 04:11, 1 more by 04:25. Rule kept: at most 2 cells on `gpu`; move 2 pending cells to
  `gpu,gpu-be` only after 6 h pending (pre-authorised).

## Next wake
`wake-when` on the first group leaving the queue: node checks (188476, 188477); arm A seed 42 trio (188439-188441);
arm C (188445, 188446, 188457); arm B (188471-188473); arm D (188442-188444); heartbeat 6 h.

## Blockers / needs the owner
- none blocking. FYI: `eval_mm_4b.sh`, `probe_attn_route.sh`, `run_tests_cpu.sh` are not under git (`.gitignore` has
  `*.sh`); force-add them when convenient. The ALERT lines about the 01:00 / 01:30 failed wakes are yours to clear
  (cause: the account's 5-hour usage limit, known_issues row).
