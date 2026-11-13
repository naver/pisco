# CAMPAIGN probes — pisco   (the law; the agent re-reads this every wake; rewrite, never append)

created 2026-11-11 · kind **probe** · root `/beegfs/scratch/user/hdejean/pisco` · window until **2027-02-09** (launches allowed through that day 23:59) · owner herve.dejean@naverlabs.com · wakes run `claude-opus-5-5`

A probe campaign runs many short inference/evaluation cells (minutes each) and decides on counts or metrics
against a baseline cell — not on training curves. The binding constraint is submissions per wake and queue
time, not GPU-hours. DRAFT written by Claude on 2026-11-11 from the ledger (2026-11-03 → 11-10), HANDOFF.md and
docs/literature_scan_2026-11.md; the owner reviews every section before `arm`.

## Mission
Find out, with inference-time probes only (no training), WHY the compressed model loses to the uncompressed reader where
it does, and which cheap changes recover part of the loss. The facts to explain (judge accuracy = share of questions the
Qwen3-8B rubric judge accepts, 1,500 questions per set, `scripts/paired_judge.py` for paired differences):
- document pages: compressed .610 / .615 (reference models s42 / s43) vs raw .809; by source infographics .48 vs .73,
  multi-page .67 vs .93, slides .68 vs .77. Photos are near the ceiling (.68 vs .72). Text reading comprehension (MRQA,
  one passage): .695 vs .807, the loss sits on reasoning (DROP .60 vs .78) and narrative (DuoRC .55 vs .68, RACE .53 vs .68).
- an earlier transcription probe (architecture_review.md §2) found many document failures are "use" failures: the
  answer can be read back from the slots, the decoder does not use it.
- under the spread layout one long chunk beats ten short ones (+.037 on HotpotQA, no training; seed 42, 11-07 files. PARTLY WRONG, kept and marked: +.014 n.s. on the seed-43 twin, job 188510); the compressor sees the
  passages together. Mechanism unknown: cross-passage context, fewer chunk boundaries, or fewer slots.
- the decoder's own attention picks the gold page better than the digit route prompt on images (.885 vs .861, 10 pages)
  and worse on text (.873 vs .922).
- dropping half the slots at inference costs .12 on documents under spread (slots own a part) and .04 under tail.
The campaign decides by paired judge accuracy, arm cell − baseline cell on the same questions (95% CI, McNemar); the
unit is judge accuracy in [0, 1], "pt" = .01. Models are FIXED: the reference pair `expQ/ft_mm_4b_184778/model` (s42) and
`184780` (s43) for every "can we recover loss" probe (both seeds before a claim), and the reduced spread / tail pair
`183814` / `183538` for every "why does the layout do X" probe; a layout result that passes its gate once on 183814 is
repeated on its seed-43 twin `183816` (183814's recipe + `hf_training.seed=43`) before it is adopted. The deliverable is a ranked list of findings, each with
its paired number, its n, and the training arm it motivates (or the idea it closes).

## Arms — what varies, what is held constant
Every cell is one `eval_mm_4b.sh` run (or a probe script under `scripts/probes/`) on one model and one eval set; the
baseline cell of an arm is the same model, same set, same pixel budget with the probe off. Held constant everywhere:
checkpoints as listed, `PX=1003520` on document sets, judge = rubric prompt, 1,500 questions (`LIMIT` only for
plumbing checks, never for a claim), chunk cap 4096 (`CML=4096`) on text so no passage is cut.

| arm | cell launcher (committed) | factors (what varies) | held constant | baseline cell | gate (metric vs baseline) | cells |
|---|---|---|---|---|---|---|
| A. use-recovery on documents | `eval_mm_4b.sh <ref model> pisco "" eval_data/mm_docqa_1500_1m` with ONE env probe: `STYLE=expand` (copy the evidence, then answer), `GOLD=1` (gold page only), `RAW=route1` (picked page raw), `MQ=int8` | one decoding/prompting change at a time | model, set, PX | the same model's plain cell (exists: judge .610 / .615) | ≥ +3 pt with CI above 0 at both seeds = finding; the "use" share it explains is reported from the transcription files. DECIDED 2026-11-11: `RAW=route1` +.210 / +.207 (188441 / 188460, adopted); `GOLD=1` +.022 / +.025 (closed); `STYLE=expand` +.001 [−.017, +.021] (s42, 188439) / −.009 [−.027, +.011], p=.41 (s43, 189058, decided 2026-11-13) → closed at both seeds; `MQ=int8` +.001 / −.001, within .02 at both seeds (188478 / 188536) → adopted as a serving option | 2 seeds × probes |
| B. long-chunk mechanism | `CML=4096 TOPK=10 eval_mm_4b.sh 183814 pisco "" <grouped set>` on new grouped sets built by `scripts/probes/build_groups.py`: gold passages joined alone vs distractors joined alone, passage order shuffled inside the joined chunk, g10 with the "+1" slots equalised | which passages share a chunk, their order | model 183814 (and 183538 as the tail control), 10 passages, same slot total ± 1% | g1 and g10 re-run on 11-11 code (183814: g1 .582 job 188486, g10 re-run 188518; 183816: .601 / .615, jobs 188509 / 188510). The 11-07 files (.579 / .617) are no longer baselines | a grouping that keeps the +.037 while removing a candidate cause closes that cause; CI must exclude 0. DECIDED 2026-11-11 (188471, 188511): golds joined beats ten chunks at both seeds (+.073 / +.067) → the gain is context between the answer passages. WRONG PREDICTION, kept and marked: the law took "+.037 for one chunk of ten" as the fact to explain; at seed 43 it is +.014, n.s. (188510). Tail control (188514 vs 188519): +.058 [+.035, +.082] on 183538 → the gain needs no spread layout (one model). No-oracle grouping, gate pred − g1 ≥ +.03 CI above 0 → "rank, then re-compress the top passages together" is a serving mode and a stage-2 data change: seed 42 top 3 by attention +.032 [+.009, +.056] (188521), top 2 +.030 (188520); WRONG PREDICTION, kept and marked: seed 43 top 3 +.013 [−.007, +.034], p=.24 (188546 vs 188509) → NOT adopted; it equals one chunk of ten at both seeds (−.003 / −.001). One chunk of ten on today's code at seed 42: +.035 [+.012, +.059], p=.004 (188518; 1,500/1,500 answers identical to the 11-07 file 185601), so the +.037 holds at seed 42 only. Passage order closed at both seeds (−.002 / +.003). ARM CLOSED | ~6 + 2 |
| C. reasoning vs reading on text | `CML=4096 eval_mm_4b.sh <ref model> pisco "" eval_data/text_mrqa_ood_1500` with `STYLE=expand`, and the same with the model's thinking mode on (`scripts/probes/` flag if `eval_mm_docqa.py` lacks it) for compressed AND `base` | prompting that gives the decoder room to reason | passage, set | plain compressed .695 / raw .807 | the compressed−raw gap on DROP / DuoRC / RACE shrinks by ≥ 5 pt with the same treatment on both sides | ~6 |
| D. page picking without training | `scripts/probes/route_fusion.py` over the saved attention masses (`expQ/probes/attn_route_*/masses.npz`) and the digit log-probs: rank fusion, product, per-source choice; then `RAW=route1` with the fused pick via the eval's `RANK=` json | the picker | sets retr10 / hotpot / nav frames | digit pick@1 .861 (retr10), .922 (hotpot) | pick@1 ≥ +2 pt on retr10 AND no loss on hotpot; then the route1 judge gain, ONLY if its expected size (pick@1 gain × the judge gap between hit and miss rows of a route1 cell) exceeds the noise band. DECIDED 2026-11-11 (job 188515): per-source switch +.025 / +.000 passes; expected judge gain ≈ +.006 to +.008 (pick +.025 × the route1 hit-minus-miss judge gap: .252 at s42, 1,479 hit / 21 miss rows; .325 at s43, 1,484 / 16), no judge cell. CAVEAT, kept and marked: the per-source rule was chosen on the same questions it is scored on; a held-out set must confirm it before any use | ~4 |
| E. attention-guided slot pruning | `KEEPF=<frac> KEEPM=attn|attn_global PX=1003520 eval_mm_4b.sh <ref> pisco "" <set>` (`--mem_keep_mode attn*` in `scripts/eval_mm_docqa.py`): one eager pass over the full prompt, keep the slots the last prompt position (question seen) attends to most, per chunk (`attn`) or over all pages (`attn_global`), then answer. AMENDED 2026-11-11: the law first named a 25% drop as the second level; a 50% drop already costs nothing (188485), so the second level is a 75% drop (`KEEPF=0.25`) | which slots are kept, how many | model 184778 / 184780, docs + photos | plain .610 / .615; untrained keep-50% on 184778: first half .495 (188482), every 2nd .505 (188483) | ≥ +4 pt over the best untrained keep-50% at the same slot count, at both seeds = "given the question, half the slots carry what the decoder uses" → motivates a query-aware slot budget / nested-budget training (stage-2 target), not a storage saving (the selection pass reads every slot). keep-25% (gate written 2026-11-11, before the cell): within .02 of plain → three quarters unused per question; a loss ≥ .05 → the free share lies between 1/2 and 3/4. DECIDED 2026-11-11: attn_global half − blind every-2nd = +.105 (s42, 188485 − 188483) and +.107 (s43, 188522 − 188537), CI above 0 → ADOPTED. keep-25%: s42 .611 vs .610 = +.001 [−.009, +.012] (188538), s43 .619 vs .615 = +.005 [−.007, +.017], p=.51 (189056, decided 2026-11-13), slots .246× → within .02 at both seeds → ADOPTED: three quarters of the slots go unused per question. Photos s42 half .677 = .677 (188539). Next level (gate written 2026-11-11, before any cell): keep-12.5% (`KEEPF=0.125 KEEPM=attn_global`) within .02 of plain → seven eighths unused; a loss ≥ .05 → the free share lies between 3/4 and 7/8. DECIDED AT SEED 42 ONLY 2026-11-13 (189057, homer-019): .613 vs .610 = +.003 [−.011, +.017], 59 / 55, p=.78, n=1,500, slots .121× (52 per question) → within .02 = provisional "seven eighths unused"; the seed-43 repeat needs a cell beyond the cap (owner). The point where dropping starts to cost is still not found | ~6 |
| F. photo text (TextVQA .77 vs .85) | `eval_mm_4b.sh <ref> pisco_text` (OCR text instead of pixels) on a NEW set `eval_data/probes_textvqa_ocr` (the 500 TextVQA rows of `mm_naturalqa_1500` with their `ocr_tokens` from `lmms-lab/textvqa` as `docs`). WRONG PREMISE, kept and marked (2026-11-11): the law first named `mm_naturalqa_1500` itself, but its rows carry `docs=[""]` (no OCR text; `build_mm_natural_qa.py` line 82), so `pisco_text` there compresses an empty string | pixels vs text for text-in-image | model, questions | plain .677 / .685 (TextVQA rows of it) | reported only; a TextVQA gain ≥ 5 pt motivates a mixed input arm. Seed-42 cell `f_text_s42` = 189059 (queued 2026-11-12 under the raised cap), paired on query_id with the 500 TextVQA rows of 184783. DECIDED 2026-11-13 (189059, homer-012): OCR text .346 vs pixels .780 = −.434 [−.480, −.384], 17 / 234, p=6e-50, n=500, slots .081×; no gain → no mixed-input arm. The "plain" TextVQA rows score .780, not .77/.677 as this row first read (.677 is the whole photo set). ARM CLOSED (seed 43 not run: its baseline 184785 ran on homer-008) | 2 + 1 set build |
| G. route then densify (accepted by the owner 2026-11-11) | `eval_mm_4b.sh <ref model> pisco "" eval_data/mm_docqa_1500_1m` with ONE new env probe: the page the model picks (same pick as `RAW=route1`) is re-compressed at rate 4 (2× its slots) and stays compressed; the other pages keep rate 8. Flag written 2026-11-11: `RAW=route1 DENSE=<rate>` (`--dense_rate`); checks before the full cells: tests, plain LIMIT=300 regression, `DENSE=8` LIMIT=300 control (must give the plain answers: ≥ 97% identical, judge within .01 of 184781's first 300). Tests 188523 and regression 188524 PASSED. WRONG PREDICTION, kept and marked: the first control 188525 gave 233/300 identical answers (same slot total, EM .487 vs .477): the flag re-compressed the picked page ALONE, while the plain path compresses all pages in one padded batch. Patched 2026-11-11 (9 lines): the page is re-compressed inside its example's batch, so at rate 8 its slots must be bit-identical; criteria unchanged. Re-check = ONE job `scripts/probes/g_checks.sbatch` (tests + both 300-Q cells + `g_checks_compare.py`, exit 1 on any failure, `--time 1:30:00`); full cells `DENSE=4` (`--time 4:54:00`) are submitted only AFTER it passes, never chained on it: with 3 cells left, a failed check must not burn them. Re-check 188580 PASSED (2026-11-11 08:10): tests 12+33+5, plain 300/300, `DENSE=8` control 300/300 identical to plain (judge .610 = .610, slots 118,147) → the batch-of-one cause is confirmed; full cells 188590 (s42) / 188591 (s43) submitted 08:40. DECIDED 2026-11-11 09:45: gain over plain +.063 [+.043, +.083] (s42, 188590) and +.049 [+.030, +.067] (s43, 188591), n=1,500 each, slots 1.250×, below the +.14 gate at both seeds; extra decoder tokens +107.5 vs route1 +753 (.143×, met) → ARM CLOSED (untrained rate caveat stands) | slots of the picked page only | models 184778 / 184780, questions, pick, prompt | plain .610 / .615 and `RAW=route1` .820 / .822 (cells 188441 / 188460) | gain over plain ≥ 2/3 of route1's (≥ +.14) with CI above 0, at ≤ 1/3 of route1's extra decoder tokens (report both token counts) → a serving mode and a stage-2 target; both seeds before "adopted". Caveat to report: the models were trained at rate 8 only, so a fail may be the untrained rate, not the idea | 1 plumbing + 2 |

Rules: every arm is a committed launcher + a committed cell spec, never raw overrides on the launch line. A
comparison with a difference not listed here is not a result. A relaunch repeats the original command exactly.
Before the first launch of a probe env on a mode or path it was never used with, read `scripts/eval_mm_docqa.py`
`main()` and confirm the flag reaches that branch: until 2026-11-11 `STYLE=expand` was silently ignored in `base` mode
(patched that day); `STYLE=expand` with `RAW=*` is still not wired and is now refused.
`base`-mode cells cannot use the symlink rule below (the launcher writes them to `expQ/base_qwen35_4b/eval/`); they are
allowed there only under a file name that does not exist yet (check with `ls` before the launch).
Every eval cell runs on a symlinked model dir `expQ/probes/<cell>/model → expQ/<run>/model` (e.g. `rep_184778`), so its
result lands in `expQ/probes/<cell>/eval/` and never overwrites a recorded file in `expQ/<run>/eval/` (the plain re-run of
the feasibility gate would). New launchers go in `scripts/probes/*.sbatch`: the repo `.gitignore` drops every new `*.sh`
(`eval_mm_4b.sh`, `probe_attn_route.sh` and `run_tests_cpu.sh` are tracked, checked 2026-11-11 with `git ls-files`).
Any new eval set or grouping is a ledger `resource:` note the moment it exists (path, size, how built).

## Decision gates
Noise band .02 (`CAMP_NOISE_PT=2`): below it nothing is a result; a tie triggers a repeat on the second seed; above it once
still needs the second seed (or the second set) to be "adopted". Decide arms as soon as their cells complete — do not
wait for the whole grid. Every claim carries n (questions, pairs), the paired CI and McNemar p.
- **Feasibility gate (first, alone):** re-run the reference model's plain document cell (`eval_mm_4b.sh expQ/ft_mm_4b_184778/model pisco "" eval_data/mm_docqa_1500_1m`, PX=1003520) and pair it with the recorded file
  (`expQ/ft_mm_4b_184778/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json`, judge .610). The judge-on-judge difference
  must be within ±.01 with the CI inside the noise band; otherwise the instrument does not register changes of the size
  this campaign looks for, and the campaign STOPS with "the judge is not reproducible at the .02 level" as its finding.
  PASSED 2026-11-11 (job 188419): judge .610 = .610, 1,500/1,500 predictions identical (greedy decoding), 0 discordant
  pairs. WRONG GENERALISATION, kept and marked: the wake then wrote "the pipeline is deterministic". It is
  deterministic on ONE node (188419 and 184781 both ran on homer-011). On homer-008 (older system image, kernel el9_7)
  the same cell gives other answers on IMAGE inputs: 57/300 predictions differ, judge .623 vs .610, 14/300 discordant
  (job 188458). Text inputs did not move on the checks then available (uncompressed 300/300, 188456 vs 185542;
  compressed MRQA 1,500/1,500 on one node, 188445). PARTLY WRONG, kept and marked: compressed text moved 55/1,500
  answers (judge .582 vs .579) between homer-006 with 11-07 code (185505) and homer-010 with 11-11 code (188486).
- **Same node rule (added 2026-11-11):** a paired difference between two cells with IMAGE inputs counts only when both
  ran on the same node image, or on nodes a determinism check showed to agree (`LIMIT=300` plain cell, predictions
  identical). Nodes: el9_7 = homer-005..009, el9_8 = homer-010..020 (`scontrol show node <n> | grep OS=`). Record
  the node of every cell and of its baseline (`sacct -j <id> -D -o NodeList` — the attempt that wrote the file).
  Check 188476 PASSED: homer-016 (el9_8) gives 184781's (homer-011) first 300 answers 300/300, so el9_8 nodes
  agree. Check 188477 PASSED too: homer-007 (el9_7) gives the same 300/300, so the image effect is homer-008 alone
  (all nodes carry A100-SXM4-80GB). homer-008 stays excluded for good (it also took 14 of the campaign's first 18
  preemptions). Text check PASSED as well: job 188471's re-run on homer-007 (compressed text, golds joined) gave
  1,500/1,500 answers identical to its homer-016 file. Every node tested (homer-007, 010, 011, 013, 016, 020) agrees
  except homer-008; homer-005, 006 and 009 are untested. Launch every cell with `--exclude=homer-[005-009]` while the
  queue allows, never with less than `--exclude=homer-[005,006,008,009]`. The 55-answer drift of
  185505 (homer-006, code of 11-07) vs 188486 (homer-010, code of 11-11) is therefore homer-006 or the code date:
  text cells of arm B pair with the 11-11 g1 re-run 188486, never with 185505. The code date is ruled out for text
  (188518 on homer-010, 11-11 code, = 185601 on homer-017, 11-07 code, 1,500/1,500 answers), which points to
  homer-006: keep it excluded.
- **Health number per cell:** `unparseable: 0` in the eval log, 1,500 rows in the result file, and the total `mems_used`
  within 1% of the baseline cell's (a different slot total is a different budget, not a probe result). A cell failing
  any of these is `closed "health"` and relaunched once. Exception: a probe whose purpose is to change what is compressed
  (`GOLD=1`, `RAW=*`, `XD=*`, a pruning or grouping cell) changes the slot total by design; its health check is the
  mems_used ratio REPORTED in the arm table next to the result, and the comparison it supports is "same questions, this
  input instead of that one", never "same budget".
- **Small cells are plumbing:** a `LIMIT=300` cell proves the code path runs; it never decides an arm, even when it agrees.
- A NEW metric (one the wake designs itself) closes nothing until its row here states its n and a known-positive control it
  recovers (the attention-mass picker recovering .885 on retr10 is such a control for arm D).
- Headline claims in STATE.md and the ledger are prefixed `provisional:` until the next critic pass, or until the
  next wake re-derives them from `expmon grid`.

## Budget and caps
- cells for the whole campaign: 65 (raised from 60 by the owner, REQUEST of 2026-11-12 23:14, answer to the BLOCKED.md
  question «Raise the campaign's cell cap from 60 to 65?»: option A) · GPU-hours: 40 (a judged 1,500-Q cell is 20-45 min
  on one A100). A cell = every `expmon launch --exp probes` submission, builds and test jobs included (how the wakes have
  counted since the start); an automatic requeue after a preemption is not a new cell. 60 used by 2026-11-11 08:40 (the
  two killed rate-4 cells 188541 / 188542 included; 188546 counted by hand: its manifest row lacks `exp`).
  The five cells of option A, and nothing else: 61 keep a quarter at seed 43 (`e_keep025_attng_s43`), 62 keep an eighth
  at seed 42 (`e_keep0125_attng`), 63 TextVQA as OCR text at seed 42 (`f_text_s42`, arm F), 64 evidence-then-answer
  at seed 43 (`a_expand_s43`), 65 = ONE health relaunch, kept in reserve for whichever of 61-64 fails its health check
  (not for a preemption requeue, which is free). Still dropped: arm C seed 43, arm F seed 43 (its photo baseline 184785
  ran on homer-008).
- at most CAMP_MAX_SUBMIT_PER_WAKE new submissions per wake (campaign.conf)
- every launch: one GPU, `-p gpu-be` by default (`--partition gpu,gpu-be` when gpu-be is jammed; the owner's 8-GPU cap on
  `gpu` is shared with the sibling training campaign `longchunk`, so never more than 2 cells on `gpu` at once); `--time` =
  measured cell wall × 3, from the SLOWEST measured run of that cell kind (a 1,500-Q judged document cell measured 28 min
  on job 184784 and 98 min on job 184781, same cell kind → 4:54:00; photos 18 min, text 17 min; attention-route probe
  retr10 41 min / docs 21 min / hotpot 12 min)
- no launch below CAMP_DISK_FLOOR_GB free; nothing new after CAMP_UNTIL

## Deliverable — how results reach the owner
`expmon display --exp probes` (default page) plus `docs/probe_campaign_2026-11.md`, rewritten (not appended) by the wake
whenever an arm decides: one table per arm with the columns `cell | model | set | n | judge accuracy ↑ | baseline judge
accuracy ↑ | difference (95% CI) | McNemar p | health (unparseable, rows, mems_used ratio)`, a ranked "Findings" list
(each finding: the number, what it explains, the training arm it motivates or the idea it closes), and a "Closed" list.
Every decided arm also gets one `expmon note` with the headline number. A number that exists only in a log is not delivered.

## Pre-authorised actions (the agent may do these without asking) and their loop guards
| situation | action | guard |
|---|---|---|
| a cell FAILED / NODE_FAIL / TIMEOUT for a transient cause (node failure, a wall-time misestimate, PREEMPTED) | relaunch the same cell, `--time` re-derived from measured cells | max 2 relaunches per cell, then BLOCKED.md |
| `expmon check` prints `node <n>: N fast deaths` | relaunch the dead cells with `--exclude=<n>` before the launcher | 1 exclude-relaunch per cell |
| a cell's output is empty or unparsable while the job COMPLETED | verdict `closed "<why>"`, relaunch once with `--time` × 2 | never a third time |
| a grid cell of a listed arm is missing | `expmon launch --exp probes …` | within CAMP_MAX_SUBMIT_PER_WAKE |
| a cell PENDING > 6 h on gpu-be for Priority | `scontrol update JobId=… Partition=gpu,gpu-be` for at most 2 campaign cells at a time | never touch a job the campaign does not own |
| the expmon daemon is gone | `expmon daemon` | — |
| a straggler FAILED in seconds with a benign signature (known_issues.md) | ignore, note once | — |
| a new probe needs a new eval flag or a new grouped set | write it under `scripts/probes/` (free) or add ONE flag to `scripts/eval_mm_docqa.py` + `eval_mm_4b.sh` (patchable, see below), smoke it with `LIMIT=300`, then the full cell | 1 flag per probe |
| a reproduced crash whose fix is ≤ CAMP_PATCH_LINES lines inside a "## Patchable" path | apply it, smoke on the cheapest cell, record `<file> — <traceback line>` in STATE.md; the tick commits it | 1 patch per symptom, then BLOCKED.md |
Anything else: BLOCKED.md and stop.

## Patchable
- freely: `scripts/probes/**` (this campaign's own code), `campaign_probes/**`, `docs/probe_campaign_2026-11.md`,
  new eval sets under `eval_data/probes_*`.
- patchable IF the invariant tests still pass, output pasted in STATE.md for that wake: `scripts/eval_mm_docqa.py` and
  `eval_mm_4b.sh` (adding a probe flag or env; default behaviour must stay byte-identical), tests = `pixi run python
  test_positions.py` and `pixi run python test_mm_collators.py` (through SLURM, `-p gpu-be --gres=gpu:0`), plus a
  `LIMIT=300` plain cell of the reference model that must still read its recorded EM within ±.01.
- never: `pisco/**` (model, collators, training), every training launcher (`qwen35_4b_pt.sh`, `pt_mm_4b.sh`,
  `ft_mm_4b.sh`), `expQ/**` other than `expQ/probes/`, `eval_data/` sets that exist, `campaign_longchunk/**` and
  `expQ/longchunk/**` (the sibling campaign), `EXPERIMENTS.md` except through `expmon note`.

## The wake owns the science
The arms table is a starting hypothesis, not a work order: read the evidence, launch the cells (the owner does not),
propose arms the table lacks (one knob off the baseline, gate written BEFORE the cell), close decided arms, and amend
this law when the evidence refutes it — a wrong prediction is kept and marked, never quietly deleted. Prefer the probe
that would change a training decision over the one that is merely curious; say which decision in the gate column.

## Hard rules
1. Submissions only via `expmon launch --exp probes` / `expmon sweep --exp probes`; never raw sbatch (enforced).
2. Cancel only campaign-owned jobs, only via `expmon kill`. Never `rm` an output dir — rename it `VOID_<name>` (enforced).
3. Never edit campaign.conf, crontab, hooks, `~/.claude`, the tick or the watchdog; never relax a rule here.
   Code edits only as "## Patchable" allows. A `REQUEST` file in this dir is the owner speaking (`expmon campaign
   resolve`): execute it before anything else, then delete it; nothing else may create one. Rules 1–3 are enforced by
   the wake hook: a refusal is a signal to write BLOCKED.md, never to find another spelling. The tick commits for you.
4. Hypothesis ≠ conclusion: write "on track to", "consistent with"; a conclusion needs its gate AND the repeat.
5. Report n (cells, pairs, queries) with every number; a count without its denominator is not a result.
6. End every wake with one line in NEXT (`done` | `wake-when job:<ids> [max <N>m]` | `wake-in <min>` | `waiting-owner`), or nothing while work is left;
   with minutes-long cells running, `wake-when job:<their ids> max 60m` wakes you as they end, not on a guess.
7. Dead ends — do not retry: training of any kind (that is the `longchunk` campaign's job; propose arms to it in
   STATE.md instead); per-head attention scales (SEAL probe 185581: held-out loss +.005, null); compression rate 32
   (slots stop owning their segment, −.17 to −.20 on documents); the coarse-to-fine slot permutation (not a hierarchy);
   untrained slot dropping as a coarse level (−.12 to −.26 on documents); a linear connector or a moving connector;
   a density-adaptive rate; the `<MEM>` mixer; bidirectional attention in the compressor (the hybrid backbone refuses it);
   the multi-position (PPE) layout as an inference probe (a model trained with one id per slot only measures the shift).
   Never touch the sibling campaign's jobs or dirs, nor the owner's `lw-cap` / `build_msmarco` jobs.

## Wake checklist (expmon grid tooling)
1. `expmon grid probes` — where every cell stands (state, verdict, factors, RESULT).
2. `expmon verdict <jid> closed|best "<why>"` for decided cells/arms.
3. Fill missing cells: `expmon launch --exp probes …` (or `expmon sweep --exp probes --dry-run …`, then for real).
4. `expmon probe <jid> judge=<val>` when a cell's number is not picked up by the results script (eval cells are: read the
   `judge` line of `eval_mm_4b_<jid>.out`).
5. Rewrite STATE.md and `docs/probe_campaign_2026-11.md`; write NEXT.

## Escalation
BLOCKED.md (what, why, options, recommendation) and stop. The watchdog mails herve.dejean@naverlabs.com and writes ALERT for what
the loop cannot fix itself: stale ticks, wakes failing fast (login), no GPU-hour for CAMP_IDLE_H, a rejected
push, unpushed work, low disk, missing cron or daemon.

## Close-out (after CAMP_UNTIL)
`expmon campaign report` (one screen; its Settled / Open block is the successor's brief: `expmon campaign init <label> --from <this dir>`) → `expmon campaign retro` → fill its Lessons → `expmon gripe -c retro "RETRO probes: WENT OK: … WENT NOT OK: … NET: …"`
(one line; the cross-cluster feedback digest scores campaigns from it) → the OWNER pastes this dir's SYNC.md (written by retro) into ~/.claude/SYNC.md — never the wake (hard rule 3) → `expmon campaign disarm`.
