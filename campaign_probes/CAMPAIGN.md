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
- under the spread layout one long chunk beats ten short ones (+.037 on HotpotQA, no training); the compressor sees the
  passages together. Mechanism unknown: cross-passage context, fewer chunk boundaries, or fewer slots.
- the decoder's own attention picks the gold page better than the digit route prompt on images (.885 vs .861, 10 pages)
  and worse on text (.873 vs .922).
- dropping half the slots at inference costs .12 on documents under spread (slots own a part) and .04 under tail.
The campaign decides by paired judge accuracy, arm cell − baseline cell on the same questions (95% CI, McNemar); the
unit is judge accuracy in [0, 1], "pt" = .01. Models are FIXED: the reference pair `expQ/ft_mm_4b_184778/model` (s42) and
`184780` (s43) for every "can we recover loss" probe (both seeds before a claim), and the reduced spread / tail pair
`183814` / `183538` for every "why does the layout do X" probe. The deliverable is a ranked list of findings, each with
its paired number, its n, and the training arm it motivates (or the idea it closes).

## Arms — what varies, what is held constant
Every cell is one `eval_mm_4b.sh` run (or a probe script under `scripts/probes/`) on one model and one eval set; the
baseline cell of an arm is the same model, same set, same pixel budget with the probe off. Held constant everywhere:
checkpoints as listed, `PX=1003520` on document sets, judge = rubric prompt, 1,500 questions (`LIMIT` only for
plumbing checks, never for a claim), chunk cap 4096 (`CML=4096`) on text so no passage is cut.

| arm | cell launcher (committed) | factors (what varies) | held constant | baseline cell | gate (metric vs baseline) | cells |
|---|---|---|---|---|---|---|
| A. use-recovery on documents | `eval_mm_4b.sh <ref model> pisco "" eval_data/mm_docqa_1500_1m` with ONE env probe: `STYLE=expand` (copy the evidence, then answer), `GOLD=1` (gold page only), `RAW=route1` (picked page raw), `MQ=int8` | one decoding/prompting change at a time | model, set, PX | the same model's plain cell (exists: judge .610 / .615) | ≥ +3 pt with CI above 0 at both seeds = finding; the "use" share it explains is reported from the transcription files | 2 seeds × probes |
| B. long-chunk mechanism | `CML=4096 TOPK=10 eval_mm_4b.sh 183814 pisco "" <grouped set>` on new grouped sets built by `scripts/probes/build_groups.py`: gold passages joined alone vs distractors joined alone, passage order shuffled inside the joined chunk, g10 with the "+1" slots equalised | which passages share a chunk, their order | model 183814 (and 183538 as the tail control), 10 passages, same slot total ± 1% | g1 (.579) and g10 (.617), files exist | a grouping that keeps the +.037 while removing a candidate cause closes that cause; CI must exclude 0 | ~6 |
| C. reasoning vs reading on text | `CML=4096 eval_mm_4b.sh <ref model> pisco "" eval_data/text_mrqa_ood_1500` with `STYLE=expand`, and the same with the model's thinking mode on (`scripts/probes/` flag if `eval_mm_docqa.py` lacks it) for compressed AND `base` | prompting that gives the decoder room to reason | passage, set | plain compressed .695 / raw .807 | the compressed−raw gap on DROP / DuoRC / RACE shrinks by ≥ 5 pt with the same treatment on both sides | ~6 |
| D. page picking without training | `scripts/probes/route_fusion.py` over the saved attention masses (`expQ/probes/attn_route_*/masses.npz`) and the digit log-probs: rank fusion, product, per-source choice; then `RAW=route1` with the fused pick via the eval's `RANK=` json | the picker | sets retr10 / hotpot / nav frames | digit pick@1 .861 (retr10), .922 (hotpot) | pick@1 ≥ +2 pt on retr10 AND no loss on hotpot; then the route1 judge gain | ~4 |
| E. attention-guided slot pruning | `scripts/probes/prune_by_attention.py`: two passes, drop the 50% / 25% of slots with the least decoder attention at the first answer token, then answer | which slots are kept | model 184778, docs + photos | keep-50% first/even cells (−.12 / −.13 docs on 183814; redo on 184778 as the baseline) | ≥ +4 pt over the best untrained keep-50% at the same slot count = a free coarse level | ~4 |
| F. photo text (TextVQA .77 vs .85) | `eval_mm_4b.sh <ref> pisco_text` (OCR text instead of pixels) on `mm_naturalqa_1500`, TextVQA rows | pixels vs text for text-in-image | model, set | plain .677 / .685 | reported only; a TextVQA gain ≥ 5 pt motivates a mixed input arm | 2 |

Rules: every arm is a committed launcher + a committed cell spec, never raw overrides on the launch line. A
comparison with a difference not listed here is not a result. A relaunch repeats the original command exactly.
Any new eval set or grouping is a ledger `resource:` note the moment it exists (path, size, how built).

## Decision gates
Noise band .02 (`CAMP_NOISE_PT=2`): below it nothing is a result; a tie triggers a repeat on the second seed; above it once
still needs the second seed (or the second set) to be "adopted". Decide arms as soon as their cells complete — do not
wait for the whole grid. Every claim carries n (questions, pairs), the paired CI and McNemar p.
- **Feasibility gate (first, alone):** re-run the reference model's plain document cell (`eval_mm_4b.sh expQ/ft_mm_4b_184778/model pisco "" eval_data/mm_docqa_1500_1m`, PX=1003520) and pair it with the recorded file
  (`expQ/ft_mm_4b_184778/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json`, judge .610). The judge-on-judge difference
  must be within ±.01 with the CI inside the noise band; otherwise the instrument does not register changes of the size
  this campaign looks for, and the campaign STOPS with "the judge is not reproducible at the .02 level" as its finding.
- **Health number per cell:** `unparseable: 0` in the eval log, 1,500 rows in the result file, and the total `mems_used`
  within 1% of the baseline cell's (a different slot total is a different budget, not a probe result). A cell failing
  any of these is `closed "health"` and relaunched once.
- **Small cells are plumbing:** a `LIMIT=300` cell proves the code path runs; it never decides an arm, even when it agrees.
- A NEW metric (one the wake designs itself) closes nothing until its row here states its n and a known-positive control it
  recovers (the attention-mass picker recovering .885 on retr10 is such a control for arm D).
- Headline claims in STATE.md and the ledger are prefixed `provisional:` until the next critic pass, or until the
  next wake re-derives them from `expmon grid`.

## Budget and caps
- cells for the whole campaign: 60 · GPU-hours: 40 (a judged 1,500-Q cell is 20-45 min on one A100)
- at most CAMP_MAX_SUBMIT_PER_WAKE new submissions per wake (campaign.conf)
- every launch: one GPU, `-p gpu-be` by default (`--partition gpu,gpu-be` when gpu-be is jammed; the owner's 8-GPU cap on
  `gpu` is shared with the sibling training campaign `longchunk`, so never more than 2 cells on `gpu` at once); `--time` =
  measured cell wall × 3 (a 1,500-Q judged document cell measured 28-45 min, photos 18 min, text 17 min)
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
