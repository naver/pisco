# STATE — campaign probes   (rewritten WHOLESALE by every LLM wake; never append)
last wake: 2026-11-12 01:30-01:40 KST (alert: a false "loop not armed" watchdog alarm; nothing else changed since
2026-11-11 09:50)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Showing the routed page raw recovers the whole document gap, at both seeds. Reference models, documents
  (`mm_docqa_1500_1m`, 4 pages, PX=1003520), n=1,500 each: seed 42 judge .820 vs plain .610 = +.210 [+.187, +.233],
  350 up / 35 down, McNemar p<1e-60, slots .749× plain + one raw page (job 188441, verdict best); seed 43 .822 vs .615
  = +.207 [+.183, +.232], 349 / 38, slots .748× + one raw page (job 188460, verdict best). The pick is right on 98.6% /
  98.9% of questions; the all-raw reader scores .809.
- Twice the slots on the picked page is NOT enough (arm G closed). Picked page re-compressed at rate 4, other pages
  rate 8, documents, n=1,500 each, slots 1.250× plain: seed 42 .673 vs plain .610 = +.063 [+.043, +.083], 158 / 64,
  p<.001 (job 188590, homer-011 = baseline's node, verdict closed); seed 43 .663 vs .615 = +.049 [+.030, +.067],
  147 / 74, p<.001 (job 188591, homer-012 el9_8 vs baseline homer-016 el9_8, verdict closed). Gate ≥ +.14 failed at
  both seeds. Vs the raw page: −.147 [−.169, −.126] (s42), −.159 [−.180, −.137] (s43). Extra decoder prompt tokens
  per question vs plain: +107.5 (rate 4) vs +752.9 (raw page) = .143×. The models were trained at rate 8 only.
- Removing the distractor pages instead gains only about 2 pt, below the +3 pt gate at both seeds: gold page only,
  compressed, slots .270×, seed 42 .632 vs .610 = +.022 [+.005, +.040], p=.020 (job 188440); seed 43 .639 vs .615 =
  +.025 [+.006, +.043], p=.010 (job 188459).
- Slots stored in int8 change nothing, at both seeds: seed 42 .611 vs .610 = +.001 [−.003, +.006], 1,445/1,500
  identical (job 188478); seed 43 .613 vs .615 = −.001 [−.005, +.003], 4 / 6, 1,444/1,500 identical, n=1,500 (job
  188536, verdict best; homer-011 vs baseline homer-016, both el9_8). Adopted as a serving option.
- Given the question, the most-attended half of the slots carries what the decoder uses, at both seeds (arm E
  adopted). Documents, n=1,500 each, slots .496×: seed 42 .610 vs .610, +.000 [−.009, +.010] (job 188485, homer-016
  vs baseline homer-011, both el9_8); seed 43 .615 vs .615, +.000 [−.009, +.009], 25 / 25 (job 188522, verdict best).
  Blind every-2nd half at the same slot count: seed 42 .505 (job 188483), seed 43 .508 (job 188537, verdict closed),
  so attention minus blind = +.105 [+.085, +.126], 205 / 47, p=1e-24 and +.107 [+.087, +.127], 203 / 43, p=5e-26.
  Gate (≥ +4 pt, CI above 0, both seeds) met.
- The long-chunk gain is context between the two answer passages, at both seeds. HotpotQA, 1,500 Q each, golds
  joined in one chunk (others alone) vs ten separate chunks on today's code: seed 42 (183814) .655 vs .582 = +.073
  [+.051, +.095], p=1.5e-10 (jobs 188471 vs 188486); seed 43 (183816) .669 vs .601 = +.067 [+.046, +.088], 186 / 85,
  p=8e-10 (jobs 188511 vs 188509, verdict best). Slots .998× vs ten chunks. Vs one chunk of all ten (slots 1.020×):
  seed 42 +.038 [+.015, +.061], 185 / 128, p=.002 (vs 188518); seed 43 +.053 [+.029, +.076], p=9e-6 (vs 188510).
- One chunk of all ten vs ten chunks is above the band at seed 42 only: seed 42 .617 vs .582 = +.035 [+.012, +.059],
  184 / 132, p=.004, slots .978× (job 188518, verdict closed; 1,500/1,500 answers identical to the 11-07 file 185601);
  seed 43 .615 vs .601 = +.014 [−.008, +.035], p=.23 (job 188510).
- The no-oracle grouping is not adopted. Top 3 passages by 183814's attention joined, others alone: seed 42 .614 vs
  .582 = +.032 [+.009, +.056], p=.007 (job 188521, verdict closed); seed 43 .615 vs .601 = +.013 [−.007, +.034],
  139 / 119, p=.24, n=1,500 (job 188546: verdict closed in verdicts.jsonl 07:35, no grid row — known issue; its file
  `expQ/probes/b_pred3_attn_s43/eval/` has 1,500 rows, judge .6147). Against one chunk of ten: −.003 (s42), −.001 (s43).
- Passage order inside a long chunk is unused, both seeds on today's code: seed 42 .615 vs .617 = −.002 [−.023,
  +.020], 141 / 144, p=.91 (188473 vs 188518); seed 43 .618 vs .615 = +.003 [−.021, +.026], p=.87 (188513 vs 188510).
- Joining the eight distractor passages is not adopted: seed 42 .604 vs .582 = +.022 [+.003, +.041], p=.027 (job
  188472 vs 188486); seed 43 .617 vs .601 = +.016 [−.003, +.035], p=.11 (job 188512 vs 188509).
- Blind slot halving on 184778 (documents, 1,500 Q): every 2nd slot (slots .504×) −.105 [−.127, −.085] (job 188483);
  first half of each chunk (slots .496×) .495, −.115 [−.136, −.095] (job 188482).
- Arm G flag checks: first version's rate-8 control FAILED, 233/300 identical (job 188525, verdict closed); patched
  version (page re-compressed inside its example's padded batch) PASSED on homer-013 (job 188580, verdict closed):
  tests 12/12 + 33/33 + 5/5, plain 300/300, rate-8 control 300/300 identical to plain (judge .610 = .610, slots 118,147).
- Data: three grouped HotpotQA sets (job 188465), six no-oracle grouped sets (job 188516), 1,500 rows each;
  `eval_data/probes_textvqa_ocr` (job 188545): 500 TextVQA rows, 500/500 questions matched.

**Provisional (one seed, or a tie the law wants repeated)**
- provisional (a tie at one seed; the law's repeat at seed 43 was not run for the cell cap): evidence-then-answer on
  documents gains nothing at seed 42: .611 vs .610 = +.001 [−.017, +.021], 108 / 106, p=.95, n=1,500, slots 1.000×
  (job 188439, verdict closed; file from the homer-011 attempt, same node as 184781).
- provisional (one seed): keeping only the most-attended QUARTER still ties plain. 184778, documents, n=1,500, slots
  .246×: .611 vs .610 = +.001 [−.009, +.012], 37 / 35, p=.91, 1,197/1,500 identical (job 188538, verdict best,
  homer-013). Seed 43 not run (cell cap: BLOCKED.md asks the owner).
- provisional (one seed, no blind control on photos): the attention half ties plain on photos. 184778,
  `mm_naturalqa_1500`, n=1,500, slots .492×: .677 vs .677, +.000 [−.013, +.013], 49 / 49 (job 188539, verdict closed).
- provisional: the instrument gives the same answers on re-run on every node tested except homer-008: plain document
  cell 1,500/1,500 on homer-011 (job 188419); first 300 questions 300/300 on homer-016 (188476, 188524), homer-007
  (188477) and homer-013 (188580); compressed text 1,500/1,500 between homer-016 and homer-007 (188471 rerun) and
  between 11-07 and 11-11 code (188518 vs 185601); homer-008 changed 57/300 image answers (188458). homer-006 is
  suspected (by elimination) of the 55/1,500 compressed-text drift of 185505 vs 188486; homer-005 and 009 are untested;
  homer-012 (el9_8) carries 188522, 188537, 188546, 188591 without a check of its own. All el9_7 nodes are excluded.
- provisional: page picking (arm D, model 183814, n=1,500 per set, job 188515): 10-page documents digit .861,
  attention .885 (+.025 [+.010, +.039]); HotpotQA digit .922, attention .873. The per-source switch that passes the
  gate was chosen on the same questions. Expected judge gain ≈ +.006 to +.008 (under the .02 band), no judge cell.
- provisional, one model: the cross-passage gain needs no spread layout. Tail-layout model 183538, golds joined
  .627 vs ten chunks .569 = +.058 [+.035, +.082], 202 / 115, p<.001, slots .998× (jobs 188514 vs 188519, verdict
  best). The grid lists 188514's parent as 185505 (a label from its launch), its baseline is 188519.
- provisional: evidence-then-answer does not close the text gap at seed 42 (MRQA, n=1,500): compressed .699 vs plain
  .718 (−.019 [−.041, +.003], p=.11, job 188446); uncompressed .807 = .807 (job 188457). The law's ".695" baseline is
  model 183814 with 256-token chunks (job 185775); the arm uses the reference model 184778 with one chunk per passage
  (job 188445, .718).
- provisional (not a gate, derived from the two arm G cells): per extra decoder token, rate 4 on the picked page
  recovers about twice what the raw page does: +.063 / 107.5 tokens ≈ .059 per 100 tokens (s42) vs +.210 / 753 ≈ .028.

**Open, and running** (0 campaign jobs at 2026-11-12 01:35)
- none running. Every arm is decided or closed for the cell cap (60 of 60 used).
- owner: BLOCKED.md asks to raise the cell cap 60 → 65 (keep a quarter at seed 43 is the main reason).

## What happened this wake
- ALERT (01:00): "crontab has no 'campaign tick' line — the loop is not armed". False alarm: the ticks at 01:00 and
  01:30 both ran and recorded `cron=3`, the 01:17 watchdog said OK, and `expmon campaign` reports `armed: yes`. One
  empty `crontab -l` read by the 01:00 watchdog (known expmon defect). ALERT deleted, known_issues row added, gripe
  filed, ledger decision written.
- ✗ lines in the brief: none of this campaign's (the one ✗ belongs to another campaign).
- Verifier (09:30 STATE): 15 supported, 8 provisional, 0 unsupported. Every provisional line it lists is already
  marked provisional below; nothing to fix.
- The tick resubmitted the expmon daemon at 01:00 (it had exited); no action needed from this wake.
- Nothing launched: all 60 cells are used. BLOCKED.md (raise the cap 60 → 65) still has no answer.

## Owned jobs (jobid | arm | state at 09:45 | node | note)
(none running)

## Adopted jobs
(none. 188422, 188466-188470 belong to the sibling `longchunk` campaign; 188582-188585 to `freeze`.)

## Open arms and the next decision for each
| arm | status | next action |
|---|---|---|
| A. use-recovery | decided: route1 adopted, int8 adopted, gold-only closed, evidence-then-answer closed at one seed (provisional tie) | expand s43 only if the cap is raised |
| B. long-chunk mechanism | closed | none |
| C. reasoning vs reading | closed for budget | none |
| D. page picking | decided, caveat: rule chosen in-sample | none |
| E. attention-guided pruning | half adopted (both seeds); quarter met at s42 | if the cap is raised: `e_keep025_attng_s43` (→ 184780, parent 184784, `KEEPF=0.25 KEEPM=attn_global PX=1003520`, `--time 4:54:00`) and `e_keep0125_attng` (→ 184778, parent 184781, `KEEPF=0.125`); dirs to create with the symlink rule |
| F. photo text vs pixels | set built (188545) | only if the cap is raised: `f_text_s42` → 184778 symlink, `pisco_text "" eval_data/probes_textvqa_ocr`, PX=1003520, `--time 2:12:00`, paired on query_id with the TextVQA rows of 184783's photo file (n=500) |
| G. route then densify | closed (both seeds below the gate) | none |

## Health and budget
- cells: 60 of 60 used (every submission counts, killed ones too; 188546 counted by hand). A launch needs the owner.
- GPU-hours of 40: tick sensor 15.8 GPU-h in the last 24 h at 2026-11-12 01:30 (no campaign GPU job since 11-11 09:15).
- gpu-be preempted 8+ campaign attempts on 2026-11-11; the two arm G cells were not preempted.

## Next wake
Only the owner can unblock (NEXT = `waiting-owner`). If the cap is raised: launch the cells in the arm table above
(quarter s43 first), each `--exclude=homer-[005-009]`, symlink rule, dry-run first. If the owner chooses to keep 60:
close-out is NOT due before 2027-02-09; the owner disarms.

## Blockers / needs the owner
- BLOCKED.md: raise the cell cap 60 → 65 (recommended) so the keep-a-quarter result gets its second seed.

## Proposals for the sibling `longchunk` campaign (training is not this campaign's job)
- Stage-1/2 data that packs related passages in one chunk (both gold passages of a multi-hop question) rather than
  random long chunks: +.067-.073 with the golds together (both seeds), +.058 on the tail layout. Joining all ten:
  +.035 (seed 42) / +.014 n.s. (seed 43). A ranker's top 3 joined does no better than all ten.
- A query-aware slot budget (nested-budget training): half the slots, picked by the decoder's attention with the
  question, lose nothing at both seeds; a quarter loses nothing at seed 42.
- A mixed-rate stage 2 (the picked page at rate 4, others at rate 8): untrained, rate 4 on the picked page gains
  +.063 / +.049 at 1/7 of the raw page's extra decoder tokens; whether training at rate 4 closes more of the raw
  page's +.21 is open.
