# STATE — campaign probes   (rewritten WHOLESALE by every LLM wake; never append)
last wake: 2026-11-13 09:30-09:55 KST (woken as the keep-an-eighth cell left the queue; it ties plain at seed 42;
every arm is now decided or closed; the seed-43 repeat needs a cell beyond the cap → BLOCKED.md for the owner)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Showing the routed page raw recovers the whole document gap, at both seeds. Reference models, documents
  (`mm_docqa_1500_1m`, 4 pages, PX=1003520), n=1,500 each: seed 42 judge .820 vs plain .610 = +.210 [+.187, +.233],
  350 up / 35 down, McNemar p<1e-60, slots .749× plain + one raw page (job 188441, verdict best); seed 43 .822 vs .615
  = +.207 [+.183, +.232], 349 / 38, slots .748× + one raw page (job 188460, verdict best). The pick is right on 98.6% /
  98.9% of questions; the all-raw reader scores .809.
- Asking the decoder to copy the evidence before answering gains nothing on documents, at both seeds (arm A
  evidence-then-answer closed). n=1,500 each, slots 1.000×: seed 42 .611 vs .610 = +.001 [−.017, +.021], 108 / 106,
  p=.95 (job 188439, homer-011 = baseline's node, verdict closed); seed 43 .606 vs .615 = −.009 [−.027, +.011],
  98 / 111, p=.41, 889/1,500 answers identical, infographics −.036 [−.070, −.002] (job 189058, homer-011 vs baseline
  184784 on homer-016, both el9_8, verdict closed). Gate ≥ +3 pt failed at both seeds.
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
- Even the most-attended QUARTER of the slots carries what the decoder uses, at both seeds (gate: within .02 of
  plain → three quarters unused per question; met). Documents, n=1,500 each, slots .246× (106 per question instead
  of 431): seed 42 .611 vs .610 = +.001 [−.009, +.012], 37 / 35, p=.91, 1,197/1,500 identical (job 188538, homer-013,
  verdict best); seed 43 .619 vs .615 = +.005 [−.007, +.017], 45 / 38, p=.51, 1,200/1,500 identical (job 189056,
  homer-019 vs baseline 184784 on homer-016, both el9_8, verdict best).
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
  139 / 119, p=.24, n=1,500 (job 188546: verdict closed in verdicts.jsonl, no grid row — known issue; its file
  `expQ/probes/b_pred3_attn_s43/eval/` has 1,500 rows, judge .6147). Against one chunk of ten: −.003 (s42), −.001 (s43).
- Passage order inside a long chunk is unused, both seeds on today's code: seed 42 .615 vs .617 = −.002 [−.023,
  +.020], 141 / 144, p=.91 (188473 vs 188518); seed 43 .618 vs .615 = +.003 [−.021, +.026], p=.87 (188513 vs 188510).
- Joining the eight distractor passages is not adopted: seed 42 .604 vs .582 = +.022 [+.003, +.041], p=.027 (job
  188472 vs 188486); seed 43 .617 vs .601 = +.016 [−.003, +.035], p=.11 (job 188512 vs 188509).
- Blind slot halving on 184778 (documents, 1,500 Q): every 2nd slot (slots .504×) −.105 [−.127, −.085] (job 188483);
  first half of each chunk (slots .496×) .495, −.115 [−.136, −.095] (job 188482).
- Arm F decided (reported only; the law allows seed 42 only, its seed-43 baseline ran on homer-008): reading photo
  text as OCR words instead of pixels is far worse. Model 184778, TextVQA, 500 questions paired on query_id with the
  TextVQA rows of the pixel cell 184783 (homer-011): judge .346 vs .780 = −.434 [−.480, −.384], 17 up / 234 down,
  McNemar p=6e-50; EM .292 vs .714; slots .081× (4.4 vs 54.2 per question); health unparseable 0, 500/500 rows,
  homer-012 (job 189059, verdict closed). The gold answer is in the OCR words on only .596 of rows (job 188545).
  Gate (gain ≥ 5 pt motivates a mixed-input arm) not met → no mixed-input arm.
- Arm G flag checks: first version's rate-8 control FAILED, 233/300 identical (job 188525, verdict closed); patched
  version (page re-compressed inside its example's padded batch) PASSED on homer-013 (job 188580, verdict closed):
  tests 12/12 + 33/33 + 5/5, plain 300/300, rate-8 control 300/300 identical to plain (judge .610 = .610, slots 118,147).
- Data: three grouped HotpotQA sets (job 188465), six no-oracle grouped sets (job 188516), 1,500 rows each;
  `eval_data/probes_textvqa_ocr` (job 188545): 500 TextVQA rows, 500/500 questions matched.

**Provisional (one seed, or no control)**
- provisional (one seed, no blind control on photos): the attention half ties plain on photos. 184778,
  `mm_naturalqa_1500`, n=1,500, slots .492×: .677 vs .677, +.000 [−.013, +.013], 49 / 49 (job 188539, verdict closed).
- provisional: the instrument gives the same answers on re-run on every node tested except homer-008: plain document
  cell 1,500/1,500 on homer-011 (job 188419); first 300 questions 300/300 on homer-016 (188476, 188524), homer-007
  (188477) and homer-013 (188580); compressed text 1,500/1,500 between homer-016 and homer-007 (188471 rerun) and
  between 11-07 and 11-11 code (188518 vs 185601); homer-008 changed 57/300 image answers (188458). homer-006 is
  suspected (by elimination) of the 55/1,500 compressed-text drift of 185505 vs 188486; homer-005 and 009 are untested;
  homer-012 and homer-019 (el9_8) carry cells without a check of their own. All el9_7 nodes are excluded.
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
- provisional (not a gate, a wake-designed ratio from the two arm G cells, no CI, no control): per extra decoder token,
  rate 4 on the picked page recovers more than the raw page does: seed 42 +.063 / 107.5 tokens ≈ .059 per 100 tokens
  vs raw +.210 / 753 ≈ .028 (about 2×, jobs 188590 / 188441); seed 43 .049 / 107.5 ≈ .046 vs .207 / 753 ≈ .027
  (about 1.7×, jobs 188591 / 188460).
- provisional (seed 42 only, the seed-43 repeat needs a cell beyond the cap): even the most-attended EIGHTH of the
  slots carries what the decoder uses. 184778, documents (`mm_docqa_1500_1m`, PX=1003520), n=1,500, slots .121× (52.2
  per question instead of 430.7; mems_used 78,269 vs 646,042): .613 vs plain .610 = +.003 [−.011, +.017], 59 / 55,
  p=.78, 1,107/1,500 identical; vs keep-25% (188538) +.001 [−.011, +.014], 48 / 46 (job 189057, homer-019 vs
  baseline 184781 on homer-011, both el9_8, verdict closed). Gate (within .02 → seven eighths unused) met at one seed.

**Open, and running**
- (none running.) One question for the owner (BLOCKED.md): use the spare 65th cell for keep-an-eighth at seed 43
  (option A), plus one new cell for keep-a-sixteenth at seed 42 (option B, recommended), or stop (option C).

## What happened this wake
- Keep-an-eighth cell 189057 left the queue. sacct: first attempt preempted on `gpu-be` homer-013 at 01:34 (nothing
  written); last attempt on `gpu` homer-019, 08:36-09:06, `.batch` COMPLETED exit 0; result file written 09:06 (job
  label PREEMPTED from the first attempt: known_issues row, 189057 added). homer-019 is el9_8 (`scontrol show node`),
  like the baseline's homer-011.
- Health: `unparseable: 0`, 1,500 rows, mems_used ratio .121× (pruning cell, changes the slot total by design).
- Paired with `scripts/paired_judge.py` against `expQ/ft_mm_4b_184778/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json`
  and against the keep-25% file 188538. Recorded `expmon probe 189057 judge=0.6127 em=0.516`, verdict closed
  (provisional, one seed), ledger result note.
- CAMPAIGN.md arm E row: keep-12.5% marked decided at seed 42 only. `docs/probe_campaign_2026-11.md`: status, finding 3,
  arm E table and gate text rewritten. `expmon display --exp probes` refreshed (still 0 table rows: known issue).
- BLOCKED.md written: the noise-band rule says a tie is repeated on the second seed; the cap gives no cell for it.
- Verifier: 0 unsupported; its six items are all labelled provisional and stay so.

## Owned jobs (jobid | cell | arm | state at 11-13 09:50 | intent — also in the ledger)
| job | cell dir (`expQ/probes/…`) → model | arm | state | intent |
|---|---|---|---|---|
| 189057 | `e_keep0125_attng` → 184778 | E | done 09:06, decided (seed 42, provisional) | see Provisional |
| 189058 | `a_expand_s43` → 184780 | A | done 08:03, decided | see Settled |
| 189056 | `e_keep025_attng_s43` → 184780 | E | done 07:32, decided | see Settled |
| 189059 | `f_text_s42` → 184778 | F | done 02:47, decided | see Settled |

## Adopted jobs
(none. 188466, 188707-188712, 189691-189695 belong to the sibling `longchunk` campaign; 189094, 189095-189099,
189434, 189445-189450 to the sibling `freeze` campaign.)

## Open arms and the next decision for each
| arm | status | next action |
|---|---|---|
| A. use-recovery | decided at both seeds: route1 adopted, int8 adopted, gold-only closed, evidence-then-answer closed | none |
| B. long-chunk mechanism | closed | none |
| C. reasoning vs reading | closed for budget | none |
| D. page picking | decided, caveat: rule chosen in-sample | none |
| E. attention-guided pruning | half adopted, quarter adopted (both seeds); eighth ties plain at seed 42 (provisional) | owner answer to BLOCKED.md: A → launch `e_keep0125_attng_s43` (184780) in the 65th cell; B → that plus `e_keep00625_attng` (184778, `KEEPF=0.0625`, gate in BLOCKED.md); C → done |
| F. photo text vs pixels | decided 11-13 (−.434), closed | none |
| G. route then densify | closed (both seeds below the gate) | none |

## Health and budget
- cells: 64 of 65 used (61-64 = 189056-189059, all passed health). The 65th (the one health relaunch) is unused and,
  by the owner's rule, may not be spent on anything else without the owner.
- GPU-hours of 40: tick sensor 2.5 GPU-h in the last 24 h.
- Widened cells: none now (189057 is gone).
- Pairing nodes: every image cell needs its node from the attempt that wrote its file (same node rule; compare the
  `eval/` file mtime with `sacct -j <id> -D -o NodeList,End`; a job-level PREEMPTED label can hide a completed attempt).

## Next wake
- NEXT = `waiting-owner`: only the owner's answer to BLOCKED.md (REQUEST / answer) moves the campaign.
- If A or B: launch with the measured rule (`--time 4:54:00`, `-p gpu-be`, `--exclude=homer-[005-009]`, symlinked
  model dir `expQ/probes/<cell>/model`), same env as 189057 (`KEEPF=… KEEPM=attn_global PX=1003520`); then
  `wake-when job:<ids> max 12h`. Seed-43 baseline: 184784 (`expQ/ft_mm_4b_184780/eval/…mm_docqa_1500_1m.json`,
  homer-016, el9_8).
- If C: write `done`; the close-out steps of CAMPAIGN.md apply after 2027-02-09.

## Blockers / needs the owner
- BLOCKED.md: spend the spare 65th cell (and one more) to settle the eighth at seed 43 and find where pruning starts to
  cost? Recommendation B.

## Proposals for the sibling `longchunk` campaign (training is not this campaign's job)
- Stage-1/2 data that packs related passages in one chunk (both gold passages of a multi-hop question) rather than
  random long chunks: +.067-.073 with the golds together (both seeds), +.058 on the tail layout. Joining all ten:
  +.035 (seed 42) / +.014 n.s. (seed 43). A ranker's top 3 joined does no better than all ten.
- A query-aware slot budget (nested-budget training): a quarter of the slots, picked by the decoder's attention with
  the question, lose nothing at both seeds (+.001 / +.005); an eighth loses nothing at seed 42 (+.003, provisional).
- A mixed-rate stage 2 (the picked page at rate 4, others at rate 8): untrained, rate 4 on the picked page gains
  +.063 / +.049 at 1/7 of the raw page's extra decoder tokens; whether training at rate 4 closes more of the raw
  page's +.21 is open.
- Not proposed: evidence-then-answer labels (no gain on documents at either seed, none on text at seed 42); OCR text
  as an extra input for photos (arm F: OCR words instead of pixels −.434 on TextVQA).
