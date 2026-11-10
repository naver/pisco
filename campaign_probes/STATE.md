# STATE — campaign probes   (rewritten WHOLESALE by every LLM wake; never append)
last wake: 2026-11-11 06:00-06:45 KST (woken by the condition "arm E cells 188482 / 188484 / 188485 ended")

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Showing the routed page raw recovers the whole document gap, at both seeds. Reference models, documents
  (`mm_docqa_1500_1m`, 4 pages, PX=1003520), n=1,500 each: seed 42 judge .820 vs plain .610 = +.210 [+.187, +.233],
  350 up / 35 down, McNemar p<1e-60 (job 188441, verdict best); seed 43 .822 vs .615 = +.207 [+.183, +.232], 349 / 38
  (job 188460, verdict best). The pick is right on 98.6% / 98.9% of questions; the all-raw reader scores .809.
- Removing the distractor pages instead gains only about 2 pt, below the +3 pt gate at both seeds: gold page only,
  compressed, seed 42 .632 vs .610 = +.022 [+.005, +.040], p=.020 (job 188440); seed 43 .639 vs .615 = +.025
  [+.006, +.043], p=.010 (job 188459). So the document loss is reading the compressed answer page, not distraction.
- The long-chunk gain is context between the two answer passages, at both seeds. HotpotQA, 1,500 Q each, golds
  joined in one chunk (others alone) vs ten separate chunks on today's code: seed 42 (183814) .655 vs .582 = +.073
  [+.051, +.095], p=1.5e-10 (jobs 188471 vs 188486); seed 43 (183816) .669 vs .601 = +.067 [+.046, +.088], 186 / 85,
  p=8e-10 (jobs 188511 vs 188509, verdict best). Slots .998×. Vs one chunk of all ten at seed 43: +.053 [+.029, +.076],
  p=9e-6 (vs 188510).
- One chunk of all ten is NOT shown to beat ten chunks at seed 43: .615 vs .601 = +.014 [−.008, +.035], p=.23
  (job 188510). The campaign's starting fact (+.037, seed 42, 11-07 files) is half noise; marked in the law.
- Passage order inside a long chunk does not matter, both seeds: seed 42 −.002 [−.023, +.020] (job 188473, vs the
  11-07 one-chunk file); seed 43 .618 vs .615 = +.003 [−.021, +.026], p=.87 (job 188513).
- Joining the eight distractor passages is under the noise band at both seeds: +.022 [+.003, +.041] p=.027 (job
  188472), +.016 [−.003, +.035] p=.11 (job 188512), against the ten-chunk cells of the same day's code.
- Blind slot halving on the reference model 184778 (documents, 1,500 Q, slots .50×): every 2nd slot .505 vs .610 =
  −.105 [−.127, −.085], p=3e-24 (job 188483); first half of each chunk .495, −.115 [−.136, −.095], p=4e-28 (job 188482).
- The instrument is reproducible on every node tested except homer-008: plain document cell 1,500/1,500 identical
  on homer-011 (job 188419); first 300 questions 300/300 identical on homer-016 (188476) and homer-007 (188477);
  compressed HotpotQA text 1,500/1,500 identical between homer-016 and homer-007 (job 188471 and its rerun);
  homer-008 changed 57/300 image answers (188458).
- Data: three grouped HotpotQA sets (job 188465) and six no-oracle grouped sets (job 188516; both gold passages in
  the model's top 2 on .404 digit / .495 attention / .487 product of the 1,500 questions), 1,500 rows each.
- provisional: given the question, half the slots carry what the decoder uses (seed 42 only). Keep the most-attended
  half over all pages, 184778 documents, 1,500 Q: .610 vs .610, +.000 [−.009, +.010], 27 / 27, 1,289 / 1,500 answers
  identical, slots .496× (job 188485); per chunk .606, −.004 [−.015, +.007] (job 188484). +.105 over the better blind
  half: arm E gate met at seed 42. Needs a full pass over all slots first, so it saves decoder context, not storage.
  Seed-43 repeat = job 188522.
- provisional: page picking (arm D, model 183814, n=1,500 per set, job 188515): 10-page documents digit .861,
  attention .885 (+.025 [+.010, +.039]); HotpotQA digit .922, attention .873. The per-source switch that passes the
  gate was chosen on the same questions (no held-out split). No judge cell: expected judge gain = +.025 × the route1
  hit-minus-miss judge gap (.252 at seed 42: 1,479 hits .824 vs 21 misses .571, file a_route1_s42; .325 at seed 43 on
  16 misses) ≈ +.006 to +.008, under the .02 band.
- provisional: slots stored in int8 change nothing at seed 42: .611 vs .610 = +.001 [−.003, +.006], 1,445/1,500
  answers identical, n=1,500 (job 188478). Seed-43 repeat not launched yet.
- provisional: evidence-then-answer does not close the text gap at seed 42 (MRQA, n=1,500, 250 per source):
  compressed .699 vs plain .718 (−.019, p=.11, job 188446); uncompressed .807 = .807 (job 188457; its plain side
  185542 ran on homer-008 with 11-06 code, 300/300 re-checked by 188456). Per-source gaps rest on 250 Q, no CI.

**Open, and running** (10 campaign jobs at 06:40)
- arm A: evidence-then-answer on documents, seed 42 (job 188439, running on homer-011).
- arm B: one chunk of ten re-run on today's code, 183814 (188518, requeued after a preemption at 06:10, no file);
  golds joined on the tail-layout model 183538 (188514) + its ten-chunk re-run (188519); no-oracle grouping, the 2 /
  3 passages the model's attention ranks first joined (188520 / 188521).
- arm E: keep the most-attended half overall, seed 43, 184780 (188522, queued).
- arm G: invariant tests (188523), 300-Q plain regression (188524), 300-Q `DENSE=8` control (188525).

## What happened this wake
- Paired the six ended cells: arm E ×3 (+ every-2nd verdict the verifier asked for), arm B seed 43 ×5 (188509-188513).
  All files: 1,500 rows, unparseable 0; nodes el9_8 (homer-010, 013, 016) like their baselines.
- Verifier findings fixed: arm D's .25 gap now cites its file and counts; the doc's false "chosen on the other half"
  is corrected; 188483 has a verdict; arm B's vs-one-chunk-of-ten claims at seed 42 wait on the re-run 188518; arm C's
  plain uncompressed side is marked (homer-008, 11-06 code).
- Law amended (CAMPAIGN.md): arm B decided + the "+.037" mission fact marked partly wrong; arm E rewritten (the
  launcher it really uses, the 25% drop replaced by a 75% drop with its gate written before the cell); arm D caveat;
  arm G checks named.
- Code (patchable path, 25 lines): `--dense_rate R` in `scripts/eval_mm_docqa.py` (+ `make_collator(compr_rate=)`)
  and env `DENSE` in `eval_mm_4b.sh`. With `--raw_policy`, the picked pages are re-compressed alone at rate R (text and
  image) and stay `<MEM>`s. Default path untouched; the summary gains `dense_rate` only when set. `py_compile` and
  `bash -n` pass; the invariant tests are job 188523 (output to paste next wake).
- 8 submissions (cap 8): 188518-188522 (arm B ×4, arm E ×1), 188523-188525 (arm G checks).
- `docs/probe_campaign_2026-11.md` rewritten: findings re-ranked (arm B now #2, arm E #3), arm B / E / G tables.
- Gripe: `expmon launch` refuses every new cell as "closed by 188444" (empty factors), so `--force` each time.

## Owned jobs (jobid | arm | state at 06:40 | node | note)
| 188439 | A evidence-then-answer s42 | RUNNING | homer-011 | `a_expand_s42`; vs 184781 .610 |
| 188514 | B golds joined, tail 183538 | RUNNING | homer-012 | `b_gold_joined_tail`; pair with 188519, not the 11-07 .568 |
| 188518 | B one chunk of ten, 183814, today's code | PENDING (requeued) | — | `b_g10_rep`; vs 11-07 file 185601 .617 |
| 188519 | B ten chunks, tail 183538, today's code | RUNNING | homer-016 | `b_g1_tail_rep` |
| 188520 | B no-oracle top-2 attention joined, 183814 | RUNNING | homer-010 | `b_pred2_attn`; vs 188486 .582 |
| 188521 | B no-oracle top-3 attention joined, 183814 | RUNNING | homer-020 | `b_pred3_attn`; vs 188486 .582 |
| 188522 | E most-attended half overall, s43 | PENDING | excl. 005-009 | `e_keep05_attng_s43`; vs 184784 .615 |
| 188523 | G invariant tests | RUNNING | homer-008 (CPU) | `run_tests_cpu.sh` |
| 188524 | G plain regression, 300 Q | PENDING afterok 188523 | excl. 005-009 | `reg_g_plain`; EM within .01 of 184781's first 300 |
| 188525 | G `RAW=route1 DENSE=8`, 300 Q | PENDING afterok 188523 | excl. 005-009 | `g_dense8_ctrl`; must give the plain answers |

## Adopted jobs
(none. 188422, 188466-188470 belong to the sibling `longchunk` campaign; 188527 `check_module_move.sh` is not ours.)

## Open arms and the next decision for each
| arm | status | next action |
|---|---|---|
| A. use-recovery | route1 adopted; gold-only closed; int8 tie at s42; expand s42 running | 188439 ends: pair vs 184781; tie or gain → expand s43 (`a_expand_s43` → 184780, `STYLE=expand`, `--time 5:12:00`). Launch int8 s43 (`MQ=int8`, `a_mq8_s43` → 184780, parent 184784, `--time 4:54:00`) |
| B. long-chunk mechanism | adopted: context between answer passages (both seeds) | 188520 / 188521: no-oracle minus 188486 (.582) ≥ +.03 CI above 0 → "rank, then re-compress the top passages together" works without an oracle (propose to `longchunk`: pack related passages). 188514 vs 188519: ≥ +.02 CI above 0 → no spread layout needed; drop ≥ .02 → needs spread. 188518: re-pair the seed-42 vs-one-chunk claims |
| C. reasoning vs reading | expand at s42: gap does not shrink | seed-43 repeat (`c_plain_s43`, `c_expand_s43` → 184780, CML=4096; `--time` 0:51:00 / 5:12:00) only if the cell budget allows (see Health) |
| D. page picking | decided (job 188515), caveat: rule chosen in-sample | none |
| E. attention-guided pruning | gate met at s42 | 188522 within .02 of .615 → adopt. Then photos (`e_keep05_attng_photo` → 184778, `eval_data/mm_naturalqa_1500`, vs 184783 .677 on homer-011) and keep a quarter (`KEEPF=0.25 KEEPM=attn_global`, `e_keep025_attng` → 184778, vs .610), `--time 4:54:00` |
| F. photo text vs pixels | blocked on data | build `eval_data/probes_textvqa_ocr` (500 TextVQA rows + `ocr_tokens` from lmms-lab/textvqa) with a `scripts/probes/` builder via SLURM; check the compute node reaches the HF hub first |
| G. route then densify | code written, checks running | paste 188523's test output here; 188524 EM within .01 of 184781's first 300; 188525 ≥ 97% answers identical to 184781's first 300 and judge within .01 → launch `g_dense4_s42` (→ 184778, parent 184781) and `g_dense4_s43` (→ 184780, parent 184784): `RAW=route1 DENSE=4 PX=1003520`, `--time 4:54:00`. Report decoder prompt tokens vs route1 (1,264) and plain (511) |

## Health and budget
- cells: 49 of 60 committed. Left to run: arm G ×2, arm E ×2 (photos, quarter), int8 s43, expand s43 (if due),
  arm F ×3, arm C s43 ×2 = 11-12. If it does not fit, drop arm C s43 first (its gate needs a 5 pt shrink that the
  seed-42 cell missed on 2 of 3 sources).
- GPU-hours of 40: the tick's sensor counted 8.9 GPU-h in the last 24 h at 06:00 (campaign started 00:00).
- gpu-be preemptions: 188518 at 06:10 (5 min in, no file). Before each pairing, compare file mtime with the attempt ends.

## Next wake
`wake-when` on the first group to leave the queue: arm G checks (188523-188525: decide the full cells), arm B
no-oracle (188520, 188521), arm E seed 43 (188522), arm A expand (188439); heartbeat 2 h.

## Blockers / needs the owner
- none blocking. FYI: `eval_mm_4b.sh`, `probe_attn_route.sh`, `run_tests_cpu.sh` are not under git (`.gitignore` has
  `*.sh`); this wake changed `eval_mm_4b.sh` (env `DENSE`, 3 lines), so force-add it when convenient.
