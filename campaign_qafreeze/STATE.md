# STATE — campaign qafreeze   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-18 21:30-21:45. Question: can a compressor that HAS had one QA finetune (R = 184778 / 184780)
stay frozen while a fresh decoder LoRA trains on the same 28k rows? (arm Q) **Answer: yes — arm Q passed its gate,
and HotpotQA (reported, not gating) ties too.** The loop now waits for the owner: run arm H (decoder LoRA lr 1e-4) or close.

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Arm Q VALIDATED at both seeds (gate: every gating set Q − R ≥ −2 pt, no CI wholly below 0). Paired judge accuracy, n = 1,500 pairs per set, Q passed first so diff = Q − R (campaign_qafreeze/pairs/):
  - documents +0.5 pt [−0.6, +1.6], McNemar p .470 (s42, 191626 vs 191495); −0.1 pt [−1.1, +0.9], p .896 (s43, 191628 vs 191496).
  - photos −0.3 pt [−1.5, +0.9], p .741 (s42, 191659 vs 191531); −0.5 pt [−1.7, +0.7], p .434 (s43, 191660 vs 191532).
  - MRQA text +1.1 pt [−0.1, +2.3], p .068 (s42, 191661 vs 191675); +0.3 pt [−0.8, +1.5], p .661 (s43, 191662 vs 191676).
  - verdicts: training 191527 (s42) and 191530 (s43) → best.
- HotpotQA (reported, not gating), same code md5 on both sides, n = 1,500 pairs: Q − R −0.5 pt [−1.9, +0.9], p .562, discordant 50/57 (s42, 191766 .600 vs 191768 .605); +0.9 pt [−0.5, +2.1], p .223, 55/42 (s43, 191767 .600 vs 191769 .591) — pairs/hotpot_QminusR_s4x.txt. Campaign freeze's never-QA frozen compressor lost −3.0 / −2.8 there.
- The prediction written before any cell (documents Q − R in [−1, +2] pt) held: +0.5 / −0.1.
- Instrument control (a): R s42 documents paired with itself = +0.000, 0 discordant (pairs/docs_noop_R42.txt).
- Instrument control (b), same eval code on both sides: Q − F documents +3.3 pt [+1.1, +5.3], p .003 (s42, 191626 vs 191677) and +3.3 pt [+1.1, +5.7], p .003 (s43, 191628 vs 191678); the law asks ≥ +2 pt with CI above 0 → met. R − F on the same files +2.8 [+0.7, +4.9] p .011 / +3.5 [+1.4, +5.7] p .002, the recorded −2.8 / −3.5 of campaign freeze (pairs/docs_RminusF_s4x.txt).
- F re-run on today's eval code gives the recorded answers: documents .582 (s42, 191677; 0 of 1,500 judgements differ from the recorded file) / .580 (s43, 191678).
- R re-run on today's eval code, 1,500 rows each, every row judged, 0 unparseable: documents .610 / .615 (191495 / 191496), photos .679 / .685 (191531 / 191532), MRQA .715 / .724 (191675 / 191676; recorded .718 / .724), HotpotQA .605 / .591 (191768 / 191769; recorded .603 / .593).
- Hybrid start model built from saved checkpoints with no new code; feasibility gate PASS (191450-191453); 20-step smoke PASS (191493 / 191494); after the full epoch the frozen side moved 0.0%, decoder lora_B 66.9% at both seeds (191625 / 191627).

**Open, and running**
- Nothing running. Owner question in BLOCKED.md: run arm H (decoder LoRA lr 1e-4, 2 seeds, ~22 GPU-h) or close.
- provisional: Q's held-out loss ends below R's and F's at both seeds (.400 / .371 vs R .414 / .375 vs F .435 / .413). One run per seed, held-out set of only 64 rows, no paired test; the s43 gap to R (.004) is smaller than R s43's own swings (.411 → .441 → .375). The loss is not the gate.

## Deliverable table
Judge accuracy ↑ (Qwen3-8B rubric judge), seed 42 / seed 43; difference to R in pt, paired, 95% CI; n = 1,500 per set.
Compression: 8 tokens per slot, ~123 slots per 1M-px page.
| model | what trains | documents ↑ | docs − R (pt) | photos ↑ | photos − R (pt) | MRQA text ↑ | MRQA − R (pt) | HotpotQA ↑ (not gating) | HotpotQA − R (pt) |
|---|---|---|---|---|---|---|---|---|---|
| R reference | compressor LoRA + connector + decoder LoRA | .610 / .615 | — | .679 / .685 | — | .715 / .724 | — | .605 / .591 | — |
| F frozen never-QA compressor (campaign freeze) | decoder LoRA | .582 / .580 | −2.8 [−4.9, −0.7] / −3.5 [−5.7, −1.4] | — | recorded −0.9 / −1.3 | — | recorded −2.3 / −2.7 | — | recorded −3.0 / −2.8 |
| A2 connector frozen (campaign freeze, recorded) | compressor LoRA + merger + decoder LoRA | — | recorded +2.3 / −0.6 | — | recorded −0.6 / −1.3 | — | recorded +0.1 / +0.3 | — | — |
| **Q frozen QA-trained compressor** | decoder LoRA | **.615 / .613** | +0.5 [−0.6, +1.6] / −0.1 [−1.1, +0.9] | **.677 / .679** | −0.3 [−1.5, +0.9] / −0.5 [−1.7, +0.7] | **.726 / .727** | +1.1 [−0.1, +2.3] / +0.3 [−0.8, +1.5] | **.600 / .600** | −0.5 [−1.9, +0.9] / +0.9 [−0.5, +2.1] |
| uncompressed decoder | — | recorded .809 | — | recorded .717 | — | recorded .807 | — | — | — |

Per source, Q − R (pt), s42 / s43: documents infovqa −0.2 / −1.2, mpdocvqa +1.8 / 0.0, slidevqa −0.2 / +0.8; photos
realworldqa 0.0 / −0.2, textvqa −1.2 / −0.6, vizwiz +0.4 / −0.8; MRQA RACE −1.6 / +0.8, RelationExtraction 0.0 / +1.2,
DuoRC +3.2 / 0.0; HotpotQA bridge −0.4 / +0.9 (1,197 rows), comparison −0.7 / +0.7 (303 rows). 250 per MRQA source,
500 per other source; no CI excludes 0 except BioASQ / DuoRC s42 touching 0.
Discordant pairs (Q right only / R right only): documents 38/31, 28/30; photos 39/43, 36/44; MRQA 47/30, 44/39; HotpotQA 50/57, 55/42.

## Training health — held-out eval_loss by epoch (lower is better; 1 epoch = 28k rows)
| run | .23 ep | .46 ep | .69 ep | .92 ep | 1.0 ep |
|---|---|---|---|---|---|
| Q s42 (191527) | .413 | .411 | .398 | .400 | .400 |
| R s42 (184778) | .513 | .455 | .393 | .394 | .414 |
| F s42 (freeze ft_frozen_s42) | .455 | .444 | .438 | .434 | .435 |
| Q s43 (191530) | .399 | .382 | .375 | .372 | .371 |
| R s43 (184780) | .471 | .439 | .411 | .441 | .375 |
| F s43 (freeze ft_frozen_s43) | .442 | .431 | .416 | .415 | .413 |
No NaN; no probe above the run's first probe. Read from each run's `checkpoint-437/trainer_state.json`.

## Module movement (entries changed %, against that run's init)
| run | compressor.* | connector.* | decoder lora_B | decoder lora_A |
|---|---|---|---|---|
| Q smoke s42 (check 191494) | 0.0 | 0.0 | 76.0 | — |
| Q s42 (check 191625) | 0.0 | 0.0 | 66.9 | 14.8 |
| Q s43 (check 191627) | 0.0 | 0.0 | 66.9 | 14.8 |
| F full runs (campaign freeze) | 0.0 | 0.0 | 66.3-66.7 | — |

## Training cost
| run | wall | s/step | peak GPU memory |
|---|---|---|---|
| Q s42 (191527) | 8:58:42 | 73.6 | 49.9 GB (sacct gpumem; GPU util max 92%) |
| Q s43 (191530) | 9:17:07 | 76.2 | 49.9 GB (GPU util max 93%) |
| F (campaign freeze) | 8.7 / 9.3 / 8.7 h | 71.5-76.1 | 49.7 GB (smoke) |
| R (184778 / 184780) | 11.0 / 13.8 h | 88-90 (smoke) | 31.9 GB (smoke) |
Q trains ~18% faster per step than R and needs ~18 GB more GPU memory (the frozen side is not offloaded).

## Config diff of the full cells (law: only the allowed fields)
`diff expQ/ft_mm_4b_184778/training_config.yaml expQ/qafreeze/ft_frozen_s42/training_config.yaml` → `+freeze_compressor: true`,
`model_name_or_path: …/archq3/pt_mm_5M_spread/model → …/qafreeze/init_s42`. Same two lines for s43 vs 184780 (init_s43). Nothing else.

## Owned jobs
| job | arm | state | elapsed | note |
|---|---|---|---|---|
| 191450-191453 | B build + checks | COMPLETED 0:0 | ≤ 3:07 | closed; gate PASS |
| 191493 / 191494 | Qs smoke + check | COMPLETED 0:0 | 29:19 / 0:44 | closed; PASS |
| 191527 / 191530 | Q full s42 / s43 | COMPLETED 0:0 | 8:58:42 / 9:17:07 | best (gate PASS) |
| 191625 / 191627 | Q weight checks | COMPLETED 0:0 | ~1 min | closed; PASS |
| 191495 / 191496 | R documents | COMPLETED 0:0 | 27:13 / 28:04 | closed; .610 / .615 |
| 191531 / 191532 | R photos | COMPLETED 0:0 | 18:59 / 18:10 | closed; .679 / .685 |
| 191675 / 191676 | R MRQA | COMPLETED 0:0 (preempted after the work finished, file reused) | 15:41 / 16:26 | closed; .715 / .724 |
| 191677 / 191678 | F documents | COMPLETED 0:0 (191677 preempted once) | 27:30 / 28:03 | closed; .582 / .580 |
| 191626 / 191628 | Q documents | COMPLETED 0:0 | 28:51 / 28:42 | closed; .615 / .613 |
| 191659 / 191660 | Q photos | COMPLETED 0:0 | 19:47 / 19:11 | closed; .677 / .679 |
| 191661 / 191662 | Q MRQA | COMPLETED 0:0 | 16:55 / 16:33 | closed; .726 / .727 |
| 191766 / 191767 | Q HotpotQA s42 / s43 | COMPLETED 0:0, homer-011 | 22:36 / 22:55 | closed; .600 / .600 |
| 191768 / 191769 | R HotpotQA s42 / s43 | COMPLETED 0:0, homer-011 / homer-020 | 22:15 / 24:26 | closed; .605 / .591 |
No campaign job is in the queue. Every other queue job belongs to sibling campaigns (longdoc, spread, slotcheck) or is the expmon daemon.

## Adopted jobs
None this wake. 191675-191678 were adopted on 2026-11-18 21:00 from the 18:00 wake that the 5-hour usage limit cut.

## Next
| when | what |
|---|---|
| owner says yes to arm H (REQUEST) | add ONE `frozen_hi` case to `scripts/qafreeze/ft_qafreeze.sbatch` (lr 1e-4, else as `frozen`), `bash -n`, 20-step smoke s42 + config diff (only `learning_rate` may differ from arm Q's), then both seeds chained to their 8 judged evals |
| owner says no | final table, RETRO.md, `expmon campaign standby` |

## Proposed text for the owner's chats (the loop does not write these files)
- `docs/training_plan.md:103-106`, replace the "**Open:** …" sentence with: "**Settled (campaign qafreeze, 2026-11-18):**
  a compressor that has had ONE QA finetune (184778 / 184780) CAN stay frozen: a fresh decoder LoRA trained on the same
  28k rows ties the joint reference on documents (+0.5 / −0.1 pt), photos (−0.3 / −0.5), MRQA text (+1.1 / +0.3) and
  HotpotQA (−0.5 / +0.9), n = 1,500 per set, every CI includes 0; the never-QA frozen compressor of campaign freeze
  scores 3.3 pt lower on documents (p .003). So stage B can train the decoder alone on cached `<MEM>`s of a stage-A
  model that already had its QA finetune; the `<MEM>`-cache engineering is triggered (hand-off 1ebc7c). Not shown:
  transfer to data the frozen compressor never saw." Also in the next paragraph: "unless the successor test … passes"
  → "the successor test passed, so routing rows and long-context rows may go to a decoder-only stage B on a QA-trained
  frozen compressor."
- `HANDOFF.md` Decisions pending 1: "`campaign_qafreeze` PASSED (QA-trained compressor frozen ties R on all three gating
  sets and on HotpotQA, both seeds); its owner checkpoint is open in BLOCKED.md: run the decoder lr 1e-4 arm (~22 GPU-h)
  or close." And item 2: drop "decoder-only training costs ~3 pt on documents unless `campaign_qafreeze` passes; read it
  first" → "decoder-only training on a QA-trained frozen compressor ties the reference (campaign_qafreeze)".

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| Q s42, s43 | VALIDATED; table complete (4 sets × 2 seeds) | none |
| Q3 seed 44 | not needed (no tie) | closed unless the owner asks |
| H lr 1e-4 | waits for the owner (BLOCKED.md) | REQUEST yes / no |

## Health evidence this wake (numbers, not adjectives)
- Code md5 unchanged (`campaign_qafreeze/code_md5.txt`, re-run 21:35): eval_mm_4b.sh b3d1cda1, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748, judge_prompts.py bae44fa7, paired_judge.py a3cc13b2, pisco/model.py 994cff3d. All four HotpotQA cells ran on these.
- Each HotpotQA cell: 1,500 records, `judge` on every row, 0 unparseable, no "exceeds" error, log ends `EXIT CODE 0`, nodes homer-011 / homer-020 (outside homer-[005-009]).
- Verifier pass of 21:10: 13 of 14 claims supported, 1 provisional (the loss line, kept provisional with its caveats above), 0 unsupported.
- GPU-h spent ≈ 25.5 of 55 (sensor gpuh24 22.9 + 4 reference cells of 18:03 + 4 HotpotQA cells ≈ 1.5).
- `expmon display --exp qafreeze` refreshed; its summary says "0 table rows" while the page has 29 rows (expmon defect, +1 on gripe be8057a1).
- Disk 7.6 T free.

## Blockers / needs the owner
- BLOCKED.md: run arm H (decoder LoRA lr 1e-4, 2 seeds, ~22 GPU-h) or close the campaign. Recommendation: yes.
