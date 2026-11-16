# STATE — campaign longchunk   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-16 18:30 (woken by the owner's REQUEST: answer A, "stop the 2048". The owner's chat had already run `expmon kill` on the 2,048-token stage-1 run at 18:22. This wake recorded where it stopped, closed it, deleted REQUEST and the answered BLOCKED.md, and ran the close-out: both arms decided → retro, standby. No submission.)

## TL;DR — where this leaves the question (readable as an email)
**Answer: long compressor chunks in stage 1 do NOT help.** The campaign is closed.

**Settled**
- Stage-1 corpus = MS MARCO documents: 43.4% of documents ≥ 1,024 tokens, 20.4% ≥ 2,048 (107,120 of 3.2M sampled) — data job 188358, closed. SmolLM2 rejected (23.8% / 8.3% of 100,000 sampled) — data job 188339, closed.
- Fit tests: the 2,048-token arm fits at batch 3 per GPU (~72 GB per GPU, 58.6 s per step, 50/50 steps) — job 188420, closed; the 1,024-token arm at batch 6 (~71 GB, 42.1 s per step) — job 188421, closed.
- Control (256-token chunks) stage 1 FINISHED: 1M documents (5,208 steps, 1.0 epoch), 16:43 h, exit 0 — job 188422, closed. Slot ownership .972 on text (3,053 test slots; gate ≥ .95) — job 188470, closed. Stage 2 FINISHED 100k rows, held-out .785 at the end — job 188467, closed. Finetunes FINISHED 28k rows each — jobs 188468 (seed 42) / 188469 (seed 43), closed.
- Control gating scores, judge accuracy ↑ (Qwen3-8B rubric judge), n = 1,500 questions per cell, rate 8 tokens per slot: long text g10 .611 (188707) / .607 (188710); documents .615 (188708) / .603 (188711); photos .667 (188709) / .670 (188712), seed 42 / 43 — closed.
- 1,024-token arm (L1024) stage 1 FINISHED: 1M documents (5,208 steps), exit 0, 60.2 h on 4 A100 — job 188466, closed. Ownership .969 (3,053 test slots) → health PASS — job 189691, closed. Stage 2 FINISHED 100k rows, held-out .818 at the end — job 189692, closed. Finetunes FINISHED 28k rows each, held-out .615 / .470 at the end — jobs 189694 / 189695, closed.
- **L1024 CLOSED by the law's gate at BOTH seeds.** Paired per question against the control of the same seed (`scripts/paired_judge.py`, n = 1,500 Q per set, rate 8):

  | set (judge accuracy ↑) | seed 42: arm / control | arm − control, 95% CI, McNemar p | seed 43: arm / control | arm − control, 95% CI, McNemar p |
  |---|---|---|---|---|
  | g10 grouped HotpotQA, one long chunk | .600 (190270) / .611 (188707) | −1.1 pt [−3.2, +1.2], p .38 | .574 (190273) / .607 (188710) | **−3.3 pt [−5.4, −1.1], p .006** |
  | documents `mm_docqa_1500_1m` | .553 (190271) / .615 (188708) | **−6.2 pt [−8.6, −3.9], p < .001** | .561 (190274) / .603 (188711) | **−4.3 pt [−6.7, −1.8], p < .001** |
  | photos `mm_naturalqa_1500` | .670 (190272) / .667 (188709) | +0.3 pt [−1.5, +2.2], p .79 | .670 (190275) / .670 (188712) | 0.0 pt [−1.9, +1.8], p 1.0 |

  Both close conditions hold at each seed: the g10 CI excludes +2 pt, AND documents ≤ −3 pt with the CI below 0.
- **2,048-token arm (L2048) STOPPED by the owner** (REQUEST 2026-11-16 18:22, option A) at stage-1 step 3,561 of 5,208 = 683,712 of 1M documents, 57.6 h on 4 A100; healthy when stopped (held-out .526 at step 2,000, no collapse); never evaluated, so no claim about 2,048 tokens — job 189693, closed (probe @3561 from the job log).

**Open, and running**
- (none) — nothing left to launch or decide.
- provisional (documents pairs not on same-node-checked nodes): node pairs arm / control — seed 42 documents homer-020 / homer-013, seed 43 homer-020 / homer-016; no node-agreement check exists for them. The g10 close condition (text only) holds on its own at both seeds, so the verdict does not depend on it.
- provisional (64 held-out rows, not a gate): L1024 was behind the control at every image/QA stage — stage 2 held-out .818 vs .785 (189692 vs 188467); finetune held-out higher at all 5 eval points at both seeds (checkpoint-437 trainer_state.json of ft_cml1024_s42/s43 and ft_cml256_s42/s43).
- provisional (n = 300, step 100, previews never gate): documents seed 42 −5.7 pt [−10.7, −0.7] (190574 vs 188980), seed 43 −2.7 [−7.3, +2.0] (190768 vs 188982); g10 seed 42 −0.7 [−5.3, +4.0] (190575 vs 188981), seed 43 +2.7 [−2.3, +7.7] (190769 vs 188983) — the seed-43 g10 preview pointed the wrong way.
- provisional (unpaired, other corpus — never a comparison): the MS MARCO control scores within 2 pt of the SmolLM2 reference on every set (documents .615 / .603 vs .599 / .618; photos .667 / .670 vs .671 / .671; g10 seed 42 .611 vs .617).
- provisional: held-out loss jumps at the last stage-1 step in all three finished runs (SmolLM2 reference .920 → 1.023, control .877 → 1.007, L1024 .612 → .850); cause untested, 64 held-out rows.

## Owned jobs
None running. Last one: 189693 (L2048 stage 1) CANCELLED 2026-11-16 18:22 by the owner's chat (`expmon kill`, ledger 18:23), out dir `expQ/longchunk/pt_text_1M_cml2048_189693` (checkpoint-2000 kept, not renamed — nothing to void).
Code md5 at the last evals (unchanged since the control evals): `eval_mm_4b.sh` 50abc167, `scripts/eval_mm_docqa.py` dfe32aa0, `scripts/judge_mm_docqa.py` 02bc8748, `pisco/model.py` 994cff3d.
Resolved-config diffs against the control: stage 1 differs ONLY in batch per GPU (24 → 6 / 3), grad accumulation (2 → 8 / 16), compressor chunk length (256 → 1,024 / 2,048), decoder max length (512 → 2,048 / 4,096); effective batch 192 documents held. Stage 2 and finetunes: only the init model, plus `freeze_part: null` (a no-op field, known_issues.md).

## Adopted jobs
(none)

## Hand-offs
- 5f4e2f → the pisco chats (2026-11-13): make `eval_mm_4b.sh` skip generation and only judge when the result file already holds every row. Still open; not blocking (the campaign is closed).

## Results table (deliverable; judge accuracy ↑, n = 1,500 Q per set; compression rate 8 tokens per slot)
| row | corpus | chunk (tokens) | slots per chunk | ownership ↑ | g10 s42 / s43 | documents s42 / s43 | photos s42 / s43 |
|---|---|---|---|---|---|---|---|
| SmolLM2 reference (183814 / 183816) | SmolLM2 | 256 | 33 | .977 | .617 / — | .599 / .618 | .671 / .671 |
| C256 control (188468 / 188469) | MS MARCO | 256 | 33 | .972 | .611 / .607 | .615 / .603 | .667 / .670 |
| L1024 (189694 / 189695) — CLOSED | MS MARCO | 1,024 | 129 | .969 | .600 / .574 | .553 / .561 | .670 / .670 |
| L1024 − C256, paired, 95% CI | | | | | −1.1 [−3.2, +1.2] / **−3.3 [−5.4, −1.1]** | **−6.2 [−8.6, −3.9]** / **−4.3 [−6.7, −1.8]** | +0.3 [−1.5, +2.2] / 0.0 [−1.9, +1.8] |
| L2048 — STOPPED at 68% of stage 1 | MS MARCO | 2,048 | 257 | — | not run | not run | not run |
| uncompressed decoder (ledger) | — | — | — | — | — | .809 | .717 |

g1 and MRQA were not run: the law runs them only for an adopted arm. Text tokens seen in stage 1 (estimates): control ≈ 0.45 B, L1024 ≈ 1.04 B, L2048 ≈ 0.89 B when stopped. More tokens did not help.

Proposed text for `docs/multimodal_results.md` (the owner's chats write it): "Long compressor chunks in stage 1 (1,024 tokens instead of 256, MS MARCO documents, same 1M documents and recipe otherwise) do NOT help: paired against a 256-token control, long text −1.1 / −3.3 pt, document pages −6.2 / −4.3 pt, photos 0 (two finetune seeds, 1,500 Q per set, rate 8). A 2,048-token run was stopped at 68% of stage 1 and not evaluated. Chunk length stays a free inference-time choice under the spread layout; training on long chunks costs pages."

## Budget (38 GPU-days in the law)
Spent ≈ 638 GPU-h ≈ 26.6 GPU-days over 51 runs (`expmon campaign retro`); the stop saved ≈ 224 GPU-h of the 2,048 chain. Nothing left to spend.

## Close-out (done this wake)
- `expmon campaign report` + `expmon campaign retro` → REPORT.md, RETRO.md (Lessons filled), SYNC.md.
- Retro journalled with `expmon gripe -c retro`; `expmon campaign standby` written → later ticks are no-ops.

## Blockers / needs the owner
- Paste `campaign_longchunk/SYNC.md` into `~/.claude/SYNC.md` (the wake may not touch `~/.claude`).
- Run `expmon campaign disarm` when convenient (the cron line stays until then; standby makes it free).
- Optional: put the proposed text above into `docs/multimodal_results.md`.
