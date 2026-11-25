# Successor brief — stage-2 rebuild that teaches the image slots to keep the page's words

Written by the slotcheck wake of 2026-11-25 19:00 for the pisco chat that drafts and arms the new campaign (hand-off
in `expmon handoff list`). The wake may not run `expmon campaign init` or `arm` (guard, hard rule 3).

## The owner's decision (verbatim, REQUEST of 2026-11-25)
- 18:40, answer to «open a stage-2 rebuild that teaches the image slots to keep the page's words?»: **"A"** — "a new
  campaign, stage 2 with a larger share of page transcription on document pages (one change) plus a second seed of
  the current stage-2 recipe as control; gate 1 = page word recall on the 100 untrained dense pages and on eval
  pages, stop at ~63 GPU-h if it fails; gate 2 = finetune + paired judge on documents, photos, text; ~90 GPU-h in all.
  Not C."
- 18:54, addition: "build BOTH stage-2 runs (the transcription-share change and the control seed) on the spread
  <MEM> layout, i.e. the recipe of expQ/archq3/pt_mm_5M_spread."

## Why (evidence from slotcheck, details in `docs/slot_checks.md`)
- Adding each page's transcription as text slots beside its image slots: **+5.0 pt** judge accuracy [+2.5, +7.6],
  1,000 paired questions, two disjoint reads (jobs 192223 / 192239). The whole net gain is on questions whose answer
  is written in the page's words.
- About 120 slots per page CAN hold a dense page's words once trained on it: word recall .921 [.902, .938] vs .702
  before, 100 pages, one training (job 192209). So the limit is what stage 2 teaches, not the slot count.

## The two stage-2 runs (both on the spread recipe)
| run | recipe | one change |
|---|---|---|
| control | `expQ/archq3/pt_mm_5M_spread/training_config.yaml` exactly: start `expQ/archq3/pt_text_5M_spread/model`; data pdfa 80k + wiki 30k + natural 50k = 160k rows; task mix image AE .2 / text .5 / image continuation .05 / desc .15 / find .1; bs 8 × accum 8, lr 3e-5, bf16, KD .5, merger LoRA, 1,003,520 px | seed 43 (the recorded run has no seed field = HF default 42) |
| change | the same file | a larger image-AE (page transcription) share on document pages |

Open design points the drafting chat settles before launch (one recommendation each):
1. **There is no per-source task mix.** The mix is drawn per row for every image row, documents and photos alike
   (`pisco/mm_collator.py:313-318`). "On document pages" therefore needs either (a) a per-source mix in the collator
   (a few lines in `pisco/`, a chat's change, with a test) or (b) raising `p_image_ae` for all image rows, which also
   moves photos. Recommendation: (a), so photos keep today's mix and gate 2's photo read stays a guard.
2. **How large a share.** Today image AE is 20% of image rows (about 16k of the 80k pdfa rows). Recommendation:
   fix the new share and what it is taken from (text rehearsal .5 is the largest) in the law before launch; never
   tune it on gate 1.
3. **Seed of the change run.** Recommendation: seed 42, so change − recorded pt_mm_5M_spread is one variable, and
   control seed 43 − recorded gives the seed floor. Caveat: the recorded run is from 2026-11-07 code; the control
   also measures code drift.
4. **AE target cap.** `ae_target_max_tokens: 768`; the dense pages average ≈ 750 tokens (build 191891), so some
   targets are cut. Keep it (one variable); report the share of document AE targets at the cap.
5. **Dense pages must stay untrained.** `eval_data/slotcheck_dense100` and `_heldout` exclude the 47,498 pdfa pages
   the recorded stage 2 trained on (build 191891, `excluded_trained`). Both new runs must draw the same `path:80000`
   slice, or the dense sets must be rebuilt against the new slice.

## Gates (owner's)
- **Gate 1 (before any finetune), page word recall ↑** from the slots against the page's uncompressed transcription,
  `scripts/longdoc/mem_recall.py` through `scripts/slotcheck/recall.sbatch` (run, never edit), paired per page,
  sign test: on the 100 untrained dense pages (`eval_data/slotcheck_dense100_heldout`; and `slotcheck_dense100`)
  and on eval pages (the gold pages of `mm_docqa_1500_1m`). Baselines on file for the recorded model (job 191893):
  dense100 .702, held-out .670; text-layer ceiling .974 / .983. The bar is the new law's to write before launch
  (change − control, both seeds' floor). Fail → stop at ≈ 63 GPU-h, no finetune.
- **Gate 2:** finetune each stage-2 model with the 184778 / 184780 recipe (28k rows, bs 1 × accum 64, lr 2e-5, bf16),
  then paired Qwen3-8B rubric judge (`scripts/paired_judge.py`) on documents `mm_docqa_1500_1m`, photos
  `mm_naturalqa_1500`, text `text_mrqa_ood_1500`, beside the uncompressed decoder on the same questions and the
  compression-rate column (8 tokens per slot; ≈ 108 slots per page).

## Budget (owner's): ≈ 90 GPU-h
≈ 30 per stage-2 run × 2 (160k rows at ≈ 5,300 rows/h) + ≈ 3 for recall = ≈ 63; then ≈ 10 per finetune × 2 + ≈ 6
for judged evals. Stage 2 is a 1-GPU A100 bf16 run on `gpu` (the shared 8-GPU cap); shake it out with a 20-step
smoke first (s/step, peak memory) and derive `--time` from it.

## Reusable from slotcheck (read-only, md5 in the ledger)
`scripts/slotcheck/recall.sbatch`, `build_dense100.py`, `transcribe_pages.py` (eval-page transcriptions already in
`expQ/slotcheck/transcribe/` and `transcribe_r2/`, 2,520 pages), `eval_data/slotcheck_dense100{,_heldout}`.
Rules that still bind: no route-then-expand or raw page in the prompt (owner, 2026-11-18); one variable per run;
dead ends in `campaign_slotcheck/CAMPAIGN.md` hard rule 12.
