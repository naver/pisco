# Stage-2 rebuild — results (campaign rebuild)

Rewritten whenever a gate decides. Law: `campaign_rebuild/CAMPAIGN.md`. Live state: `campaign_rebuild/STATE.md`.

**Question:** can a rebuilt stage 2 (pretraining on page images) teach the image slots to keep the page's words?
Arms: K = control (today's recipe, seed 43), S = larger share of page transcription on document pages, W = transcription
of a random window of long pages, P = fp32 master weights, L = finetune with the judged-wrong teacher labels removed.

**Metric of gate 1:** page word recall ↑ in [0, 1] = share of the words of the page's uncompressed transcription found in
the text the model rebuilds from the page's slots alone (`scripts/longdoc/mem_recall.py`, plain `<AE>` prompt, greedy,
768 new tokens), paired per page by `scripts/longdoc/recall_pair.py` (bootstrap 95% CI, sign test).

## Feasibility gate — can gate 1 be read on today's code? (2026-11-25; result: one bar missed, owner question open)

Set `eval_data/slotcheck_dense100`: 100 dense document pages, untrained by every model below except the overfit one.
Compression rate: 8 visual tokens per slot, 116-123 slots per page (mean 119.9). Text-layer ceiling .974 (job 191893).

| check | model | n pages | word recall ↑ today | word recall ↑ recorded | today − recorded (95% CI, up / down pages, sign p) | bar | health | verdict |
|---|---|---|---|---|---|---|---|---|
| R2 = recorded stage 2, the start of every arm | `expQ/archq3/pt_mm_5M_spread/model` | 100 | **.715** (job 193565) | .702 (job 191893) | +1.3 pt [−2.4, +5.2], 52 / 45, p .54 | .702 ± .01 | 100 rows, 120 slots/page as recorded, stride 8 | **MISS by 0.3 pt** |
| known positive: overfit on these 100 pages | `expQ/slotcheck/o_bf16/model` | 100 | .921 (job 193566) | .921 (job 192209) | 0.0 pt, 100 / 100 texts identical | .921 ± .01 | 100 rows, stride 8 | pass |
| reference transcription, paired with itself | uncompressed Qwen3.5-4B | 100 | 1.000 (job 193567) | 1.000 | fresh vs recorded reference: 1.000 (9 pages differ by < 0.1 pt) | 1.000 | 100 rows | pass |

**What it means for the stage-2 recipe:** the readout and today's code give the recorded numbers, but one read of a
non-overfit model moves by a lot from run to run, so gate 1 on 100 pages can only see large gains.

Why the R2 bar was missed (evidence, not a guess):
- The code path is unchanged. The only change to `pisco/model.py` since the recorded run (994cff3d → 2a1cc007) is the
  opt-in one-row pad guard, off by default and off in R2. `mem_recall.py` and `probe_mem_recall.py` have the same md5.
- The same code reproduces the recorded run exactly on one node: the 5-page check pass of job 193565 (homer-019) wrote
  the same 5 texts as job 191893 (homer-016). Its 100-page pass (homer-017, all A100-SXM4-80GB) changed 99 of 100 texts.
- The overfit model does not move at all (100 / 100 identical texts): its outputs are very confident. R2's are not:
  10 of 100 rebuilds differ from the first character and 21 within the first 20 (the model starts at another part of
  the page); the median page diverges after 78 characters of a ~2,800-character rebuild.
- Size of this run-to-run noise on R2, per page: standard deviation of the difference .20; mean absolute difference .12;
  28 of 100 pages move by more than 10 pt; the largest move is 70 pt. On 100 pages that alone gives a paired CI of about
  ±4 pt, so a true +3 pt gain cannot clear "CI above 0" on the dense sets; on the 360 eval pages about ±2 pt.
- Not settled: the node-level cause. A candidate is the Triton kernels of Qwen3.5's linear-attention layers
  (flash-linear-attention picks its kernel settings by timing them), which changes bf16 rounding; untested.

## Gate 1, gate 2

Not started: no training runs before the owner answers the feasibility question (`campaign_rebuild/BLOCKED.md`).
