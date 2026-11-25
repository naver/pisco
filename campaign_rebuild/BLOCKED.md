# BLOCKED — campaign rebuild — 2026-11-25 23:00

## The first check missed one bar by 0.3 point: start training anyway?
In plain words: if you say yes, the four stage-2 trainings start their short test runs at the next wake (about 2
GPU-hours, then about 120 for the full runs); if you say no, the campaign stops before any training.

**What happened.** Before any training, the law asks three re-reads of known numbers on 100 dense document pages, to
prove that "page word recall" (share of a page's words the model rebuilds from its slots, higher is better) still
gives the recorded values on today's code:

| check | today | recorded | bar | result |
|---|---|---|---|---|
| the recorded stage-2 model (R2), word recall ↑ | .715 | .702 | .702 ± .01 | missed by .003 |
| a model trained on these 100 pages (known positive), word recall ↑ | .921 | .921 | .921 ± .01 | pass: 100 / 100 rebuilds identical |
| the reference transcription paired with itself | 1.000 | 1.000 | 1.000 | pass |

**Why the miss is not a code change.** The R2 code path is unchanged: the only edit to `pisco/model.py` since the
recorded run is an opt-in switch that is off. On one node, today's code wrote the exact recorded text for the first 5
pages; on another node (same GPU model) the 100-page pass changed 99 of 100 rebuilds. Today − recorded = +1.3 pt
(95% CI −2.4 to +5.2, 52 pages up / 45 down, sign test p .54): no real difference. The rebuild is long (768 tokens,
greedy) and R2 is not confident, so a tiny rounding difference sends it down another path. The overfit model is very
confident and does not move. Cause at node level not tested (candidate: kernel settings that Qwen3.5's linear-attention
layers pick by timing).

**Why it matters later.** This run-to-run noise is large per page (standard deviation .20 between two reads of the same
model). On a 100-page set it alone gives a ±4 pt interval, so the dense-page sets can only see big gains; the 360 eval
pages (about ±2 pt) will carry gate 1.

**Options**
| option | what happens | GPU-h now |
|---|---|---|
| A (recommended) | count the check as passed; next wake starts the four 20-step stage-2 test runs; at gate 1 every model of one page set is read in ONE job (one node), with a fresh read of R2 beside the control, and the R2 read-to-read difference is reported beside the seed floor | 0 now; +1.3 at gate 1 |
| B | measure the noise first: R2 re-read on two more nodes, plus 5-page passes on four nodes, then ask again | ≈ 2 |
| C | stop: gate 1 cannot be read on today's code | 0 |

**Recommendation: A.** It relaxes the ±1 pt bar the law set for R2, so it needs your word (it is not covered by the
standing ≤ 5 GPU-hour rule). It spends nothing now and keeps every gate-1 and gate-2 bar.

Heads-up on cost, no decision now: recall measured 23 s per page (the law assumed 14 s), so gate-1 recall projects
≈ 21 GPU-h against the 12 budgeted (total ≈ 200 of 190). The wake will ask before going over, at gate 1.
