# BLOCKED — campaign probes (2026-11-13 09:50 KST)

## Use the spare cell (and one more) to finish the slot-pruning question?

**What.** Every arm is decided or closed. The last cell finished: keeping only the eighth of the slots the decoder
attends to most (52 slots per question instead of 431) scores the same as keeping all of them. Documents, model
184778 (seed 42), 1,500 questions: judge accuracy .613 vs .610, difference +.003 (95% CI −.011 to +.017), McNemar
p = .78 (job 189057). The law says a tie must be repeated on the second seed before it counts. That repeat needs a
cell the law does not give.

**Why it needs you.** The cap is 65 cells; 64 are used. The 65th was reserved by your 2026-11-12 answer for "one health
relaunch, and nothing else". All four of its candidates passed their health check, so it is free, but using it for
another purpose changes your rule.

**Why it matters for training.** The proposal to the sibling `longchunk` campaign is a query-aware slot budget
(nested-budget training: the model learns to put what matters in a small, fixed part of its slots). Its target size
depends on this: a quarter is settled at both seeds; an eighth is seed 42 only; where the loss starts is unknown.

**Options.**
| option | cells | what it settles | cost |
|---|---|---|---|
| A. Spare cell → keep an eighth at seed 43 (model 184780) | 65th (no raise) | "seven eighths unused per question" at both seeds, or not | about 30 min on one A100 |
| B. A, plus one new cell: keep a sixteenth at seed 42 (27 slots per question) | 65th + raise to 66 | also where dropping starts to cost. Gate written now, before any cell: within .02 of plain → fifteen sixteenths unused; a loss ≥ .05 → the free share lies between 7/8 and 15/16 | about 1 h of A100 in total |
| C. Stop here | none | the eighth stays provisional (one seed); the campaign is finished | none |

GPU-hours are not the limit: about 2.5 GPU-h used in the last 24 h, of a 40 GPU-h budget.

**My recommendation: B.** Two cells of about 30 min each answer the arm's own open question ("where does dropping start
to cost") and give the training campaign its target budget. A is the minimum if you prefer not to raise the cap again.

**How to answer:** `expmon campaign resolve` with `launch:` lines, or `expmon answer <id> "A" | "B" | "C"`.
