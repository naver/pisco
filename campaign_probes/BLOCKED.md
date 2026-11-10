# BLOCKED — campaign probes (2026-11-11 07:50 KST)

## Raise the campaign's cell cap from 60 to 65?
**What.** 58 of the 60 cells are used. The last 2 are reserved for arm G (the picked page at twice the slots, both
seeds), which only runs if its re-check (job 188580, running) passes. The strongest new result has no cell left to
confirm it.

**Why it matters.** Keeping only the quarter of the slots the decoder attends to most (given the question) scored
.611 vs .610 for the plain compressed model on 1,500 document questions (seed 42, slots .246×, job 188538). The
half is already adopted at both seeds. If the quarter holds at seed 43, "three quarters of the slots go unused per
question" becomes a finding, and it motivates a query-aware slot budget / nested-budget training in stage 2. The law
needs the second seed before it can say so.

**Options** (each cell = one GPU, 20-100 min on gpu-be):
| option | cells | what it buys |
|---|---|---|
| A. +5 (recommended) | keep a quarter at seed 43; keep an eighth at seed 42 (gate already written in CAMPAIGN.md arm E); TextVQA as OCR text at seed 42 (arm F, set built); evidence-then-answer at seed 43 (its seed-42 tie needs a repeat under the law); 1 health relaunch | confirms the quarter result, finds where slot dropping starts to cost, closes arm F and arm A cleanly |
| B. +2 | keep a quarter at seed 43; keep an eighth at seed 42 | the slot-budget question only |
| C. keep 60 | — | the quarter result stays "provisional, one seed" |

**Recommendation:** A. The GPU cost is about 3-4 GPU-hours (the law's 40 GPU-hour budget has room: the tick counted 13.0 GPU-hours used at 07:30).
To answer: `expmon answer` on this question, or edit `CAMPAIGN.md` "Budget and caps" yourself.
