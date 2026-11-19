## The four slot checks are done: open a stage-2 rebuild that teaches the image slots to keep the page's words?

In plain words: if you say yes, a new campaign retrains stage 2 (the page-image pretraining) with more
page-transcription work on document pages, first checks that the slots now hold more of a page's words, and only then
finetunes and scores it; one change plus its control costs about 90 GPU-hours, and if the first check fails it stops at
about 63 (no finetunes).

**What the four checks found.** Judge accuracy ↑ (Qwen3-8B rubric judge, 0 to 1). The document gap: .610 compressed vs
.809 uncompressed on 1,500 questions (19.9 pt). Compression rate everywhere: 8 tokens per slot; about 108 image slots
per page (431 per 4-page question). Full tables: `docs/slot_checks.md`.

| check | question | result (n) | share of the gap it explains |
|---|---|---|---|
| 3 | add each page's own words (its transcription) as text slots beside its image slots | image + text **+5.0 pt** [+2.5, +7.6] vs image slots, .667 vs .617, uncompressed .811 (1,000 paired questions, two disjoint reads, both above 0); the words alone, without the image: **−13.7 pt** | about 5 of 19.4 pt; the whole net gain is on questions whose answer is written in the words (+50 of 878 questions; 0 of 122 elsewhere). Costs 1.52× the slots (654 per question) |
| 4 | can about 120 slots per page hold a dense page's words? | page word recall ↑ **.921** [.902, .938] after training on 100 pages vs .702 before; 100 untrained pages stay flat (one training) | not a judge number: the slot count is not the limit, training is |
| 2 | does the eval's padding differ from training's in a way that costs answers? | +0.5 / +0.9 pt, both intervals inside ±3 (1,500 questions × 2 seeds) | none |
| 1 | does the finetune's 4-document cap cut the answer pages? | 0 of 20,000 document rows lose every answer page | none |

**Why it is your call.** A stage-2 training is outside this campaign: its law allows one 100-page overfit only (hard
rule 9). It is also a choice of direction, and it costs far more than 5 GPU-hours.

**Options**

| option | what | GPU-hours | what it buys |
|---|---|---|---|
| **A (recommended)** | a new campaign: stage 2 with a larger share of page transcription on document pages (one change), and beside it a second seed of today's stage-2 recipe as the control. Gate 1, before any finetune: page word recall on the 100 untrained dense pages and on eval pages. Gate 2: finetune, then paired judge on documents, photos and text | about 30 per stage-2 run × 2 + 3 for recall = 63; then about 10 per finetune × 2 + about 6 for judged evals = about 90 in all; if gate 1 fails, it stops at about 63 | tests the cause checks 3 and 4 point to: the slots can hold the words but are not taught to keep them |
| B | a text side channel: compress each page's transcription beside its image slots (works today: +5.0 pt) | about 0 to adopt in evals; about 10 for a finetune that also trains the decoder on text slots | +5 pt now, but every page needs a transcription by the 4B model (9-12 s per page on one A100) and each question 1.5× the slots |
| C | firm up check 4 first: a second 100-page overfit with another seed (one training + one recall) | about 5 (this campaign's cell cap 32 → 34) | check 4 rests on one training; its interval [.902, .938] sits just above the 90% bar |
| D | stop here: close the campaign with the four verdicts, no rebuild now | 0 | nothing new |

**Recommendation: A**, about 90 GPU-hours, with the word-recall gate after about 63 so a change that does not help
stops before its finetunes. It aims at the same 5 points as B without an extra transcription pass per page, and it follows the plan
written before this campaign (`docs/review_2026-11-18.md`, Plan step 2). C is cheap but does not change the choice:
even at the low end of its interval, the overfit holds 90% of the words. B stays the fallback if A's slots do not
learn to keep the words.

Your decision: A, B, C or D (C can be added to any of the others).
