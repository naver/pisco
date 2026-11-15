# BLOCKED — campaign longchunk (2026-11-16 04:00)

## Should the 2,048-token chunk run finish, or stop now?

**What.** The 1,024-token chunk arm is closed: it lost against the 256-token control on BOTH finetune seeds.
Judge accuracy, paired on the same 1,500 questions per set, compression rate 8 tokens per slot:

| set | seed 42 (arm − control) | seed 43 (arm − control) |
|---|---|---|
| long text, one long chunk per question (g10) | −1.1 pt [−3.2, +1.2] | −3.3 pt [−5.4, −1.1] |
| document pages | −6.2 pt [−8.6, −3.9] | −4.3 pt [−6.7, −1.8] |
| photos | +0.3 pt [−1.5, +2.2] | 0.0 pt [−1.9, +1.8] |

The 2,048-token arm is the same idea at twice the chunk length. Its stage 1 (text pretraining, 4 A100) is healthy and
half done: step 2,678 of 5,208 (≈ 514k of 1M documents), ends ≈ 11-17 evening.

**Why I ask.** The law lets the loop kill a run only when it collapses. This one is healthy, so stopping it is your call.

**Options.**

| option | GPU cost still to spend | what you learn | results by |
|---|---|---|---|
| A. Stop it now (recommended) | 0 (saves ≈ 224 GPU-h ≈ 9 GPU-days) | nothing more; the campaign closes with "long chunks in stage 1 do not help" at 1,024 | now |
| B. Finish it with the full chain (the law's default) | ≈ 163 stage 1 + 61 (images stage, 2 finetunes, evals) | whether 2,048 tokens behaves like 1,024 (a dose check) | ≈ 11-21 |
| C. Finish stage 1, then one finetune seed only | ≈ 163 + 50 | the same dose check, one seed (cannot close or adopt under the gate) | ≈ 11-20 |

**My recommendation: A.** The trend from 256 to 1,024 tokens is down on long text and on pages, so 2,048 most likely
closes too. Budget is not the problem (≈ 24 of 38 GPU-days spent; B ends at ≈ 34): the 4 A100 are better used elsewhere.

**If you say nothing**, the loop follows the law (option B): when stage 1 ends it submits the rest of the chain.
To answer: `expmon campaign resolve "A"` (or B / C) — or `expmon answer` from the queue.
