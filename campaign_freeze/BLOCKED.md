# BLOCKED — campaign freeze   (2026-11-12, wake 10)

## Gate 0 failed — run the split test, accept the cost, or close?

**What.** Freezing the compressor and the connector during the question-answering finetune (only the decoder LoRA
trains) costs judge accuracy. Paired per question, n = 1,500 per set, three seeds (jobs 188650, 188701):

| set | judge Δ frozen − reference, s42 / s43 / s44 [95% CI] | reading |
|---|---|---|
| documents | −.028 [−.049, −.007] / −.035 [−.057, −.014] / −.041 [−.062, −.021] | FAILED (≤ −3 pt, CI below 0, at 2 of 3 seeds) |
| MRQA text | −.023 / −.027 / −.025, CI below 0 at all three | misses the −2 pt bar |
| photos | −.009 / −.013 / +.005, CI spans 0 at all three | tie |
| HotpotQA (not gating) | −.028 / −.023 / −.026 | lower |

Frozen training is ~25% faster (71.5 vs 95.1 s/step) but uses 50 vs 34 GB of GPU memory.

**Why you.** The law (CAMPAIGN.md "After the gate", "if failed") forbids variants of freezing and sends the next step
to you: the split test needs a new flag in `pisco/`, which is never patchable by this loop.

**Options.**

| option | what it means | cost | what it tells you |
|---|---|---|---|
| A. split test (recommended) | a `pisco/` flag that freezes the compressor and the connector separately (written by a pisco chat, with a `test_freeze.py` case). Then 2 arms × 2 seeds: (A1) compressor frozen, connector trains; (A2) connector frozen, compressor LoRA trains. Compared with the reference of the same seed, same gates as gate 0 | ~2 GPU-days (4 finetunes ~10 h + 16 evals); campaign used ~2.2 of 8 | A1 is the one that keeps stage B alive: the compressor output can still be cached once, and only the small connector + decoder train. A2 is diagnostic: it says where the 3 pt come from |
| B. accept the cost | go to stage B as proposed: compress once, train the decoder only, ~3 pt lower on documents and text, photos unchanged | 0 GPU now; `<MEM>` caching is engineering in `pisco/` | nothing new; the trade-off is known |
| C. close the campaign | record the FAILED result in docs/training_plan.md and HANDOFF.md (proposed text in STATE.md), stand the loop down | 0 | — |

**Recommendation: A, A1 first.** The connector moved only 20.7% of its entries (.0073 of its norm) in the reference
seed-44 finetune, and the compressor LoRA moved 74% of lora_B entries. So the gap more likely sits in the compressor.
But only A1 is useful for stage B, so if A1 ties the reference, stage B is still open at almost the same cost.

**How to answer.** Write `campaign_freeze/REQUEST` (via `expmon campaign resolve`) with, e.g.,
`verdict: split test, A1 and A2` once a pisco chat has committed the flag (give the sha as `fixed: <sha>`), or
`verdict: close`. On A, the loop writes both arms' gates into CAMPAIGN.md before launching anything.
