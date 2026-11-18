# BLOCKED — campaign qafreeze   (owner checkpoint; one question)

## Run the higher learning-rate arm (decoder LoRA lr 1e-4, 2 seeds), or close the campaign here?
In plain words: the frozen-compressor test passed, so caching `<MEM>`s is safe; saying yes spends about 22 GPU-hours (two ~9-hour trainings plus eight evals) to learn whether a 5x higher decoder learning rate beats the passing model by 2 points or more.

**What passed (2026-11-18, n = 1,500 questions per set, paired judge accuracy, seed 42 / seed 43).**
The QA-trained compressor + connector of the reference finetunes 184778 / 184780 stayed frozen. A fresh decoder LoRA
was trained on the same 28k rows. The result ties the reference on every gating set at both seeds:
| set | frozen − reference, s42 (pt, 95% CI) | s43 (pt, 95% CI) |
|---|---|---|
| documents ↑ | +0.5 [−0.6, +1.6] | −0.1 [−1.1, +0.9] |
| photos ↑ | −0.3 [−1.5, +0.9] | −0.5 [−1.7, +0.7] |
| MRQA text ↑ | +1.1 [−0.1, +2.3] | +0.3 [−0.8, +1.5] |
The instrument sees an effect of this size: the same frozen setup on a compressor that never had QA training scores
3.3 pt lower on documents at both seeds (p .003). The `<MEM>`-cache build went to the pisco chats as an engineering
hand-off (1ebc7c). Nothing is switched on.

**Why ask now.** The law (CAMPAIGN.md "Owner checkpoint") stops the loop before arm H. Arm H changes one thing: the
decoder LoRA learning rate goes from 2e-5 to 1e-4. 2e-5 is a full-finetune value; LoRA usually trains at 1e-4
(docs/training_review.md §4). Every later decoder-only training on cached `<MEM>`s would inherit this choice.
Gate, written before any cell: ≥ +2 pt over the passing model on documents with the CI above 0, and no gating set
≤ −2 pt vs the reference, at both seeds.

**Options**
| option | what happens | cost |
|---|---|---|
| yes | the loop adds one `frozen_hi` case to `scripts/qafreeze/ft_qafreeze.sbatch`, runs a 20-step smoke, then both seeds and their eight judged evals | ~22 GPU-h of the +25 the law reserves; 1 GPU on `gpu` per training, ~9.5 h wall |
| no | the loop writes the final table and stands down | 0 |

**Recommendation: yes.** It costs ~22 GPU-hours, can be undone, and settles the learning rate for every later
decoder-only training on cached `<MEM>`s. Reply via `expmon answer` or a REQUEST.
