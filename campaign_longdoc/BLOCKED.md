## Final checkpoint: close the long-input diagnosis now, or fund one repeat of the slot-reading probe (~15 runs, ~17 GPU-hours)?
In plain words: saying "close" ends the campaign with its ranked answer (the compression itself costs most; the extra
loss on 40-page documents is about half "the decoder reads thousands of compressed slots badly", not yet proven);
saying "repeat" spends about 17 GPU-hours and 15 runs over the cap to re-test that half on new documents, with only
about even odds of a clear answer.

**What the campaign found** (provisional; judge = Qwen3-8B rubric, paired per question, seeds 42 / 43; one `<MEM>` slot
= 8 tokens, ~120 slots per page; yardstick = LongDocURL, one real PDF per question, 1,407 questions, answer cap 512)
| rank | cause | evidence | questions it costs (of 1,407) | training arm it motivates |
|---|---|---|---|---|
| 1 | the compression itself, at every length | compressed − uncompressed is already −16.6 / −18.3 pt at 10 pages, −17 to −20 pt at the shortest size of every set; seeds agree within 2 pt in every cell | ~233 / ~258 at 10 pages | better slots (stage 2; campaigns slotcheck and rebuild) — not a long-input fix |
| 2 | extra loss with length, page images only | 40 vs 10 pages costs −6.5 / −5.3 pt more (project prompt −5.7 / −6.1, string match −4.6 / −3.6); 20 pages, mixed documents and text meet no bar | ~91 / ~75 at 40 pages | split in 2a / 2b |
| 2a | … the decoder reads thousands of slots worse than a few hundred | keeping only the most-attended eighth of the slots (~580 of ~4,800) gains +2.5 / +3.8 pt at 40 pages and nothing at 10 pages (−0.4 / +0.5): net +2.8 [+0.4, +5.3] / +3.3 [+0.8, +6.0] pt. Same sign under the second judge prompt (+1.8 / +3.8), string match (+1.9 / +1.9) and exact match (+1.8 / +1.9). The cause bar is ≥ +3 pt at BOTH seeds: seed 42 misses it by 0.2 pt | ~40 / ~47 at 40 pages | train the decoder on long inputs: finetune / stage-2 rows with 10-40 pages (1,000-5,000 slots; training today shows ≤ 488) |
| 2b | … position numbers past the trained range | moving a 10-page document's slots to its 40-page position numbers costs −0.6 / −0.1 pt (second prompt −0.4 / −0.1): not a cause | 0 | none |
| 3 | long-chunk stage-1 training (1,024-token chunks) | hurts image pages (−6.7 / −4.9 pt on the gold page alone; the loss enters in stage 2); long text: seeds disagree (+4.6 / 0.0) | ~93 / ~65 of 1,500 on 4-page documents | keep the 256-token stage 1 |
| — | the evaluation | sound on LongDocURL (nothing cut, same pages in both modes, needle and closed-book checks pass); the shared 32-token answer cap underrates the uncompressed reader — already handed to the pisco chats | — | — |
Details and tables: `docs/longdoc_diagnosis.md` (section 0 = this table). Budget used: 96 of 96 runs, 98.3 of 114 GPU-h.

**Why I stop:** every funded run is in. Your rule for an effect that is positive at both seeds but short of the bar
is ONE repeat on new questions. That needs runs beyond the cap, and LongDocURL has no unused questions left.

**Options**
| option | runs | GPU-h | what you get |
|---|---|---|---|
| A. close now | 0 | 0 | the ranked answer above; rank 2a stays "consistent with, not confirmed". A long-input training arm would re-measure it on this yardstick (it should drop to ~0 if the decoder learns to read many slots) |
| B. one repeat of rank 2a on new documents | ~15 (cap 96 → ~111): build a 10 / 40-page set from MMLongBench-Doc (licence not yet checked), its closed-book / uncompressed / needle checks, 8 slot-reading runs, 1 second-judge re-score | ~17 (114 → ~116) | a yes / no on rank 2a. But that source gives about 700-1,000 usable questions, so the 95% interval on the effect would be about ±3.5 pt: a true +3 pt effect clears the bar only about half the time |

**Recommendation: A.** The repeat is under-powered for a 3-pt effect, and rank 2a matters only if you train a
long-input decoder, which re-tests it for free. Note: A skips the one-repeat rule for this probe; it is your call.
