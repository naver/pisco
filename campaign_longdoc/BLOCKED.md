## Final checkpoint: fund the last two probes (9 runs, ~13 GPU-hours), or close the campaign?
In plain words: approving 9 more evaluation runs (about 13 GPU-hours, inside the 114 GPU-hours you already approved,
but 9 runs over the 87-run cap) would tell whether the extra loss on 40-page documents comes from the decoder reading
too many compressed slots or from position numbers it never saw in training; saying no closes the campaign with
that one question open.

**What the campaign found** (provisional; judge = Qwen3-8B rubric, paired per question, both seeds 42 / 43; one
`<MEM>` slot = 8 tokens; the yardstick is LongDocURL, one real PDF per question, 1,407 questions, answer cap 512):
| rank | cause | evidence (n) | questions it costs | training arm it motivates |
|---|---|---|---|---|
| 1 | the compression itself, at every length | compressed − uncompressed is already −17 to −20 pt at the shortest size (10 pages, 10 mixed items, ~4k text tokens) and the two seeds agree within 2 pt in every cell | ~233 of 1,407 on 10-page documents (353 only uncompressed right vs 120 only compressed right) | better slots (campaign slotcheck, docs/review_2026-11-18.md) — not a long-input fix |
| 2 | length, on page images only | 40 vs 10 pages costs a further −6.5 / −5.3 pt (rubric judge), −5.7 / −6.1 (project prompt), −4.6 / −3.6 (string match), every CI below 0; exact match flips the sign because the uncompressed decoder answers in longer sentences (its exact match .19 → .08 while its judge stays .67 → .66). 20 pages, mixed documents and text meet no bar | ~80 of 1,407 at 40 pages | unknown until the two probes below run: both "too many slots" and "untrained positions" are open, and training never shows the decoder more than 488 slots (every arm E question is beyond that) |
| 3 | long-chunk stage-1 training (1,024-token chunks) | hurts image pages: −6.7 / −4.9 pt with the gold page alone, the loss sits in the compressed page and enters in stage 2; on long text the seeds disagree (+4.6 / 0.0 pt, 2,246 questions; the +4.6 is one weak 256-chunk model) | ~65-93 of 1,500 on 4-page documents | keep the 256-token stage 1 |
| — | the evaluation | sound for arm E: nothing truncated, both modes see the same pages, needle and closed-book checks pass. But the shared 32-token answer cap cuts the uncompressed answers mid-sentence (pages .340 → .717 when raised to 256, 300 rows): handed to the pisco chat | — | — |
Details: `docs/longdoc_diagnosis.md`, figure `expQ/longdoc/figures/length_curve.svg`.

**Why I stop:** the law runs two more probes where a length loss shows (page images, 40 vs 10 pages, both seeds), but
only 2 of the 87 runs are left. Rule: never trim the grid to fit; ask.
- Slot-reading probe: keep only the most-attended eighth of the slots, at 10 and 40 pages, both seeds — 4 runs,
  ≤ 10.2 GPU-h (measured walls 1:13 and 3:53). If keeping fewer slots recovers the 40-page loss, the decoder reads
  many slots worse than few.
- Position probe: one eval flag that shifts the slot position numbers of a 10-page document to where a 40-page
  document puts them (≤ 40 lines in `scripts/eval_mm_docqa.py`, off by default, checked byte-identical on 300 rows
  first) — 4 runs (tests, the 300-row check, 2 seeds), ~2.8 GPU-h. If the shift alone costs the 5-6 pt, positions are
  the cause and an inference-only position remap may fix it.
- Second-judge re-score of whatever decides — 1 run, ~0.1 GPU-h.

**Options**
| option | runs | GPU-h | what you get |
|---|---|---|---|
| A. fund both probes | +9 (cap 87 → 96) | ~13 (≈ 103 of 114 after) | which of the two causes the 40-page loss; the training arm it motivates |
| B. close now | 0 | 0 | the ranked list above, with rank 2's cause open |

**Recommendation: A** — cap 96 runs, GPU-hours unchanged at 114. It is the law's planned next step, reversible, and
the position probe can turn up a fix that needs no training.
