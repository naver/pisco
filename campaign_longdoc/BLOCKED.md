## Arm E model cells: approve about 50 more GPU-hours and 16 more cells?
In plain words: if you say yes, the campaign measures on real long documents (LongDocURL texts of 4k-32k tokens, PDFs of 10-40 pages, and mixed image + text pages) how much the compressed model loses against the uncompressed one at each length, at both seeds, and whether long-chunk training helps on long text; it costs about 75 GPU-hours (50 above the campaign's 60) and 26 cells (16 above its 70).

**What is ready.** The long-document sets are built and sound (all 300-row trials, cap 512 tokens):
- the document is needed: uncompressed with the document beats closed-book by +27 to +54 pt in every bucket (bar 20);
- the uncompressed decoder can read that far: a planted 6-digit code is found in 97-100% of rows at every length
  (text up to 33.5k tokens, 40 pages, mixed up to 68.7k tokens; bar 90%);
- no input is cut, both modes get the same pages (audit, every row).
The trials already hint at the answer (300 rows, decides nothing): R − uncompressed ≈ −16 pt on pages, text and mixed;
on pages and mixed the gap grows with length (pages −13 → −23 pt, mixed −12 → −21 pt from 10 to 40), on text it does not.

**Cost of the grid the law asks for** (GPU-h = measured wall per 300 judged rows × rows; one GPU per cell):
| cell | rows | GPU-h per cell | cells | GPU-h |
|---|---|---|---|---|
| uncompressed pages (also the soundness reference) | 4,221 | 8.0 (cap-256 wall 31:01 × 1.10, the text 256 → 512 ratio) | 1 | 8.0 |
| uncompressed text | 2,246 | 2.9 | 1 | 2.9 |
| uncompressed mixed | 4,221 | 10.3 | 1 | 10.3 |
| R pages, seeds 42 and 43 | 4,221 | 7.3 | 2 | 14.6 |
| R text, seeds 42 and 43 | 2,246 | 3.8 | 2 | 7.6 |
| R mixed, seeds 42 and 43 | 4,221 | 7.3 | 2 | 14.7 |
| long-chunk on text: C256 and L1024 finetunes, CML 1,024, s42 and s43 | 2,246 | 3.8 | 4 | 15.2 |
| total | | | 13 | ≈ 73 |
Left after this wake: ≈ 32 GPU-h (25.8 used + ≈ 2.2 for the two running seed-43 length cells) and 14 cells.

**Why 26 cells, not 13.** gpu-be preempts jobs, and `eval_mm_4b.sh` restarts a preempted generation from zero. A
7-10 h cell would likely restart several times. So every cell above 4 h (uncompressed pages and mixed, R pages and
mixed at both seeds) runs as 3 cells, one per length bucket (same rows, written once as per-bucket sets by one CPU
job). That is 18 bucket cells + 7 whole cells + 1 CPU build = 26. A preemption then loses one bucket at most.

**Options**
| option | GPU-h added | cells added | what you get | what is lost |
|---|---|---|---|---|
| 1. full grid, bucket cells (recommended) | +50 (60 → 110; 73 + ≈ 2 for model loads + ≈ 7 for one relaunch) | +16 (70 → 86; 12 needed + 4 relaunches) | gap and Δ_len (the gap at a length minus the gap at the shortest) per bucket, both seeds, three modalities; the long-chunk text answer | — |
| 2. as 1 without the long-chunk text cells | +35 (→ 95) | +12 (→ 82) | the length answer on real documents | arm T's text half stays open (long-chunk stage 1 is already stopped) |
| 3. seed 42 only (40 GPU-h of cells, 15 cells) | +16 (→ 76) | +3 (→ 73) | one-seed gaps | the law needs both seeds to conclude: nothing could be settled |
| 4. stop arm E here | 0 | 0 | the trial hints above | no long-document number decides anything |

**Recommendation: option 1** (+50 GPU-hours, +16 cells). Order: the three uncompressed cells first (soundness on every
row, and the reference), then R s42, R s43, then C256 / L1024. Every cell on gpu-be outside homer-[005-009],
`--time` = measured wall × 3, at most 4 submissions per wake.
