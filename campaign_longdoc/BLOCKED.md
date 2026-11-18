## Arm E model cells: approve about 54 more GPU-hours and 17 more cells?
In plain words: if you say yes, the campaign measures on real long documents (LongDocURL texts of 4k-32k tokens, PDFs of 10-40 pages, and mixed image + text pages) how much the compressed model loses against the uncompressed one at each length, at both seeds, and whether long-chunk training helps on long text; it costs about 82 GPU-hours (54 above what is left of the campaign's 60) and 30 cells (17 above what is left of its 70).

**New since the first version of this question (2026-11-19 08:30): the earlier length results are not clean.**
The length runs on our usual eval sets (adding distractor pages or passages from other documents) all ran with the old
32-token answer cap. That cap cuts the uncompressed reader's answers mid-sentence, and more at long inputs: 5.7% of its
answers at 4 pages, 16.3% at 20 pages; 33% at 10 text passages, 47% at 40 (bar 5%; R ≤ 5.9%). So:
- the reading "at 20 pages the gap to uncompressed SHRINKS" (+4.3 pt at seed 42, +3.1 at seed 43) is withdrawn: on the
  questions no cap touches it becomes −0.6 / −2.0 pt;
- on text, the same restricted read leans toward a real length loss (−4.8 / −3.4 pt from 10 to 40 passages);
- these restricted reads are chosen on the answers, so they decide nothing. Arm E already runs at a 512-token cap,
  so it is now the only clean length measurement the campaign has.

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
Used so far: 32.0 GPU-h (sacct over the 57 campaign jobs, every attempt; the seed-43 20-page cell was preempted 4 times and took 3.2 GPU-h for a 1.5 h run)
and 57 cells. Left: ≈ 28 GPU-h and 13 cells.

**Why 26 cells, not 13.** gpu-be preempts jobs, and `eval_mm_4b.sh` restarts a preempted generation from zero (seen
again this wake: 4 restarts). So every cell above 4 h (uncompressed pages and mixed, R pages and mixed at both seeds)
runs as 3 cells, one per length bucket (same rows, written once as per-bucket sets by one CPU job). That is 18 bucket
cells + 7 whole cells + 1 CPU build = 26, plus 4 for relaunches. A preemption then loses one bucket at most.

**Options**
| option | GPU-h added | cells added | what you get | what is lost |
|---|---|---|---|---|
| 1. full arm E grid, bucket cells (recommended) | +54 (60 → 114; 73 + ≈ 2 model loads + ≈ 7 relaunch) | +17 (70 → 87) | gap and Δ_len (the gap at a length minus the gap at the shortest) per bucket, both seeds, three modalities, at a clean cap; the long-chunk text answer | the distractor-set length runs stay undecided |
| 2. as 1, plus the distractor-set length runs again at the 512 cap (documents 4 / 20 pages, text 10 / 40 passages, uncompressed + both seeds: 12 cells) | +66 (→ 126) | +31 (→ 101) | also settles the 20-page and 40-passage reads on our usual sets | — |
| 3. as 1 without the long-chunk text cells | +39 (→ 99) | +13 (→ 83) | the length answer on real documents | arm T's text half stays open (long-chunk stage 1 is already stopped) |
| 4. stop arm E here | 0 | 0 | the trial hints above | no length number decides anything (the distractor-set runs are capped) |

**Recommendation: option 1** (+54 GPU-hours, +17 cells). Real long documents answer the length question better than
distractor pages from other documents, and arm E already runs at the clean cap. Order: the three uncompressed cells
first (soundness on every row, and the reference), then R s42, R s43, then C256 / L1024. Every cell on gpu-be outside
homer-[005-009], `--time` = measured wall × 3, at most 4 submissions per wake.
