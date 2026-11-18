# Slot checks — campaign slotcheck (2026-11)

Why: the document loss — judge accuracy .610 / .615 compressed vs .809 uncompressed on the same 1,500 questions
(`eval_data/mm_docqa_1500_1m`) — sits in what one page's `<MEM>` slots hold. Four checks, diagnosis only, before a
stage-2 rebuild (law: `campaign_slotcheck/CAMPAIGN.md`). Compression rate everywhere: 8 tokens per slot; on the eval
documents ≈ 108 slots per page (430.7 per 4-page question, 646,042 / 1,500).

| check | question | status |
|---|---|---|
| 1 | does the finetune collator's 4-document cap drop the answer page? | **decided: not a cause** (0 of 20,000 rows) |
| 2a | does left padding change the `<MEM>` vectors? | **decided: not inert** (120 pages): pads move the slots on both batch paths; why is being split (job 191781) |
| 2b | does training-style padding (to 64) change judge accuracy? | running: wrapper check (300 Q), then 1,500 Q × 2 seeds |
| 3 | do text slots of the page's transcription keep what image slots lose? | not started |
| 4 | can stage 2 memorise 100 dense pages (slot capacity vs training)? | not started |

## Check 1 — answer pages dropped by `topk_docs=4` in the finetune set (decided 2026-11-18)

Cell: CPU audit, job 191663 (`scripts/slotcheck/audit_ft_rows.py`), every row of the two sets the reference finetunes
184778 / 184780 trained on (`data/mm_ft_docpages_20k_1m_teacher`, first 20,000 rows = all;
`data/mm_ft_natural_8k_teacher`, 8,000 = all). "Loses every answer page" = every gold position ≥ 4 (definition fixed
in the law before the count was read). Controls passed: 50 / 50 sampled rows through the committed `example_chunks`
give min(documents, 4) blocks (25 of them with 5 documents); a planted 6-document row with golds at 4 and 5 counts as
dropped, [0, 5] as partial, [1] as kept. Health: rows read = set size, per-source counts sum to the total.

| source | trained rows | rows with 5 documents (1 is cut) | rows losing EVERY answer page ↓ | rows losing one of two source pages ↓ | gate |
|---|---|---|---|---|---|
| docmatix | 10,000 | 4,745 (47.5%) | **0 (0.00%)** | 1,910 (19.1%) | — |
| docvqa | 7,000 | 0 | 0 | 0 | — |
| infographic_vqa | 2,000 | 0 | 0 | 0 | — |
| chartqa | 1,000 | 0 | 0 | 0 | — |
| **document rows, total** | **20,000** | 4,745 (23.7%) | **0 (0.00%)** | 1,910 (9.55%) | ≤ 2% → not a cause |
| photos (aokvqa / textvqa / vqav2) | 8,000 | 0 (1 image per row) | 0 | 0 | — |

Why "every page" is 0: a 5-document Docmatix row holds 2 pages of the source PDF + 3 distractors, shuffled; only
index 4 is cut, so at most one of the two source pages can go. The record marks both source pages as gold and does not
say which one holds the answer.

**For the stage-2 rebuild:** the cap never trains the student on a row whose answer pages are all missing, so it is not
a cause of the document gap. One data note: 1,910 Docmatix rows (9.55% of document rows) lose one of their two source
pages while the teacher label was written from all of them; a rebuild with `topk_docs=5` removes that doubt at the
cost of one more page on 24% of rows.

## Check 2a — do left pads change a page's `<MEM>` vectors? (decided 2026-11-18: not inert)

Cell: job 191672 (`scripts/slotcheck/pad_probe.py`), R seed 42 (`expQ/ft_mm_4b_184778/model`), the first 120 pages
of `eval_data/mm_docqa_1500_1m` (PX 1,003,520, 651-903 compressor tokens, 73-101 slots per page), each page cut and
compressed by the eval's own code, with 0 / 17 / 63 extra left pad tokens (attention mask 0). Training pads every
compressor batch to a multiple of 64; the eval does not. Metric: cosine between the padded and the unpadded `<MEM>`
vector (connector output, what the decoder reads), median over a page's slots, then median over pages. Not a judge
number: no uncompressed column applies (the set's uncompressed judge accuracy is .809).

| path | n | median cosine ↑ | worst slot cosine ↑ | max abs difference ↓ | reference ↑ | health | read |
|---|---|---|---|---|---|---|---|
| same input twice (noise floor) | 120 pages | 1.0000 | 1.0000 | 0 | — | pass | control |
| another page (planted difference) | 120 pages | .739 | .600 | 2.73 | must be < .99 | pass | control |
| one page alone + 17 pads | 120 pages | .934 | .632 | 2.65 | 1.0000 (bar ≥ .9999) | slot count same | moves |
| one page alone + 63 pads | 120 pages | .934 | .631 | 2.80 | 1.0000 (bar ≥ .9999) | slot count same | moves |
| a question's 4 pages, padded to 64 as in training | 30 questions | .9966 | .637 | 2.39 | 1.0000 (bar ≥ .9999) | — | moves |

Two paths, two sizes. A batch of ONE row (a photo question in training: one image) leaves the pads in the recurrent
state of Qwen3.5's linear-attention layers: the installed transformers zeroes pad positions there only when the batch
has more than one row (`modeling_qwen3_5.py:198`). A batch of several rows (a document question) zeroes them, and the
slots still move, less (median .9966). Job 191781 tests both: the 1-row case with the zeroing forced on, and the same
comparisons in fp32 to tell rounding from a real change.

**For the stage-2 rebuild:** not yet known. If check 2b shows a judge cost, the rebuild trains and evaluates with
the same padding (or none); if the 1-row leak is confirmed, photo rows were trained on slots the eval never produces.
