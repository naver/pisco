# Slot checks — campaign slotcheck (2026-11)

Why: the document loss — judge accuracy .610 / .615 compressed vs .809 uncompressed on the same 1,500 questions
(`eval_data/mm_docqa_1500_1m`) — sits in what one page's `<MEM>` slots hold. Four checks, diagnosis only, before a
stage-2 rebuild (law: `campaign_slotcheck/CAMPAIGN.md`). Compression rate everywhere: 8 tokens per slot; on the eval
documents ≈ 108 slots per page (430.7 per 4-page question, 646,042 / 1,500).

| check | question | status |
|---|---|---|
| 1 | does the finetune collator's 4-document cap drop the answer page? | **decided: not a cause** (0 of 20,000 rows) |
| 2a | does left padding change the `<MEM>` vectors? | **decided: not inert, split** (120 + 40 pages): a one-row padded batch leaks the pads (latent: no training or eval run takes that path); a several-row batch moves the slots only by bf16 rounding, as much as batching with any other page |
| 2b | does training-style padding (to 64) change judge accuracy? | running: wrapper check passed (300 Q); 1,500 Q × 2 seeds judged now |
| 3 | do text slots of the page's transcription keep what image slots lose? | not started |
| 4 | can stage 2 memorise 100 dense pages (slot capacity vs training)? | queued: page set build, 20-step training smoke, recall of the start model before training |

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

## Check 2a — do left pads change a page's `<MEM>` vectors? (decided 2026-11-18: not inert; cause split)

Cells: job 191672 (`scripts/slotcheck/pad_probe.py`, 120 pages) and job 191781 (`scripts/slotcheck/pad_split.py`,
the first 40 of them, bf16 and fp32). Model R seed 42 (`expQ/ft_mm_4b_184778/model`); pages of
`eval_data/mm_docqa_1500_1m` (PX 1,003,520, 651-903 compressor tokens, 73-101 slots per page), each page cut and
compressed by the eval's own code, with 0 / 17 / 63 extra left pad tokens (attention mask 0). Training pads every
compressor batch to a multiple of 64; the eval does not. Metric: cosine between the padded and the unpadded `<MEM>`
vector (connector output, what the decoder reads), median over a page's slots, then median over pages. Not a judge
number: no uncompressed column applies (the set's uncompressed judge accuracy is .809).

| path (63 pads unless named) | n pages | median cosine bf16 ↑ | median cosine fp32 ↑ | worst slot bf16 ↑ | max abs bf16 ↓ | read |
|---|---|---|---|---|---|---|
| same input twice (noise floor) | 120 / 40 | 1.0000 | 1.0000 | 1.0000 | 0 | control passes |
| another page (planted difference) | 120 | .739 | — | .600 | 2.73 | control passes (< .99) |
| same page in a 2-row batch, NO pads (batching floor) | 40 | .9943 | 1.0000 | .662 | 2.15 | bf16 rounding of a different batch shape |
| one row + pads, as installed | 120 / 40 | .934 (17 pads: .934) | .953 | .631 | 2.80 | **leaks**: far below the floor in both precisions |
| one row + pads, pad zeroing forced on | 40 | .995 | 1.0000 | .642 | 2.18 | = floor: the leak is the zeroing rule |
| 2-row batch + pads (masked, how training and the eval batch) | 40 | .9948 | 1.0000 | .665 | 2.08 | = floor: rounding, not a leak |
| a question's 4 pages, padded to 64 as in training | 30 questions | .9966 | — | .637 | 2.39 | several-row path, real questions |

What it means. (1) The installed transformers zeroes pad positions in Qwen3.5's linear-attention layers only when the
batch has more than one row (`modeling_qwen3_5.py:198`), so a ONE-row padded batch leaks the pads into the page's
slots (fp32 .953). No recorded run takes that path: every training batch has ≥ 2 compressor rows (stage 2 bs 8,
finetune bs 2; all chunks are padded together), and the eval pads nothing (a one-row batch has no pads). Hand-off
2de131 asks the pisco chat to guard it before any code compresses one page at a time with padding. (2) On the
several-row path the pads are masked: in fp32 the slots are unchanged (median 1.0000, worst .985), and in bf16 they
move exactly as much as putting the page in a batch with any other page without pads (.9948 vs .9943). The law's
inert bar (.9999 in bf16) cannot be met even by that floor, so check 2b asks whether this bf16 movement changes
answers: in its 300-question wrapper check, 67 / 300 answers changed with training padding (233 identical).

**For the stage-2 rebuild:** padding is not a cause by itself on the paths we run; bf16 batch-shape rounding moves
slots by a cosine of about .995 per question. Check 2b says whether that changes the judge score.
