Owner (2026-11-18): "no progress has been done after many tries to improve the models. double check again training set,
models architecture, evaluation dataset and protocol." A four-audit review (this chat, 2026-11-18) found no single bug but
found that the document loss (judge accuracy .61 compressed vs .80 uncompressed on eval_data/mm_docqa_1500_1m) sits in what
ONE page's <MEM> slots hold, and that no training arm so far aimed at it. The owner approved four checks ("yes go 1-4 with camp").
Diagnosis only: inference probes, CPU audits and one small overfit training. No full training run.

Owner rule (2026-11-18): NO route-then-expand, no raw-page or raw-tile expansion, no picked-page tricks, as arms or as
proposals. Raw-page scores may be quoted as a ceiling only. Every arm must be about what the slots hold.

Reference models: expQ/ft_mm_4b_184778 (seed 42) and expQ/ft_mm_4b_184780 (seed 43); stage-2 start expQ/archq3/pt_mm_5M_spread/model.

The four checks, cheapest first:
1. CPU count: in the finetune set data/mm_ft_docpages_20k_1m_teacher, how many rows have more than 4 documents, and in how
   many of those is an answer (gold) page at position >= 4, i.e. dropped by the collator's topk_docs=4 cap
   (pisco/mm_collator.py:489; builder scripts/build_mm_finetune.py puts up to 2 gold + 3 distractor pages, shuffled; the
   teacher label was written from up to 8 pages, scripts/relabel_mm_teacher.py:60). Report per source.
2. Padding probe: training pads the compressor input to a multiple of 64 (pisco/mm_collator.py:431), the eval passes
   pad_to_multiple_of=None (scripts/eval_mm_docqa.py:90); the compressor is left-padded. (a) compress the same pages with
   0, 17 and 63 pad tokens and compare the <MEM> vectors (cosine, max abs difference) against a same-input repeat as the
   noise floor; (b) one eval of each reference model on documents with padding to 64 as in training, paired against the
   recorded unpadded file re-run on the same code and node.
3. Text-slot probe (no training): for all 4 pages of each document question, compress the page's TEXT transcription as text
   slots (the uncompressed model's own transcription, or the text layer where the data has one) and answer from those
   slots; also image slots + text slots together. Paired against image slots only. Tells whether a text side channel next
   to the image slots is worth building. No raw page ever enters the decoder prompt.
4. Overfit test: train stage 2 (multimodal pretraining collator, page transcription task only) on 100 dense pages until the
   training loss stalls, at 8 tokens per slot, then measure page word recall (scripts/probe_mem_recall.py /
   scripts/longdoc/mem_recall.py) on those same 100 pages. Recall near 100% = slot capacity is fine and training is the
   limit; recall staying near the current 65% = capacity or architecture limit. Run it in bf16 and in fp32 master weights
   if the budget allows (one variable: precision).

Deliverable: docs/slot_checks.md with one table per check (dataset x metric, each column naming its metric and direction,
compression rate column, uncompressed baseline on the same questions), and for each check a one-line verdict on what it
means for the stage-2 rebuild that would follow.
