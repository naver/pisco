# Owner's question (verbatim, 2026-11-16, pisco chat)

"we need to understand why longchunk is failing. stop the 2048. and prepare a loop for understanding why the work
using long documents is always failing. which point is the bottleneck: training data, training itself, evaluation, ???"

The 2,048-token stage-1 run (job 189693) was stopped by the owner on 2026-11-16. This is a DIAGNOSIS campaign:
find WHERE long inputs lose accuracy, not train another long-chunk arm. The owner's standing rules: probe before
training (inference-only probes on trained models first), one variable per round, paired judge on 1,500 questions,
results tables that name each metric and its direction, with the uncompressed decoder on the same questions and a
compression-rate column; every capability tested on text, images and mixed documents.

# Context gathered by the pisco chat (pointers to read; facts with their source, not decisions)

Long inputs have lost accuracy in every round so far:
- campaign_longchunk (STATE.md, CAMPAIGN.md): 1,024-token stage-1 chunks vs 256, paired judge, 2 seeds: long text
  (g10 = HotpotQA, 10 passages in one long chunk) -1.1 / -3.3 pt, document pages -6.2 / -4.3 pt, photos 0.
  Stage-1 held-out loss was LOWER for the long arm (.612 vs .877) and slot ownership healthy (.969), yet stage-2
  and finetune held-out losses were HIGHER at every eval point.
- In that campaign only stage 1 changed chunk length: stage 2 keeps 256-token text chunks, and the finetune data
  (FT_DATA = mm_ft_docpages_20k_1m_teacher + mm_ft_natural_8k_teacher) holds page images and photos, no long text.
  The g10 eval runs with CML=4096 (one long chunk per question).
- Whole documents of 11-20 pages: route-then-expand -.115 vs uncompressed (memory: pisco-multimodal-campaign;
  docs/architecture_review.md section 1b).
- Transcription probe (scripts/probe_mem_recall.py, analyze_mem_recall.py): multi-page documents KEEP the answer in
  the <MEM>s but the decoder does not use it (27 unused vs 6 lost); infographics lose content in compression.
- Long text: g10 compressed ~.61 vs uncompressed HotpotQA ~.75; MRQA .695 vs raw .807; LongBench F1 .275 vs .263
  (low in absolute terms). Long chunks were "free at inference" under the spread layout (+.037, one 1,270-token chunk
  vs ten 130-token chunks), with a tail drop under the old layout (.399 vs .568).
- OpenEQA (memory: pisco-openeqa-nav-test): more compressed frames barely help (k16/32/64/all 48-52 vs raw 32 frames 57).
- Earlier reviews: docs/data_review.md (stage-2 collator cuts bias toward the top of the page; 17% of PDF text never
  supervised), docs/training_review.md, docs/mem_tokens_review.md, docs/literature_scan_2026-11.md.
- Reference models: expQ/ft_mm_4b_184778 (s42) / 184780 (s43); longchunk control finetunes 188468 / 188469 and the
  closed L1024 finetunes 189694 / 189695 (both trained, both judged on the same 1,500-question sets).

Candidate places the loss could come from (the owner's list; the loop decides how to test each):
training data (what lengths and tasks the decoder ever sees with many <MEM>s), training itself (objective, stages,
positions, optimiser), evaluation (chunking, truncation, decoder_max_length, the judge, the eval sets), and anything
else the probes point to (e.g. the decoder's ability to read many <MEM>s at once).
