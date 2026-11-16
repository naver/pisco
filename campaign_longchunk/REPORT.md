# Campaign report — longchunk (2026-11-16 18:31)
Does training stage 1 (text pretraining of the compressor + connector + decoder LoRA) with LONG compressor chunks (1,024 or 2,048 tokens instead of 256) make the compressed model better at reading long passages, without losing on pages and photos? Background: under the spread `<MEM>` layout, chunk length is already free at inference — one
window 2026-11-11 → 2026-11-16 (until 2027-02-08) · 276 ticks · 23 wakes (4 rc≠0) · 638.0 GPU-h over 51 runs
⚠ The campaign idled 85.6 h of its 137.5 h window (62%) — with work outstanding, that is the main operational fact.

## Where this leaves the question
**Settled**
- pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no training); results in the ledger (spread g10 .617 vs g1 .579; tail .399); not a cell of the 2026-11-10 campaign — jobs 185499, 185500, 185501 … 185603 (16 cells), closed
- data check, SmolLM2: 23.8% of docs >= 1,024 tokens, 8.3% >= 2,048 -> corpus rejected for long chunks (feasibility gate a) — job 188339, closed
- data build + check, MS MARCO docs: 43.4% >= 1,024, 20.4% >= 2,048 -> corpus adopted; data/msmarco_docs built (feasibility gate a PASSED) — job 188358, closed
- feasibility gate (b) PASS at L2048 (CML 2048, DML 4096, BS 3 x ACCUM 16 x 4 A100-80GB): 50/50 steps, exit 0, GPU memory peak 288 GB over 4 GPUs (~72 GB each, sacct gpumem), host RAM 108 of 160 GB, GPU util 99.5%; 58.6 s/step steady (steps 10-49, 39 steps in 2287 s); full cell 5,208 steps = 84.8 h -> --time 106 h, 339 GPU-h; eval_loss 2.026 at step 50 (9,600 docs) — job 188420, closed
- fit + speed test PASS at L1024 (CML 1024, DML 2048, BS 6 x ACCUM 8): 50/50 steps, exit 0, GPU memory peak 285 GB over 4 GPUs (~71 GB each), host RAM 96 GB, util 99.7%; 42.1 s/step steady (steps 10-49, 1641 s); full cell 5,208 steps = 60.9 h -> --time 76 h, 244 GPU-h; eval_loss 2.096 at step 50 (9,600 docs) — job 188421, closed
- C256 stage 1 COMPLETED 16:43 h, 1M docs (5,208 steps); held-out loss .801/.877/1.007 at steps 2000/4000/5208 (64 rows); SmolLM2 reference 180486 had the same end rise .924/.920/1.023, so not a collapse; train loss ~.94 — job 188422, closed
- … 15 more

**Open, and running**
- (none)

## Arms
| arm | job | state | GPU-h | verdict | headline (last verdict, else latest probe) |
|---|---|---|---|---|---|
| kind=eval_mm topk=10 ckpt=ft_mm_4b_183814 mode=pisco data=text_hotpotqa_1500 image_rate=ckpt | 185499 | CANCELLED | 0.0 | closed | pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no tr |
| kind=eval_mm topk=10 ckpt=ft_mm_4b_183814 mode=pisco data=text_hotpotqa_1500_g3 image_rate=ckpt | 185500 | CANCELLED | 0.0 | closed | pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no tr |
| kind=eval_mm topk=10 ckpt=ft_mm_4b_183814 mode=pisco data=text_hotpotqa_1500_g10 image_rate=ckpt | 185501 | CANCELLED | 0.0 | closed | pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no tr |
| kind=eval_mm topk=10 ckpt=ft_mm_4b_183538 mode=pisco data=text_hotpotqa_1500 image_rate=ckpt | 185502 | CANCELLED | 0.0 | closed | pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no tr |
| kind=eval_mm topk=10 ckpt=ft_mm_4b_183538 mode=pisco data=text_hotpotqa_1500_g3 image_rate=ckpt | 185503 | CANCELLED | 0.0 | closed | pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no tr |
| kind=eval_mm topk=10 ckpt=ft_mm_4b_183538 mode=pisco data=text_hotpotqa_1500_g10 image_rate=ckpt | 185504 | CANCELLED | 0.0 | closed | pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no tr |
| kind=eval_mm topk=10 ckpt=ft_mm_4b_183814 mode=pisco data=text_hotpotqa_1500 image_rate=ckpt limit=all max_pixels=451584 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 | 185505 | PREEMPTED | 0.4 | closed | pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no tr |
| kind=eval_mm topk=10 ckpt=ft_mm_4b_183814 mode=pisco data=text_hotpotqa_1500_g3 image_rate=ckpt limit=all max_pixels=451584 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 | 185506 | FAILED | 0.0 | closed | pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no tr |
| kind=eval_mm topk=10 ckpt=ft_mm_4b_183814 mode=pisco data=text_hotpotqa_1500_g10 image_rate=ckpt limit=all max_pixels=451584 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 | 185507 | FAILED | 0.0 | closed | pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no tr |
| kind=eval_mm topk=10 ckpt=ft_mm_4b_183538 mode=pisco data=text_hotpotqa_1500 image_rate=ckpt limit=all max_pixels=451584 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 | 185509 | PREEMPTED | 0.4 | closed | pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no tr |
| kind=eval_mm topk=10 ckpt=ft_mm_4b_183538 mode=pisco data=text_hotpotqa_1500_g3 image_rate=ckpt limit=all max_pixels=451584 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 | 185510 | FAILED | 0.0 | closed | pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no tr |
| kind=eval_mm topk=10 ckpt=ft_mm_4b_183538 mode=pisco data=text_hotpotqa_1500_g10 image_rate=ckpt limit=all max_pixels=451584 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 | 185511 | FAILED | 0.0 | closed | pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no tr |
| kind=eval_mm topk=10 ckpt=ft_mm_4b_183814 mode=pisco data=text_hotpotqa_1500_g3 image_rate=ckpt limit=all max_pixels=451584 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 | 185600 | COMPLETED | 0.3 | closed | pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no tr |
| kind=eval_mm topk=10 ckpt=ft_mm_4b_183814 mode=pisco data=text_hotpotqa_1500_g10 image_rate=ckpt limit=all max_pixels=451584 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 | 185601 | COMPLETED | 0.3 | closed | pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no tr |
| kind=eval_mm topk=10 ckpt=ft_mm_4b_183538 mode=pisco data=text_hotpotqa_1500_g3 image_rate=ckpt limit=all max_pixels=451584 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 | 185602 | COMPLETED | 0.3 | closed | pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no tr |
| kind=eval_mm topk=10 ckpt=ft_mm_4b_183538 mode=pisco data=text_hotpotqa_1500_g10 image_rate=ckpt limit=all max_pixels=451584 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 | 185603 | COMPLETED | 0.3 | closed | pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no tr |
| kind=data_check dataset=smollm2_10b rows=5000000 every=50 tokenizer=qwen3.5-4b | 188339 | COMPLETED | 0.0 | closed | data check, SmolLM2: 23.8% of docs >= 1,024 tokens, 8.3% >= 2,048 -> corpus reje |
| kind=data_build dataset=msmarco_docs every=30 tokenizer=qwen3.5-4b | 188358 | COMPLETED | 0.0 | closed | data build + check, MS MARCO docs: 43.4% >= 1,024, 20.4% >= 2,048 -> corpus adop |
| kind=pt_text backbone=qwen3.5-4b data=msmarco_docs samples=1000000 cml=2048 dml=4096 bs=3 accum=16 rate=8 layout=spread extra=++hf_training.max_steps=50 | 188420 | COMPLETED | 3.8 | closed | feasibility gate (b) PASS at L2048 (CML 2048, DML 4096, BS 3 x ACCUM 16 x 4 A100 |
| kind=pt_text backbone=qwen3.5-4b data=msmarco_docs samples=1000000 cml=1024 dml=2048 bs=6 accum=8 rate=8 layout=spread extra=++hf_training.max_steps=50 | 188421 | COMPLETED | 2.9 | closed | fit + speed test PASS at L1024 (CML 1024, DML 2048, BS 6 x ACCUM 8): 50/50 steps |
| … 31 more | | | | | |

## Findings (from verdicts, not prose)
- pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no training); results in the ledger (spread g10 .617 vs g1 .579; tail .399); not a cell of the 2026-11-10 campaign (jobs 185499, 185500, 185501 … 185603, closed)
- data check, SmolLM2: 23.8% of docs >= 1,024 tokens, 8.3% >= 2,048 -> corpus rejected for long chunks (feasibility gate a) (job 188339, closed)
- data build + check, MS MARCO docs: 43.4% >= 1,024, 20.4% >= 2,048 -> corpus adopted; data/msmarco_docs built (feasibility gate a PASSED) (job 188358, closed)
- feasibility gate (b) PASS at L2048 (CML 2048, DML 4096, BS 3 x ACCUM 16 x 4 A100-80GB): 50/50 steps, exit 0, GPU memory peak 288 GB over 4 GPUs (~72 GB each, sacct gpumem), host RAM 108 of 160 GB, GPU util 99.5%; 58.6 s/step steady (steps 10-49, 39 steps in 2287 s); full cell 5,208 steps = 84.8 h -> --time 106 h, 339 GPU-h; eval_loss 2.026 at step 50 (9,600 docs) (job 188420, closed)
- fit + speed test PASS at L1024 (CML 1024, DML 2048, BS 6 x ACCUM 8): 50/50 steps, exit 0, GPU memory peak 285 GB over 4 GPUs (~71 GB each), host RAM 96 GB, util 99.7%; 42.1 s/step steady (steps 10-49, 1641 s); full cell 5,208 steps = 60.9 h -> --time 76 h, 244 GPU-h; eval_loss 2.096 at step 50 (9,600 docs) (job 188421, closed)

## What went wrong
- The campaign idled 85.6 h of its 137.5 h window (62%) — with work outstanding, that is the main operational fact.
- 4 of 23 LLM wakes returned rc≠0.
- Alert raised: no campaign GPU-second for .h (cap h) — a gate failing CLOSE|2026-11-11.
- 4 run(s) ended FAILED/TIMEOUT/NODE_FAIL: 185506 185507 185510 185511.
- BLOCKED.md is standing: BLOCKED — campaign longchunk (2026-11-16 04:00).

## Left, and what it costs
- 0 arm(s) without a verdict
- window: until 2027-02-08

## Artifacts
- (no best/runner-up run yet)
- campaign dir: /beegfs/scratch/user/hdejean/pisco/campaign_longchunk (STATE.md, BLOCKED.md)
- ledger: 109 entries of this campaign — expmon ledger (linked, never pasted)
