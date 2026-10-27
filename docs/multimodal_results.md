# Multimodal PISCO: what compresses well

State on 2026-10-20. Backbone Qwen3.5-4B (compressor and decoder), LoRA, one A100 per run.
PISCO compresses each image into a few `<MEM>` vectors; the decoder answers from those vectors only, never from
pixels. Scores are judge correctness (see "How the evaluation is done"), paired with the uncompressed Qwen3.5-4B on
the same questions, with 95% intervals. Speed is in [latency.md](latency.md).

## Bottom line

Compression works when the answer depends on large, salient content, and struggles when it depends on one small detail
among many. It is not a "document versus photo" divide. Giving the decoder the one page that holds the answer as a raw
image (the model picks it from the `<MEM>`s itself), the other pages staying compressed, closes the document gap with no
training: .801 vs .809 uncompressed, with a third of the tokens ([architecture_review.md](architecture_review.md)).

Best model `expQ/ft_mm_4b_173550/model` (stage-2 v4+KL at 1M px, generic teacher-label finetune), 1,500 questions per
family (500 per source), rubric judge:

| image type | eval | compressed | uncompressed | difference (95% CI) | share kept | `<MEM>` per image | rate |
|---|---|---|---|---|---|---|---|
| nav frames (rendered scans of real houses) | memory QA, 928 Q, 8 frames per Q | .806 | .792 | +.014 (−.010, +.038) | ~100% | 25 | 16× |
| blurry user photos | VizWiz, 500 Q | .564 | .572 | −.008 (−.050, +.032) | 99% | 56 | 8× |
| photos with text | TextVQA, 500 Q | .788 | .854 | −.066 (−.102, −.032) | 92% | 56 | 8× |
| scenes | RealWorldQA, 500 Q | .642 | .726 | −.084 (−.122, −.044) | 88% | 56 | 8× |
| slides | SlideVQA, 500 Q | .648 | .774 | −.126 (−.168, −.084) | 84% | 123 | 8× |
| multi-page documents | MP-DocVQA, 500 Q | .682 | .926 | −.244 (−.288, −.202) | 74% | 123 | 8× |
| infographics | InfographicVQA, 500 Q | .404 | .726 | −.322 (−.370, −.272) | 56% | 123 | 8× |
| **all photos** | 1,500 Q | **.665** | **.717** | −.053 (−.074, −.031) | 93% | | |
| **all documents** | 1,500 Q | **.578** | **.809** | −.231 (−.258, −.206) | 71% | | |

Rate = image tokens / `<MEM>` vectors (`<MEM>` per image = image tokens ÷ rate + 1). Pages and slides use a 1M-pixel
budget (about 980 image tokens), photos 451K pixels (about 440), nav frames about 400. Counting the question and the
prompt template, which stay uncompressed, a whole prompt is 6 to 11 times shorter.

Caveats: the nav questions are simple (which object, in what order, on which side) and the frames are renders, not
photos; the nav "uncompressed" number is the backbone LoRA-finetuned on the same rows (the fair ceiling,
`scripts/train_base_mm.py`). Navigation model: `expQ/ft_nav_4b/model`.

Under the earlier Yes/No judge the same files read: documents .594 vs .817, photos .669 vs .718, nav .810 vs .797
(the `judge_project` fields). The two judges never change a ranking; they differ on long free-text answers (VizWiz).

## Compression beats fewer pixels

The training-free alternative to compression is to shrink the image until the raw model sees as many visual tokens as
PISCO has `<MEM>` vectors. Same 1,500 questions, same judge, same prompt length:

| family | tokens per question | PISCO | raw, same tokens | raw, 2× tokens | raw, full resolution |
|---|---|---|---|---|---|
| documents (4 pages) | 431 `<MEM>` vs 439 / 915 / 3416 visual | **.578** | .403 (−.175, CI −.203 to −.147) | .607 (+.029, p = .04) | .809 |
| photos (1 image) | 53 vs 48 / 105 / 416 | **.665** | .548 (−.117, CI −.144 to −.090) | .632 (−.033, p = .009) | .717 |
| nav (8 frames, finetuned base) | 200 vs 192 | **.806** | .769 (−.037, CI −.061 to −.012) | — | .792 |

At an equal token budget compression wins everywhere; the raw model needs about twice the tokens to match it
(documents: a little better at 2×, photos: still behind). Multi-page documents gain most from compression
(.682 vs .338 at equal tokens), text in photos next (.788 vs .546); scenes (RealWorldQA) and blurry photos are where
low resolution hurts least. Files: `expQ/base_qwen35_4b/eval/mm_docqa_base_px{125952,251904,57344,114688}_*.json`,
`expQ/ft_nav_base_4b/eval/mm_docqa_base_k8_px28672_nav_ft_val_v2.json` (`PX`/`MINPX` of `eval_mm_4b.sh`).

## How the evaluation is done

- **Eval sets:** `eval_data/mm_docqa_1500_1m` (SlideVQA, InfographicVQA, MP-DocVQA from the VisRAG test sets, 500
  questions each, gold pages + 3 distractors, 1M px) and `eval_data/mm_naturalqa_1500` (VizWiz, RealWorldQA, TextVQA,
  500 each). The earlier 500-question sets are exact subsets (same ids, same pages, same distractors). Nav:
  `data/nav_ft_val_v2`, 928 questions.
- **Judge:** Qwen3-8B with the rubric prompt of `scripts/judge_prompts.py` (JSON verdict + a short reason, kept as
  `judge_raw`). Chosen by the judge study of 2026-10-20 (`expQ/judge_study`, `scripts/judge_study.py`,
  `scripts/judge_agreement.py`): 3,730 pooled (question, gold, answer) items from strong and weak models, 7 judge
  conditions (Qwen3-8B and Qwen3-32B × Yes/No prompt / rubric / thinking, Gemma-3-4B). Findings:
  every model ranking is the same under every judge; agreement between conditions 92–97%; on objective anchors the
  rubric prompt errs least (8B: says "no" to 0.3% of exact matches and "yes" to 0.9% of answers whose numbers are all
  wrong; the earlier prompt: 1.3% and 5.0%; thinking makes judges stricter, not more accurate). A 114-item
  disagreement sheet for human labelling is `expQ/judge_study/human_sheet.md`.
- **Comparisons are paired on the same questions** with a bootstrap interval and McNemar's test
  (`scripts/paired_judge.py`; `scripts/show_mm_results.py --ci` against the baseline).
- **Sample size:** 500 questions are enough for compressed-vs-uncompressed gaps, not for arm-vs-arm differences of
  .03: on the first 500 doc questions the balanced stage-2 arm looked −.032 (slides "significant", p = .007); on 1,500
  the two arms are equal (.578 vs .579, p = 1.0). Treat any effect under ~.04 measured on 500 questions as unproven.
- **Answers are GPU-state sensitive:** the same model and code reproduce 60/60 short document answers but only
  45/60 long VizWiz answers on another GPU/library state. Compare arms within one eval campaign.

## How each component was settled

Earlier rounds, 500 questions, Yes/No judge; each row changed one thing, paired on the same questions
(standard error about .02). Effects of .03 and under are suggestive only (see "Sample size").

**Finetuning**

| change | result | verdict |
|---|---|---|
| targets = the uncompressed backbone's own answers ("teacher labels", `scripts/relabel_mm_teacher.py`) instead of dataset gold, and learning rate 2e-5 instead of 1e-4 (which overfits and drifts verbose) | .318 → .488 | essential; both changes were made together |
| 4× more `<MEM>` per page (rate 2), more finetune data (65K), query-dependent compression | .466 / .474 / .468 vs .488 | no gain (on the old stage-2) |
| 2× more `<MEM>` per page (rate 4) at finetune, from the 1M stage-2; 1,500 Q, rubric judge (job 174693) | docs .612 vs .578 (+.034, CI +.015 to +.055); photos .670 vs .665 | small doc gain for 1.8× the tokens |
| KL to the teacher's token probabilities during finetuning | .59 vs .64 on 100 Q | fails: the KL stalls, the `<MEM>`s lack the information |
| training the vision tower (LoRA on the ViT) | .480 vs .510 | no gain; keep the merger-only LoRA |
| adding 8K photo-QA rows to the doc finetune | doc +.014 | costs documents nothing |

**Stage-2 pretraining (what the compressor learns to keep)**

| change | result | verdict |
|---|---|---|
| Wikipedia screenshots → dense PDF pages | .488 → .510; multi-page .45 → .54 | helps multi-page only |
| raise the pixel budget at finetune only (451K → 1M) | .508 vs .510 | null: the compressor never saw large images |
| raise the pixel budget in stage-2 (1M pages + photos, "v3") | .508 → .558 (+.050, 2.5 SE) | **the first clear doc gain** |
| add *describe* (teacher descriptions) and *find* (which image holds this text) tasks ("v4") | photos +.032 (2.0 SE), docs ±0 | helps photos (suggestive) |
| add KL to the uncompressed teacher on the chat tasks ("v4+KL") | docs +.032 (1.8 SE), photos ±0 | suggestive only |
| balanced diet (documents 28% / photos 37% / charts 23% / nav frames 11%) instead of document-heavy (69% / 31%) | **1,500 Q:** docs .579 vs .578, photos .669 vs .669 | **no difference** (the 500-Q "−.03 on docs" was sampling) |
| mix pages and nav frames in one stage-2 | docs −.03, nav −.055 | domains compete at a fixed budget |
| nav-specific stage-2 versus starting from the document model | nav .810 vs .730 | the stage-2 diet must match the domain |

The uncompressed model itself gains from more pixels: doc .692 → .776 from 451K to 1M pixels, infographics .55 → .745.

**Text compression is preserved**: LongBench F1 .201 (v3) vs .197 (previous stage-2) vs .231 (text-only model).

**Reinforcement-learning track (separate session):** filtering finetune data on teacher correctness and RAFT
self-training gave .498 to .518 against .510, all within noise. GRPO was deferred until these stage-2 results.

## Open questions

1. **Documents.** Expanding the routed page reaches the uncompressed score at 4, 8 and 10 candidate pages
   ([architecture_review.md](architecture_review.md)). Untested: hard negatives (same-document or retrieved pages), more
   than 10 pages, a decoder finetuned on mixed raw + `<MEM>` prompts.
2. **GRPO with an evidence-then-answer format**, for the multi-page questions where the answer is in the `<MEM>`s but
   is not used (the prompt alone gives +.025).
3. **External nav benchmarks** (OpenEQA, VSI-Bench): not started.
4. **Human check of the judge:** label `expQ/judge_study/human_sheet.md`, then
   `python3 scripts/judge_agreement.py --pool expQ/judge_study/pool.json expQ/judge_study/q*.json expQ/judge_study/gemma*.json --human <sheet>`.

## Where things are

- Code: collators `pisco/mm_collator.py` (stage-2 tasks image-AE, continuation, text, *describe*, *find*, KL teacher
  inputs), prompts shared with the teacher `pisco/mm_tasks.py`, tests `test_mm_collators.py` (run through SLURM).
- Data builders: `scripts/build_mm_pretrain_pdfa.py`, `scripts/build_mm_pretrain.py`, `scripts/build_mm_pretrain_natural.py`
  (photos and charts from the_cauldron), `scripts/teacher_describe_vllm.py` + `scripts/attach_teacher_text.py` (teacher
  descriptions), `scripts/build_mm_natural_qa.py` (natural-image QA eval and finetune sets; `--per_source`),
  `scripts/build_mm_eval.py` (doc eval sets; `--extend` keeps an older set as a subset).
- Fair ceiling (uncompressed backbone finetuned on the same rows): `scripts/train_base_mm.py`.
- Evals: `eval_mm_4b.sh` (env `TOPK`, `PX`, `ADAPTER`, `MERGE`), results with `python scripts/show_mm_results.py --ci`,
  arm comparisons with `scripts/paired_judge.py`.
- Session record: `EXPERIMENTS.md` (ledger) and `HANDOFF.md`.
