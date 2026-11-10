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

**Current best (2026-11-10): the spread `<MEM>` layout at full budget**, `expQ/ft_mm_4b_184778/model` (s42) and
`184780` (s43): documents .610 / .615, photos .677 / .685 (judge, 1,500 Q each), equal to 174908 / 176242 within noise
with one stage-2 pass instead of three ([architecture_review.md](architecture_review.md), archq3). Slots are placed at
the centre of the segment or image tile they compress; the gain over the tail layout is +.08 documents / +.02 photos at
a reduced budget, and chunk length becomes free at inference (see "Long chunks" below).

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

## How much can we compress? (inference curve, 2026-11-02)

Best model (rate 8), slots removed at inference, judge accuracy on 1,500 questions, difference to the full set:

| kept slots per chunk (effective rate) | documents, first slots | documents, every k-th | photos, first slots | photos, every k-th |
|---|---|---|---|---|
| 75% (10.7×) | −.023 | | −.024 | |
| 50% (16×) | −.128 | −.130 | −.087 | −.043 |
| 25% (32×) | −.283 | −.254 | −.233 | −.159 |

Without retraining, 16× costs .13 on documents and .04 to .09 on photos; 32× is destructive. Spreading the kept slots
over the chunk beats keeping the first ones on photos (coverage), not on documents. Trained at rate 16 the text model
tied rate 8 on RAG, so a trained rate-16 image chain would recover part of this; rate 4 (twice the vectors) gave
nothing and made dense pages worse. Files: `expQ/ft_mm_4b_174908/eval/*keep0*` (`eval_mm_4b.sh` env `KEEPF`, `KEEPM`).

**Trained rate curve (spread layout, reduced-budget chains, label `rate`, 2026-11-06/10).** Judge accuracy, 1,500 Q,
two seeds (s42 / s43), paired against the spread rate-8 chain at equal budget:

| rate (slots per 1M-px page) | documents, judge accuracy ↑ | vs rate 8 (95% CI) | photos, judge accuracy ↑ | vs rate 8 (95% CI) | stage-1 slot ownership |
|---|---|---|---|---|---|
| 8 (56) | .599 / .618 | | .671 / .671 | | .977 |
| 16 (28) | .535 / .544 | −.065 (−.087 to −.043) / −.074 (−.095 to −.053) | .643 / .642 | −.027 (−.047 to −.007) / −.029 (−.048 to −.007) | .978 |
| 32 (14) | .427 / .420 | −.172 (−.197 to −.147) / −.198 (−.224 to −.174) | .587 / .585 | −.084 (−.107 to −.061) / −.086 (−.108 to −.063) | .547 (none) |

Rate 16 equals the tail layout at rate 8 (documents +.028 / +.001, photos n.s.) at half the tokens; it is the floor
for photos only (infographics fall to .40). Rate 32 is not usable for QA and its slots no longer own a segment. The
image rate curve is not flat, unlike text RAG. Rate round closed.

**Slot dropping by layout (reduced chains, judge, A = kept − full, 2026-11-10):** spread: documents keep 50% first
−.121, every 2nd −.129, keep 25% −.261; photos −.061 / −.043 / −.153. Tail: documents −.043 / −.103 / −.083; photos
−.019 / −.045 / −.058. Tail slots are redundant (all read the whole chunk), so dropping them is cheap; spread slots
each own a part, so dropping them loses content. Neither gives a free coarse level; a trained hierarchy (nested-budget
training) would have to beat the tail's cheap degradation to be worth it.

## Long chunks (probe 2026-11-10, no training)

Same content, grouped differently, judge accuracy on 1,500 HotpotQA questions with 10 passages (~1,270 tokens joined),
models trained on 256-token chunks only, chunk cap 4,096 so nothing is cut:

| chunks per question | spread rate 8, judge accuracy ↑ | vs ten chunks (95% CI) | tail rate 8, judge accuracy ↑ | vs ten chunks (95% CI) |
|---|---|---|---|---|
| ten (one passage each, ~130 tokens) | .579 | | .568 | |
| three (~400 tokens) | .578 | −.001 (−.023 to +.021) | .495 | −.073 (−.098 to −.048) |
| one (~1,270 tokens) | **.617** | **+.037 (+.014 to +.062)** | .399 | −.169 (−.198 to −.141) |

With the spread layout each slot stays ~4 tokens from its text whatever the chunk length, so a long chunk costs
nothing and even helps multi-passage questions (the compressor sees all passages together). Under the tail layout
every slot sits at the chunk end and the start of a long chunk is lost. Confirmed on MRQA (one ~900-token passage):
spread one chunk .694 = split .695; tail .639 vs .689; production .631 vs .697. Files: `expQ/ft_mm_4b_18381{4}/eval/
*cml4096_text_hotpotqa_1500*`, `eval_data/text_hotpotqa_1500_g3`, `_g10` (`eval_mm_4b.sh` env `CML`, `CHUNK`).

## External benchmarks: the sets of "Position IDs Matter" (2026-11-07/10)

Built by `scripts/build_mm_eval_mcq.py` (ScienceQA-IMG 500 + MMBench-EN dev 500, one page, lettered options) and
`scripts/build_text_eval_mrqa.py` (MRQA out-of-domain, 6 × 250, one passage). Judge accuracy, paired:

| set | spread rate 8 (183814) | tail rate 8 (183538) | production 174908 | uncompressed | spread − tail (95% CI) | spread − uncompressed (95% CI) |
|---|---|---|---|---|---|---|
| image multiple choice (1,000 Q) | .864 | .838 | .839 | .895 | +.026 (+.010 to +.042) | −.031 (−.051 to −.013) |
| MRQA, passages split into 256-token chunks (1,500 Q) | .695 | .689 | .697 | .807 | +.006 n.s. | −.111 (−.135 to −.087) |
| MRQA, one passage = one chunk | .694 | .639 | .631 | .807 | +.055 (+.031 to +.077) | −.113 |

MRQA per source (spread, split): BioASQ .88, RelationExtraction .79, TextbookQA .82, DROP .60, DuoRC .55, RACE .53 vs
raw .92 / .90 / .87 / .78 / .68 / .68: the gap is reasoning and narrative, not retrieval. A first MRQA run with the
finetune collator's 256-token truncation read .63 and is void. Not comparable model for model with the paper
(Llama-3.2 / LLaVA-1.5 there), same questions and metrics.

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

## External navigation test: OpenEQA (2026-10-30)

OpenEQA EM-EQA, HM3D half: 557 human-written questions (object recognition, attributes, state, localization, spatial,
functional, world knowledge) over 63 house tours of 69 to 400 frames (163 on average); memory only, no walking.
Regime label for every number: **"OpenEQA EM-EQA HM3D 557 Q, memory only"**. Metric: LLM-Match (0-100, mean of
(mark−1)/4 over a 1-5 mark, OpenEQA's own prompt), judge Qwen3-14B, every arm judged in the same pass; k = frames per
question, evenly spaced over the tour (OpenEQA's rule). The uncompressed arms show the raw frames to the same Qwen3.5-4B.
Paired against the uncompressed backbone with 16 frames (sign test on the non-zero per-question differences).

| arm (regime above) | tokens per question | LLM-Match ↑ | vs uncompressed 16 frames (95% CI) | p |
|---|---|---|---|---|
| uncompressed Qwen3.5-4B, 16 frames | 7,200 | 50.0 | — | — |
| uncompressed Qwen3.5-4B, 32 frames | 14,400 | **57.0** | +7.0 (+4.3, +10.0) | <.001 |
| uncompressed, nav-finetuned (`ft_nav_base_4b`), 16 frames | 7,200 | 54.3 | +4.3 (+1.0, +7.6) | .036 |
| generic PISCO 174908 (57 `<MEM>`/frame), 16 frames | 920 | 48.2 | −1.8 (−5.4, +1.9) | .94 |
| generic PISCO 174908, 64 frames | 3,650 | 49.6 | −0.4 (−4.4, +3.5) | .84 |
| generic PISCO 174908, 32 frames / whole tour (69-400 frames, `--compress_slice 32`) | 1,830 / 4,000-23,000 | 50.9 / **51.9** | +1.0 (−2.9, +4.7) / +2.0 (−1.7, +5.8) | .30 / .03 (sign test) |
| nav PISCO `ft_nav_4b` (29 `<MEM>`/frame), 16 / 32 / 64 frames | 470 / 930 / 1,860 | 35.0 / 34.7 / 37.3 | −15.0 / −15.2 / −12.6 | <.001 |
| silver-nav `campaign_eqa`, Qwen3-VL-2B: blind / 15 even / 15 retrieved frames | 0 / ~12k / ~12k | 29.5 / 46.1 / 54.0 | | |
| silver-nav `campaign_eqa`, Qwen3-VL-4B: 15 even / 15 retrieved frames | ~12k | 54.2 / **59.3** | +4.2 / +9.4 | .002 / <.001 |

What it says:
- **The generic compressor keeps what 16 raw frames give, at an eighth of the tokens** (48.2 vs 50.0, not
  distinguishable on 557 questions). The compressed decoder reads 64 frames in half the tokens of 16 raw frames.
- **A longer compressed memory is not exploited.** Raw frames gain +7 from 16 to 32; `<MEM>`s gain +1.4 from 16 to
  64 and +3.7 from 16 to the whole tour (every frame, 51.9: the only compressed arm above raw 16 frames, and still 5
  points under raw 32). The decoder was finetuned on 4-8 documents; reading 64-400 compressed frames is out of its
  training range.
- **Retrieval still wins on this task**: 15 frames picked by an embedding retriever reach 54.0 (2B) / 59.3 (4B
  answerer). The compressed memory is cheaper, not better, until the decoder learns to use many `<MEM>`s.
- **The nav-specific finetune is a dead end for open questions**: trained on templated yes/no, order and side
  questions, `ft_nav_4b` answers in 1.1 words and loses 29 points on object recognition (22 vs 51). It is the right
  model for the templated memory QA (.806) and the wrong one for human questions.
- Per category (uncompressed 16 → generic PISCO 16): attributes 64 → 62, object state 66 → 64, world knowledge 55 → 56,
  functional 36 → 40, object recognition 51 → 43, localization 41 → 38, spatial 35 → 34. Fine object identity is
  what compression loses, as on documents (infographics).

Files: `eval_data/openeqa_hm3d_k{16,32,64,all}` (`scripts/build_mm_eval_openeqa.py`), results
`expQ/*/eval/mm_docqa_*_openeqa_hm3d_k*.json`, the campaign's cells re-judged under `expQ/openeqa/campaign/`,
table `scripts/openeqa_results.py table … --ref <uncompressed k16 json>`; judge prompt `openeqa` in
`scripts/judge_prompts.py` (`eval_mm_4b.sh` env `JPROMPT=openeqa JMODEL=Qwen/Qwen3-14B`). Frames:
`/beegfs/scratch/user/hdejean/open-eqa/data/frames/hm3d-v0` (HF mirror `the-Lin/openeqa`).

## Open questions

1. **Documents.** Expanding the routed page reaches the uncompressed score at 4, 8 and 10 candidate pages
   ([architecture_review.md](architecture_review.md)). Untested: hard negatives (same-document or retrieved pages), more
   than 10 pages, a decoder finetuned on mixed raw + `<MEM>` prompts.
2. **GRPO with an evidence-then-answer format**, for the multi-page questions where the answer is in the `<MEM>`s but
   is not used (the prompt alone gives +.025).
3. **External nav benchmarks.** OpenEQA done (section above): next is a decoder that uses many `<MEM>`s (finetune
   with 32-64 compressed frames per question, or route-then-expand over the tour), then VSI-Bench and place retrieval
   over the silver-nav memrep bank.
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
