# Training data, pretraining tasks and labels: review (2026-10-30)

What this covers: the *data side* of PISCO training only. What the rows are, what the targets are, what the stage-2
tasks ask for, and what to change. Architecture, optimiser and RL are in [architecture_review.md](architecture_review.md),
[training_review.md](training_review.md) and [mem_tokens_review.md](mem_tokens_review.md). Every claim about our
pipeline was checked in the code, the configs, the run directories (`expQ/*/.hydra/overrides.yaml`) or the ledger
(`EXPERIMENTS.md`). Paper numbers carry the model and setting. **(ours)** marks our reading or extrapolation.
**†** marks a number copied from [training_review.md](training_review.md) and not re-read in the paper here.

## 0. What we train on today (verified)

| stage | rows | targets | tasks and mix | notes found while checking |
|---|---|---|---|---|
| **1, text** (`pt_lora`, `qwen35_4b_pt.sh`) | `EleutherAI/SmolLM2-135M-10B`, 1M or 5M samples, 256-token chunks | the text itself | autoencoding 50% / continuation 50% (`ae_ratio` .5), rate 8 | one domain (web text), one length |
| **2, multimodal** (`pt_lora_mm`, run `pt_mm_v4kd_4b`) | 160k: PDFA pages 80k + Wikipedia screenshots 30k (both 1M px) + photos 50k (localized narratives, 451k px); each row = image + its text + a teacher description | page text (AE, continuation), teacher description (`desc`), "Document j" (`find`) | `p_text` .5, image-AE .2, image-continuation .05, `desc` .15, `find` .1; KL to the uncompressed decoder on the chat rows only (`kd_weight` .5); lr 3e-5 | **AE target is cut at 768 tokens** (`ae_target_max_tokens`); continuation reads the first 0–32 tokens and the next 512; `find` snippets come from the **first 300 words** of the page. All three favour the **top of the page** (ours, section 2.3) |
| **finetune** (`finetune_mm`, 174908 / 176242) | 28k: 20k document questions = Docmatix 10k + the_cauldron **train** splits DocVQA 7k / InfographicVQA 2k / ChartQA 1k (1–2 gold pages + 3 pages of *other* documents of the same source, 1M px) + 8k photo questions (VQAv2 3k, TextVQA 2.5k, A-OKVQA 2.5k) | the uncompressed Qwen3.5-4B's greedy answer on the raw pages (`relabel_mm_teacher.py`, 48 tokens max, brief-answer prompt); dataset gold kept in `answers` | CE on the answer, 4 documents per question | not OpenDocVQA (gated, ledger 2026-10-08). Teacher is right on **68%** of rows (Qwen3-8B judge: Docmatix .60, DocVQA .80, InfoVQA .74, ChartQA .53). Teacher median 3 words, gold 6 |
| text RAG rows | none in the mm finetune. The released text set `maxoul/pisco_finetuning_data` (= PISCO's 453k `dmrau/multi_qa`, Mistral-7B labels) was used only in the null 65k arm (10k rows) | | | the "46% miss the gold" figure is a containment artefact (ledger 2026-09-10); two LLM judges put the wrong-target share at ~45% with a different composition: MS MARCO / WikiQA long gold are mostly *right* |
| nav | `nav_ft_train_v2`: templated questions (presence / order / position / final object) answered from the HM3D semantic sensor; `nav_pretrain(_desc)`: templated frame descriptions | synthetic gold | | the templated set collapsed open questions (OpenEQA −15); no train-scene overlap with OpenEQA (checked: 0 of 63 OpenEQA scenes in `nav_hm3d_train`; 24 of 63 are in `nav_hm3d_val`, eval-eval only) |

Eval sets and their source splits (for the deduplication list in 3.4): InfoVQA and MP-DocVQA = **validation** splits
(VisRAG 2410.10594, stated in the paper), SlideVQA = original test split; VizWiz val, RealWorldQA test, TextVQA
validation; HotpotQA distractor **validation**; OpenEQA HM3D (HM3D val scenes).

## 1. Labels

### 1.1 What the literature says about targets for compression models

| paper | setting | finding on labels | what it says for us |
|---|---|---|---|
| PISCO 2501.16075 | Mistral-7B / Llama-3.1-8B / Solar-10.7B, 16×, 453k questions (`dmrau/multi_qa`, Wikipedia-KILT, k=5 docs) | all training uses the *teacher's* answer ("sequence-level distillation"); best teachers Mistral-7B or Solar-10B; removing distillation "leads to much lower results" (figure only, no table value); .68 vs .71 uncompressed | matches our own largest win: teacher labels (+ lr) .318 → .488 |
| COCOM 2407.09252 | Mistral-7B-Instruct, 4–128×, 493,473 examples from 10 QA sets | trains on **dataset gold** (NQ, MS MARCO, HotpotQA, SQuAD, …); NQ EM .554 / .539 / .511 at 4 / 16 / 128× vs .597 uncompressed; no pretraining .519 → .490 at 128× | gold labels work when they are short span answers of the same style as the model's; our Docmatix gold is sentence-style, which drifted answers verbose |
| xRAG 2405.13792 | Mistral-7B, one token per document | instruction tuning + **self-distillation KL** from the model reading the raw text; without it 44.58 → 41.49† | the token-level version of "teacher labels"; ours stalls at finetune, helps in stage 2 |
| DEX-Comp 2609.05152 | Mistral-7B, 16× | distil only on rows the teacher got **right** (LLM judge), RL on the rest; 67.75 → 72.02 vs 69.49 full context† | our SFT-only version of the filter was null (pd .510 vs .510 at 500 Q); the RL half is training_review proposal 1 |
| AdaMem 2609.22100 | OSCAR-style, 16–64× | keeps a row only if the gold is in a passage **and** in the teacher answer | a stricter filter than ours; untested here |
| No Mean Feat 2510.20797 | Qwen3-0.6B–8B, 4–128× | **no labels at all**: the whole loss is KL to the teacher's next-token distribution on the answer steps; Qwen3-8B 4× 71.66 vs teacher 74.33† | the extreme end: labels are only a way to pick the positions where the KL is applied |
| GKD 2306.13649 | T5-XL → T5-small/base/large | **on-policy** distillation (teacher scores the *student's* samples) beats distillation on fixed teacher outputs: relative gains 2.1× (XSum), 1.7× (WMT), 1.9× (GSM8K) over baseline KD | a fixed teacher string is the weakest form of distillation; our RAFT (student samples judged correct → SFT) was the cheap on-policy proxy and was null |
| RAFT 2403.10131 | LLaMA-2-7B, 1 gold + 4 distractors | labels = chain of thought that **quotes the evidence verbatim** (`##begin_quote##`), then the answer: without / with CoT PubMed 68.30 / 73.30, HotpotQA 25.62 / 35.28, HuggingFace 59.07 / 74.00, Torch Hub 86.56 / 84.95, TF Hub 83.21 / 86.86; best share of rows *with* the gold document 40–100% | the strongest published evidence for evidence-then-answer labels, on an uncompressed reader |
| ACL 2026 long 682 (Memorization-Utilization Gap)† | Qwen2.5-7B, 16× | CoT-SFT on 50k distilled traces 66.73 < RL 72.66, and below plain SFT on three sets; RL answers re-copy evidence by themselves (20-gram expansion 31.5% vs 2.8%) | a warning: trained long-form labels can hurt a *compressed* reader; the gain came from RL, not from the label format |

Reading: teacher labels are settled (every compression paper uses them or goes further to KL). The open questions are
the **format** of the teacher label and **what to do with rows where the teacher is wrong**.

### 1.2 Rows where the teacher is wrong (32% of our finetune rows)

| option | evidence | our status | verdict |
|---|---|---|---|
| drop them (teacher-correct only) | DEX-Comp stage 1 | `mmrl_ft_20k_pd`: .510 vs .510 (500 Q) | null as SFT; keep the split for RL |
| replace the label by the short gold | our `fix` arm | .518 vs .510 (500 Q, noise) | null |
| best-of-k teacher, judge-verified (sample k answers from the uncompressed teacher, keep the first the judge accepts) | GKD-style use of the teacher's distribution; our RAFT probe found 36% of teacher-failed rows recoverable at G=8 with the *compressed* model | untested on the **teacher** side | cheap relabel job; raises the S+ share from 68% toward ~80% without changing style **(ours)** |
| keep them as they are | PISCO (no filtering, 453k rows) | our default | fine as long as the goal is "imitate the uncompressed model", wrong if the goal is "beat it" |
| turn them into "not in the documents" rows | RAFT's P% (rows without the gold document), 2503.02199 | untested | only for rows where the answer is truly absent; a judge cannot tell "absent" from "missed" |

### 1.3 Label format

| format | evidence | our measurement | cost in tokens at inference |
|---|---|---|---|
| short answer (today) | PISCO, COCOM, DEX-Comp | .578 docs | 1–3 tokens: the format the speed numbers assume ([latency.md](latency.md): the gain shrinks once the answer exceeds ~1/30 of the saved prompt tokens) |
| **evidence, then answer** | RAFT +5 to +15 points (above); ACL-682: RL converges to it by itself | **+.025 (CI +.005 to +.045) with the prompt alone, no training** (`--answer_style expand`) | 20–60 tokens: halves the throughput gain on 4-page prompts **(ours, from latency.md)** |
| chain of thought | ACL-682 CoT-SFT below plain SFT on three sets; RAFT's CoT is evidence-quoting, not free reasoning | not tested | 100+ tokens |
| answer with document pointer ("Document 2: 1998") | DocR1 2508.07313 (evidence-page reward, F1 over pages) | our `find` task trains the pointer separately | 3–5 tokens; also trains routing |

Recommendation: a **two-mode** label set, the brief mode kept as it is (speed) and an evidence mode under a distinct
system prompt (the eval's `EXPAND_SYSTEM_PROMPT`), so the deployed mode is a prompt choice. Recipe in 5.2.

### 1.4 Relabel the text set and the nav set with our own decoder?

- **Text** (`maxoul/pisco_finetuning_data`, Mistral-7B labels). Reasons for: style match with Qwen3.5-4B (the mm
  relabel + lr fixed a .17 gap, mostly the lr); the labels are a 7B Mistral's, our decoder is better on long gold
  (MS MARCO / WikiQA sentences). Reasons against: text rows barely helped (LongBench 25.0 vs 24.3 with 10k rows), and
  the text model's accuracy is within 1% of the paper. Cost: 453k prompts in vLLM ≈ 10–15 A100-hours **(ours)**.
  Verdict: only when text RAG rows join the mm finetune (shortlist #1 needs ~2k text routing rows: relabel those).
- **Nav**: the problem is not the label style but the **question set** (templated yes/no, order, side). For open
  questions the decoder needs open labels: teacher-written answers to teacher-written questions on the frames (OpenEQA
  categories: object, attribute, state, location, spatial, function). The uncompressed nav-finetuned base already shows
  the ceiling for templated QA (.792). Verdict: a relabel is the wrong tool; a new QA set is (3.1, "nav open QA").

### 1.5 Multi-reference and judge-filtered labels

- The eval already scores against all `answers` aliases; training uses one string. Several judged-correct teacher
  samples per row (1.2, best-of-k) give multi-reference labels for free: train on the shortest, or on all correct
  variants as augmentation. No compression paper tests this; GKD's on-policy result is the nearest evidence. **(ours)**
- Judge choice matters more than the filter: containment mislabelled 27.8k text rows against two LLM judges
  (ledger 2026-09-10); the Qwen3-8B rubric judge errs ≤1% on objective anchors (multimodal_results.md). Any filter
  must use the rubric judge, never containment or F1.

## 2. Tasks for stage 2 (and stage 1)

### 2.1 Auxiliary objectives: evidence for soft and visual-token compression

| objective | who, result | our status | reading |
|---|---|---|---|
| reconstruction (AE) share | ARC-Encoder 2510.20535: 0 / 20 / 50 / 100% → 39.8 / 41.6 / 41.5 / 37.5†; LCLM ~17%†; CLaRa drops it† | stage 1 50%, stage 2 20% image-AE + 25% text-AE (half of `p_text`) | stage 2 is in the good range; stage 1 is on the high side **(ours)** |
| description / caption of the compressed input | ours: `desc` + `find` photos +.032 (2.0 SE), docs 0; CLaRa paraphrase targets†; VLM pretraining uses captions by default | 15% of stage 2 | keep; 200-word descriptions cannot carry a dense page (below) |
| "find" / pointer (which document holds this snippet) | ours: the untrained router picks the gold page 98% of the time on 4 pages, 87% on 10 retrieved, 77–81% on 11–20 pages; DocR1: evidence-page F1 reward, MultiChartQA 54.26 → 61.54 and TATDoc 60.03 → 64.51 vs answer-only GRPO (Qwen2.5-VL-7B, format also changed) | 10% of stage 2, ≤4 random pages of the batch, text snippets, never a question | the one task with a measured gap to close (routing on long documents and the bridge paragraph in HotpotQA: both gold in top 2 only .42) |
| contrastive / retrieval loss over the `<MEM>`s | Latent Memory 2606.10572: InfoNCE + KL + reconstruction, −1.1 to −1.7 EM without reconstruction†; 2602.21202: memory tokens 54.4 vs clustering 56.7 vs full 60.0 nDCG (Qwen2.5-VL-3B, 64 vectors)† | none | the `find` task is the generative form of this; a contrastive head is only needed if `<MEM>`s must also serve as an index |
| question generation from the `<MEM>`s | no soft-compression paper found that uses it as a training task (search 2026-10-30); LLM-generated queries are standard for *retriever* training (VisRAG 239k GPT-4o queries on 40k pages; ColPali 37% synthetic) | none | unproven as an objective; proven as a **data generator** (3.1) |
| counterfactual pairs (gold page swapped, answer must change / "not in the documents") | 2503.02199: SFT with conflicting text reduces blind trust in text†; RAFT: 0–60% of rows without the gold document helps; HII-DPO 2602.10425 counterfactual images as preference pairs†; CounterCredit 2609.22910 +6.3–9.4 over outcome-only GRPO† | none; every finetune row has its gold page | aimed exactly at P2 ("the answer is in the vectors and the decoder does not use it") |
| multi-hop across pages | DocSkill 2610.36759 +5.5† (cross-page evidence paths, 60k QA); MMLongBench-Doc: 33.2% of questions cross-page; DocHop-QA 2508.15851 11,379 LLM-generated multi-document Q (PubMed) | none; Docmatix rows have ≤2 gold pages, questions are single-page | needed for the text multi-hop loss (HotpotQA route2 .751 only when both paragraphs are found) |
| layout / region tasks ("what is in the top-left", read the text in box b) | DocOwl 1.5 2403.12895 (uncompressed 7B VLM): no structure learning → parsing → parsing + multi-grained localisation: DocVQA 72.8 → 77.7 → 81.6, ChartQA 65.0 → 67.5 → 70.5, TabFact 72.9 → 76.5 → 80.4; word / phrase / line / block granularity | none | the published lever for dense pages; the compressed analogue is "transcribe region r of Document j" **(ours)** |
| OCR-style transcription of dense pages | DeepSeek-OCR 2510.18234: 97% OCR precision when text tokens ≤ 10× vision tokens, ~60% at 20× | our image-AE *is* OCR from `<MEM>`s, but its target stops at 768 tokens and infographics rebuild only 58% of their words | at 123 `<MEM>` per page a 1,000-token page is 8×: inside DeepSeek-OCR's 97% band, so the loss is not a hard capacity ceiling **(ours)** |
| routing / selection over 10–20 pages | ours: pick@1 .77–.81 at 11–20 pages (−.115 route1); DocR1 EviBench trains on up to full documents | `find` ≤4 pages | a data problem first: the model never saw more than 4 documents in training |

### 2.2 Mixing ratios and curricula

| lever | evidence | us |
|---|---|---|
| task ratios | ARC: reconstruction a minority; MM1 2403.09611: "a careful mix of image-caption, interleaved and text-only data is crucial" (abstract; ratios in the paper's tables not re-read); our balanced-diet arm: **no difference** at 1,500 Q | the *domain* mix is not a lever here; the *task* mix was changed once (v4) and helped photos only |
| rate schedule (pretrain high, finetune low) | ARC: pretrain 8× then finetune 4× best at both†; Activation Beacon samples the rate per step† | one rate everywhere; a rate-8 model read at 16 loses .070 | cheap to add to the next stage-2 run: sample rate ∈ {8, 16} per row **(ours)**; buys flexibility, not accuracy |
| length / page-count curriculum | REFRAG 2509.01092: 9-stage curriculum "essential" for reconstruction†; LLaVolta heavy → light† | stage 1 = 256-token chunks only; stage 2 = 1 image (AE) or 2–4 (`find`); finetune = 4 pages; eval goes to 20 | a curriculum 1 → 4 → 12 → 20 documents inside stage 2 / finetune is untested **(ours)** |
| hard-example mining | DEX-Comp and Lychee: RL on failed rows†; our RAFT on recovered rows: null | SFT on recovered rows does nothing; the rows the model fails are the RL set |
| epochs vs more data | Muennighoff 2305.16264: up to 4 epochs of repeated data ≈ unique data (pretraining scale); our 65k same-kind rows: null | a 2-epoch control of the 28k finetune is free and untested |

### 2.3 Three task flaws found in the collator (cheap to fix, stage-2 cost to test)

1. **Page-top bias (ours).** Image-AE targets are the first 768 tokens of the page text (`ae_target_max_tokens`),
   continuation reads tokens 0–32 and predicts the next 512, `find` snippets are drawn from the first 300 words, and
   Wikipedia screenshots are cropped to the top (`crop_aspect` 1.6). A PDFA page holds up to 6,000 characters
   (~1,400 tokens): the bottom third of a dense page is **never a target**. InfographicVQA answers sit anywhere.
   Fix: AE of a random window with its position stated ("Transcribe Document 0 from 'xxx'"), snippets from the whole
   text, `decoder_max_length` 2048 for AE rows. Cheap check first: the transcription probe split by the answer's
   vertical position on the page (`probe_mem_recall.py` output + OCR line boxes from PDFA metadata).
2. **`find` distractors are the batch's other rows**: random pages of mixed sources (a PDF page against a photo).
   The eval's hard cases are same-document and retrieved pages. Fix: build `find` rows offline with same-document
   pages (PDFA gives 2 pages per PDF; Docmatix whole PDFs) and ColQwen2 neighbours.
3. **Descriptions are capped at 200 words** (DESC prompt), fine for photos, a summary for a dense page. Keep for
   photos; for pages use the page text (AE) and region tasks instead.

## 3. Data

### 3.1 Sources we could add

Sizes and licences as read on 2026-10-30 (dataset cards / papers); **verify** marks what the card did not state.

| source | what | size | licence | targets | dedup risk vs our evals |
|---|---|---|---|---|---|
| **PixMo-Docs** (allenai/pixmo-docs) | code-rendered charts 118k / documents 72k / tables 47.5k / diagrams 17.6k, QA written **from the code** (exact ground truth) | 255k images | ODC-BY (Claude / OpenAI terms noted on the card) | P1 (dense pages, charts), stage 2 + finetune | none (synthetic) |
| **CoSyn-400K** 2502.14846 | same idea, 9 categories (charts, documents, math, tables, diagrams, vector graphics, music, circuits, chemistry); 7B model: ChartQA 81.4 → 86.3, DocVQA 87.9 → 90.0, InfoVQA 68.2 → 70.5, TableVQA 53.6 → 65.8 vs the same model without it | 400k images, 2.7M rows | **verify** on the HF card | P1 | none |
| **DUDE** 2305.08455 | multi-page documents, 5.72 pages avg, extractive 42% / abstractive 38% / list 7% / **unanswerable 13%** | train 3,010 docs / 23,728 Q | CC BY 4.0 | P2, routing, "not in the documents" rows | low (DUDE is not in our evals) |
| **MP-DocVQA** train (lmms-lab-encoder/MP-DocVQA) | DocVQA questions with their 1–20-page document and `answer_page_idx` (gold page label) | ~36k Q | DocVQA terms (UCSF IDL), **verify** | routing on whole documents | **high**: the MP-DocVQA paper says pages appear in more than one split (same documents); our eval is its val split; hash every page |
| **Docmatix** (already used, 10k of 9.5M QA) | whole PDFs (PDFA) with GPT-written QA, multi-page | 1.3M docs | MIT (secondary source) | routing on 8–20 same-document pages: free | none |
| **VisRAG in-domain train** (openbmb/VisRAG-Ret-Train-In-domain-data) | (query, page) pairs from the **train** splits: ArxivQA 25,856, PlotQA 56,192, InfoVQA 17,664, MP-DocVQA 10,624, SlideVQA 8,192, ChartQA 4,224 | 122,752 | not stated | routing rows (no answer column) | pages of the same corpora as our evals but train splits; hash anyway |
| **ColPali train set** (vidore/colpali_train_set) | 63% academic (DocVQA, InfoVQA, TAT-DQA, ArxivQA) + 37% synthetic queries on web PDFs (Sonnet), **with answers** | 127k | **verify** (card truncated) | routing + QA | same as above |
| **SlideVQA** train decks | 2,619 decks × 20 slides, 14.5k Q | NTT **evaluation licence** (internal evaluation only, no redistribution, no modification; HF gate) | routing on 20 slides | our SlideVQA eval is the test split | the licence allows internal evaluation of *their* method; training a released model on it is doubtful: do not |
| **ArxivQA** 2403.00231 | figure QA from arXiv papers | 100k | arXiv paper licences vary; **verify** | charts / figures | ArxivQA is a VisRAG source we do not eval on |
| **TAT-DQA** 2207.11871 | financial reports, numeric reasoning | 16k Q | **verify** | tables, numbers (ChartQA .53 teacher accuracy says numbers are weak) | none |
| **EviBench** (DocR1) | 4.8k multi-page Q with evidence pages, Gemini-2.5-Flash labels, from DUDE / MP-DocVQA / SlideVQA / TATDoc | 4.8k | not stated; includes SlideVQA | routing | contains MP-DocVQA and SlideVQA: hash |
| **Leopard-Instruct** 2410.01744 | 925k multi-image text-rich rows (documents 208k, tables + charts 401k, web 55k), GPT-4o synthesis | 925k | not stated; redistributes SlideVQA, DUDE, MP-DocVQA, TAT-DQA, InfographicVQA | multi-page QA | **high**: contains InfographicVQA and MP-DocVQA rows of unknown split |
| **Doc-750K** (Docopilot 2507.14675) | 758k Q over Sci-Hub / arXiv / OpenReview papers, cross-page tasks, 31.6% real QA | 758k | not stated; **Sci-Hub source** | multi-page | legal risk: do not |
| **MMLongBench-Doc** 2407.01523 | 1,062 Q, 130 PDFs, 49.4 pages avg, 33.2% cross-page, 22.8% unanswerable | eval only | benchmark | an **eval** for 20–100-page routing, not training data | — |
| **HotpotQA** train | 90k Q, 2 gold of 10 paragraphs | 90k | CC BY-SA 4.0 | text routing with **2 gold** | our eval is the validation split; also in `dmrau/multi_qa` (88,869 rows) |
| **MuSiQue** | 2–4-hop, 20 paragraphs | 20k | CC BY 4.0 | text multi-hop routing | none |
| **NQ / MS MARCO / TriviaQA** | in `dmrau/multi_qa` already | 453k | CC BY-SA 3.0 / non-commercial (**verify**) / Apache 2.0 | text rehearsal | bergen evals only |
| **teacher-generated QA on our own pages** | the uncompressed Qwen3.5-4B writes questions + answers for PDFA / Wikipedia / photo rows (VisRAG and ColPali did this for retrieval; Docmatix for QA) | any | our data | P2 on any page type; nav open QA (1.4) | none; filter questions answerable from the page only (judge) |

### 3.2 Hard negatives

| method | evidence | status |
|---|---|---|
| same-document pages | our eval: on the 379 same-document questions route-then-expand .818 vs uncompressed .805, routing 99 → 95% | free for Docmatix / DUDE / MP-DocVQA / PDFA (2 pages per PDF); the finetune set uses *other* documents only |
| retriever top-k (ColQwen2) | pick@1 drops to .87 (10 retrieved pages); Nemotron ColEmbed V2 2602.03992 caps negatives at .95 × the positive score to avoid false negatives† | `build_mm_eval_retrieved.py` ranks a VisRAG corpus; the same code can rank the finetune pages (a few GPU-hours per 20k-page corpus) |
| BM25 / OCR-text neighbours | standard for text (dmrau/multi_qa used retrieval over KILT) | PDFA and Docmatix carry text: cheap on CPU |
| false-negative guard | ColEmbed cap; ours: never place two pages of the same question's gold set as negatives of each other | needed for Docmatix (QA per document, pages shared) |

### 3.3 How much data, how many epochs

- Ours: 20k → 65k same-kind rows null; teacher labels + lr the only large win; the finetune changes the decoder, not
  the compressor (training_review 1.6). More rows of the same distribution are not the lever; **different rows**
  (more pages, counterfactuals, routing targets) are the open question.
- Literature: PISCO 453k (pretraining useless at that size, useful at small sizes); DEX-Comp distillation 1.5 GPU-h†;
  ACL-682 60k RL questions†; Muennighoff: repeating data up to 4 times costs nothing at pretraining scale. A **2-epoch
  control** of the 28k set is a free rider on any finetune arm and settles "are we under-trained on these rows".

### 3.4 Deduplication: overlaps to check before any new data build

The 1,500 + 1,500 + 1,500 eval questions and their pages must stay out of training. Checks, cheapest first
(image hashes on the stored JPEGs are exact within a set because every page is resized once by `prepare_page`; use a
perceptual hash across sets with different pixel budgets):

| training source | eval set | why | check |
|---|---|---|---|
| the_cauldron `docvqa` (train) | MP-DocVQA val (500 Q, `mm_docqa_1500_1m`) and the whole-document set `mm_docqa_longdoc_1m` (1–20 pages) | MP-DocVQA was built by adding the neighbouring pages of the **same UCSF documents**; its paper says pages appear in more than one split | pHash of the 7k gold pages vs every eval page; UCSF document ids where available |
| MP-DocVQA train (if added) | same | same, stronger | mandatory page hash + document id |
| the_cauldron `infographic_vqa` (train) | InfoVQA val | disjoint splits by construction | pHash spot check |
| the_cauldron `textvqa` (train); stage-2 localized narratives (Open Images) | TextVQA validation (Open Images) | same image pool; stage 2 has no QA but the eval image could be an AE target | pHash of 50k stage-2 photos + 2.5k TextVQA train rows vs 500 eval images |
| VQAv2 / A-OKVQA (COCO) | VizWiz, RealWorldQA | different image pools | none needed |
| Docmatix / PDFA | all document evals | Common-Crawl PDFs vs UCSF / infographics / SlideShare | pHash spot check only |
| `dmrau/multi_qa` (HotpotQA train 88,869 rows) if text rows are added | `text_hotpotqa_1500` (validation) | disjoint splits; same Wikipedia | question-string dedup |
| Wiki-SS screenshots | HotpotQA paragraphs | both Wikipedia; stage 2 AE only | none needed |
| VisRAG / ColPali train sets, EviBench, Leopard (if added) | InfoVQA, MP-DocVQA, SlideVQA | mixed splits, redistributed sets | mandatory page hash; drop any row whose page matches |
| `nav_hm3d_train` | OpenEQA HM3D (63 scenes), `nav_ft_val_v2` | **checked: 0 shared scenes** with train; 24 of 63 OpenEQA scenes are in `nav_hm3d_val` | never train on `nav_hm3d_val` renders |
| new teacher-generated QA on PDFA / wiki pages | none | our own pages | none |

Also keep the question strings of all eval sets in one blocklist (`eval_data/*/query`) and run every new build against
it.

## 4. Candidates: problem, evidence, cost, cheap check first

| candidate | targets | evidence and effect size | cost | cheap check first |
|---|---|---|---|---|
| **A. routing / selection rows on real questions**: 8–20 documents, same-document + ColQwen2 negatives, 1–2 gold, text + pages + frames, the eval's `ROUTE_QUESTION` prompt | P2 at 11–20 pages (−.115), HotpotQA bridge (.42 both found), many-`<MEM>` use (OpenEQA 64 frames unused) | DocR1 evidence reward +4 to +7 on multi-chart / TATDoc; ours: pick@1 is the whole route-then-expand loss | data build (ColQwen2 over ~30k pages, a few GPU-h) + finetune 2 seeds (~11 h each) | already measured at inference (pick@1 per page count); a 2-seed finetune is the test |
| **B. evidence-then-answer teacher labels**, two modes | P2 "use" failures (54 of 93) | RAFT +5 to +15 (uncompressed); our prompt-only +.025; ACL-682 warns CoT-SFT can lose to plain SFT | relabel job (28k prompts, ~2 GPU-h in vLLM) + finetune 2 seeds | the teacher's own accuracy under the evidence prompt on 1,500 eval Q (one base eval, `STYLE=expand`): if the teacher does not gain, the labels will not either |
| **C. counterfactual rows** (gold page swapped for a near neighbour; label = teacher's answer on that prompt, usually "not in the documents") | P2, hallucination | RAFT P% 40–100%; 2503.02199†; CounterCredit† | relabel of ~6k rows + finetune 2 seeds | **inference-only, zero training**: swap the gold page on the 1,500 doc Q and count unchanged answers = how often the decoder ignores the page |
| **D. dense synthetic pages + region / window tasks in stage 2** (PixMo-Docs, CoSyn; AE of a random window with its position; "read region r") | P1 | CoSyn +2.3 InfoVQA / +4.9 ChartQA / +12 TableVQA (uncompressed 7B); DocOwl 1.5 +8.8 DocVQA from structure tasks (uncompressed); our dense PDFA pages +.022, 1M px +.050 | stage 2 (22 h) + finetune 2 seeds | transcription probe by vertical position (2.3, flaw 1) and by page density: if the bottom third rebuilds far worse than the top, the bias is real |
| **E. best-of-k judge-verified teacher labels** (+ aliases) | P3 noise, S− rows | GKD direction; our pd / fix arms null | relabel job (k=4 samples × 6.4k S− rows) + finetune 2 seeds | share of S− rows recovered by the teacher at k=4 (one vLLM job): below 30% it is not worth a finetune |
| F. teacher-generated QA on PDFA / wiki / photo pages (+ nav open QA) | P2 on new page types; OpenEQA | VisRAG / ColPali / Docmatix as generators; no compression paper | generation job + judge filter + finetune | 2k generated questions judged for answerability from the page; teacher accuracy on them |
| G. multi-hop rows (DUDE, MuSiQue, DocSkill-style paths) | text multi-hop, cross-page | DocSkill +5.5†, HotpotQA route2 needs both gold | data + finetune | HotpotQA both-gold@2 after arm A (A contains 2-gold text rows) |
| H. rate ∈ {8, 16} sampled in stage 2; page-count curriculum | flexibility, long inputs | ARC†, Beacon†, REFRAG† | rides on the next stage-2 run | none needed |
| I. relabel the text set with Qwen3.5-4B | style match | ours: label style mattered for mm | 10–15 GPU-h | only with text rows in the mix |
| J. 2 epochs | P3 | Muennighoff | free rider | none |

## 5. Ranked shortlist (at most 5)

| # | candidate | expected effect (judge, 1,500 Q) | confidence | why this rank |
|---|---|---|---|---|
| 1 | **A. routing rows on real questions, 8–20 documents, hard negatives, mixed modalities** | route1 on 11–20-page documents .78 → ≥ .85; pick@1 on 10 retrieved .87 → ≥ .92; HotpotQA both-gold@2 .42 → ≥ .60; all-`<MEM>` 4-page docs ±.02 **(ours)** | medium that routing improves (a direct supervised target with a measured gap); low that all-`<MEM>` accuracy moves | the only candidate whose target is both measured and large; it serves documents, text and video routing at once (HANDOFF decision 1) |
| 2 | **B. evidence-then-answer labels, two modes** | +.02 to +.04 docs in evidence mode; brief mode unchanged **(ours)** | low-medium: RAFT and our prompt effect say yes, ACL-682 says trained long labels may not add to the prompt effect | the one format change with a prompt-only gain already banked; also the cold start the RL proposal needs |
| 3 | **C. counterfactual rows** | +.01 to +.03 docs; the main value is a measurable drop in "page ignored" answers **(ours)** | low-medium; the probe decides | zero-cost probe first; targets the dominant failure on multi-page documents |
| 4 | **D. dense synthetic pages + window / region tasks (stage 2)** | infographics +.02 to +.05 **(ours)** | low-medium: dense pages and 1M px were the only stage-2 data wins (+.072 together); synthetic text-rich data is the published lever | expensive (stage 2); goes with the next stage-2 round only, after flaw 1 of 2.3 is confirmed by the probe |
| 5 | **E. best-of-k judge-verified teacher labels** | 0 to +.02 **(ours)** | low: two filter arms were null | cheapest relabel; worth doing only as the label set for #2, not alone |

### 5.1 Recipe for #1: routing rows ("find v2") in the finetune

- **Sources and counts** (≈8k routing rows added to the 28k QA rows; one epoch; 173550 recipe, norm on):
  - 3k **Docmatix whole PDFs**: question + its gold page(s) + the document's other pages up to 20 (same-document
    negatives), padded with other-document pages when the PDF is short. Gold = `gold_positions`.
  - 2k **the_cauldron single pages with ColQwen2 negatives**: rank the 20k finetune pages per source with
    `build_mm_eval_retrieved.py`'s ranker (same corpus = same source), keep top-k negatives below .95 × the gold score,
    n ∈ {8, 12, 16, 20}.
  - 2k **text**: HotpotQA train (2 gold of 10) and MuSiQue train (2–4 gold of 20); labels relabelled by our decoder
    only where a QA row is also kept.
  - 1k **frames**: `nav_ft_train_v2` windows with gold frames from `add_nav_gold.py`, 8–16 frames.
  - Optional 500 **DUDE** train rows for "no page" (unanswerable 13%): answer "None".
- **Prompt**: the eval's routing prompt verbatim (`ROUTE_QUESTION`: *Which document contains the answer to this
  question: "{q}"? Answer with the document number.*), answer `Document j` or `Documents j and k` (2 gold), brief system
  prompt. Same `Document j:` + `<MEM>` background as every other row, so the finetune trains the inference-time router.
- **Config**: `decoder_max_length` 4096 (20 pages × 123 `<MEM>` + prompt), `topk_docs` 20 for these rows; the QA rows
  keep 4 documents. Stage-2 `find` is left as is (one variable per round).
- **Filters**: drop routing rows whose gold page hash matches any eval page; never use two gold pages of one question
  as each other's negatives; cap same-document negatives at the gold's ColQwen2 score (false-negative guard).
- **Control arm**: 28k QA rows + 8k **more QA rows** of the same sources (so "more steps" is controlled), same seeds.
  References: 174908 (s42) / 176242 (s43).
- **Judging**: pick@1 and route1 / route2 on `mm_docqa_longdoc_1m` (by page bucket), `mm_docqa_retr10_1m`,
  `text_hotpotqa_1500` (both-gold@2), `nav_ft_val_v2_gold`; all-`<MEM>` on `mm_docqa_1500_1m` and
  `mm_naturalqa_1500` must stay within .02 of the control; OpenEQA k64 as the many-`<MEM>` check. A gain counts when
  pick@1 rises at both seeds and all-`<MEM>` does not fall.
- **Build**: extend `build_mm_finetune.py` with a `--routing` mode (same `ft_features` schema, `gold_positions`
  already there) and a ColQwen2 ranking pass reusing `rank_source`; HotpotQA / MuSiQue through
  `build_text_eval_hotpot.py`'s row format. CPU builds through SLURM (`gpu-be`, `gres=gpu:0`); the ranker on one GPU.

### 5.2 Recipe for #2: evidence-then-answer labels, two modes

- **Teacher pass** (`relabel_mm_teacher.py` with a `--system` option): the uncompressed Qwen3.5-4B on the raw pages
  with the eval's `EXPAND_SYSTEM_PROMPT` (*First copy, word for word, the text of the documents that contains the
  answer, after 'Evidence:'. Then give the final answer, as briefly as possible, after 'Answer:'.*), greedy, 96 new
  tokens, on the 28k rows (photos included: the evidence is what the image shows). Keep both label columns:
  `mistral_label` (brief, existing) and `label_expand`.
- **Cheap check before training**: base eval with `STYLE=expand` on the 1,500 doc Q: the teacher's accuracy in
  evidence mode vs brief mode (.809). If it does not gain ≥ .01, stop; the labels would teach the format without the
  benefit.
- **Filter**: Qwen3-8B rubric judge on the part after the last `Answer:` (the eval's extractor) vs `answers`; keep
  rows whose evidence label is judged right **or** whose brief label is judged right (one of the two modes must be
  correct); drop the rest (~25% expected (ours)). Also drop evidence labels whose `Evidence:` span is not a substring
  of the page text (PDFA / Docmatix rows have text; for the_cauldron pages use the teacher's own transcription from
  `probe_mem_recall.py` ref mode) to keep the "copy, word for word" contract honest.
- **Training rows**: every kept question appears **twice**, once in brief mode (default system prompt, brief label)
  and once in evidence mode (`EXPAND_SYSTEM_PROMPT`, evidence label); the `system_prompt` column of the collator
  already supports this. Same 437-step budget → `data.samples` doubles, or 1 epoch over the doubled set (then the
  control must match the step count).
- **Control arm**: the same filtered rows, brief labels only, doubled (same steps), 2 seeds. This separates the format
  from the filter and from the step count.
- **Judging**: all-`<MEM>` docs / photos in **both** modes (two eval runs per model, `STYLE=brief|expand`); the
  transcription probe's "use failure" count (54 of 93 today); answer length and refusal rate in each mode; route1 must
  stay at .81. Success = evidence mode above the control's brief mode by > .03 at both seeds **and** brief mode within
  .02 of the control (the speed mode is not damaged).
- **Follow-up**: this label set is the cold start for the decoder-only RL of training_review 3.1 (reward on the
  `Answer:` part; the format is already learned).

## 6. Ruled out or already done (from `EXPERIMENTS.md`)

| item | result | status |
|---|---|---|
| teacher-written labels + lr 2e-5 | .318 → .488 | adopted |
| dense PDF pages in stage 2; 1M px in stage 2 | +.022; +.050 | adopted; 1M px at finetune only was null |
| `desc` + `find` tasks | photos +.032, docs 0 | adopted |
| KL to the uncompressed decoder in stage 2 (chat rows) | docs +.032, single run | adopted, unproven at 2 seeds |
| more finetune rows of the same kind (65k, Docmatix 30k + six cauldron sets + 10k text) | .474 vs .488 | null |
| balanced stage-2 diet (docs 28 / photos 37 / charts 23 / nav 11%) | .579 vs .578 at 1,500 Q | null |
| mixing nav frames and pages in one stage 2 | docs −.03, nav −.055 | negative |
| query-dependent compression at finetune (OSCAR) | .468 vs .488 | null |
| teacher-correct filtering (pd), gold-retargeted S− (fix), RAFT on self-recovered rows | .510 / .518 / .498 vs .510 (500 Q) | null (SFT form); RL form open |
| text RAG rows in the mm finetune (10k) | LongBench 25.0 vs 24.3 | barely helps |
| templated nav QA for open questions | OpenEQA −15 | ruled out for open questions |
| KL at finetune | stalls, .59 vs .64 (100 Q) | ruled out at this stage |
| rate 2 / rate 4 / ViT LoRA | no gain / +.034 s42, −.022 s43 / −.030 | not data levers |
| containment-based teacher filtering on the text set | measures gold length, not correctness | never use containment for a filter |
| OpenDocVQA corpus; SlideVQA decks | gated; NTT evaluation licence | not available |
| **running now** | archq3 stage-1 position layout (180486 / 180487), linear connector finetunes (180488 / 180489), precision grid (180560–180564) | wait for them before opening a data round |

## Appendix: papers and datasets used

| id | paper / dataset | setting | number used |
|---|---|---|---|
| 2501.16075 | PISCO | Mistral-7B / Llama-3.1-8B / Solar-10.7B, 16×, 453k Q from `dmrau/multi_qa`, k=5 | teacher labels throughout; .68 vs .71 (Mistral); pretraining helps only at small sample sizes |
| 2407.09252 | COCOM | Mistral-7B-Instruct, 4–128×, 493,473 gold-labelled QA rows | NQ EM .554 / .539 / .511 vs .597; no pretraining .519 → .490 (128×) |
| 2405.13792 | xRAG | Mistral-7B, 1 token per document | self-KL ablation 44.58 → 41.49† |
| 2609.05152 | DEX-Comp | Mistral-7B, 16× | distil on teacher-correct, RL on the rest; 67.75 → 72.02 vs 69.49† |
| 2609.22100 | AdaMem | OSCAR-style | keep rows with gold in passage and in teacher answer† |
| 2510.20797 | No Mean Feat | Qwen3-0.6B–8B | KL-only loss, no labels† |
| 2306.13649 | GKD | T5-XL → T5-small/base/large | on-policy vs fixed-output KD: 2.1× / 1.7× / 1.9× relative gains (XSum / WMT / GSM8K) |
| 2403.10131 | RAFT | LLaMA-2-7B, 1 gold + 4 distractors | CoT with quoted evidence: PubMed 68.30 → 73.30, HotpotQA 25.62 → 35.28, HF 59.07 → 74.00, Torch Hub 86.56 → 84.95, TF Hub 83.21 → 86.86; P 40–100% |
| ACL 2026 long 682 | Memorization-Utilization Gap | Qwen2.5-7B, 16× | CoT-SFT 66.73 < RL 72.66; RL re-copies evidence† |
| 2403.12895 | mPLUG-DocOwl 1.5 | 7B VLM, DocStruct4M (4M) | no structure / parsing / + localisation: DocVQA 72.8 / 77.7 / 81.6, ChartQA 65.0 / 67.5 / 70.5, TabFact 72.9 / 76.5 / 80.4 |
| 2502.14846 | CoSyn | 7B, 400k synthetic images, 2.7M rows | ChartQA 81.4 → 86.3, DocVQA 87.9 → 90.0, InfoVQA 68.2 → 70.5, TableVQA 53.6 → 65.8, avg 77.3 → 80.9 |
| allenai/pixmo-docs | PixMo-Docs | 255k code-rendered images | charts 118k / documents 72k / tables 47.5k / diagrams 17.6k; ODC-BY |
| 2510.18234 | DeepSeek-OCR | 3B-MoE decoder | 97% OCR precision below 10× text/vision tokens, ~60% at 20× |
| 2508.07313 | DocR1 | Qwen2.5-VL-7B, EviBench 4.8k (Gemini-2.5-Flash labels) | EviGRPO vs GRPO: MultiChartQA 54.26 → 61.54, TATDoc 60.03 → 64.51, DUDE 53.14 → 52.93; vs SFT: DUDE 52.72 → 54.39 |
| 2410.10594 | VisRAG | retriever training | in-domain train 122,752 pairs (train splits), synthetic 239,358 GPT-4o queries on 40k pages; eval MP-DocVQA and InfoVQA from the **validation** splits |
| vidore/colpali_train_set | ColPali train set | 127k query-page pairs with answers | 63% academic (DocVQA, InfoVQA, TAT-DQA, ArxivQA) + 37% synthetic (Sonnet) |
| 2305.08455 | DUDE | 5,019 documents, 41.5k Q | 5.72 pages avg; unanswerable 12.74%; train 3,010 docs / 23,728 Q; CC BY 4.0 |
| 2407.01523 | MMLongBench-Doc | 130 PDFs, 1,062 Q | 49.4 pages avg, 33.2% cross-page, 22.8% unanswerable; GPT-4o F1 42.7; benchmark only |
| 2507.14675 | Docopilot / Doc-750K | InternVL2-based 2B / 8B | 758k Q, 31.6% real, 4.8% GPT-4o; Sci-Hub / arXiv / OpenReview; MMLongBench-Doc 28.8 acc |
| 2410.01744 | Leopard-Instruct | 925k rows, GPT-4o synthesis | documents 208k, tables + charts 401k, web 55k; redistributes SlideVQA, DUDE, MP-DocVQA, TAT-DQA, InfographicVQA |
| 2508.15851 | DocHop-QA | 11,379 multi-document Q, PubMed, LLM-generated | CC BY 4.0 (article) |
| 2610.36759 | DocSkill | cross-page evidence paths, 60k QA | +5.5† |
| 2212.05935 | MP-DocVQA | 46,176 Q, 47,952 pages, 5,928 documents | built from DocVQA documents; pages may appear in more than one split |
| 2602.03992 | Nemotron ColEmbed V2 | retriever hard negatives | cap at .95 × positive† |
| 2503.02199 / 2602.10425 / 2609.22910 | Blind Faith in Text / HII-DPO / CounterCredit | utilisation gap in VLMs | SFT with conflicting text reduces the bias†; counterfactual images as preference pairs†; +6.3–9.4 over outcome RL† |
| 2510.20535 / 2606.09659 / 2511.18659 / 2401.03462 / 2509.01092 | ARC-Encoder / LCLM / CLaRa / Activation Beacon / REFRAG | objectives and curricula | reconstruction share 0 / 20 / 50 / 100% → 39.8 / 41.6 / 41.5 / 37.5†; ~17% reconstruction†; no token reconstruction†; sampled rates†; curriculum "essential"† |
| 2606.10572 / 2602.21202 | Latent Memory / multi-vector index compression | contrastive objectives | −1.1 to −1.7 EM without reconstruction†; memory tokens 54.4 vs clustering 56.7 vs full 60.0† |
| 2305.16264 | Scaling Data-Constrained LMs (Muennighoff) | pretraining | up to 4 epochs of repeated data ≈ unique data |
| 2403.09611 | MM1 | VLM pretraining | "a careful mix of image-caption, interleaved and text-only data is crucial" (abstract only) |
| SlideVQA LICENSE (github nttmdlab-nlp/SlideVQA) | NTT software evaluation licence | annotations + scripts | internal evaluation only, no redistribution or modification |
