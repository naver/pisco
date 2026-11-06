# Improving the training pipeline: literature and design review (2026-10-30)

What this covers: how we *train* PISCO (stages, objectives, data, distillation, RL, optimisation), and only that. The
architecture is settled ([architecture_review.md](architecture_review.md)), the connector is a linear map
([connector_probe.md](connector_probe.md)), the memory tokens have their own review
([mem_tokens_review.md](mem_tokens_review.md)). About 60 papers, 2023 to October 2026, arXiv ids opened. Every paper
number carries the model size and the compression rate. **(ours)** marks our reading or extrapolation, not a measured
result. **†** marks a number copied from the literature sweep that I did not re-read in the paper itself.

## The three problems this review is about

| problem | what we measured | where it lives |
|---|---|---|
| **P1 dense-page capacity** | infographics keep 56% (.404 vs .726); the transcription probe loses the answer *at compression* on 25 of 42 infographic failures | compressor, stage 2 |
| **P2 the decoder does not use the vectors** | multi-page documents: the answer is in the `<MEM>`s on 27 of 33 failures; "copy the evidence, then answer" as a prompt gives +.025 (CI +.005 to +.045); pass@8 .698 vs greedy .510 | decoder, finetune |
| **P3 noise and undertrained modules** | seed 42 vs 43 differ by .029 on documents; in bf16 80% of connector entries and 84% of LoRA-A entries never change in a finetune; the connector is a stage-2 product (moves < 1%) | optimiser, every stage |

## 0. Our pipeline today (verified in `pisco/configs/*.yaml` and the launchers)

| stage | trains | objective | data | lr, batch, steps | precision |
|---|---|---|---|---|---|
| 1 text pretrain (`pt_lora`, `qwen35_4b_pt.sh`) | compressor LoRA r64, decoder LoRA r64, connector, `<MEM>`/`<AE>` rows | autoencoding ⊕ continuation, `ae_ratio` .5, chunk 256, rate 8 | SmolLM corpus, 5M or 1M samples | 5e-5, eff. batch 192 (4 GPUs), 1 epoch | bf16 weights |
| 2 multimodal pretrain (`pt_lora_mm`, `pt_mm_4b.sh`) | same + merger LoRA (ViT frozen) | image AE .5, text rehearsal .3, image continuation .2, plus `desc` and `find` (v4); KL to the uncompressed decoder on the chat rows only, `kd_weight` .5 | 160k rows (dense PDF pages, wiki screenshots, photos, charts), 1M px | 3e-5 (yaml default 5e-5), eff. batch 64, 1 epoch | bf16 |
| finetune (`finetune_mm`, `ft_mm_4b.sh`) | same adapters; connector norm gain (bf16: frozen at init) | CE on teacher-written answers, 4 pages per question | 28k rows (20k OpenDocVQA pages + 8k photos) | 2e-5, eff. batch 64, 437 steps | bf16 |

Facts the rest of the document rests on: LoRA is attached with `PreTrainedModel.add_adapter`, which calls PEFT's
`inject_adapter_in_model`; that path has **no fp32 upcast** (checked in peft 0.19.1 / transformers 5.5.0), unlike
`get_peft_model`, whose `autocast_adapter_dtype=True` default has upcast fp16/bf16 adapters to fp32 since PEFT 0.12.
So every trainable tensor is stored in bf16 unless `frozen_base_dtype=bfloat16 torch_dtype=float32` ("option 2") is
set, which promotes them (`PISCO.__init__`). `connector_lr` exists (`PISCOTrainer.create_optimizer`), tested in jobs
180488/180489.

## 1. The landscape, 2024 to October 2026

### 1.1 Objectives

| objective | who uses it, with what result | what it says for us |
|---|---|---|
| **Autoencoding + continuation pretraining** | ICAE 2307.06945 (Llama-7B, 4×: pretrained vs not = 60.4% win / 9.5% loss); ARC-Encoder 2510.20535 (3B encoder → frozen Llama-3.1-8B, 4–8×: pretraining worth +16 at 20k steps, +19 at 80k over finetune-only; reconstruction share 0 / 20 / 50 / 100% → 39.8 / 41.6 / 41.5 / 37.5); COCOM 2407.09252 (Mistral-7B: no pretraining NQ .519 → .490, worst at 128×); OSCAR 2504.07109 (1B compressor: no pretraining −4 to −14 points per set). The exception: PISCO 2501.16075 (Mistral-7B 16×, both sides LoRA, 453k teacher-labelled questions) found "pretraining has little benefit". | Pretraining pays when the compressor is small or separate, or the decoder is near-frozen. Reconstruction must be a **minority** of the mix: ACL-682 (below) uses 15%, LCLM 2606.09659 ~17%, CLaRa 2511.18659 drops token reconstruction for QA and paraphrase targets and still beats PISCO (38.96 vs 37.83, 16×, Mistral-7B). Our stage 2 is 50% image AE: on the high side **(ours)**. |
| **Sequence-level distillation (teacher-written labels)** | PISCO (gold labels hurt; teacher labels fix it); OSCAR; CLaRa; AdaMem 2609.22100 keeps only rows whose gold is in a passage *and* in the teacher answer; DEX-Comp 2609.05152 distils on teacher-**correct** rows only. | We already do this (+.17 with lr 2e-5, our largest win). Filtering by teacher correctness was null for us at 500 Q (pd .510 vs .510). |
| **Token-level KL from the decoder reading the raw input** | xRAG 2405.13792 (Mistral-7B, 1 token/doc: without self-KL 44.58 → 41.49); KV-Distill 2503.10337 (Llama-3-8B, 20% KV: mixed KL 86.0 vs AE+LM 79.1, forward-only 83.4, reverse-only 82.7); No Mean Feat 2510.20797 (Qwen3-0.6B–8B, 4–128×: the **whole** loss is KL on answer steps, no CE; Qwen3-8B 4× pooling 71.66 vs teacher 74.33); Boa 2610.33006 (Mamba encoder → frozen Qwen3-1.7B, 4–16×: KL only, τ = 2; F1 81.3 / 80.5 / 80.2 vs 83.7); Latent Memory 2606.10572 (KL along the teacher's own answer); LatentPress 2609.01507 (CE + forward KL, λ 1). The negative: ARC-Encoder (frozen decoder, trainable encoder): "did not help, large overhead". | KL helps most when the **decoder side** is what learns. Ours stalled at finetune ("the `<MEM>`s lack the information") and gave +.03 (single run) in stage 2 on the chat rows only. Variants we have not tried: KL on all stage-2 tasks; KL restricted to answer tokens where the teacher is right and confident (Token-Budget Distillation 2608.28138, LLaVA-Video at 10% tokens: cumulative 58.3 → 58.6 → 58.8 → 58.9, i.e. small); on-policy self-distillation with no labels (CAFD 2609.39953, Qwen2.5-Omni-7B: +1.44 points, 26.9% of the gap recovered, 120/125 conditions). |
| **Hidden-state / attention distillation** | VoCo-LLaMA 2406.12275 (attention from raw vision tokens → VoCo tokens); DVTR-KD 2610.17853 (reconstruct the teacher's dense visual tokens and final states; no numbers in the abstract); EPIC 2510.00515 (token- and layer-consistency distillation, NeurIPS 2025, no numbers in the abstract); CMC 2609.25537 (contrastive to the teacher's answer-span states; removing KL or contrastive falls back to phase-1 level, 0.670 → 0.589); CLaRa (MSE between mean document states and mean memory states). | Plausible, not quantified against token KL anywhere. Needs a stage-2 retrain. Not shortlisted. |
| **Contrastive / ownership losses** | Latent Memory (InfoNCE over positives and hard negatives, λ .2); per-slot reconstruction in 2412.17483 | Covered in [mem_tokens_review.md](mem_tokens_review.md) proposal 2. |

### 1.2 Curricula

| curriculum | evidence | our situation |
|---|---|---|
| **rate sampled per step** | Activation Beacon 2401.03462 ({2…32}, Llama-2-7B ≈ full finetune on LongBench); KV-Distill (0.1–80%); No Mean Feat multi-ratio {4…128}: helps the high ratios a little (16×: 56.85 vs 56.22), costs at 4× (65.18 vs 66.97); TokenFLEX 2504.03154 (+1.6 / +1.0 / +0.4 at 64 / 144 / 256 tokens); MQT-LLaVA 2405.19315; M3 2405.17430 | A one-rate model does not transfer (rate 8 read at 16: −.070). Multi-rate buys flexibility, not accuracy at the main rate. |
| **pretrain at a higher rate, finetune at the target** | ARC-Encoder: pretrain 8× then finetune 4× is best at both 4× and 8× | Cheap to apply if a stage-1/2 rerun opens: stage 2 at 16, finetune at 8 **(ours)**. |
| **length / difficulty curriculum** | REFRAG 2509.01092 (RoBERTa-large → Llama-2-7B): 9 geometric stages over chunk counts, "essential" for reconstruction (Table 11: log-ppl 0.135 with vs 1.599 without†); LLaVolta 2406.20092 (heavy → light compression across stages); EPIC (progressive) | Our stage 1 trains on 256-token chunks only; the finetune then reads 4 pages. A page-count curriculum (1 → 4 → 10 pages in stage 2) is untested here **(ours)**. |
| **hard-example mining** | DEX-Comp: RL only on teacher-failed rows; LycheeMemory 2602.08382: drop questions the base solves zero-shot; CORE-RAG 2508.19282: keep summaries only if they help the frozen generator | Our RAFT variant (self-recovered S− rows as SFT targets) was null; the RL form is untested. |

### 1.3 Reinforcement learning

Three soft-compression papers put RL on top of a distilled checkpoint and all three cross or approach the
uncompressed model. None gets this from more SFT data.

| paper | setting | SFT → RL → uncompressed | recipe |
|---|---|---|---|
| **"Bridging the Memorization-Utilization Gap"** (ACL 2026 long 682, Ting et al.; no arXiv) | Qwen2.5-7B / 32B, same model encodes and decodes, 16× mean pooling | 7B in-domain **75.57 → 82.10** vs 83.07 (98.8%); out-of-domain 58.46 → 67.00 vs 71.94; 32B 81.50 → 86.20 vs 87.90. Reconstruction ROUGE-1 .98 at the same time: the gap is named from this contrast. CoT-SFT on 50.3k distilled traces: 66.73 vs RL 72.66 (SFT 64.88) | RL **trains the decoder only, memory embeddings precomputed** from the SFT checkpoint, 60k QA; DAPO (GRPO advantages, dynamic sampling, soft overlong penalty), reward = binary judge correctness (Qwen3-32B), G 16, **KL β 0**, lr 1e-6, 600-token answers, vLLM rollouts on prompt embeddings. RL answers re-copy evidence: 20-gram expansion rate 31.5% vs 2.8% for SFT |
| **DEX-Comp** 2609.05152 | Mistral-7B, separate LoRAs, 16×, 5 QA sets, top-5 to top-30 | LLM-judged avg, top-30: pure distillation **67.75 → 72.02** vs untuned full context 69.49; PISCO 64.79 | stage 1 KL distillation on teacher-correct rows (S+, judged by Gemini 3 Flash), lr 1e-4; stage 2 GRPO on teacher-failed rows (S−), reward = binary containment EM, G 16, β .02 to the stage-1 checkpoint, lr 1e-6, 2 epochs, 32-token answers. Cost on 8 GPUs: distillation 1.5 GPU-h, +RL 13, full-context RL 38 |
| **LycheeMemory** 2602.08382 | Qwen2.5-3B / 7B, 3 LoRAs, 4× | 3B sub-EM: SFT 59.38 → RL 66.80 → end-to-end RL 68.75† | GSPO, strict EM reward, G 12, β 1e-3, lr 3e-5 (LoRA) |

VLM-side RL trains the model to **select and expand** (route-then-expand) rather than to read compressed tokens
better: VisionThink 2507.13348 (GRPO decides when to request the high-resolution image; Qwen2.5-VL-7B ≈ 101% of
base†), DocR1 2508.07313 (evidence-page reward), MM-Doc-R1 2604.13579 (Qwen3-4B / 8B, +10.4% MMLongBench-Doc),
Doc-V* 2604.13731, LensVLM 2605.07019 (Qwen3.5-9B-Base, learned expand tools, full-text parity at 4.3×). The most
transferable reward idea for P2 is **CounterCredit** 2609.22910: each visual call is scored against two
counterfactuals (answer without the call; the call with random patches); 6.3–9.4 points over outcome-only GRPO,
fewer calls. For us the analogue is a reward that pays only when the answer *changes* with the gold page's `<MEM>`s
**(ours)**.

### 1.4 Data

| data lever | evidence | us |
|---|---|---|
| synthetic page QA at scale | Docmatix 2408.12637 (9.5M QA from 1.3M PDFs via OCR text; Florence-2 DocVQA 60.1 → 71.4 ANLS†); PixMo-Docs 2409.17146 (QA written from the *rendering code*: exact ground truth) | our 20k OpenDocVQA rows; 65k rows of the same kind were null |
| multi-page / multi-hop construction with evidence labels | DocSkill 2610.36759 (page-element graphs → cross-page evidence paths, 60k QA, +5.5 acc over the best baseline); DocHop-QA 2508.15851; MMLongBench-Doc 2407.01523 (33% cross-page, 23% unanswerable) | our `find` task has ≤ 4 random pages and no questions; routing is the weak point on 11–20 pages (−.115) |
| hard negatives from a page retriever | ColPali 2407.01449; Nemotron ColEmbed V2 2602.03992 (teacher top-k negatives, capped at .95 × positive score†) | we have ColQwen2 top-k pages for the eval sets only |
| counterfactual / conflicting evidence | 2503.02199 (10 VLMs trust text over image; SFT with text augmentation reduces it); HII-DPO 2602.10425 (counterfactual images → preference pairs); 2603.10360 (pruned-token latents as negatives) | untested; directly aimed at P2 |

### 1.5 Optimisation

| fact | source | us |
|---|---|---|
| LoRA's optimal lr is ~10× full finetuning's; LoRA tolerates large batches worse; RL needs very low capacity ("even rank 1"); apply LoRA to MLPs, not attention only | LoRA Without Regret (Thinking Machines, 2025) | our finetune lr 2e-5 is a *full-FT* number; it works only because bf16 freezes most entries (section 4) |
| A and B are not symmetric: a random frozen A does about as well as a trained A; B carries the adaptation | Asymmetry in LoRA 2402.16842; LoRA-FA 2308.03303 | bf16 has silently made us LoRA-FA (84% of A frozen) |
| different lrs for A and B (B higher) give +1–2% and up to 2× faster convergence | LoRA+ 2402.12354 | untested |
| compressors trained fully beat LoRA'd ones when the compressor is separate | OSCAR ("more effective than LoRA"); LCLM (frozen-decoder and LoRA variants "substantially underperform") | our compressor is LoRA r64 in bf16 (66–76% of A frozen in stage 2) |
| module-wise lrs are standard in staged recipes | LCLM: adapter 1e-3 → encoder 6e-5 → decoder 1e-5 → SFT 3e-5†; ARC-Encoder: encoder 1e-5 / MLP 5e-5† | `connector_lr` exists; LoRA-A/B and compressor/decoder share one lr |
| nobody reports EMA or weight averaging; 1 epoch is the norm, 2 for RL (DEX-Comp) and some SFT (COCOM) | the whole set | a 2-epoch control is cheap |
| seed noise is now acknowledged: ±0.8 points around the baseline across 5 data scales (LLaVA-UHD v4 2605.08985†); 5-seed reporting (LatentPress ±.024) | | our .029 spread; 2 seeds per arm is the minimum |

### 1.6 Stage design: what belongs where, and can a finetune change the compressor?

- **Stage 1 / 2 own the compressor and the connector.** Our probes: the finetune moves the connector 0.65%
  (cosine .9999), left new connector columns 50× too small (arm 1 null test), and cannot move LoRA-A at all in bf16.
  The literature agrees that what the vectors *contain* is learned in pretraining (ARC +16 to +19; OSCAR −4 to −14
  without it) and that the finetune adapts the decoder.
- **Can a finetune change the compressor?** In our regime, barely: compressor LoRA-B moves (63–68% of entries), LoRA-A
  and the connector do not. With fp32 masters or a higher compressor lr it can, but then the decoder must re-learn
  the read-out at the same time, on 28k rows. A compressor change therefore belongs in stage 2 (22 h) with the
  finetune as the test, which is what the archq3 chain does **(ours)**.
- **The finetune owns P2.** The three RL papers train the decoder *only* (ACL-682 precomputes the embeddings) and
  that alone closes most of the gap. This is also the cheapest stage to iterate (11 h, or far less for decoder-only
  RL with cached `<MEM>`s).
- **Stage 2's KD is in the right place** (it failed at finetune, helped in stage 2), but it covers the chat rows only.

## 2. Candidates against our three problems

| candidate | targets | evidence and effect size | cost here | cheap check first |
|---|---|---|---|---|
| **A. decoder-only RL on cached `<MEM>`s** (GRPO/DAPO, binary reward, on rows the model gets wrong) | P2 | +4.3 (DEX-Comp, 7B 16×), +6.5 in-domain / +8.5 OOD (ACL-682, 7B 16×), +7.4 (Lychee 3B 4×); all from the SFT checkpoint. Ours: prompt alone +.025; pass@8 .698 vs greedy .510 | 1–2 GPU-days; rollouts = decoder only (vLLM prompt embeddings exist) | 20-step GRPO shake-out: reward mean rises, LoRA-B entries moved (bf16 vs fp32); pass@k already measured |
| **B. precision and lr regime** (fp32 masters or stochastic rounding at a re-tuned lr; LoRA-B lr > LoRA-A; connector lr) | P3 | Zamirai 2010.06192: nearest rounding cancels small updates, SR/Kahan recover up to 7% abs; Ozkara 2502.20566 (AISTATS 2025, to 6.7B): bf16+SR beats (bf16, fp32) mixed precision; LoRA+ +1–2%. Ours: fp32 at 2e-5 is −.043 | 20-min shake-outs, then 4 finetunes | per-module fraction of entries moved and update/weight ratio at 20 steps for each lr |
| **C. stage-2 distillation upgrade** (KL on all tasks; KL only on answer tokens where the teacher is right and confident; `kd_weight` sweep; reconstruction share down to 20–30%) | P1, P2 | KL is the whole loss in No Mean Feat / Boa (≈ −2.7 to −3.5 from the teacher at 4×); gated KL +1.0 relative (TBD); reconstruction share: ARC 20–50% best, 100% −4. Ours: KD in stage 2 +.03 (single run) | one stage-2 pass (22 h) + finetune 2 seeds | inference-only: teacher per-token reliability on finetune rows (share of answer tokens with teacher top-1 = label); KL(teacher‖student) on S+ vs S− rows of the current model |
| **D. data for the use problem** (evidence-then-answer teacher labels; counterfactual rows where the gold page is swapped and the answer must change or become "not in the documents"; 8–16 pages with ColQwen2 hard negatives; cross-page QA) | P2, routing on long documents | SFT with conflicting evidence reduces "blind faith" (2503.02199); HopGen +5.5; CounterCredit's counterfactual credit +6–9 over outcome RL. Ours: more same-kind data was null; `find` ≤ 4 random pages | data build 1 day + finetune 2 seeds | inference-only: swap the gold page for a distractor on 1,500 Q and count unchanged answers (how often the decoder ignores the page) |
| **E. compressor capacity in stage 2** (fp32 LoRA or full-parameter compressor; rate schedule 16 → 8; density-adaptive rate is in mem_tokens_review) | P1 | OSCAR: full compressor > LoRA; LCLM: LoRA underperforms; ARC: pretrain high rate → finetune lower. Ours: rate 4 null over 2 seeds | stage 2 + finetune; full-FT compressor needs 8-bit Adam or 2 GPUs | compressor LoRA-B norms and AE loss per page vs page density on the current checkpoint |
| F. hidden-state / attention distillation | P1 | no number vs token KL anywhere | stage 2 | — (not now) |
| G. rate / page-count curriculum in stage 1 | P1, P2 | REFRAG "essential" for reconstruction; EPIC | stage 1 (16 h on 4 GPUs) | only with the next stage-1 round |
| H. EMA, 2 epochs, LoRA+ ratio | P3 | no compression paper reports EMA; LoRA+ +1–2% | free riders on any finetune | add to the shake-out grid of B |

## 3. Ranked shortlist

| # | candidate | expected effect on documents (1,500 Q, judge) | confidence | why this rank |
|---|---|---|---|---|
| 1 | **A. decoder-only RL on cached `<MEM>`s, evidence-then-answer** | +.03 to +.06 **(ours)** | medium-high that it beats the SFT checkpoint (three papers, same mechanism, same gap signature as our probe); medium that it clears .03 at both seeds | the only lever with published gains above the uncompressed model; cheapest stage; our headroom is measured (pass@8 .698) |
| 2 | **B. precision / lr regime** | sign unknown by itself; it is the **precondition** for A (RL updates of 1e-6 to 1e-5 vanish in bf16) and for any connector or compressor arm | high that it changes which weights move; low-medium that it alone gains .03 | every null test so far (arm 1, fp32 at 2e-5, connector in finetunes) traces back to it |
| 3 | **C. stage-2 distillation upgrade** | +.01 to +.03 **(ours)** | low-medium | our single-run +.03 and the literature agree on the direction; the gated/answer-only variants are untested anywhere at our scale |
| 4 | **D. counterfactual and multi-page data** | +.01 to +.03 on 4-page documents; the real target is 10–20 pages where we lose .05 to .115 **(ours)** | low-medium | same-distribution data was null; different-distribution data is the open question; it also serves the `find` upgrade |
| 5 | **E. compressor capacity in stage 2** | +.02 to +.04 on infographics only **(ours)** | low | the rate-4 null says slots are not the limit; full-FT or fp32 compressor is the untested half |

### 3.1 Proposal 1: decoder-only RL with an evidence-then-answer format

**Data.** Rows: the 28k finetune rows. Keep the existing S+/S− split (`data/mmrl_ft_20k_pd`, 13,598 teacher-correct /
6,402 teacher-failed, Qwen3-8B judge) and add the rows where the current model (174908) is wrong at greedy but right
at least once in 8 samples (the recoverable "use failures"; probe A found pass@8 − greedy = .19). RL set ≈ 8–10k
questions, held out from every eval set. `<MEM>`s of all pages are **precomputed once** with the frozen compressor
and connector (ACL-682 does this; DEX-Comp's cost table shows RL on cached embeddings at a third of full-context
RL).

**Format and cold start.** Arm A keeps the current prompt (no format change; ACL-682 ran RL from a plain-answer SFT
checkpoint and the answers grew to re-copy evidence on their own). Arm B uses the "copy the evidence, then answer"
prompt that gives +.025 at inference; no SFT cold start, since the model already follows it. A CoT-SFT cold start
is **not** recommended (ACL-682: CoT-SFT 66.73 < RL 72.66, and below plain SFT on three sets).

**Loss.** GRPO with DAPO details: G = 8 (16 if memory allows), token-mean loss, clip 0.2 / 0.28, soft overlong
penalty above 96 tokens, dynamic sampling (drop groups with all-equal rewards). Reward: binary containment EM against
the gold aliases where the gold is short (DocVQA, InfographicVQA, ChartQA), the Qwen3-8B rubric judge otherwise
(cached per unique answer string). No format reward. KL to the SFT checkpoint β = .02 (DEX-Comp) in arm A, 0 in a
variant if arm A is stable (ACL-682). Trainable: **decoder LoRA only** (compressor, connector and norm frozen;
"RL requires very low capacity"). Precision: **fp32 masters for the LoRA** (option 2) — at lr 1e-5 a bf16 update is
dropped on every entry with |w| > 0.003, i.e. the whole run would be a null test (section 4).

**Schedule.** lr 1e-5 (ACL-682 and DEX-Comp use 1e-6 for near-full finetuning; LoRA wants ~10×), 10 warmup steps,
batch 64 prompts × G rollouts, 150 to 300 updates, 1 epoch over the RL set; checkpoints every 50 updates. Rollouts
through vLLM with prompt embeddings (`scripts/bench_vllm.py` path), answers ≤ 96 tokens. Budget: about 1 GPU-day
per arm per seed **(ours)**.

**Control arms.** (1) The SFT checkpoint itself (174908 s42 / 176242 s43). (2) Same rows, same steps, SFT with CE on
the teacher label (rules out "more steps on these rows"). RAFT (best-of-8 SFT) was already null, so it need not rerun.

**Judging.** Paired judge accuracy on `mm_docqa_1500_1m` and `mm_naturalqa_1500` against the same-seed control; a
gain counts above .03 on documents at both seeds. Secondary: the transcription probe's "use failure" count (54 of 93
today), greedy vs pass@8 gap (should shrink), answer length and refusal rate (reward hacking), route-then-expand
with the RL decoder (must not regress .810). Shake-out before launch: 20 updates on 256 prompts: mean reward rises,
KL to the reference < .1, > 50% of decoder LoRA-B entries moved (fp32) vs the bf16 run of the same 20 steps.

### 3.2 Proposal 2: fix the precision and learning-rate regime

**What changes.** Trainable tensors in fp32 (`frozen_base_dtype=bfloat16 torch_dtype=float32`, already implemented;
memory cost is a few hundred MB for two r64 adapters and the connector) **or** bf16 weights with stochastic rounding
(torchao's optimisers expose `bf16_stochastic_round=True`, "for full BF16 training where we don't keep an FP32
copy"; optimi's Kahan-summation AdamW is the other drop-in). Per-module lrs: connector 5e-5 (running), LoRA-B at
2–4× LoRA-A (LoRA+), norm gain frozen or at a tenth of the lr (run D's spikes).

**Grid.** Phase 1, shake-outs (20 steps, ~20 min each, 1 GPU, 173550 recipe): bf16 2e-5 (reference) · fp32 at
{2e-6, 5e-6, 1e-5, 2e-5} · bf16+SR at 2e-5 · fp32 5e-6 with LoRA+ ratio 4. Record per module (connector, LoRA-A,
LoRA-B, compressor vs decoder, gain): share of entries changed, mean |Δw|/|w|, loss and grad norm at step 20
(`expQ/archq2/check_connector_move.sh` pattern). The fp32 lr whose realised update size matches bf16 2e-5 on
LoRA-B, and moves A and the connector by 0.5–2%, is the candidate; expect 5e-6 to 1e-5 **(ours)**.
Phase 2: the two best settings at seeds 42 and 43 (4 finetunes, ~11 h each) plus bf16+SR at one seed.

**Control.** bf16 2e-5 at the same seeds (174908 / 176242). **Judging.** As in 3.1; plus held-out eval loss at every
checkpoint (the fp32 runs rose from .49 to .62, the control fell to .43) and no grad-norm spike above 2.

**What success means.** Not a gain by itself. Success = a regime in which (a) the connector and LoRA-A move when
asked to, (b) training loss does not rise, (c) accuracy is within noise of bf16 2e-5. That regime is then the
default for proposal 1, the archq3 finetunes and every connector/compressor arm; a null test like arm 1 cannot
happen again.

## 4. The bf16 problem

**Mechanics (ours, from the numbers).** bf16 keeps 8 significant bits, so the spacing between neighbouring values
is 0.4–0.8% of |w|. A nearest-rounded update is dropped when |Δw| is below half that spacing, roughly
|Δw| < 0.002–0.004·|w|. Adam's step is ≈ lr per entry once the moments settle. So at lr 2e-5 an entry with
|w| > ~0.005–0.01 never moves; at 3e-5 (stage 2) the threshold is ~0.008–0.015; at 5e-5 (stage 1) ~0.013–0.025.
LoRA-A (Kaiming-uniform, scale 1/√2560 ≈ 0.02) and the connector's Linear init (≈ 0.02) sit above the finetune
threshold; LoRA-B starts at 0 and stalls once an entry reaches ~0.005. This reproduces the ledger: finetune 80%
connector / 83–85% LoRA-A / 32–37% LoRA-B never changed; stage 2 at a higher lr 70% / 66–76% / 21–26%. The regime
is a magnitude-gated sparse update: only small entries learn, and LoRA-A is frozen — which the literature says is
harmless for A (Asymmetry 2402.16842; LoRA-FA 2308.03303) and is why the finetune still works.

**What the literature and practice say.**

| source | claim |
|---|---|
| Revisiting BFloat16 Training 2010.06192 (Zamirai et al.) | nearest rounding of bf16 weight updates "often cancels small updates"; stochastic rounding or Kahan summation recover up to 7% absolute accuracy and land within −0.1 / +0.2% of fp32 on seven applications |
| Stochastic Rounding for LLM Training 2502.20566 (AISTATS 2025, to 6.7B) | bf16 Adam with SR on the update **outperforms (bf16, fp32) mixed precision** on validation perplexity, 1.54× throughput, ~30% less memory |
| Collage 2405.03637 (ICML 2024) | multi-component float removes the fp32 master copy at equal or better loss |
| Scaling Laws for Precision 2411.04330 | training in lower precision lowers the model's *effective parameter count* (465 runs, to 1.7B) — the same phenomenon seen from the loss side |
| torchao `optim` README; optimi | `bf16_stochastic_round=True` exists "for full BF16 training, where we don't keep an FP32 copy"; optimi enables Kahan summation by default on bf16 models |
| PEFT ≥ 0.12 docs | `get_peft_model` upcasts fp16/bf16 adapters to fp32 by default ("typically required for stable training"); the transformers `add_adapter` → `inject_adapter_in_model` path we use does not (verified in the installed source) |
| soft-prompt practice | prompt tuning and gists train the soft parameters at high lrs (xRAG bridge 6e-3; LCLM adapter 1e-3; ICAE 1e-4): soft parameters are expected to move far more than LoRA, which bf16 at 2e-5 forbids |

**Is "fp32 is worse at the same lr" expected? Yes (ours).** Three things change at once when the masters go fp32 at
lr 2e-5: (1) every entry moves, so the realised update per step is several times larger than in bf16 — an lr of
1e-4 had already overfit and drifted verbose in bf16; (2) LoRA-A unfreezes and re-learns the features LoRA-B was fitted
against (the asymmetry papers say moving A adds little signal); (3) the 21M-parameter connector, a stage-2 product,
starts moving on 28k rows. The lr was tuned under the gated regime and is simply too high once updates land. The
expected optimum is 3–5× lower (5e-6 to 1e-5), which also matches "LoRA lr ≈ 10× full-FT lr" with the RL papers'
full-FT 1e-6 at the low end. Run D's instability (norm gain trainable in fp32, grad spikes 5–7.5, docs −.167) is
not explained by this alone; a per-dimension gain on every `<MEM>` has a scale ambiguity with LoRA-B and the connector,
and Adam moves it by lr every step. Keep the gain frozen (it is a measured constant) or give it a tenth of the lr
until a sweep says otherwise.

**Consequence for RL.** DEX-Comp and ACL-682 train RL at lr 1e-6 (near-full finetuning); for LoRA that means
~1e-5. In bf16 an update of 1e-5 is dropped on every entry with |w| above ~0.003–0.005: the policy would barely move
and the run would read as "RL does nothing". Any RL arm must run with fp32 masters or stochastic rounding.

**Sweep design.** Section 3.2 above: a 20-step shake-out grid first (fraction of entries moved per module is the
readout), then 2 lrs × 2 seeds. Keep bf16 2e-5 as the control at both seeds; accept a setting that matches it within
noise while moving the connector and LoRA-A by 0.5–2%.

## 5. Ruled out or already done (from `EXPERIMENTS.md`)

| item | result | do not re-propose |
|---|---|---|
| teacher-written labels + lr 2e-5 (instead of gold + 1e-4) | .318 → .488, the largest win | adopted |
| dense PDF pages in stage 2 | +.022 (multi-page +.09) | adopted |
| 1M px in stage 2 | +.050 (2.5 SE) | adopted; 1M px at finetune only was null |
| `desc` + `find` tasks in stage 2 | photos +.032, documents 0 | adopted |
| KL to the uncompressed decoder in stage 2 (chat rows) | documents +.032 (1.8 SE, single run) | adopted; counts as unproven at 2 seeds |
| connector output RMSNorm | photos +.021 at both seeds, documents 0 | adopted |
| KL at finetune | .59 vs .64 on 100 Q; the KL stalls | ruled out at this stage |
| query-dependent compression (OSCAR) at finetune | .468 vs .488 | ruled out |
| more finetune data of the same kind (65k) | .474 vs .488 | ruled out |
| filtering finetune rows by teacher correctness; RAFT self-training | .510 / .518 / .498 vs .510 at 500 Q | null (SFT form); the RL form is proposal 1 |
| balanced stage-2 diet | .579 vs .578 on 1,500 Q | null |
| rate 4 (finetune) | documents +.034 at s42, −.022 at s43; photos +.017 for 1.8× tokens | not replicated |
| rate 2, training the ViT LoRA | .466 / .480 vs .488 / .510 | ruled out |
| fp32 master weights at lr 2e-5 | documents −.043; with a trainable gain −.167 and grad spikes | ruled out at that lr; the sweep is proposal 2 |
| multi-layer read-out trained in a finetune (arm 1) | null test: new connector columns 50× too small | must be trained in stage 1/2 |
| decoder-side `<MEM>` position stride (arm 2) | flat to −.026 at 2 seeds | ruled out |
| mixing nav frames and pages in one stage 2 | documents −.03, nav −.055 | domains compete at a fixed budget |
| templated nav finetune for open questions | OpenEQA −15 points | ruled out |
| **running now** | archq3 (spread position layout from stage 1, 180486/180487); linear connector + `connector_lr` 5e-5 (180488/180489) | wait for them |

## Appendix: papers used

| id | paper | setting | number used |
|---|---|---|---|
| ACL 2026 long 682 (no arXiv) | Bridging the Memorization-Utilization Gap (Ting et al.) | Qwen2.5-7B/32B, 16× mean pooling, decoder-only RL on cached embeddings | 7B SFT 75.57 → RL 82.10 vs 83.07; OOD 58.46 → 67.00 vs 71.94; 32B 81.50 → 86.20 vs 87.90; CoT-SFT 66.73 vs RL 72.66; DAPO, judge reward, G 16, β 0, lr 1e-6 |
| 2609.05152 | DEX-Comp | Mistral-7B, 16×, separate LoRAs | top-30 LLM-judged 72.02 vs 69.49 full context, PISCO 64.79; distillation-only 67.75; GRPO on S−, CEM reward, G 16, β .02, lr 1e-6; 1.5 / 13 / 38 GPU-h |
| 2602.08382 | LycheeMemory | Qwen2.5-3B/7B, 4× | SFT 59.38 → RL 66.80 → e2e 68.75†; GSPO, G 12, β 1e-3 |
| 2510.20797 | No Mean Feat | Qwen3-0.6B–8B, Gemma2-2B, Llama3.2-1B, 4–128× | KL-only loss; 6-model avg 4×: causal 59.48 / bidir 62.70 / pooling 66.97 vs teacher 70.14; multi-ratio 65.18 (4×), 56.85 (16×) |
| 2405.13792 | xRAG | Mistral-7B, 1 token/doc | without self-KL 44.58 → 41.49 |
| 2503.10337 | KV-Distill | Llama-3-8B, 20% KV | mixed KL 86.0 vs fwd 83.4 vs rev 82.7 vs AE+LM 79.1 (base 87.6) |
| 2610.33006 | Boa | Mamba-2 → Qwen3-1.7B, 4–16× | KL only, τ 2; F1 81.3 / 80.5 / 80.2 vs 83.7 |
| 2606.10572 | Latent Memory | 1B compressor → frozen 8B, 1 token/evidence | KL along the teacher answer; without reconstruction −1.1 to −1.7 EM |
| 2609.01507 | LatentPress | frozen Qwen2.5-7B etc., 4–16× | CE + forward KL; 5 seeds; LongMemEval .504 ± .024 vs .490 |
| 2609.39953 | CAFD | Qwen2.5-Omni-7B | label-free on-policy self-distillation: +1.44 points, 26.9% of the gap, 120/125 conditions |
| 2608.28138 | Token-Budget Distillation | LLaVA-Video, 10% tokens | answer-region KL gated by teacher reliability; 58.3 → 58.9 cumulative |
| 2510.20535 | ARC-Encoder | 3B encoder → frozen Llama-3.1-8B, 4–8× | reconstruction share 0/20/50/100% → 39.8/41.6/41.5/37.5; pretraining +16/+19; pretrain 8× → finetune 4×; context distillation at finetune did not help |
| 2606.09659 | LCLM | 0.6B encoder → Qwen3-4B, 4–16× | staged recipe; frozen-decoder / LoRA "substantially underperform"; ~17% reconstruction† |
| 2511.18659 | CLaRa | Mistral-7B, 16× | no token reconstruction; 38.96 vs 37.67 uncompressed vs PISCO 37.83 |
| 2501.16075 | PISCO | Mistral-7B, 16× | .68 vs .71; "pretraining has little benefit"; gold labels hurt |
| 2504.07109 | OSCAR | Mistral-7B, 16× | no pretraining −4 to −14; compressor full FT > LoRA; frozen generator −.06 |
| 2407.09252 | COCOM | Mistral-7B, 4–128× | frozen decoder NQ .519 → .421; no pretraining .519 → .490 |
| 2307.06945 | ICAE | Llama-7B, 4× | AE weight .4–.6; pretrained vs not 60.4% win / 9.5% loss |
| 2401.03462 | Activation Beacon | Llama-2-7B, 2–32× sampled | LongBench ≈ full finetune |
| 2509.01092 | REFRAG | RoBERTa-large → Llama-2-7B | 9-stage geometric curriculum "essential" (Table 11: 0.135 vs 1.599 log-ppl†); RL pointer policy, perplexity reward |
| 2504.03154 / 2405.19315 / 2405.17430 / 2406.20092 | TokenFLEX / MQT-LLaVA / M3 / LLaVolta | VLM token budgets | +1.6 / +1.0 / +0.4 at 64 / 144 / 256 tokens; nested budgets; stage-wise compression |
| 2510.00515 / 2610.17853 / 2406.12275 | EPIC / DVTR-KD / VoCo-LLaMA | VLM compression distillation | progressive token+layer consistency; dense-token reconstruction; attention distillation (no numbers in abstracts) |
| 2507.13348 / 2508.07313 / 2604.13579 / 2604.13731 / 2605.07019 | VisionThink / DocR1 / MM-Doc-R1 / Doc-V* / LensVLM | RL to select and expand | ≈ 101% of base†; evidence-page reward; +10.4% MMLongBench-Doc (Qwen3-4B/8B); SFT → GRPO; Qwen3.5-9B full-text parity at 4.3× |
| 2609.22910 | CounterCredit | Qwen3-VL-8B | counterfactual credit per visual call: 6.3–9.4 over outcome-only GRPO; 75.4 → 80.8 |
| 2408.12637 / 2409.17146 / 2610.36759 / 2602.03992 | Docmatix / PixMo-Docs / DocSkill / Nemotron ColEmbed V2 | data | 9.5M QA (DocVQA 60.1 → 71.4†); QA from rendering code; cross-page evidence paths +5.5; hard negatives capped at .95 × positive† |
| 2503.02199 / 2510.17771 / 2602.10425 / 2603.10360 | Blind Faith in Text / Seeing but Not Believing / HII-DPO / One Token Two Fates | the utilisation gap in VLMs | SFT with conflicting text reduces the bias; correct attention, wrong answer; counterfactual images as preference data |
| 2010.06192 / 2502.20566 / 2405.03637 / 2411.04330 | Revisiting BFloat16 Training / SR for LLM Training / Collage / Scaling Laws for Precision | precision | lost small updates; SR/Kahan ≈ fp32; bf16+SR > mixed precision to 6.7B; lower precision = fewer effective parameters |
| 2402.12354 / 2308.03303 / 2402.16842 | LoRA+ / LoRA-FA / Asymmetry in LoRA | LoRA optimisation | B needs a higher lr (+1–2%); frozen A is fine; random A ≈ trained A |
| thinkingmachines.ai/blog/lora | LoRA Without Regret | LoRA practice | optimal LoRA lr 10× full-FT; less tolerant of large batches; RL works at rank 1; all layers |
| 2508.02645 / 2605.08985 | Variance in VQA benchmarks / LLaVA-UHD v4 | noise | seed and framework variance; ±0.8 points around the baseline across data scales† |
