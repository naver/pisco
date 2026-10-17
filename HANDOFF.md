# HANDOFF — PISCO (multimodal)

_Updated 2026-10-17 18:00 (NLE). mmrl chain DONE (null, see below); Rounds 5/6/7 (stage-2 v3/v4/v4kd/balanced) queued as `afterok` chains — run
`expmon jobs` before relaunching anything. Science record: `EXPERIMENTS.md` (`expmon ledger`). Results table:
`expmon results --exp mmpisco` (= `scripts/show_mm_results.py`, every row beside the uncompressed baseline on the same
questions). Design/invariants of the multimodal code: `CLAUDE.md` → "Multimodal PISCO"._

## Goal
Multimodal PISCO: compress document-page images and text into `<MEM>` embeddings (decoder never sees pixels).
Round 3 asks whether a dense-page stage-2 diet closes the gap on multi-page/infographic pages, and starts a
navigation-memory track (Habitat/HM3D frames at ~25 `<MEM>` per frame).

## Status

### Settled (full 500-Q doc-QA eval, compressed / uncompressed, judge = Qwen3-8B Yes/No)
| model | judge | EM | ANLS | note |
|---|---|---|---|---|
| **reference `expQ/ft_mm_4b_171745/model`** (rate 8, 20K TEACHER labels, lr 2e-5) | **.488 / .692** | **.388 / .576** | **.501 / .635** | slides .635/.750, infovqa .360/.550, mpdocvqa .450/.860 |
| **stage-2 v2 dense pages** → ft `expQ/ft_mm_4b_172671/model` (same recipe) | **.510** | .366 | .490 | slides .640, infovqa .365, **mpdocvqa .540** (+.09); only mpdocvqa moved; LongBench F1 unchanged vs v1 (.23/.21/.18/.17 per length bucket) |
| rate 2 (224 MEM/page), same recipe (172296) | .466 | .362 | .473 | 4× MEM = no gain |
| 65K teacher set (172402) | .474 | .334 | .448 | peaked at ckpt-100, drifted |
| query-dependent (172297) | .468 | .338 | .463 | no gain |
| original (gold labels, lr 1e-4) (171654) | .318 | .222 | .276 | superseded |
Diagnosis (user's hint confirmed): finetune targets must be the uncompressed backbone's own answers
(`scripts/relabel_mm_teacher.py`), and lr 1e-4 overfits; prompt mismatch ruled out. Dense pages are bottlenecked
UPSTREAM (stage-2 saw only Wikipedia screenshots). Text LongBench F1 mean: 171745 24.3 vs text ref 31.7.
Habitat frame probe (`scripts/probe_mm_frames.py`, `expQ/probe_habitat/`): 25 MEM/frame confirms .72–.80 of the objects
the uncompressed model sees (13 MEM: .5); 25 is the nav budget.

### Round 3 results (2026-10-15, all chain jobs COMPLETED; pipeline ran as SLURM `afterok` chains)
**Dense pages (Chain 1):** stage-2 v2 `expQ/pt_mm_4b_172617/model` (16 h, loss .87→?) → ft 172671 → eval 172673. Judge
.510 vs .488 (ref 171745); the gain is all mpdocvqa (.45→.54, ceiling .86); infovqa (.36 vs .55) and slidevqa flat; EM/ANLS flat.
Dense diet helps multi-page retrieval a little, not infographics. Text-forgetting gate: LongBench F1 same as v1 (no extra loss).
**Navigation (Chain 2):** `expQ/pt_nav_4b/model` → ft `expQ/ft_nav_4b/model` (rate 16 = 25 MEM/frame, 8 frames/Q) →
eval 172672 on `data/nav_ft_val_v2` (928 Q, TOPK=8) beside the uncompressed k8 ceiling 172674:
| | judge | EM | presence | position | order |
|---|---|---|---|---|---|
| uncompressed Qwen3.5-4B, raw 8 frames, zero-shot (172674) | .693 | .573 | .822 | .615 | .604 |
| **PISCO ft_nav_4b, 25 MEM/frame (172672)** | **.810** | **.806** | .878 | .718 | .848 |
Caveat: the ceiling is ZERO-SHOT on raw frames while the PISCO model was finetuned on 10,488 in-domain rows, so this is
not compressed-vs-uncompressed at equal training; the fair ceiling (uncompressed backbone LoRA-finetuned on the same
rows) does not exist yet. Object probe of `pt_nav_4b` (`expQ/probe_habitat/probe_pt_nav_4b.json`, 113 frames, 651 Q):
yes/no .81 at 25 MEM vs .83 raw (v1 ckpt .71), but object recall .30 (v1 .39) — nav pretrain made it more conservative.
**Bug fixed on resume (2026-10-14):** `eval_mm_4b.sh` hardcoded 4 docs/Q; nav windows carry 8 frames → old ceiling 172624
invalid. Launcher takes `TOPK=<n>` (env, default 4, suffix `_k<n>`). Doc-QA evals (4 pages/Q) unaffected.

## Round 4 (2026-10-15) — results so far
**C. Fair nav ceiling landed — PISCO matches the uncompressed model trained the same way.** `nav_ft_val_v2`, 928 Q, 8 frames/Q:
| model | judge | EM | presence | position | order |
|---|---|---|---|---|---|
| uncompressed Qwen3.5-4B, zero-shot (172674) | .693 | .573 | .822 | .615 | .604 |
| **uncompressed, LoRA-ft on the same 10,488 rows (FAIR ceiling, 173039, `expQ/ft_nav_base_4b`)** | **.797** | .792 | .861 | .704 | .843 |
| **PISCO `ft_nav_4b`, 25 MEM/frame (~16× fewer tokens) (172672)** | **.810** | .806 | .878 | .718 | .848 |
| PISCO nav-ft from the DOC stage-2 172617 (D, 173021, `expQ/ft_nav_from_doc_4b`) | .730 | .725 | .839 | .590 | .774 |
| PISCO doc-ft 172671 zero-shot on nav (D, 173016) | .599 | .537 | .725 | .459 | .618 |
→ the nav stage-2 diet is worth +.08 judge over starting from the document model; documents do not transfer to frames.
**B. Pixel budget is THE infographics bottleneck (uncompressed ceiling, same 500 Q rebuilt at 1M px, 173031):**
| budget | judge | EM | ANLS | slidevqa | infovqa | mpdocvqa |
|---|---|---|---|---|---|---|
| 451 584 px (≈440 visual tokens) | .692 | .576 | .635 | .750 | .550 | .860 |
| **1 003 520 px (≈980 tokens)** | **.776** | .652 | .690 | .735 | **.745** | .920 |
**Compressed 1M-px finetune = null:** 173030 (123 MEM/page, 1M-px teacher labels) judge .508, infovqa .360 vs .510 / .365 at
451K px, while the uncompressed ceiling rose to .776 / .745. The compressor cannot use the extra resolution: stage-2 never saw
>440-token spans. Exploiting 1M px needs a stage-2 rebuilt at 1M px (pretrain data is 451K).
**A. vision LoRA = null.** 173014 (ViT trainable, +25 % train time): judge .480 / EM .334 / ANLS .448 vs .510 / .366 / .490 with the ViT frozen; infovqa .370 vs .365. Keep `train_vision_lora=merger`. (`expQ/ft_mm_4b_173014`, eval 173020)
**E. joint pretrain = negative for both.** `expQ/pt_mm_joint_4b` (173098: 70K dense pages + 45K nav frames, from 172617, rate 8)
→ doc ft 173101: judge .482 (infovqa .295) vs .510; nav ft 173099: judge .755 vs .810 from the nav-only stage-2 (928 Q, ~4 SE).
Confound: the joint pretrain ran at rate 8, the nav finetune at 16. Single-domain stage-2 per task wins at this budget.
**Launchers fixed:** every launcher now ends `exit $RC` (they used to exit 0 after a python failure, so `afterok` chains ran on);
`pt_mm_4b.sh` quotes `data.training_dataset` (Hydra rejects a bare comma).
New knobs: `eval_mm_4b.sh` env `TOPK`, `PX`, `ADAPTER`; `relabel_mm_4b.sh` env `PX`, `SHARD=merge`; `build_mm_1m.sh <eval|finetune> [PX]`;
`train.py` `data.training_dataset='path:N,path:N'` (quote for Hydra); `scripts/train_base_mm.py` + `ft_base_mm_4b.sh` (fair ceiling).

## Latency
Full write-up (numbers, method, pitfalls, reproduce commands): **`docs/latency.md`**. In one line: compression pays for
short answers at any load (first token 2.6-9× sooner alone; vLLM 9-19× more req/s than raw images at 64-256 concurrent
requests); for one request it stops paying after ~30-180 generated tokens. `--merge_lora` was a silent no-op before
`pisco.model.merge_lora_inplace` (fixed). Doc sweep 173425 hit its time limit; 4/8 pages + 16 pages at 1 request recovered from its log.

## Round 5 — stage-2 v3, generic compressor (queued 2026-10-16 19:30, user: "launch the training" + natural images)
Data (`build_stage2_v3.sh`, CPU jobs): `data/mm_pretrain_pdfa_1m` (pdfa pages re-rendered at 1 003 520 px, shards 173492-96
→ merge 173497), `data/mm_pretrain_wiki_1m` (Wiki-SS at 1M, 173498), `data/mm_pretrain_natural` (NEW
`scripts/build_mm_pretrain_natural.py`: the_cauldron `localized_narratives` photos at 451K px + human descriptions as AE
targets; probe 173491 → build 173499, 60K rows). Training chain (afterok): shake 173958 (300 rows, mostly 1M pages; memory
check) → pt 173501 `expQ/pt_mm_v3_4b` (from 172617; pdfa 80K + wiki 30K + natural 50K = 160K rows, bs 8×8, lr 3e-5,
p_text .5 / ae .4 / cont .1, `image_max_pixels=1003520`) → LongBench 173502 + doc ft 173503 (1M teacher set, bs 1×64)
→ eval 173504 on `eval_data/mm_docqa_500_1m` (compare: 173030 .508, 1M ceiling .776).
Open: no natural-image QA eval or finetune rows yet (finetune set is doc-only) — needed to show the "generic" part.

## Round 6 — stage-2 v4: new tasks + KL (queued 2026-10-16 21:30, user: "go for all")
Why: KL at FINETUNE failed (mmkd 172793: slidevqa-100 judge .59 vs .64; KL .81→.50 then flat = the MEMs lack the info)
→ fix what stage-2 teaches. New in `MultimodalPretrainingCollator` (tests: `python test_mm_collators.py`, 28 OK):
- `desc` (p_desc): 1..4 images, finetune chat template, "Describe Document j in detail." → row `teacher_text`
  (uncompressed Qwen3.5-4B description; prompts in `pisco/mm_tasks.py`, shared with the generator
  `scripts/teacher_describe_vllm.py`; full chat == generation prompt + answer, checked on the real tokenizer).
- `find` (p_find): 2..4 images, 'Which document contains this text: "<6-12 words>"?' → "Document j".
- `teacher_inputs=True`: KL on the chat rows only (`teacher_kd_rows` mask, honoured in `PISCO.forward`).
Pipeline (afterok; gpu QOS caps 24 queued jobs/user → later jobs on gpu-be):
teacher descriptions 173522-26 (`data/teacher_desc/*.jsonl`: natural 50K, pdfa 0-40K, wiki 0-15K) → attach 173528-30
(`data/mm_pretrain_{natural,pdfa_1m,wiki_1m}_desc`, `scripts/attach_teacher_text.py`) → v4 shake 173542 → pt 173543
`expQ/pt_mm_v4_4b`; v4+KL (kd_weight .5) shake 173548 → pt 173549 `expQ/pt_mm_v4kd_4b`. Same init (172617), rows and
steps as v3; mix text .5 / ae .2 / cont .05 / desc .15 / find .1 (v3: .5/.4/.1).
Repairs 2026-10-16 22:30: pdfa shards 1/3 died on a truncated tar download → builder now streams PER TAR (skips a bad
tar), partial shards deleted, rebuilt as 173638/173639, merge 173497 rewired. 2026-10-17: 173639 (shard 3) was preempted,
requeued, and refused its own partial output (the builder cannot resume) → merge never ran, every stage-2 arm sat blocked
~17 h. Fix: `build_stage2_v3.sh pdfa_finalize 3` (173953) keeps the 17.5K rows written → `pdfa_1m` = 92.5K rows (all
consumers cap ≤ 80K); merge 173497 rewired `afterok:173953`. Then the v3 shake 173500 died on a Hydra error
(`+model_config_overrides.image_max_pixels` while `pt_lora_mm.yaml` already set it) → key dropped from the yaml (all
checkpoints store 451584 = default); shake resubmitted 173958, pt 173501 rewired; v4/v4kd/balanced shakes need nothing. Both probes (173491, 173521) finished but
HUNG at exit (streaming threads) → cancelled; streaming builders now `os._exit` after main. Teacher descriptions were
Markdown reports, 99% cut at 512 tokens → `DESC_SYSTEM_PROMPT` now asks for plain text, ≤200 words; wiki redone 173642
→ attach 173643 → v4 shakes 173542/173548 rewired (`scontrol update JobId=… Dependency=…` keeps chains intact).
Natural-image QA (`scripts/build_mm_natural_qa.py`): probe 173521 (cancelled, schemas OK) → eval 173531 `eval_data/mm_naturalqa_500` (VizWiz 150,
RealWorldQA 150, TextVQA val 200; TextVQA photos are Open Images = mild overlap risk with pretraining photos) and ft
173532 → relabel 173533-37 → merge 173538 `data/mm_ft_natural_8k_teacher`. 
Generic finetune (doc-1M teacher 20K + natural teacher 8K) + evals (doc-1M, natural): v3 173539→173540/41, v4
173545→173546/47, v4kd 173550→173551/52. Natural ceilings: uncompressed 173553, doc-only PISCO 172671 173554.

## mmrl — DEX-Comp-style RL track (queued 2026-10-16 22:50, user: "go as you say")
Paper: DEX-Comp, arXiv 2609.05152 (soft compression; stage 1 distils only on teacher-correct rows, stage 2 GRPO on
teacher-failed rows, G 16, reward = gold containment, lr 1e-6, β .02; +3 judge pts from the filter, +4 from RL, text QA).
Why here: 172671's errors are confident wrong values (lost .24 of 500 Q, refusals 3 %, median 1 word), and teacher labels
miss gold on 25-50 % of short-gold rows. RL can only sharpen what the model sometimes gets right, so probes gate it.
Code: `eval_mm_docqa.py --num_samples G` (+ greedy log-prob; greedy tokens unchanged, smoke 173654), `judge_mm_docqa.py
--samples`, `scripts/mmrl_teacher_split.py` (dump/build, `--out_rest`), `scripts/mmrl_probe_report.py <probe> [--base
<uncompressed json>]`, `scripts/mmrl_raft_build.py`, `pisco.reward.is_refusal`; launchers `mmrl_split_4b.sh`,
`mmrl_raft_4b.sh` (CPU), `eval_mm_4b.sh` env `NSAMP` (suffix `_s<G>`).
**Done:** split 173666 — judge says teacher right on 13 598 / 20K (docmatix .60, docvqa .80, infovqa .74, chartqa .53) →
`data/mmrl_ft_20k_pd` (S+), `data/mmrl_ft_20k_fix` (S+ + 2 360 gold-retargeted short-gold S-), `data/mmrl_sminus_probe_2k`,
`data/mmrl_sminus_rest` (other 4 402 S-). Probe A 173667 (172671, 500 Q, G 8): greedy .510, pass@2 .529, @4 .616,
**@8 .698** (uncompressed .692) → headroom gate PASSED (+.19, same on numeric and text gold; recovered Qs median 1/8
right; maj@8 .496). AUROC .79 but the raw-page router is weak (least-confident 30 % → .586 vs random .566) → dropped.
Judge caveat: Qwen3-8B sometimes accepts non-answers ("the text does not provide ..."): 3/1 721 probe-A hits, 162
teacher labels in pd (left in); RAFT targets filter them with `is_refusal`.
**Result (500 Q, judge / EM / ANLS; uncompressed .692): all null** (standard error ≈ .022):
| arm | judge | EM | ANLS |
|---|---|---|---|
| 172671 reference | .510 | .366 | .490 |
| pd 173668 (teacher-correct rows only, 13 598) | .510 | .374 | .496 |
| fix 173669 (pd + 2 360 gold-retargeted) | .518 | .384 | .498 |
| pd + RAFT 173710 (+2 323 self-recovered S- rows, median 3 words) | .498 | .372 | .483 (infovqa .355→.310) |
Probes B/C (pd model, G 8 on the 6 402 teacher-failed rows): RAFT recovered 36 % (docmatix 35, docvqa 37, infovqa 46,
chartqa 37 %). Greedy did not move although pass@8 = .698 → GRPO is the only untested lever; owner's call.

## Round 7 — balanced generic arm (queued 2026-10-16 23:40, user: are documents too specific for a generic PISCO?)
Same init / rows / steps / task mix as v4; only the image sources change: docs 28% (pdfa_1m_desc 30K + wiki_1m_desc 15K),
photos 37% (natural_desc 60K), charts+diagrams 23% (NEW `data/mm_pretrain_charts`: the_cauldron chart2text / vistext /
diagram_image_to_text at 1M px, ~37K, build 173684), nav frames 11% (`nav_pretrain_desc` 18K). Teacher descriptions:
photos 50-60K 173686, frames 173687, charts 173688/173689 → attach photos (now 60K rows; v4 still reads the first
50K) 173698, charts 173691, frames 173692 → shake 173700 → pt 173701 `expQ/pt_mm_bal_4b` → generic ft 173703 → evals
173704 (doc-1M), 173706 (natural). Question it answers: v4 vs balanced on natural QA (does a doc-heavy diet cost photos?)
and on doc QA (does it cost documents?). Note: `afterok` on a job that finished minutes ago fails with "Job dependency
problem" (it left the controller): depend only on jobs still in the queue.

## vLLM
Running the real PISCO decoder in vLLM 0.19.1 (export, hybrid hooks, launcher settings) and its results:
`docs/latency.md`. Exported decoders: `expQ/ft_nav_4b/vllm_decoder`, `expQ/ft_mm_4b_172671/vllm_decoder`.

## Next step
Wait for Round 5/6 (2-3 days): the v3 / v4 / v4+KL pretrains start once their data jobs finish (fail-fast watch:
`expmon babysit` log in the session scratchpad; or `expmon jobs`). Then compare the three on doc-1M and natural-QA beside
the ceilings (1M uncompressed .776; natural uncompressed 173553). Open science question: the doc gap (.51 vs .69/.78).
Nav is solved at this scale (PISCO .810 ≥ fair ceiling .797); external nav evals (OpenEQA, VSI-Bench) not started.

## Data (all built, final dirs only; `*/shards` deleted)
| dir | rows | use |
|---|---|---|
| `data/mm_pretrain_160k` | 160K Wiki-SS screenshot+text | stage-2 v1 |
| `data/mm_pretrain_pdfa` → `data/mm_pretrain_v2` | 100K pdfa dense pages (+40K wiki) | stage-2 v2 (`scripts/build_mm_pretrain_pdfa.py`, needs pypdfium2 — added to pixi.toml) |
| `data/mm_ft_docpages_20k_teacher` | 20K page-QA, teacher labels | the finetune recipe |
| `data/mm_ft_docpages_65k[_teacher]`, `_20k` (gold) | | superseded |
| `eval_data/mm_docqa_500` | 500 Q (slidevqa 200, infovqa 200, mpdocvqa 100, VisRAG test sets) | doc-QA eval |
| `data/nav_hm3d_{train,val}` | 1,450 / 180 rendered episodes, 61 frames, semantic GT (`scripts/nav/render_hm3d.py`) | raw |
| `data/nav_pretrain` | 44,950 frames + GT descriptions | nav stage-2 |
| `data/nav_ft_train_v2` / `nav_ft_val_v2` | 10,488 / 928 memory-QA rows (8-frame windows; presence/order/position) | nav ft / eval (v1 sets = ill-posed, kept) |

## Key commands
- Launch only via `expmon launch --exp <mmpisco|mmnav> …`; `-p gpu,gpu-be` takes the first free slot; training is
  resumable (PISCOTrainer + `--requeue`), 100-step checkpoints. `gpu` QOS: 8 running jobs / 8 GPUs per user.
- Eval: `eval_mm_4b.sh <model dir> <pisco|base> [limit] [data] [image_rate]` (judge chained; outputs tagged by
  checkpoint/rate/dataset). Checkpoint previews: `scripts/auto_preview_ckpts.sh <run_id[:rate]> …` (nohup loop).
- Tests: `python test_mm_collators.py` (15 tests). Habitat env: `pixi run --manifest-path
  /beegfs/scratch/user/hdejean/cosmos_nav_test/habitat_env/pixi.toml python …` (python 3.9, no pisco imports).

## Open questions / decisions pending
- **Disk (2026-10-15 cleanup, +412 GB → 856 GB free):** deleted all `expQ/*/checkpoint-*` where `model/` exists (47 dirs), the
  superseded data sets (`mm_ft_docpages_65k[_teacher]`, `_20k` gold, `mm_pretrain_160k`, `mm_pretrain_pdfa`, `nav_ft_{train,val}` v1),
  the `model/` of concluded mm ablations (171654/55/56/97, 171733/36, 172296/97, 172402 — their `eval/*.json` are kept, the
  results table still works) and the HF `datasets/generator` caches. NOT touched (user decision): `Reranking/cache/hub` 465 GB
  (Qwen3-32B 62 GB, Qwen3.6-27B 53 GB, bge-multilingual-gemma2-data 31 GB, SmolLM2-10B 24 GB + 40 GB processed, …) — other
  projects' models; `refs/` mtimes are not a reliable last-use signal (the daily judge Qwen3-8B shows April).
- If stage-2 v2 does not move mpdocvqa/infovqa: next suspects are the pixel budget (eval set + teacher labels would
  need rebuilding at the higher budget) and training the vision tower (`train_vision_lora=all`).
- Nav: external evals not started (OpenEQA EM-EQA HM3D via HF mirror `the-Lin/openeqa`, VSI-Bench); OpenDocVQA corpus
  is gated (needs HF access + token).
- Code (September RAFT/judge work + all October multimodal work, 36 changed/new paths) is UNCOMMITTED on branch `arc`.

## Older open items (unchanged)
- [raft] gpt-4.1 relabel of the 453K text ft set (~$93 batch) not launched; needs `OPENAI_API_KEY`.
- [bidi_ablation] causal 4B→4B finetune 162668 finished 2026-09-10, never evaluated, output dir not under `expQ/`.

## Don't-break list
- Image spans are atomic (never chunk/truncate); pixel row order == image-chunk row order; `bidirectional` + images refused.
- `mask_before_mem` is a no-op in base `PretrainingCollator` (do NOT change); the multimodal collators use the fixed order.
- Qwen3.5-4B has no generation_config.json: `PISCO.decoder_eos_ids()` must stay in every generate call.
- Any extra decoder forward with `input_ids` + images (teacher pass, generation) leaves `rope_deltas` cached on the
  Qwen3.5 model; the `inputs_embeds` student forward reuses it. Snapshot/restore it (see `PISCO.kd_loss`).
- `*.sh` launchers are gitignored; durable logic lives in `scripts/` and `pisco/configs/`.
- Agent-QA numbers before 2026-07-15 are on buggy labels (`my_qa_fixed.json`).
