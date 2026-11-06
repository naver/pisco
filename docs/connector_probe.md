# Probing an architecture component at inference: the connector (archq2, 2026-10-29)

How we decide whether a piece of the model (here the connector) is worth redesigning, without training anything:
replace or perturb that piece on the trained model, measure what the decoder loses, and look at what the piece
actually computes. One GPU, a few minutes, on real evaluation rows. Code: `expQ/archq2/probe_connector.py`,
launcher `expQ/archq2/probe_connector.sh`. Pattern inherited from the norm probe (`expQ/norm_probe/norm_probe.py`),
which settled the connector output norm the same way before its training arm.

## Why

- The connector is the simplest part of PISCO: a 2-layer ReLU MLP (2560 → 4096 → 2560) plus, since October 2026, an
  RMSNorm with a learned gain. The norm alone moved photos by +.02, so the read-out path has slack, and the owner
  asked whether the projector is too simple.
- The literature does not settle it. In vision-language models connector depth and type barely matter (Honeybee:
  linear 67.1, 2-layer MLP 68.3, 6-layer MLP 68.5 on MMBench; MM1: little effect; LLaVA-UHD v4: MLP beats
  resamplers), but GMSA reports a huge gain from a transformer block initialised from the decoder's first layer,
  and StateBridge shows that between a model and itself a training-free rotation plus scale is nearly enough.
- A training arm costs about 12 GPU-hours per seed and needs two seeds to clear the seed spread (.03 on documents).
  The probe answers most of the question first and tells which arm, if any, is worth that price.

## How

**Rows.** Real evaluation rows through the training collator, so the `<MEM>` layout is the training one:
document pages (`eval_data/mm_docqa_1500_1m`, 4 pages per question, about 123 `<MEM>` each), photos
(`eval_data/mm_naturalqa_1500`, 56 `<MEM>` per image), text (`eval_data/text_hotpotqa_1500`, 4 paragraphs; the
document eval rows carry no page text, so text `<MEM>`s must come from a text set). Rows are split 70 / 30; every fit
uses the 70%, every loss is measured on the 30% the fit never saw.

**What is measured.**

| part | question | measurement |
|---|---|---|
| weights | is the width used, did the finetune change the connector? | singular spectrum and effective rank of W1, W2 and W2·W1; relative movement of the weights since the stage-2 checkpoint |
| ReLU | is the nonlinearity used? | share of hidden units active per `<MEM>`, dead units, units active on < 1% of inputs, effective rank of the hidden activations |
| linearity | is the trained map linear? | ridge regression from the compressor's `<MEM>` state to the pre-norm connector output (bias included, λ chosen on a validation fifth of the fit rows); R² and output cosine on held-out rows; the same fit truncated to rank 64 / 256 / 1024 |
| spaces | is the map near identity? do outputs look like tokens? | cosine between input state and output; nearest decoder-token cosine of outputs and of inputs |
| effect | what does each replacement cost? | answer NLL on the finetuning labels when the `<MEM>`s are swapped at inference |

**Replacements** (all go through the connector's norm, so the scale is the trained one):

- `linear_fit`: the ridge map in place of the MLP. If this costs nothing, the nonlinearity is dispensable.
- `no_relu`: the MLP's own weights with the ReLU removed. Separates "a linear map works" from "this linear map".
- `linear_rank{64,256,1024}`: how many directions carry the map.
- `stage2_connector`: the connector of the stage-2 checkpoint in place of the finetuned one.
- `h_through_norm`: the compressor state itself, only rescaled (no projector, Coconut-style).
- `shuffle_in_chunk`: the `<MEM>`s of each chunk permuted. Does the decoder read their order?
- `chunk_mean`: every `<MEM>` of a chunk replaced by the chunk mean. Does per-vector content matter?

Each row's loss is paired with the untouched model, so the report gives the mean difference, its standard error and
the share of rows that got worse. A difference under about .02 with a standard error of the same size is noise.

**Checks built in.** The probe path must reproduce `PISCO.compress` on the first row (relative difference < 1e-3; the
first version used plain matmuls and failed this in bf16, which is why it uses the connector's own modules). A
ridge fit of the input onto itself must give R² ≈ 1. A set whose `<MEM>`s have effective rank 1 (the document rows
read as text) is a data problem, not a result. A fit with fewer `<MEM>`s than features (1.5k photo `<MEM>`s for 2,561
features) overfits and its R² is meaningless: use 6× more rows.

**Run.** `expmon launch --exp archq2 --stage probe expQ/archq2/probe_connector.sh <ckpt/model> [N_ROWS] [REF_CKPT]`.
N_ROWS is per set for documents; the text set takes 4× and photos 6× that many rows. A text-only checkpoint
(no image processor) switches to the plain `FineTuningCollator` and the text set. Output: a JSON under
`expQ/archq2/probe_connector_<jobid>/` and a summary in the job log.

## Results

Checkpoint 174908 (best multimodal, norm on), job 178401. Fit / held-out `<MEM>`s: documents 12.0k / 5.5k (40 rows),
text 7.6k / 3.2k (160 HotpotQA rows), photos 8.9k / 3.8k (240 rows). NLL differences are paired, ± one standard error.

| probe | documents | text | photos |
|---|---|---|---|
| finetune moved W1 / W2 since stage 2 | 0.6% / 0.9% of their norm (cosine .9999) | same weights | same weights |
| hidden units active per `<MEM>` (of 4,096) | 59%; 101 dead, 830 active on < 1% | 60%; 252 dead, 920 < 1% | 58%; 148 dead, 802 < 1% |
| effective rank input / hidden / output | 304 / 280 / 268 | 199 / 181 / 128 | 269 / 262 / 228 |
| linear fit of the map, held-out R² (output cosine) | .978 (.997) | .991 (.999) | .968 (.995) |
| cosine input → output; nearest token cosine of outputs | .09; .08 | .10; .08 | .11; .08 |
| NLL untouched | 1.177 | 1.516 | 1.950 |
| `linear_fit` | −.006 ± .007 | −.002 ± .004 | −.006 ± .004 |
| `stage2_connector` | −.006 ± .020 | +.002 ± .011 | −.004 ± .007 |
| `no_relu` | +.80 ± .27 | +.03 ± .08 | +.23 ± .11 |
| `linear_rank1024` | +.03 ± .02 | −.02 ± .02 | +.02 ± .01 |
| `linear_rank256` | +.84 ± .25 | +.04 ± .09 | +.25 ± .11 |
| `h_through_norm` (no projector) | +1.09 ± .31 | +.07 ± .10 | +.78 ± .20 |
| `shuffle_in_chunk` | −.16 ± .17 | −.03 ± .03 | .00 ± .03 |
| `chunk_mean` | +.93 ± .30 | +.11 ± .08 | +.65 ± .16 |

- **A refit linear map equals the trained connector on all three modalities** (within .006, standard errors
  smaller). The MLP's own linear part does not (`no_relu`), so the ReLU is used to *implement* a linear map, not for
  anything a linear map cannot do. The map needs about a thousand directions (rank 1,024 costs ≤ .03, rank 256 is
  broken on images).
- **The finetune's connector is the stage-2 connector**: swapping it back changes nothing. Connector training
  happens in stage 2; finetuning only adapts LoRA.
- **The decoder does not read the order of `<MEM>`s inside a chunk**, on any modality (shuffle: 0 ± .03 on photos,
  −.03 ± .03 on text, −.16 ± .17 on documents). It does read each vector's content (`chunk_mean` costs .65 to .93 on
  images).
- Text `<MEM>`s matter less in this set (every edit costs little, even no projector): the HotpotQA rows keep 4 of 10
  paragraphs, so many rows have no gold paragraph among the compressed ones. Text conclusions rest on the
  agreement with the two image sets.
**Same-backbone caveat (owner).** Compressor and decoder are the same Qwen3.5-4B, which could be why a linear map
suffices. The probe was run on text checkpoints of other sizes and families (HotpotQA rows, 4 paragraphs, 400 rows;
jobs 178431/178433/178435, 179294/179296/179297). `linear_fit` = paired NLL difference of the refit linear map
against the trained MLP, ± one standard error; R² on held-out `<MEM>`s.

| compressor → decoder | hidden sizes | trained how | linear fit R² | `linear_fit` NLL | `shuffle_in_chunk` |
|---|---|---|---|---|---|
| Qwen3.5-4B → 4B (149908, causal override) | 2560 → 2560 | pretrain + finetune | .953 | −.004 ± .005 | −.021 ± .013 |
| Qwen3-8B → 8B (135866) | 4096 → 4096 | pretrain only | .996 | +.001 ± .004 | +.007 ± .023 |
| Ministral 8B → 8B (131513) | 4096 → 4096 | pretrain + finetune | .986 | −.013 ± .010 | −.008 ± .017 |
| **Qwen3-4B → 8B** (135922) | 2560 → 4096 | pretrain only | .995 | −.004 ± .005 | −.007 ± .020 |
| **Ministral-3B → 8B** (133994) | 3072 → 4096 | pretrain only | .988 | +.017 ± .014 | −.172 ± .035 |
| **Qwen3.5-4B → 9B** (149907, causal override) | 2560 → 4096 | pretrain + finetune | .941 | −.008 ± .005 | +.004 ± .013 |

The linear map equals the MLP whether or not the two backbones share a size or a family: the connector's job is a
rotation plus a scale between two spaces, not a nonlinear transform. (The Ministral 3B → 8B row is a pretrain-only
model read through the RAG prompt it never saw: its `<MEM>`s hurt the answer, so averaging or shuffling them *helps*;
its fit R² still says the map is linear.) The 4B → 4B and 4B → 9B Qwen3.5 rows run the compressor causally although
they were trained bidirectionally, which the current code refuses on the hybrid backbone; both sides of each paired
difference carry the same handicap.

## Is position already in the vectors? (`expQ/archq2/probe_mem_position.py`, job 179327)

A ridge probe from each `<MEM>` vector to its place in its chunk (0 = first, 1 = last), fitted on 70% of the rows,
scored on the rest. Also the mean cosine between `<MEM>`s of the same chunk and of other chunks of the same prompt.

| set | R² from the connector output | first / second half accuracy | cosine same chunk / other chunk |
|---|---|---|---|
| document pages | .92 | .92 | .75 / .73 |
| text (HotpotQA) | .97 | .96 | .79 / .77 |
| photos | .92 | .95 | .76 / - (one chunk) |

Each vector carries its own position: a linear read-out recovers it almost exactly. That is why shuffling costs
nothing (the decoder does not need the order, the vectors say where they come from) and why decoder-side position
ids (arm 2) add information the decoder already has. The `<MEM>`s are also very alike everywhere (cosine .75 to .79,
against about .08 to the nearest token embedding) and only .02 more alike within a chunk than across chunks: one
large shared direction, a small chunk-specific part. Whether the vectors of a chunk are *redundant* in content is
measured directly by the fewer-`<MEM>`s variants of the connector probe (`keep_even_mems`, `keep_first_half`,
`keep_second_half`: the prompt loses the matching `<MEM>` tokens).

## Are the `<MEM>`s of a chunk redundant? (job 179335)

Fewer `<MEM>`s per chunk at inference; the prompt loses the matching `<MEM>` tokens. Paired NLL difference to the
untouched model, held-out rows (documents 12 rows, text 48, photos 72):

| variant | documents | text | photos |
|---|---|---|---|
| keep every second `<MEM>` (half the vectors) | +.17 ± .13 | +.07 ± .07 | −.03 ± .03 |
| keep the first half of each chunk | +.21 ± .19 | −.03 ± .07 | +.09 ± .07 |
| keep the second half of each chunk | +.53 ± .36 | +.07 ± .07 | +.34 ± .11 |
| chunk mean (one vector repeated) | +.92 ± .29 | +.10 ± .07 | +.65 ± .16 |

- Half the vectors carry most of the information. **Caution (2026-11-02):** these NLL differences on tens of rows
  understate accuracy costs: keeping the first 75% of slots read as free here (docs −.003 ± .011, photos .000 ± .017)
  but costs −.023 / −.024 judge accuracy on the 1,500-question sets (jobs 183652 / 183653). Treat a null NLL probe as
  "under about .02 accuracy", and confirm any serving claim on the judged sets. (Compressing at rate 16 with the rate-8 model is a
  different operation and lost .07 accuracy: the compressor's write pattern is rate-specific, the content is not.)
- The early `<MEM>`s of a chunk carry more than the late ones, on every modality. The compressor is causal: a late
  `<MEM>` can read the early ones and tends to repeat them.
- This is the premise of a chunk-level mixing step in the connector (merge or re-allocate what the late vectors
  duplicate), and of pruning instead of a bigger connector.

## Reading the result

- The nonlinearity in the connector buys nothing on documents; a linear map of rank about 1,000 is the connector.
  Replacing the MLP by a deeper or wider MLP is therefore not where a gain would come from.
- The decoder does not read the order of `<MEM>`s inside a page because each vector already encodes its position
  (R² .92 to .97). Arm 2 (spread positions on the decoder side) therefore adds nothing the decoder lacks; a
  position change can only matter on the compressor side, where it decides what each `<MEM>` attends to.
- What could still help is a component that changes what the vectors *contain*, not how they are projected: more
  compressor layers (arm 1), a token-mixing step over a chunk's `<MEM>`s (Interlat-style self-attention adapter), or
  GMSA's decoder-initialised block, which is a decoder layer, i.e. token mixing plus a decoder-shaped MLP.
