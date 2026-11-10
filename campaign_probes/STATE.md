# STATE — campaign probes   (rewritten WHOLESALE by every LLM wake; never append)
last wake: 2026-11-11 03:00-03:35 KST (woken because the feasibility-gate job ended)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- The instrument is reproducible: re-running the reference document cell (model 184778, 1,500 questions, PX=1003520)
  gives judge accuracy .610 = the recorded .610 (job 184781), paired difference +.000, CI [+.000, +.000], 0 discordant
  pairs, n=1,500; all 1,500 answers byte-identical, mems_used 646,042 both. Job 188419, verdict closed (PASS).
  Consequence: decoding is greedy and the whole eval is deterministic, so any paired difference between two cells comes
  from the probe itself.
- plumbing only: the page-picking fusion script reproduces the saved route numbers exactly — route .8607, attention
  .8853, n=1,500 rows, model 183814, 10-page document set (selftest job 188427 vs probe 185577's summary.json).

**Open, and running** (all 8 submitted this wake, all PENDING on gpu-be at 03:30; SLURM start estimates 08:00-16:30,
which have been pessimistic: the gate was estimated 09:55 and started at 02:09)
- arm A (use-recovery on documents), seed 42, model 184778, 1,500 questions each, baseline .610:
  evidence-then-answer (188439), gold page only (188440), picked page raw = route-then-expand (188441).
- arm C (reasoning vs reading on text), model 184778, MRQA out-of-domain 1,500 questions, one chunk per passage:
  plain baseline (188445; no 184778 cell existed), evidence-then-answer (188446).
- arm D (page picking without training), model 183814: attention-route re-runs that now save the full digit scores,
  10-page documents (188442) and HotpotQA 10 passages (188443), then fusion chained afterok on both (188444, CPU).

## Owned jobs (jobid | arm | state | elapsed | rows | eta | note)
| 188439 | A expand s42 | PENDING (Resources) | 0 | 0/1500 | est. start 08:08, then ~1.7 h | `expQ/probes/a_expand_s42/eval/`; --time 5:12:00 = 174852 (104 min) × 3 |
| 188440 | A gold s42 | PENDING (Priority) | 0 | 0/1500 | est. 13:15, then ~0.5 h | `expQ/probes/a_gold_s42/eval/`; --time 4:54:00; slot total smaller by design |
| 188441 | A route1 s42 | PENDING (Priority) | 0 | 0/1500 | est. 14:30, then ~1 h | `expQ/probes/a_route1_s42/eval/`; --time 4:54:00 |
| 188442 | D attn route retr10 | PENDING (Priority) | 0 | 0/1500 | est. 14:30, then ~41 min | → `expQ/probes/attn_route_188442/`; control: digit .861, attention .885 |
| 188443 | D attn route hotpot | PENDING (Priority) | 0 | 0/1500 | est. 16:30, then ~12 min | → `expQ/probes/attn_route_188443/`; control: digit .922, attention .873 |
| 188444 | D fusion (CPU) | PENDING (Dependency afterok 188442:188443) | 0 | — | after both | writes `fusion.json` + `fused_<picker>.json` in each dir |
| 188445 | C plain s42 | PENDING (Priority) | 0 | 0/1500 | ~17 min once started | `expQ/probes/c_plain_s42/eval/`; CML=4096; --time 0:52:00 |
| 188446 | C expand s42 | PENDING (Priority) | 0 | 0/1500 | unmeasured | `expQ/probes/c_expand_s42/eval/`; --time 5:12:00 (expand doc cell × 3; no text expand cell measured) |

## Adopted jobs
(none. The sibling `longchunk` campaign's 188420-188422 run on `gpu` — not ours, never touched.)

## Open arms and the next decision for each
| arm | status | next action |
|---|---|---|
| A. use-recovery | seed 42 cells queued (3) | when each ends: health (unparseable 0, 1,500 rows, mems_used ratio REPORTED for gold/route1) → `paired_judge.py` vs `expQ/ft_mm_4b_184778/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json`; then the seed-43 cell (model 184780, baseline .615 in `expQ/ft_mm_4b_184780/eval/`). Not yet launched: `MQ=int8` s42 (expected null; calibrates the instrument) |
| B. long-chunk mechanism | not started | write `scripts/probes/build_groups.py` (no builder exists: g3/g10 were built ad hoc on 2026-11-06 and saved as DatasetDict 'train'): cells gold-passages-joined (2 golds share a chunk, 8 distractors alone) and distractors-joined (8 distractors share a chunk, golds alone), on 183814 with CML=4096 TOPK=10, vs g1 .579 / g10 .617; build through SLURM (gpu-be, gres=gpu:0), check it reproduces g10 from text_hotpotqa_1500 first |
| C. reasoning vs reading | baseline + expand queued | next: `base` mode with `STYLE=expand` on the same set (raw reader, same treatment; writes into `expQ/base_qwen35_4b/eval/` under a new name); thinking mode needs one flag in `eval_mm_docqa.py` (hard-coded `enable_thinking=False`, lines 80 and 344) — patchable with tests |
| D. page picking | re-runs + fusion queued | read `fusion.json`: control recovered (digit .861 / attention .885 on retr10)? then gate pick@1 ≥ +2 pt on retr10 AND no loss on hotpot; a passing picker → `RAW=route1 RANK=fused_<picker>.json` judge cell; then the same on 184778 / 184780 |
| E. attention-guided pruning | not started | `scripts/probes/prune_by_attention.py` not written |
| F. photo text vs pixels | blocked on data | the photo set has no OCR text (`docs=[""]`); build `eval_data/probes_textvqa_ocr` (500 TextVQA rows + `ocr_tokens` from lmms-lab/textvqa) via a SLURM job; law amended |

## Health evidence this wake
- gate cell 188419: COMPLETED rc 0, 28.6 min on an A100, unparseable 0, 1,500 rows, mems_used ratio 1.000.
- gpu-be is all A100-80GB (sinfo); 37 jobs pending on gpu-be at 03:10; 3 A100 free cluster-wide.
- budget: 10 of 60 cells committed (gate 188419, CPU selftest 188427, this wake's 8); GPU-hours spent ≈ 0.5 of 40
  (the 8 queued cells will add ≈ 6, from their measured walls).

## Changes this wake
- verdict closed PASS on 188419; 8 submissions (cap reached).
- symlinked model dirs `expQ/probes/{a_expand_s42,a_gold_s42,a_route1_s42,c_plain_s42,c_expand_s42}/model → expQ/ft_mm_4b_184778/model`.
- CAMPAIGN.md: gate marked PASSED with its numbers; health rule clarified (probes that change what is compressed report
  the mems_used ratio instead of failing on it); arm F premise marked wrong (no OCR text in the photo set) and re-pointed
  to a new set.
- `docs/probe_campaign_2026-11.md` rewritten with the gate table and the arms in flight.

## Blockers / needs the owner
- none blocking. FYI: `eval_mm_4b.sh` and `probe_attn_route.sh` are NOT under git (`.gitignore` has `*.sh`); force-add
  them when convenient. The ALERT about the 01:00/01:30 failed wakes is yours to clear (cause: usage quota).
