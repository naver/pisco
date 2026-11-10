# Probe campaign 2026-11 — why the compressed model loses, and what recovers it without training

Law: `campaign_probes/CAMPAIGN.md`. Live state: `campaign_probes/STATE.md`. This page is rewritten whenever an arm decides.
Metric everywhere: judge accuracy ↑ (share of questions the Qwen3-8B rubric judge accepts), paired against the same model's
plain cell on the same questions (95% bootstrap CI, McNemar p). Compression: 1 slot per 8 visual or text tokens
(`compr_rate: 8` in the reference models' config); on the document set at PX=1003520 the plain cell uses 430.7 slots per
question on average (646,042 slots / 1,500 questions, 4 pages each, measured on job 188419).

## Status (2026-11-11 04:25 KST)
- **Is the judge reproducible? Yes, on the same node.** A re-run on the node of the recorded run gives the same 1,500
  answers. **On another node it is not, for image inputs** (found this wake, details below). Every image comparison
  must now pair two cells from the same node image.
- 17 probe cells are open (7 running, 10 queued). No arm has decided. The `gpu-be` partition keeps taking GPUs back
  (preemption: SLURM stops the job and requeues it from zero): 7 cells were stopped between 04:11 and 04:25.

## The instrument: same answers on re-run?
| check | model | set | n | node of the re-run (system image) | node of the recorded run | answers identical | judge accuracy ↑, re-run vs recorded | judge flips (up / down) |
|---|---|---|---|---|---|---|---|---|
| gate re-run (job 188419) | 184778 (seed 42) compressed | documents, 1M px, 4 pages | 1,500 | homer-011 (el9_8) | homer-011 (el9_8) | 1,500 / 1,500 | .610 vs .610 | 0 / 0 |
| fix check (job 188458) | 184778 compressed | documents, first 300 questions | 300 | homer-008 (el9_7) | homer-011 (el9_8) | 243 / 300 | .623 vs .610 | 9 / 5 |
| fix check (job 188456) | uncompressed Qwen3.5-4B | MRQA text, first 300 questions | 300 | homer-011 (el9_8) | homer-008 (el9_7) | 300 / 300 | (EM .367 vs .367) | — |
| re-run after a late preemption (job 188445) | 184778 compressed | MRQA text, 1,500 questions | 1,110 rows seen | homer-008 (el9_7) | homer-016 (el9_8) | running scores equal to 3 decimals at 710 and 1,110 rows; preempted before the end | — | — |

Reading: the node changes compressed answers on page images (19% of answers differ, 5% of judge verdicts flip; on
1,500 questions this alone would add about ±.011 to a paired difference) and changes nothing seen so far on text.
Still to measure: two more nodes (homer-016, same image as the recorded run; homer-007, the older image), 300
questions each, so we know whether the cause is the image or homer-008 alone.

## Arms in flight (no result yet)
| arm | cells | what it answers | training decision it feeds |
|---|---|---|---|
| A. use-recovery on documents | seed 42: evidence-then-answer and picked page raw (running), gold page only (requeued after a preemption), slots stored in int8 (queued, kept off the older-image nodes). Seed 43: gold page only, picked page raw (queued) | how much of the −.20 vs raw comes from reading vs picking vs using | evidence-then-answer labels in stage 2; route-then-expand at serving |
| B. long-chunk mechanism | model 183814 on HotpotQA (10 passages): the 2 gold passages joined in one chunk, the distractors alone / the distractors joined, the golds alone / all 10 joined in a shuffled order | is the +.037 of "one long chunk" (g10 .617 vs g1 .579) context between the gold passages, fewer chunks, or order | whether the long-chunk stage 1 should group related passages or just lengthen chunks |
| C. reasoning vs reading on text | compressed plain (done once: .718, see below), compressed evidence-then-answer (running), uncompressed evidence-then-answer (running) | is the text loss (DROP, DuoRC, RACE) room-to-reason or missing content | evidence-then-answer teacher labels for text |
| D. page picking without training | attention-route re-runs on 10-page documents (running) and HotpotQA (queued), then fusion | can the decoder's attention beat the digit route prompt with no training | whether routing rows need training |

First number of arm C (provisional: the job finished once, then SLURM requeued it): the reference model, compressed,
MRQA (6 sources × 250 questions, one passage each, nothing cut):

| source | n | compressed judge accuracy ↑ (184778) | uncompressed judge accuracy ↑ (185542) | gap |
|---|---|---|---|---|
| all | 1,500 | .718 | .807 | −.089 |
| DROP (reasoning) | 250 | .580 | .784 | −.204 |
| DuoRC (narrative) | 250 | .600 | .684 | −.084 |
| RACE (narrative) | 250 | .528 | .680 | −.152 |
| BioASQ | 250 | .912 | .924 | −.012 |
| RelationExtraction | 250 | .840 | .896 | −.056 |
| TextbookQA | 250 | .848 | .872 | −.024 |

## New eval sets (arm B)
`eval_data/probes_hotpot_{gold_joined,distr_joined,g10_shuf}`, 1,500 rows each, built by `scripts/probes/build_groups.py`
(job 188465). The builder first rebuilds the old one-chunk set `text_hotpotqa_1500_g10` and matches it on 1,500/1,500 rows.

## Findings
(none yet)

## Waiting on a new eval set
- Photo text vs pixels (arm F) cannot run on the existing photo set: its rows carry no OCR text (`docs=[""]`). It needs
  a new set with TextVQA's OCR tokens (500 questions) before it can run.

## Closed
(none yet)
