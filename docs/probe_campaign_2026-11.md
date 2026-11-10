# Probe campaign 2026-11 — why the compressed model loses, and what recovers it without training

Law: `campaign_probes/CAMPAIGN.md`. Live state: `campaign_probes/STATE.md`. This page is rewritten whenever an arm decides.
Metric everywhere: judge accuracy ↑ (share of questions the Qwen3-8B rubric judge accepts), paired against the same model's
plain cell on the same questions (95% bootstrap CI, McNemar p). Compression: 1 slot per 8 visual or text tokens
(`compr_rate: 8` in the reference models' config); on the document set at PX=1003520 the plain cell uses 430.7 slots per
question on average (646,042 slots / 1,500 questions, 4 pages each, measured on job 188419).

## Status (2026-11-11 03:30 KST)
- **Feasibility gate (is the judge reproducible?) — PASSED.** 8 probe cells submitted, all waiting in the queue.

| cell | model | set | n | judge accuracy ↑ | baseline judge accuracy ↑ | difference (95% CI) | McNemar p | health (unparseable, rows, mems_used ratio) |
|---|---|---|---|---|---|---|---|---|
| gate: plain re-run (job 188419) | 184778 (seed 42) | documents, 1M px, 4 pages | 1,500 | .610 | .610 (recorded, job 184781) | +.000 [+.000, +.000] | 1.0 (0 discordant pairs) | 0, 1,500, 1.000 |

Per source, both runs: infographics .480, multi-page .666, slides .684 (500 questions each). All 1,500 answers are
byte-identical: decoding is greedy, so the pipeline is deterministic and every paired difference below comes from the probe.

## Arms in flight (no result yet)
| arm | cells waiting | what it answers | training decision it feeds |
|---|---|---|---|
| A. use-recovery on documents (seed 42) | evidence-then-answer, gold page only, picked page raw | how much of the −.20 vs raw comes from reading vs picking vs using | evidence-then-answer labels in stage 2; route-then-expand at serving |
| C. reasoning vs reading on text (seed 42) | plain one-chunk baseline, evidence-then-answer | is the text loss (DROP, DuoRC, RACE) room-to-reason or missing content | evidence-then-answer teacher labels for text |
| D. page picking without training | attention-route re-runs on 10-page documents and HotpotQA, then fusion | can the decoder's attention beat the digit route prompt with no training | whether routing rows need training |

## Findings
(none yet)

## Waiting on a new eval set
- Photo text vs pixels (arm F) cannot run on the existing photo set: its rows carry no OCR text (`docs=[""]`). It needs
  a new set with TextVQA's OCR tokens (500 questions) before it can run.

## Closed
(none yet)
