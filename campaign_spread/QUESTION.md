Owner's question (2026-11-17, in the owner's words): "i also need a camp to investigate why the spread layout was not
validated: in which cases is it ok, not ok. try some probes to better understand the results/evaluations. and propose
and test hypothesis to get it working."

What "not validated" refers to (from the record; verify every number before the law quotes it):
- Reduced budget, one variable (1M-text stage 1 -> stage-2 v4kd 100k -> 28k finetune; only the <MEM> layout differs):
  spread - tail, paired judge, 1,500 Q per set: documents +.093 / +.075 (seeds 42 / 43), photos +.023 / +.021.
  Models: spread chain expQ/ft_mm_4b_183814 / 183816, tail chain expQ/ft_mm_4b_183538 / 183539 (HANDOFF.md, Models table).
- Full budget (5M-text spread stage 1 -> ONE stage-2 pass 160k -> 28k finetune), expQ/ft_mm_4b_184778 / 184780, vs the
  production tail models 174908 / 176242: documents +.011 [-.014,+.035] / +.024 [-.002,+.048], photos -.009 / +.010:
  equal within noise, so the +.08 did not carry to full budget (EXPERIMENTS.md line 518). That comparison is NOT one
  variable: production went through three stage-2 passes (170810 + 172617 + v4kd) and possibly another stage 1. The
  ledger's reading "stage-2 data covers what the layout gave" is a hypothesis nobody tested.
- Other spread vs tail facts to explain or use: under spread a long chunk is free at inference (one 1,270-token chunk
  .617 vs ten chunks .579; tail .399), MRQA one long chunk spread .694 vs tail .639; dropping half the slots costs spread
  -.12 on documents vs tail -.04 (docs/multimodal_results.md); training stage 1 with 1,024-token chunks under spread
  LOST (campaign_longchunk); campaign_probes (all on spread models) found the most-attended eighth of the slots keeps the
  plain score and showing the picked page raw recovers the document gap; campaign_longdoc (running now, inference only)
  diagnoses where long inputs lose.
Read: HANDOFF.md, docs/multimodal_results.md, docs/architecture_review.md, docs/probe_campaign_2026-11.md,
docs/longdoc_diagnosis.md, EXPERIMENTS.md (expmon ledger), campaign_probes/ and campaign_longchunk/ (STATE.md,
known_issues.md), campaign_longdoc/CAMPAIGN.md (a live sibling: share its eval sets and reference cells, never its jobs).

What the owner asks for, in order: (1) where spread is better, equal or worse than tail (per data source, page type,
document length, number of pages, question type, slot position, text vs image vs mixed) and whether the evaluation
itself hides or inflates the difference; (2) probes, inference-only first, to understand those results; (3) hypotheses
for why the gain vanished at full budget and how to get it back, each tested, cheapest first; training arms only after
a probe supports them, one variable per arm.
Standing rules of this owner: probe before training (an inference probe on the trained models before any training arm);
one variable per round (never stack a second change into a running arm); every capability on text, image and mixed
documents; paired judge on >= 1,500 Q with 2 seeds for any training claim under ~.03; every results table names its
metric and direction, the compression rate, and the uncompressed baseline on the same questions. Arming is the owner's
decision.
