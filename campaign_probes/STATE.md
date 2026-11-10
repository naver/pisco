# STATE — campaign probes   (rewritten WHOLESALE by every LLM wake; never append)
last wake: 2026-11-11 ~02:05-02:35 KST (first wake that ran; the 00:30 / 01:00 / 01:30 wakes hit the Claude 5-hour usage limit)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- nothing yet. The campaign has not measured anything of its own; it waits on the instrument check below.
- plumbing only: the page-picking fusion script reproduces the saved route numbers exactly — route .8607, attention .8853,
  n=1500 rows, model 183814, 10-page document set (selftest job 188427 vs probe 185577's summary.json).

**Open, and running**
- feasibility gate (is the judge reproducible at the .02 level?): re-run of the reference model's document cell
  (model 184778, 1,500 questions, PX=1003520) = job 188419, PENDING on gpu-be (Priority, SLURM start estimate 09:55 KST).
  Pass = judge within ±.01 of the recorded .610 (job 184781) and the paired CI inside ±.02. Every arm waits on it (law: "first, alone").

## Owned jobs (jobid | arm | state | elapsed | epoch or step/total | eta | note)
| 188419 | feasibility gate | PENDING gpu-be (Priority) | 0 | 0/1500 rows | start ~09:55, then 30-100 min | output → `expQ/probes/rep_184778/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json` (symlinked model, so the recorded file is NOT overwritten); --time 4:54:00 = slowest measured run of this cell (184781, 98 min) × 3 |
| 188427 | arm D plumbing | COMPLETED, verdict closed PASS | < 1 min | — | — | `route_fusion.py --selftest` |

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none. The queue also holds the sibling `longchunk` campaign's 188420 / 188421 / 188422 on `gpu` — not ours, never touched.)

## Open arms and the next decision for each
| arm | status | next action (after the gate passes) |
|---|---|---|
| gate | running (188419) | pair with `scripts/paired_judge.py` against `expQ/ft_mm_4b_184778/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json`; pass → launch arms; fail → campaign STOPS with "the judge is not reproducible at the .02 level" |
| A. use-recovery on documents | not started | 4 probes (STYLE=expand, GOLD=1, RAW=route1, MQ=int8) × seed 42 first, each on a symlinked model dir `expQ/probes/<cell>/model` |
| B. long-chunk mechanism | not started | `scripts/probes/build_groups.py` not written yet |
| C. reasoning vs reading on text | not started | check whether `eval_mm_docqa.py` has a thinking-mode flag |
| D. page picking | code ready: `probe_attn_route.py` now saves the full digit log-probs + query_id per row; `route_fusion.py` (rank fusion, product, per-source choice, paired CI + McNemar vs the digit route) passed its selftest | re-run `probe_attn_route.sh` on retr10 (41 min measured) and hotpot (12 min), then `route_fusion.sbatch` on both dirs |
| E. attention-guided slot pruning | not started | `scripts/probes/prune_by_attention.py` not written |
| F. photo text | not started | 2 cells |

## Health evidence this wake (numbers, not adjectives)
- queue: gpu-be has 51 pending jobs; the gate cell sits at Priority. Not moved to `gpu`: the sibling training campaign has two
  4-GPU jobs pending there, which already fill the owner's 8-GPU cap on `gpu`; one more GPU there would delay them.
  Law rule: move to `gpu,gpu-be` if still PENDING after 6 h (≈ 08:05 KST).
- free: 3 A100 cluster-wide; disk 14 T free.
- the three failed wakes: tick logs say `You've hit your 5-hour limit` (quota, not login). known_issues.md row added; gripe filed.

## Changes this wake
- `scripts/probes/probe_attn_route.py`: meta rows also carry `digit_lp` and `query_id` (2 fields added; nothing else changed).
- new `scripts/probes/route_fusion.py` + `scripts/probes/route_fusion.sbatch` (CPU, gres=gpu:0).
- CAMPAIGN.md tightened: `--time` from the SLOWEST measured run (document cells measured 28 and 98 min); every eval cell
  runs on a symlinked model dir under `expQ/probes/`; new launchers as `scripts/probes/*.sbatch`.
- verdicts `closed` on the pre-campaign rows of the grid (185577 / 185579 / 185580 / 185581 / 185783-185785 / 188338) and 188427.

## Blockers / needs the owner
- none blocking. FYI: `eval_mm_4b.sh` and `probe_attn_route.sh` are NOT under git (`.gitignore` has `*.sh`), although the law
  calls them "committed launchers"; force-add them when convenient. The ALERT about the failed wakes is yours to clear
  (cause: usage quota, nothing to fix).
