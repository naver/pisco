# STATE — campaign slotcheck   (rewritten WHOLESALE by every LLM wake; never append)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Check 1 (arm C, feasibility gate PASS): the finetune collator's 4-document cap never removes ALL answer pages of a
  trained row — 0 / 20,000 document rows, 0 / 8,000 photo rows → not a cause of the document gap. Beside it: 1,910 /
  20,000 document rows (9.55%; all Docmatix, 19.1% of its 10,000) lose one of their two source pages; the record does
  not say which page holds the answer. Job 191663 (CPU audit, 56 s, verdict closed). Table: `docs/slot_checks.md`.

**Open, and running**
- Check 2a (arm P, padding probe): plumbing cell 191671 done (exit 0, 6 pages, controls pass: repeat cos 1.0000 /
  max abs 0; another page cos .742). provisional, not a read: one page left-padded by 17 / 63 tokens moved its
  `<MEM>`s to median cos .930 / .941 (min .657); a 4-page batch padded to 64: median .996, min .685 (n = 2 questions).
  The single-page number is likely a probe fault: installed transformers zeroes pad states in Qwen3.5's
  linear-attention layers only when the batch has > 1 row (`modeling_qwen3_5.py:198`), so a 1-row padded batch leaks
  pads into the recurrent state. That same 1-row case happens in TRAINING for photo rows (1 image, padded to 64), never
  in the eval (no pads at batch size 1). Full cell 191672 (120 pages, same design) running on homer-011.
- Checks 2b, 3, 4: not started (2b only if 2a is not inert).

## Owned jobs (jobid | arm | state | elapsed | epoch or step/total | eta | note)
| job | arm | state | note |
|---|---|---|---|
| 191663 | C (check 1) | COMPLETED 0:0, 56 s, homer-017 | verdict closed; MaxRSS 0.7 GB |
| 191671 | P (2a) plumbing | COMPLETED exit 0, ≈ 4 min, homer-011 | 6 pages + 2 four-page questions; decides nothing; no verdict yet |
| 191672 | P (2a) full | RUNNING homer-011 (started 17:59) | 120 pages; `--time 0:45` (unmeasured), `--mem 16G`; its single-page arm carries the 1-row confound |

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none — the running eval_mm_4b / rejudge jobs belong to spread, longdoc and qafreeze)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| C. finetune-set count (check 1) | **decided: not a cause** (0 / 20,000), 1 of 2 cells used | none (the +1 re-run is not needed) |
| P. padding (check 2a / 2b) | 2a running (2 of 6 cells) | when 191672 ends: (1) read `compressor_spread_position_ids` (pisco/model.py) for left-pad handling; (2) patch `pad_probe.py` to measure two paths apart — 1-row batch (the photo TRAINING path) and ≥ 2-row batch (page + a twin row, the masked path) — and rerun (one cell); a 2a read decides only on the ≥ 2-row path vs the law's bar; the 1-row leak is reported as its own finding (training-only, photo rows), a hand-off candidate, never patched here (pisco/** is never-patchable). If the ≥ 2-row path is not inert: write `scripts/slotcheck/eval_pad64.sbatch`, its LIMIT=300 invariant run, then 2 seeds × padded / unpadded |
| O. overfit capacity (check 4) | not started | write `build_dense100.py` (100 + 100 held-out dense pages of `data/mm_pretrain_pdfa_1m`) and `pt_overfit.sbatch`; 20-step smoke; config diff vs `expQ/archq3/pt_mm_5M_spread/training_config.yaml` |
| X. text slots (check 3) | not started | last (biggest cost); first shard measures the 12 GPU-h checkpoint |

## Health evidence this wake (numbers, not adjectives)
- 191663: rows read 20,000 = set size and 8,000 = set size; per-source sums = totals; image rows complete 20,000 /
  20,000 (gate 1 ≥ 95%); sampled 50 / 50 rows give min(n, 4) "Document j:" blocks and as many image chunks; planted
  rows [4,5] → dropped, [0,5] → partial, [1] → kept (gate 2). Exit 0.
- Code md5 at launch: eval_mm_4b.sh b3d1cda1, scripts/eval_mm_docqa.py dfe32aa0, scripts/judge_mm_docqa.py 02bc8748,
  scripts/paired_judge.py a3cc13b2, pisco/model.py 994cff3d; pad_probe.py d794607c, mm_collator.py 8224c06b,
  mm_utils.py c3ddfd97.
- 191671: `<MEM>` count per page identical across widths (97 at 867 tokens); model load 47 s; 6 pages compressed in ≈ 1 min.
- Budget: 3 of 32 cells, ≈ 0.2 of 30 GPU-h.

## Law edits this wake
- CAMPAIGN.md Check 1: fixed the definition of "loses its gold page" (every gold ≥ 4; partial reported apart) and that
  the gate reads the trained rows — written before the count was read (job started 17:40, edit 17:45, read 17:48).

## Blockers / needs the owner
(none)
