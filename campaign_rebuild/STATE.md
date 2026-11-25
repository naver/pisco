# STATE — campaign rebuild   (rewritten WHOLESALE by every LLM wake; never append)

## TL;DR — where this leaves the question (readable as an email)
**Settled** · **Open, and running** — 3-4 one-line bullets each, every bullet a claim with its number and the job/commit that backs it
(`expmon campaign report --section leaves` prints this block from verdicts + the live grid; start from it)
- Settled (pre-arm, the pisco chat, 2026-11-25): the code and launchers of every arm are proven before any GPU-hour —
  CPU job 193564: 40 / 40 unit tests OK (`TestRebuildOptions`: options off = the old collator's RNG stream draw for
  draw vs commit 69869340), and the resolved configs of stage 2 K / S / W / P and finetunes K s42 / K s43 / L s42 /
  L s43 differ from the recorded ones only in the law's keys (`RESULT preflight_cpu_ok=1`). Values read back: S mix
  .45 / .25 / .05 / .15 / .1 on `pdfa,wiki-ss`; W window p .5, cue 16 on `pdfa,wiki-ss`; P torch_dtype float32 +
  frozen base bfloat16; seed 43 everywhere.
- Open: nothing launched yet. First: the feasibility gate (recall reproduces .702 / .921), ALONE.

## Owned jobs (jobid | arm | state | elapsed | epoch or step/total | eta | note)
- 193554 | unit tests | COMPLETED rc 1 | 0:03 | — | — | one test-side miscount (fixed in the test); superseded by 193564
- 193564 | preflight (CPU) | COMPLETED rc 0 | ~0:10 | — | — | tests + 8 config resolutions, all OK

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| feasibility gate | not started | write `scripts/rebuild/recall.sbatch` (slotcheck's, outputs under `expQ/rebuild/recall/`, any model dir as ARM); run R2 + `expQ/slotcheck/o_bf16/model` on `slotcheck_dense100` + the ref self-pair |
| K / S / W / P (stage 2) | launchers committed (`scripts/rebuild/pt_stage2.sbatch`) | after the gate: 20-step smokes (`… <ARM> 20`), then the full runs side by side, `--time` from the smoke |
| L (cleaned labels) | launcher committed (`scripts/rebuild/ft.sbatch L <seed>`) | after the gate: dump + judge the 28k teacher labels, build `data/rebuild_ft_clean_{docpages,natural}` (`mmrl_teacher_split.py build --out_pd`), the R re-run check, then L s42 / s43 |
| gate-2 finetunes | `ft.sbatch <ARM> 42` | P's finetune adds bf16 overrides (its checkpoint saves fp32); its config resolution was added to `preflight_cpu.sbatch` after 193564 — run `ft.sbatch P 42 cfg` before any P finetune |

## Health evidence this wake (numbers, not adjectives)
(none yet)

## Blockers / needs the owner
(none)
