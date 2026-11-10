# STATE — campaign freeze   (rewritten WHOLESALE by every LLM wake; never append)

## TL;DR — where this leaves the question (readable as an email)
**Settled** · **Open, and running** — 3-4 one-line bullets each, every bullet a claim with its number and the job/commit that backs it
(`expmon campaign report --section leaves` prints this block from verdicts + the live grid; start from it)
- scaffolded 2026-11-11; no wake yet

## Owned jobs (jobid | arm | state | elapsed | epoch or step/total | eta | note)
| 188526 | F smoke (feasibility gate) | submitted 2026-11-11 by the owner's chat | - | 0 / 20 steps | ~45 min | `expQ/freeze/ft_freeze.sh frozen 42 20`; out `expQ/freeze/ft_frozen_s42_smoke` |
| 188527 | module-move check of the smoke | PENDING (afterok 188526) | - | - | 5 min after | `expQ/freeze/check_module_move.sh <stage-2 model> <smoke model>`; log `check_move_freeze_188527.out` |
Unit tests of the flag: job 188517, 12 + 33 + 5 pass (commit d167725). Do not relaunch the smoke: read these two.

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none)

## Open arms and the next decision for each
(copy the Arms table from CAMPAIGN.md with the current status per row)

## Health evidence this wake (numbers, not adjectives)
(expmon check / grid / occupancy facts; util%, epochs seen, probe vs incumbent at the same epoch)

## Blockers / needs the owner
(none — or point to BLOCKED.md)
