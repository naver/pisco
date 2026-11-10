# STATE — campaign longchunk   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-11 02:00 (first working wake; the 00:30 / 01:00 / 01:30 wakes died on the account's 5-hour quota)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Stage-1 corpus = MS MARCO documents: 43.4% of documents ≥ 1,024 tokens, 20.4% ≥ 2,048 (107,120 of 3.2M sampled) — data job 188358, closed. SmolLM2 rejected (23.8% / 8.3%) — data job 188339, closed.
- No training result yet: no arm has a number.

**Open, and running** (all queued 2026-11-11 02:03, none started yet)
- Fit + speed test at the 2,048-token chunk settings (50 steps, 1 h, 4 GPUs) — job 188420, PENDING (Resources).
- Fit + speed test at the 1,024-token chunk settings (50 steps, 1 h, 4 GPUs) — job 188421, PENDING (Priority).
- Control C256 stage 1 (256-token chunks, 1M documents, 5,208 steps, 23 h) — job 188422, waits for 188421 to succeed.

## Owned jobs (jobid | arm | state | elapsed | progress | eta | note)
| job | arm / stage | state | elapsed | progress | eta | note |
|---|---|---|---|---|---|---|
| 188420 | L2048 fit-and-speed test (gate b) | PENDING | 0 | 0 / 50 steps | ≤ 1 h after start | `pt_text_4b_long.sh 1000000 2048 4096 3 16 ++hf_training.max_steps=50`, partition gpu |
| 188421 | L1024 fit-and-speed test (extra, from the smoke slack) | PENDING | 0 | 0 / 50 steps | ≤ 1 h after start | `… 1000000 1024 2048 6 8 ++hf_training.max_steps=50`; measures s/step for L1024's `--time` |
| 188422 | C256 stage 1 (control) | PENDING (afterok:188421) | 0 | 0 / 1M documents | ~19 h after start | `… 1000000 256 512 24 2`; out dir `expQ/longchunk/pt_text_1M_cml256_188422` |

Recipe file `expQ/longchunk/pt_text_4b_long.sh`, md5 8d3745e9cc343b6fb30d8d4a2bc2cd8c. It is NOT in git: `expQ/` is gitignored and the tick does not force-add `CAMP_COMMIT_PATHS` (griped). The md5 is the record until it is committed.

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none. 188419 `eval_mm_4b.sh` belongs to the sibling campaign `probes`.)

## Open arms and the next decision for each
| arm | chunk / text kept per doc (tokens) | text tokens seen at 1M docs (mean kept, data job 188358) | status | next decision |
|---|---|---|---|---|
| C256 control | 256 / 512 | ≈ 0.45 B (447 per doc) | stage 1 queued (188422) | after stage 1: slot-ownership probe (`expQ/archq2/probe_mixer.sh`, control .977 on SmolLM2), then stage 2 → 2 finetunes → evals + judge |
| L1024 | 1,024 / 2,048 | ≈ 1.04 B (1,038 per doc) | fit test queued (188421) | read s/step + fit; launch stage 1 after C256 ends (one stage 1 at a time), `--time` = s/step × 5,208 × 1.25 |
| L2048 | 2,048 / 4,096 | ≈ 1.30 B (1,304 per doc) | fit test queued (188420) | fits at BS 3 → launch after L1024; OOM → one retry at BS 2 / ACCUM 24; OOM again → close "does not fit" |
| SmolLM2 reference | 256 / 512 | ≈ 0.39 B (394 per doc) | done (180486 → 183537 → 183814 / 183816) | reported as a row only, never compared |

Known confound (report, do not fix): the long arms see 2.3× / 2.9× more text tokens than the control at the same 1M documents.

## Health evidence this wake (numbers, not adjectives)
- No campaign GPU-second had run for 85 h because the law's first step (gate b) had never been launched; the earlier wakes all died on quota before acting. Fixed: three jobs queued.
- C256 `--time` 23 h = 11.06 s/step (180486, SmolLM2, 394 tokens/doc) × 447/394 (MS MARCO text kept per doc at the 512 cap) × 5,208 steps × 1.25.
- The two fit tests get the law's 1 h. At 2,048 tokens a step may take ~60 s, so 50 steps may not finish in 1 h. The log's s/it still gives the speed; a TIMEOUT there is not a failure of the fit.
- Queue: 0 nodes with 4 free A100 now; `gpucap land a100 4` says the `gpu` partition reaches one by preempting gpu-be jobs. User cap on `gpu` is 8 GPUs: the two tests use all 8 while both run.

## Blockers / needs the owner
(none)
