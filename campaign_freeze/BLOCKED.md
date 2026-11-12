# BLOCKED — campaign freeze (2026-11-13 04:45)

## May the loop cancel the queued connector-frozen smoke test so the second seed of the compressor-frozen run starts first?

**What.** The split test has two arms. Arm A1 freezes the compressor (only the connector and the decoder LoRA train).
Arm A2 freezes the connector. The law says A1 first. A1 seed 42 trains now (healthy: held-out loss .459 at a
quarter of the run, vs .513 for the reference run). A1 seed 43 cannot start, because the campaign allows 2 trainings
in flight, and the second slot is held by A2's 20-step smoke test (job 189065).

**Why it is stuck.** That smoke test sits on the preemptible `gpu-be` partition. It was preempted twice
(Restarts = 2), and each time it restarts from zero. Its new start estimate is 2026-11-14 03:45, about 23 h away.
Until it ends, A1 seed 43 waits too. The law lets the loop cancel its own jobs (hard rule 2), but its list of
pre-authorised actions has no line for "cancel a queued job to change the order", so the loop asks.

**Options.**
| option | what happens | GPU-hours lost | A1 result (both seeds judged) |
|---|---|---|---|
| 1. swap (recommended) | cancel the smoke test and its 2-min check; start A1 seed 43 now on `gpu` (7 of your 8 slots, never the last); re-run the smoke test on `gpu` when A1 seed 42 ends (~10:20 today) | 0 (the smoke test has no saved work) | ~2026-11-13 evening |
| 2. wait | keep the order | 0 | ~2026-11-14 evening or later, if the smoke test is preempted again |

**Recommendation: option 1.** Cheap (0 GPU-hours lost) and reversible. To approve, write `verdict: swap` in a
`REQUEST` file (`expmon campaign resolve`), or answer `expmon answer <id> "swap"`.
