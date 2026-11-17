# Known issues — campaign spread   (symptom → verdict → action; check here BEFORE diagnosing)

The LLM wake adds one row per NEW symptom it resolved. Seeded from the splare, rrk_vac, navmem and variety retros.

| symptom | verdict | action |
|---|---|---|
| gpu-be job shows CANCELLED then RUNNING again, Restarts>0, checkpoints keep advancing | benign preemption + requeue | verify world size (`sacct -j J -o AllocTRES`); no action |
| many probe/eval jobs FAILED within seconds, `OSError: Repo id must be in the form` | snapshot self-clean race, benign | ignore; re-run later only if the metric is needed |
| RUNNING, log empty for >10 min, `↳ hint` from expmon check | block-buffered stdout in the launcher (python \| grep/tee) | `expmon occupancy J`; if GPUs are busy, no action; fix the launcher (`python -u`) at the next relaunch |
| `starved util=NN%` on a run older than 1 h | wrong bs/ga or a data-loading bottleneck — the 15%-for-34-days class | `expmon occupancy`; kill + `expmon optimize` before any relaunch; never "just relaunch" |
| TIMEOUT at step S of N | `--time` sized from a different grad-accum or a guess | relaunch with `--time` = measured s/step × remaining × 1.25; TimeLimit can only be lowered in place |
| `Killed` / cgroup OOM in .err, no `CUDA out of memory` | HOST-RAM OOM (data prep), not GPU | raise `--mem`, not a smaller batch |
| loss pinned at ln(#candidates), or grad→0 and l0→0, or NaN | collapse / no-learning / divergence | `expmon kill`, `expmon verdict J closed "<why>"`; never retry the same cell |
| `gpucap free <type>` prints "total: 0" for hours while squeue shows that GPU type running | the type string matches no gres — a gate failing CLOSED | fix CAMP_GPU_TYPE in campaign.conf (owner), BLOCKED.md |
| LLM wake rc≠0 within seconds, tick log < 1 KB | login / auth expired | ALERT is already written; the owner must run `claude` once interactively |
| `sbatch: error: Unable to open file launchers/…` | launcher path relative to the wrong cwd | use the absolute path (expmon sweep/optimize now resolve it) |
| a comparison between arms differs in something not listed in CAMPAIGN.md Arms | confounded — not a result | list the difference, re-run one arm with it held constant, or downgrade the claim to "consistent with" |
| N jobs FAILED in < 60 s on ONE node, `cudaErrorDevicesUnavailable` / `device handle` in the log, `gpu-unavailable` from check | poisoned node: its GPUs are held by other users' processes, every landing job dies at model load and frees the slot for the next victim (navmem lost 10 jobs in 4 h) | relaunch with `--exclude=<node>` (check names it: `node <n>: N fast deaths`); it is a NODE fault, not a code crash |
| `expmon logs <id> -e` says no .err file | the launcher uses `--output` only: stderr is merged into the .log | read the .log |
| the same job shows `✗` for hours after you diagnosed and `verdict closed` it, then everything goes green at once | the check's 6 h sacct window aged out; green is not recovery | rely on verdicts and STATE.md, not on the colour changing |
| a recorded photo file and its re-run (same model, set, env, el9_8 nodes, two code dates) disagree on ~15% of answers (184778: 233/1,500, judge .677 vs .673) while documents re-run identical | photo answers drift with the code date / node, not only on homer-008 (184780's homer-008 photo file = its homer-016 re-run) | pair photo cells only when both ran on the same code (CAMPAIGN.md gates); never read a photo difference under ~1 pt across code dates as a result |
| a sacct `-X -D` listing of a requeued eval looks out of order | attempts of several jobs are interleaved by time; the LAST line of a job is the attempt that wrote the file (check against the file mtime) | slice_pairs.py takes the last attempt per job; confirm with `ls --time-style=full-iso` when it matters |
| `expmon probe --from-json` with two `--map` flags records only the last key (e.g. only n=1500) | known expmon defect (gripe 8377084c) | pass ONE `--map a=summary.a,b=summary.b`; in zsh never put the flags in an unquoted `$VAR` (zsh does not word-split it: "unknown flag") |
| `expmon display --exp spread` writes 0 table rows while the grid has judged cells | the eval json lands in `expQ/spread/<cell>/eval/` (symlinked model dir), outside the run's outdir (gripe 43db294f) | record `judge`/`em`/`n` with `expmon probe <jid> --from-json <file>`; the case table lives in docs/spread_cases.md |
