# Known issues — campaign freeze   (symptom → verdict → action; check here BEFORE diagnosing)

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
| a 20-step smoke's logged `'loss'` at step 10 is far (here .086 lower) from the full run's step-10 loss | `warmup_steps: .05` is a RATIO of max_steps: 1 warm-up step of 20 vs 22 of 438, so the smoke trains at ~3x the lr over steps 1-10; the comparison is confounded, not a fault | compare a smoke only with a smoke of the same `max_steps` (the unfrozen 20-step smoke, job 188534) |
| `check_module_move.sh` prints no `connector.norm_gain` row | the gain is absent from the stage-2 init (`connector_out_norm: false`) and retrofitted at load at `connector_out_norm_init`; the script compared only keys of the init, so the gain was silently skipped | patched 2026-11-11 (6 lines): a retrofitted gain is compared with its init value from the trained `config.json` (jobs 188578 / 188579) |
| a TRAINABLE connector norm gain reads `changed 0.0%` after a bf16 finetune (unfrozen smoke 188579) | bf16 lost update: near .461 the bf16 step is 2^-9 = .002, one Adam update at lr 2e-5 is ~1e-5, so the stored gain never changes | expected, not a leak of the frozen flag; the frozen-vs-reference difference is therefore compressor LoRA + merger + connector MLP only |
| `(eval):1: ==== not found` or `no matches found: <glob>` from a Bash call | the wake's shell is zsh: a word starting with `=` is a command lookup, and an unmatched glob aborts the whole line | quote `'====='` separators; test globs with `ls … 2>/dev/null` or use `find`; avoid zsh-only arrays in a command (the rm safety check refuses lines it cannot parse) |
