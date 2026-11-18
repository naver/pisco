# BLOCKED — campaign spread — 2026-11-19 02:30 (owner checkpoint before arm E)

## 1. Train the missing tail control (arm E)?
In plain words: if you say yes, the loop trains one tail-layout chain (one stage-2 pass, then the finetune) to learn whether the spread layout's +8-9 pt document gain at the small budget survives at the full budget; it costs about 44 GPU-hours (60 if the first seed shows a gain) and 2-3 days on the shared `gpu` partition.

**What every inference arm found** (judge accuracy ↑, spread − tail on the same 1,500 questions, seeds 42 / 43, 8 tokens
per slot; full table in `docs/spread_cases.md`):

| case | slots per question | spread − tail s42 / s43 (95% CI) | uncompressed ↑ | verdict |
|---|---|---|---|---|
| HotpotQA, 10 passages compressed as ONE chunk | 173 | +.260 (+.233, +.289) / +.247 (+.219, +.275) | .752 | better |
| MRQA, one long passage per question | 53 | +.084 (+.060, +.109) / +.098 (+.075, +.122) | .807 | better |
| HotpotQA, 10 passages as 10 chunks | 177 | +.001 / −.004; pooled −.001 (−.020, +.018) | .752 | tie (exact-match favours tail by 2 pt) |
| photos | 53 | −.007 / +.010; pooled +.001 (−.014, +.018) | .717 | equal |
| documents, 4 pages | 431 | +.011 (−.014, +.035) / +.024 (−.002, +.048); pooled +.017 (−.002, +.036) | .809 | not resolvable |
| documents, 10 / 20 pages | 1,077 / 2,153 | no growth over 4 pages (−.004 / −.007; +.009) | .801 / .754 | gap does not grow |

- What the slots hold (stage-2 models, inference probe): spread slots stay tied to their own text chunk at full budget
  (ownership .972 vs .763 for the three-pass tail model; .977 vs .799 at the small budget). The extra stage-2 data did
  not make the tail slots like the spread ones.
- Why the small-budget gain vanished, as far as existing models tell: from small to full budget the tail models gained
  +.093 / +.048 on documents, the spread models +.011 / −.003. But the full-budget tail models had THREE stage-2 passes
  and the spread ones ONE, so this is not one variable. Arm E is the one-variable test.

**Arm E.** Train tail with the spread chain's exact recipe and ONE stage-2 pass: init
`expQ/qwen35_4b_pt_5M_143228/model`; stage 2 = job 184777's recipe; finetunes = 184778 / 184780's. Launchers are
written: `scripts/spread/pt_mm_tail5m.sbatch` and `scripts/spread/ft_tail5m.sbatch` (`bash -n` OK; each checks that its
config differs from the recipe only in the init, max_steps and output paths, else exits 5).
- The two stage-1 configs differ only in the layout (`diff` of the two `training_config.yaml`: lines 46-47,
  `compressor_mem_layout: spread`, `mem_position_stride: 8`).
- Difference that cannot be removed: code date. The spread chain trained on commit 24b7a2c (Nov 4-8); arm E trains on
  today's code, which adds only options that are off by default (`freeze_compressor`, `freeze_part`, `connector_lr`).
- Cost, from measured runs: stage 2 30.2 h (184777) + seed-42 finetune 11.0 h (184778) + judged evals ≈ 2 h + smokes
  ≈ 1.5 h = **≈ 44 GPU-h**; seed 43 adds 13.8 h (184780) + 2 h = **≈ 60 GPU-h** in all. Used so far: 8.5 of 95.
  One training at a time on `gpu`: about 2 days to the seed-42 answer.
- What it can show: the law's gate is "spread − tail(one pass) on documents ≥ +3 pt with the CI above 0 at both seeds"
  → the layout gain does carry at full budget, and production's tie came from its two extra stage-2 passes. The other
  outcome, "inside ±2 pt", **cannot be certified on this set**: one seed's CI is .044-.053 wide and the seed-pooled one
  .038, so "equal" needs a difference under ~.1 pt. And no second document set exists in the same sources (only 365
  unused questions: 56 slides, 218 infographics, 91 MP-DocVQA). So arm E answers one question: does the gain carry at
  ≥ 3 pt — yes, or not shown.

**Options**
| option | what runs | GPU-h | what you learn |
|---|---|---|---|
| A (recommended) | stage 2 → seed-42 finetune → documents, photos, MRQA, one-chunk evals; the seed-43 finetune ONLY if seed 42 shows documents ≥ +3 pt with CI > 0 | ≈ 44 (≈ 60 if seed 42 passes) | yes / not shown, at the 3-pt level |
| B | the same, both seeds at once | ≈ 60 | the same answer, ~1 day sooner when seed 42 passes |
| C | no arm E; close with the case table | 0 | the vanished gain stays "consistent with tail catching up through budget" |

Why the seed-43 stop rule is safe: "better" needs both seeds; if seed 42 misses the bar, seed 43 cannot make the case
"better", and "equal" is out of reach on this set anyway.

**Fixes (arm F): none proposed.** The one large effect (long text: +8 to +26 pt) needs no training: spread already has
it. Nothing in the probes points to a one-setting recipe change for the gating sets.

**Other items, for information (no answer needed)**
- Second document set (option B of the "equal" test): not available from the same sources; it needs a new source.
- Infographics lean (+.050 at both seeds, found after the fact among three sources): a repeat could use only 218 new
  questions, under the law's 300-question floor, so it stays provisional and is not run.
- Serving: spread reads ten passages compressed as one chunk as well as ten separate chunks (+.006 / +.023); tail
  loses 23-25 pt. No default change is asked.

**Recommendation: option A, ≈ 44 GPU-h (≈ 60 if seed 42 shows the gain).**

## 2. Count "better under every scoring" as better, even when the sizes differ?
In plain words: if you say yes, "spread is better on long text" (+25 pt on one HotpotQA chunk, +8-10 pt on MRQA) becomes a settled finding instead of a provisional one; it costs nothing and changes no measurement.

- The law's instrument rule (arm J) asks every deciding case to keep its sign AND stay within 2 pt under the rubric
  judge, the project prompt and exact match (EM); otherwise the case is "judge-sensitive".
- One chunk: rubric +.260 / +.247, project prompt +.271 / +.253, EM +.151 / +.152. MRQA: rubric +.084 / +.098, project
  prompt +.070 / +.103, EM +.057 / +.074. Under every scoring the sign holds AND the "better" bar holds
  (≥ +3 pt, CI above 0, both seeds). Only the size differs, because both models rarely hit the exact answer string.
- Options: (a) apply the 2-pt rule to the verdict: a case is "better" when every scoring gives "better"; the size range
  is reported next to it. (b) keep the literal rule: these cases stay "judge-sensitive in size".
- **Recommendation: (a). 0 GPU-h.** It changes no number and no judge; it only names the verdict.
