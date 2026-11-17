# Spread vs tail `<MEM>` layout — where each wins (campaign spread)

Rewritten whenever an arm decides. Last rewrite 2026-11-17 22:45, after arm A (audit of the recorded files, job 191454,
tables `expQ/spread/audit_191454/audit.md`) and arm B seed 42 (long text chunks at full budget, jobs 191455-191457). Metric everywhere: judge accuracy ↑ (Qwen3-8B rubric judge, 1 = correct),
paired on the same questions; difference = spread − tail; CI = paired bootstrap 95%; b/c = McNemar discordant pairs
(spread-only right / tail-only right). Compression rate 8 tokens per slot in every row. "Power" = 2.8 × paired SE = the
smallest difference the set detects 80% of the time.

Models. Reduced budget (1M-text stage 1 → one 100k stage-2 pass → 28k finetune, ONLY the layout differs): spread
183814 (s42) / 183816 (s43), tail 183538 / 183539. Full budget: spread 184778 / 184780 (5M text → one 160k stage-2
pass) vs the production tail models 174908 / 176242 (three stage-2 passes) — NOT one variable.

## 1. Case table (recorded files; arm A)

| case | budget | n | slots per question | spread ↑ s42 / s43 | tail ↑ s42 / s43 | difference s42 (95% CI) | difference s43 (95% CI) | b/c s42 / s43 | uncompressed ↑ | power (pt) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| documents, 4 pages | reduced | 1,500 | 431 (108 per page) | .599 / .618 | .507 / .543 | +.093 (+.069, +.115) | +.075 (+.051, +.099) | 238/99 · 220/107 | .809 | 3.4 | **better** (recorded verdict) |
| photos | reduced | 1,500 | 53 | .671 / .671 | .648 / .649 | +.023 (+.002, +.043) | +.021 (+.002, +.039) | 139/105 · 114/82 | .717 | 2.9 | +2 pt, CI > 0 at both seeds, below the 3-pt bar |
| documents, 4 pages | full | 1,500 | 431 | .610 / .615 | .599 / .591 | +.011 (−.014, +.035) | +.024 (−.002, +.048) | 177/161 · 193/157 | .809 | 3.4 | not resolvable here: CI width .049 / .050 > the 4-pt "equal" band |
| documents, infographics only | full | 500 | — | .480 / .480 | .430 / .430 | +.050 (+.004, +.096) | +.050 (+.008, +.096) | 80/55 · 76/51 | .726 | 6.5 | provisional: better (s43 pair spans two code eras) |
| documents, multi-page (MP-DocVQA) | full | 500 | — | .666 / .678 | .702 / .682 | −.036 (−.080, +.006) | −.004 (−.050, +.040) | 52/70 · 68/70 | .926 | 6.2 | not resolvable here |
| documents, slides | full | 500 | — | .684 / .686 | .666 / .660 | +.018 (−.018, +.052) | +.026 (−.010, +.062) | 45/36 · 49/36 | .774 | 5.0 | not resolvable here |
| photos | full | 1,500 | 53 | .677 / .685 | .685 / .675 | −.009 (−.028, +.011) | +.010 (−.011, +.030) | 108/121 · 125/110 | .717 | 2.8 | not resolvable here (CI width .039 / .041) |
| MRQA, one long passage (~900 tokens) | full | 1,500 | 53 | .718 / — | .631 / — | +.087 (+.063, +.112) | missing (176242 has no file) | 245/114 | .807 | 3.4 | one seed: needs 176242's cell |
| MRQA, one long passage | reduced | 1,500 | 53 | .694 | .639 | +.055 (+.031, +.077) | — | 201/119 | .807 | 3.3 | one seed |
| MRQA, split into 256-token chunks | reduced | 1,500 | 55 | .695 | .689 | +.006 (−.015, +.026) | — | 137/128 | .807 | 2.9 | one seed |
| HotpotQA, 10 passages as 10 chunks | reduced | 1,500 | 177 | .579 | .568 | +.011 (−.012, +.034) | — | 162/145 | .752 | 3.3 | one seed; spread side ran on homer-006 |
| HotpotQA, 10 passages as 3 chunks | reduced | 1,500 | 174 | .578 | .495 | +.083 (+.058, +.109) | — | 267/142 | .752 | 3.5 | one seed |
| HotpotQA, 10 passages as ONE chunk | reduced | 1,500 | 173 | .617 | .399 | +.217 (+.189, +.245) | — | 418/92 | .752 | 3.8 | one seed |
| image multiple choice | reduced | 1,000 | 21 | .864 | .838 | +.026 (+.010, +.042) | — | 48/22 | .895 | 2.3 | one seed; spread side on homer-008 |
| HotpotQA, 10 passages as 10 chunks | full | 1,500 | 177 | .603 / running | .603 / running | −.001 (−.023, +.022) | running | 151/152 | .752 | 3.4 | reported only: the spread side is the freeze file (188615, older eval_mm_4b.sh + model.py); today's-code re-run running |
| HotpotQA, 10 passages as ONE chunk (~1,380 tokens = 173 slots × 8) | full | 1,500 | 173 | .611 / running | .351 / running | **+.260 (+.233, +.289)** | running | 488/98 | .752 | 4.2 | s42: far above the 3-pt bar; s43 running |

Arm B, full budget, seed 42 (jobs 191455-191457, today's code, homer-017/019; 1,500 rows, judge on all, 0 unparseable,
`mems_used` equal between twins): going from ten chunks to ONE chunk costs the production tail model −.253
(−.281, −.225) and the spread model nothing (+.008, −.016 to +.031). The reduced-budget collapse of tail on one chunk
(.568 → .399) is WORSE at full budget (.603 → .351): three stage-2 passes did not teach tail to read a long chunk.
Same pair under other metrics (one chunk, s42): EM +.151, match +.199, ANLS +.174 — same sign, all CIs above 0.

## 2. Evaluation table — what the gating sets can resolve

| set | CI width of one seed's difference | power (pt, 80%) | both right | both wrong | what this means |
|---|---|---|---|---|---|
| documents 1,500 Q | .044–.050 | 3.3–3.6 | .43–.52 | .28–.33 | "equal" (CI inside ±2 pt) cannot be reached: the CI is wider than the band |
| photos 1,500 Q | .037–.041 | 2.6–3.1 | .58–.60 | .24–.27 | at the band: "equal" reachable only if the difference is ~0 |
| text 1,500 Q | .041–.056 | 2.9–4.2 | .29–.60 | .21–.33 | as documents (HotpotQA one chunk full budget: both right .285, both wrong .324; ten chunks .502 / .296) |

- Re-run noise (same model, same set, two dates; `expQ/spread/rerun_checks/photo_drift.md`, `scripts/spread/compare_answers.py`):
  documents of 184778 1,500/1,500 answers identical (11-09 vs 11-11); photos of 184778 moved 233/1,500 answers, 85 judge
  flips (45/40), .677 → .673 (184783 homer-011 11-09 vs 188602 homer-016 11-11); photos of 184780 identical although the
  original ran on homer-008. So the node rule of
  campaign_probes does not explain photo drift; a photo pair across code dates carries about ±.5 pt of re-run noise.
- Judge prompt / EM / ANLS per deciding case: arm J (not run yet).

## 3. Mechanism table
Arm D (not run yet).

## 4. Hypothesis table

| hypothesis | probe or arm | n | number | gate | verdict | what it would change |
|---|---|---|---|---|---|---|
| The +.08 vanished because the spread chain lost accuracy at full budget | arm A, budget pairs (full − reduced, same layout; NOT one variable: stage 1 5M vs 1M and stage 2 160k vs 100k) | 1,500 × 2 seeds | spread +.011 (−.011, +.033) / −.003 (−.026, +.020); tail +.093 (+.068, +.117) / +.048 (+.025, +.072) | — (descriptive) | **refuted** (provisional): spread stayed where it was, the tail models gained | the question becomes "why does the tail layout need the extra budget and spread not" |
| The tail models gained on multi-page documents most | same pairs, per source | 500 × 2 seeds | tail gain MP-DocVQA +.162 / +.090, slides +.076 / +.040, infographics +.040 / +.014 | — | consistent with (provisional) | arm C (many pages) and arm E target MP-DocVQA |
| Spread is better on infographics at full budget | full pair, per source | 500 × 2 seeds | +.050 / +.050, CI > 0 both | better bar | provisional: the s43 tail file is from an older code era; re-run 176242 on today's code | a per-source claim the gating mean hides |
| Spread is better on one long text passage at full budget | full MRQA long | 1,500, s42 only | +.087 (+.063, +.112) | better bar, both seeds | open: s43 tail cell missing | a serving rule "compress long text as one chunk" |
| Tail cannot read one long text chunk even at full budget; spread can | arm B, HotpotQA one chunk vs ten chunks | 1,500, s42 (s43 running) | spread − tail one chunk +.260 (+.233, +.289); tail one − ten −.253; spread one − ten +.008 | better bar, both seeds | on track to **better** (s42 only) | a serving rule "spread compresses long text as one chunk"; the gating eval (short chunks) never asks this |
| The gating eval cannot show "equal" | arithmetic on the CIs | 23 pairs + arm B | CI width .044–.050 documents, .041–.056 text, .037–.041 photos | — | **confirmed** (arm A) | the owner chooses how "equal" is tested (BLOCKED.md) |
