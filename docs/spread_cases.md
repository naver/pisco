# Spread vs tail `<MEM>` layout — where each wins (campaign spread)

Rewritten whenever an arm decides. Last rewrite 2026-11-18 21:30, after arm C (many pages) and arm J (judge
prompt / EM check) decided. Tables: audit of the recorded files `expQ/spread/audit_191454/audit.md`; seed-pooled test
`expQ/spread/pooled_20261118_1730/pooled.md` (spec `scripts/spread/pooled_pairs.tsv`); arm J per metric
`expQ/spread/rejudge/pooled_20261118_2100/pooled*.md` (spec `scripts/spread/rejudge_pairs.tsv`); arm C growth test
`expQ/spread/did_20261118_2100/did*.md` (spec `scripts/spread/did_pairs.tsv`, `scripts/spread/did_pairs.py`). Metric everywhere: judge accuracy ↑ (Qwen3-8B rubric judge, 1 = correct), paired on
the same questions; difference = spread − tail; CI = paired bootstrap 95%; b/c = McNemar discordant pairs (spread-only
right / tail-only right). Compression rate 8 tokens per slot in every row. "Power" = 2.8 × paired SE = the smallest
difference the set detects 80% of the time.

Models. Reduced budget (1M-text stage 1 → one 100k stage-2 pass → 28k finetune, ONLY the layout differs): spread
183814 (s42) / 183816 (s43), tail 183538 / 183539. Full budget: spread 184778 / 184780 (5M text → one 160k stage-2
pass) vs the production tail models 174908 / 176242 (three stage-2 passes) — NOT one variable.

## 1. Case table

Full-budget rows: both sides of each seed ran the same code (today's eval launcher, or for s42 documents both recorded
files on commit 24b7a2c; documents re-run 0 / 1,500 answers different, three times).

| case | budget | n | slots per question | spread ↑ s42 / s43 | tail ↑ s42 / s43 | difference s42 (95% CI) | difference s43 (95% CI) | b/c s42 · s43 | seed-pooled (95% CI) | uncompressed ↑ | power (pt) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HotpotQA, 10 passages as ONE chunk (~1,380 tokens) | full | 1,500 | 173 | .611 / .616 | .351 / .369 | **+.260 (+.233, +.289)** | **+.247 (+.219, +.275)** | 488/98 · 477/106 | +.254 (+.227, +.279) | .752 | 4.2 | **better** at both seeds (p < .001 each); EM +.151 / +.152 |
| HotpotQA, 10 passages as 10 chunks | full | 1,500 | 177 | .605 / .593 | .603 / .597 | +.001 (−.021, +.025) | −.004 (−.027, +.019) | 156/154 · 154/160 | −.001 (−.0203, +.018) | .752 | 3.4 | no difference under either judge prompt; "equal" NOT certified (band edge). **Judge-sensitive**: EM pooled −.023 (−.039, −.007), tail ahead — spread answers are 0.4–0.6 words longer |
| documents, 4 pages | full | 1,500 | 431 (108 per page) | .610 / .615 | .599 / .591 | +.011 (−.014, +.035) | +.024 (−.002, +.048) | 177/161 · 193/157 | +.017 (−.002, +.036) | .809 | 3.4 | not resolvable here (width .038): neither equal nor better. Leans spread under every metric (project prompt pooled +.024 (+.006, +.042), EM +.012) but below the +3 pt bar |
| documents, infographics only | full | 500 | — | .480 / .480 | .430 / .430 | +.050 (+.004, +.096) | +.050 (+.008, +.096) | 80/55 · 76/51 | — | .726 | 6.5 | better at both seeds by the bars, but a post-hoc pick among 3 sources on the same 500 questions in both seeds → needs one repeat on new infographic questions |
| documents, multi-page (MP-DocVQA) | full | 500 | — | .666 / .678 | .702 / .682 | −.036 (−.080, +.008) | −.004 (−.050, +.042) | 52/70 · 68/70 | — | .926 | 6.2 | not resolvable here |
| documents, slides | full | 500 | — | .684 / .686 | .666 / .660 | +.018 (−.018, +.054) | +.026 (−.010, +.062) | 45/36 · 49/36 | — | .774 | 5.0 | not resolvable here |
| photos | full | 1,500 | 53 | .679 / .685 | .687 / .675 | −.007 (−.027, +.011) | +.010 (−.011, +.030) | 105/116 · 125/110 | +.001 (−.014, +.018) | .717 | 2.8 | **equal** under the rubric judge and EM; under the project prompt s42 −.023 (−.043, −.004), pooled −.006 (−.022, +.010) → equal not certified there |
| documents, 10 pages (6 distractor pages from other documents) | full | 1,500 | 1,077 | .607 / .608 (longdoc 191378 / 191390) | .600 / .591 (191666 / 191667) | +.007 (−.017, +.031) | +.017 (−.007, +.040) | 170/160 · 174/149 | — | .801 | 3.4 | not resolvable here; no growth vs 4 pages (arm C) |
| documents, 20 pages (16 distractor pages) | full | 1,500 | 2,153 | .599 / not run | .579 (191668) / not run | +.019 (−.006, +.043) | — | 181/152 | — | .754 | 3.4 | one seed; no growth vs 4 pages (arm C) |
| documents, infographics only, 10 / 20 pages | full | 500 | — | .458 / .476 (10 p); .454 (20 p s42) | .430 / .406; .398 | +.028 (−.014, +.070) 10 p | +.070 (+.030, +.114) 10 p | 64/50 · 76/41 | — | — | 5.9 | same 500 questions as the 4-page row, so NOT a repeat; 20 p s42 +.056 (+.016, +.096) |
| documents, MP-DocVQA only, 10 / 20 pages | full | 500 | — | .676 / .676; .648 | .712 / .680; .690 | −.036 (−.078, +.006) 10 p | −.004 (−.050, +.042) 10 p | 56/74 · 66/68 | — | — | 6.4 | not resolvable; leans tail at s42 (20 p −.042 (−.088, +.002)) |
| MRQA, one long passage (~900 tokens) | full | 1,500 | 53 | .715 / .724 (today's code, qafreeze 191675 / 191676) | .631 (recorded, older code) / — | +.087 (+.063, +.112) old pair | — | 245/114 | — | .807 | 3.4 | tail re-runs on today's code running (191779 / 191780) |
| documents, 4 pages | reduced | 1,500 | 431 | .599 / .618 | .507 / .543 | +.093 (+.069, +.115) | +.075 (+.051, +.099) | 238/99 · 220/107 | +.084 (+.065, +.103) | .809 | 3.4 | **better** (recorded verdict; the pooled test's control) |
| photos | reduced | 1,500 | 53 | .671 / .671 | .648 / .649 | +.023 (+.002, +.043) | +.021 (+.002, +.039) | 139/105 · 114/82 | — | .717 | 2.9 | +2 pt, CI > 0 at both seeds, below the 3-pt bar |
| MRQA, one long passage | reduced | 1,500 | 53 | .694 | .639 | +.055 (+.031, +.077) | — | 201/119 | — | .807 | 3.3 | one seed |
| MRQA, split into 256-token chunks | reduced | 1,500 | 55 | .695 | .689 | +.006 (−.015, +.026) | — | 137/128 | — | .807 | 2.9 | one seed |
| HotpotQA, 10 passages as 10 chunks | reduced | 1,500 | 177 | .579 | .568 | +.011 (−.012, +.034) | — | 162/145 | — | .752 | 3.3 | one seed; spread side ran on homer-006 |
| HotpotQA, 10 passages as 3 chunks | reduced | 1,500 | 174 | .578 | .495 | +.083 (+.058, +.109) | — | 267/142 | — | .752 | 3.5 | one seed |
| HotpotQA, 10 passages as ONE chunk | reduced | 1,500 | 173 | .617 | .399 | +.217 (+.189, +.245) | — | 418/92 | — | .752 | 3.8 | one seed |
| image multiple choice | reduced | 1,000 | 21 | .864 | .838 | +.026 (+.010, +.042) | — | 48/22 | — | .895 | 2.3 | one seed; spread side on homer-008 |

Arm B (closed 2026-11-18): going from ten chunks to ONE chunk costs the production tail model −.253 (−.281, −.225) at
s42 and −.229 (−.255, −.200) at s43, and the spread model nothing (+.006 (−.019, +.029) s42, +.023 (+.000, +.045) s43, McNemar 178/144, p .066). The
reduced-budget collapse of tail on one chunk (.568 → .399) is not cured at full budget (.603 → .351, .597 → .369):
three stage-2 passes did not teach tail to read a long chunk. On ten short chunks the two layouts tie within the
resolution of the test.

## 2. Evaluation table — what the gating sets can resolve

| set | CI width of one seed's difference | seed-pooled width | both right | both wrong | what this means |
|---|---|---|---|---|---|
| documents 1,500 Q | .044–.053 | .038 | .49 / .49 (full) | .28 / .28 (full) | "equal" only when the difference is under ~.1 pt; the full-budget +.017 is in the dead zone (not equal, not better) |
| photos 1,500 Q | .036–.041 | .031 | .61 / .60 | .24 / .24 | "equal" reachable when the difference is under ~.5 pt — reached (+.001) |
| text 1,500 Q, ten chunks | .046–.048 | .038 | .50 / .49 | .29 / .30 | a −.001 difference missed "equal" by .03 pt: this set cannot certify a tie |
| text 1,500 Q, one chunk | .052–.056 | .052 | — | — | resolves the +25 pt easily |

**Arm J — the same difference under four scorings** (job 191665 re-judged COPIES of the 16 today's-code files with the
project Yes/No prompt; the copies' rubric verdicts and answers equal the sources 16 / 16; re-pairing the copies'
rubric verdicts reproduces the rubric table exactly). Spread − tail, s42 / s43, then seed-pooled (95% CI):

| case | rubric judge ↑ | project prompt ↑ | exact match ↑ | ANLS ↑ | arm J reading |
|---|---|---|---|---|---|
| HotpotQA one chunk | +.260 / +.247; +.254 (+.227, +.279) | +.271 / +.253; +.262 (+.236, +.288) | +.151 / +.152; +.152 (+.129, +.175) | +.174 / +.173 | **better under every scoring**; the size depends on it (EM is 10 pt smaller: exact strings are rarer for both) |
| HotpotQA ten chunks | +.001 / −.004; −.001 (−.020, +.018) | −.001 / −.006; −.003 (−.022, +.016) | −.027 (p .010) / −.020 (p .058); −.023 (−.039, −.007) | −.030 (−.045, −.014) | **judge-sensitive**: both judges see no difference, string metrics favour tail. Spread answers are longer (3.43 vs 3.01 words s42, 3.91 vs 3.35 s43) and have 42 / 24 more judged-right-but-EM-0 answers |
| documents 4 pages | +.011 / +.024; +.017 (−.002, +.036) | +.019 / +.029 (+.005, +.053); +.024 (+.006, +.042) | +.001 / +.022; +.012 (−.007, +.031) | +.029 (+.012, +.046) | same sign and within 2 pt under every scoring; small lean to spread, never ≥ +3 pt at both seeds |
| photos | −.007 / +.010; +.001 (−.014, +.018) | −.023 (−.043, −.004) / +.010; −.006 (−.022, +.010) | −.009 / +.004; −.003 (−.017, +.012) | +.001 (−.012, +.014) | equal under rubric, EM and ANLS; not certified under the project prompt (s42 −2.3 pt). Signs flip between seeds under every scoring: no layout effect |

**Seed-pooled "equal" test** (owner decision 2026-11-18): per question, spread − tail averaged over seeds 42 and 43;
"equal" = its 95% CI inside ±2 pt AND each seed's own difference inside ±2 pt. Controls on every run: the
reduced-budget documents pair returns +.084 (+.065, +.103) (expected +.084) and every self-pair 0.000. PASS at 16:32
and at 17:45 on 2026-11-18.

Re-run noise (same model, same set, two code dates; `scripts/spread/compare_answers.py`):
- documents: 184778, 184780 and 176242 re-ran 1,500 / 1,500 identical answers.
- photos: 184778 moved 233 / 1,500 answers (85 judge flips, .677 → .673), 174908 moved 135 / 1,500 (34 flips,
  .685 → .687); 184780 and 176242 identical. A photo pair across code dates carries about ±.5 pt of re-run noise.
- text: 184778 ten chunks moved 58 / 1,500 (19 flips, .603 → .605); 184780 ten chunks identical (191630 = freeze file).
- `PX` is inert on photos: every `mm_naturalqa_1500` file has 79,322 slots at 451,584 and at 1,003,520 px.

**Arm C — does the layout gap grow with page count?** Per question, (spread − tail at N pages) − (spread − tail at 4
pages), same seed; control = arm B's one-chunk vs ten-chunk growth, recovered +.259 (+.222, +.294) / +.251 (+.215,
+.288); self-pairs 0.000. 10 pages: −.004 (−.027, +.019) s42, −.007 (−.031, +.015) s43; 20 pages s42 +.009 (−.015,
+.031) (EM +.003 / +.007 / +.013). Gate (≥ +3 pt, CI > 0, both seeds) cannot be met: **adding pages does not make the
spread layout pay more.** The 20-page s43 pair was not run (the gate is decided by s42 and both 10-page seeds).

## 3. Mechanism table
Arm D running (job 191785, `scripts/spread/mech_probes.sbatch`): text ownership and answer loss of the four stage-2
models (full budget: spread one pass `pt_mm_5M_spread` vs tail three passes `pt_mm_v4kd_4b`; reduced pair
`pt_mm_spread` / `pt_mm_base_from1M` as the control that must reproduce ownership .977 / .799 and answer loss on pages
1.138 / 1.334).

## 4. Hypothesis table

| hypothesis | probe or arm | n | number | gate | verdict | what it would change |
|---|---|---|---|---|---|---|
| Tail cannot read one long text chunk even at full budget; spread can | arm B, HotpotQA one chunk vs ten chunks | 1,500 × 2 seeds | spread − tail one chunk +.260 / +.247; tail one − ten −.253 / −.229 | better bar, both seeds | **better**, under the rubric judge, the project prompt and EM (arm J) | a serving rule "spread compresses long text as one chunk"; the gating eval (short chunks) never asks this |
| On short chunks the layouts tie | arm B, ten chunks | 1,500 × 2 seeds | judge −.001 (−.0203, +.018) pooled; EM −.023 (−.039, −.007) | equal (pooled test) + arm J | **judge-sensitive**: judges tie, string metrics favour tail (longer spread answers) | nothing to serve; an answer-style difference, not a reading difference |
| At full budget the layouts tie on photos | stage G, today's code | 1,500 × 2 seeds | pooled +.001 (−.014, +.018) rubric; −.006 (−.022, +.010) project | equal (pooled test) | **equal** under rubric and EM; not certified under the project prompt | the layout is not a photo lever |
| At full budget the layouts tie on 4-page documents | stage G, today's code | 1,500 × 2 seeds | pooled +.017 (−.002, +.036) rubric, +.024 (+.006, +.042) project | equal or better | **not resolvable** with 1,500 Q × 2 seeds; a 1–3 pt lean to spread under every scoring | the case for option B (a second document set) at the checkpoint |
| The +.08 vanished because the spread chain lost accuracy at full budget | arm A, budget pairs (NOT one variable) | 1,500 × 2 seeds | spread full − reduced +.011 / −.003; tail +.093 / +.048 | — (descriptive) | **refuted** (provisional: tail s42 reduced file ran on homer-009, spread s43 reduced on homer-008) | the question becomes "why does tail need the extra budget and spread not" → arm E |
| The tail models gained most on multi-page documents | same pairs, per source | 500 × 2 seeds | tail gain MP-DocVQA +.162 / +.090, slides +.076 / +.040, infographics +.040 / +.014 | — | consistent with (provisional; no between-source test) | arm C (many pages) and arm E target MP-DocVQA |
| Spread is better on infographics at full budget | stage G per source, today's code | 500 × 2 seeds | +.050 / +.050, CI > 0 both, 135 / 127 discordant | better bar | post-hoc pick among 3 sources → one repeat on new infographic questions | a per-source claim the gating mean hides |
| Spread pays at many pages | arm C, 4 / 10 / 20 pages | 1,500 × 2 seeds (20 p: s42) | growth vs 4 pages −.004 / −.007 (10 p), +.009 (20 p s42) | (spread − tail at 20 p) − (at 4 p) ≥ +3 pt, CI > 0, both seeds | **refuted** (arm C closed) | the document plateau is not a length effect of the layout; MP-DocVQA leans tail, infographics spread, at every size |
| Spread is better on one long text passage at full budget | MRQA long | 1,500 × 2 seeds | spread today's code .715 / .724; tail re-runs running | better bar, both seeds | open: 191779 / 191780 | confirms arm B on natural text |
| The gating eval cannot show "equal" | arithmetic on the CIs | 23 pairs + arm B + stage G | one seed's CI width .044–.053 documents, .046–.056 text, .036–.041 photos; pooled .031–.038 | — | **confirmed** | option B at the checkpoint |
