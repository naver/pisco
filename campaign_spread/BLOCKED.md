# BLOCKED — campaign spread (one question; the campaign keeps running its other cells meanwhile)

## How should "spread equals tail" be tested, now that 1,500 questions per seed cannot show it on documents?
In plain words: if you say yes, a layout tie can be called "equal" when the two seeds averaged per question stay within ±2 points; it costs no GPU time, and without it no document or text result of this campaign can ever be called "equal" — only "better", "worse" or "not resolvable".

**What.** The law calls a case "equal" when one seed's 95% interval of the paired difference (spread − tail, judge
accuracy, same questions) lies inside ±2 pt, at both seeds. Arm A (job 191454, 23 recorded pairs) measured how wide
that interval is on the gating sets: documents .044–.050, text .041–.055, photos .036–.041. An interval wider than .040
cannot fit inside ±2 pt, so "equal" is out of reach on documents and text whatever the models do. This hits arm B's
gate ("inside ±2 pt at ten chunks") and arm E's tie outcome ("CI inside ±2 pt at both seeds"), i.e. the very reading
"the layout buys stage-2 data, not accuracy" can never be confirmed as written.

**Why it matters now.** The full-budget comparison the ledger calls "equal" (documents +.011 / +.024, photos −.009 /
+.010) is, by this law's bar, "not resolvable" on documents. Arm B's first cells are running (191455-191457).

**Options.**
| option | what changes | GPU-h | measured on the recorded files |
|---|---|---|---|
| A. seed-pooled test (recommended) | "equal" = the 95% CI of the per-question difference averaged over the two seeds inside ±2 pt, AND each seed's own difference inside ±2 pt; "better"/"worse" bars unchanged | 0 (CPU, recorded files) | known positive (reduced documents): pooled +.084 [+.065, +.101] (recovered; seeds +.093 / +.075); self-pair 0.000; full photos +.001 [−.015, +.016] → equal; full documents +.017 [−.003, +.037] → still not resolvable (width .040) |
| B. option A + more questions | build a second, disjoint 1,500-Q document set from the same three sources and run each DECIDING document cell on 3,000 Q | ≈ +1 per document cell (≈ 4-6 for arm E's four final cells) | expected pooled width ≈ .028 |
| C. keep the law | ties stay "not resolvable here (CI width w)" | 0 | — |

**Recommendation: A**, applied to arms A-E from now on, the new statistic added to `scripts/spread/slice_pairs.py`
(stdlib) with its controls re-run on every use (known positive +.084, self-pair 0). It weakens no judge or guard (the
per-seed bars for better/worse are untouched and each seed must still sit inside ±2 pt), can be undone by deleting the
rule, and spends 0 GPU-hours. Decide B at the owner checkpoint before arm E, where it matters most.
