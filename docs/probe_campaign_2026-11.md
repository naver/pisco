# Probe campaign 2026-11 — why the compressed model loses, and what recovers it without training

Law: `campaign_probes/CAMPAIGN.md`. Live state: `campaign_probes/STATE.md`. This page is rewritten whenever an arm decides.
Metric everywhere: judge accuracy ↑ (share of questions the Qwen3-8B rubric judge accepts), paired against the same model's
plain cell on the same questions (95% bootstrap CI, McNemar p). Compression: 1 slot per 8 visual or text tokens
(`compr_rate: 8` in the reference models' config), about 97 slots per document page at PX=1003520 (388 slots for 4 pages).

## Status (2026-11-11)
- **Feasibility gate (instrument check) — running.** Re-run of the reference document cell (model 184778, 1,500 questions)
  to pair with the recorded judge accuracy .610. Nothing below is measured until it passes.

## Findings
(none yet)

## Closed
(none yet)
