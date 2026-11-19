# Campaign report — qafreeze (2026-11-19 18:02)
Successor of campaign 'freeze' (its report: /beegfs/scratch/user/hdejean/pisco/campaign_freeze/REPORT.md — not written yet; REPORT.json carries the same Settled list). Open at hand-off: nothing — the previous campaign settled every arm. **Question (owner, QUESTION.md; HANDOFF.md:66-70):** campaign freeze froze a compressor that had NEVER seen question-answering training. Can a compressor that HAS had one QA finetune then stay FROZEN while a fresh decoder LoRA is trained on the same rows? That is the case `<MEM>` caching needs: compress a collection once, train only the decoder afterwards. **Arm Q (what is tested):** the reference finetune recipe of 184778 / 184780 with ONE change — it starts from a hybrid init: R's own `compressor/` + `connector.pt` (QA-trained) with R's `decoder_state.pt` replaced by the stage-2 decoder (`expQ/archq3/pt_mm_5M_spread/model/decoder_state.pt`, R's own starting point), run with `+model_config_overrides.freeze_compressor=true`, so only the decoder LoRA trains. The hybrid is assembled by copying R's model dir and swapping that one file — the only route that needs no new code, since `PISCO.from_pretrained` reads exactly `decoder_state.pt`, `compressor/compressor_state.pt`, `connector.pt` and `config.json` (pisco/model.py:1376-1405); the two configs differ only in `connector_out_norm` / its init / the image-processor path (`expQ/ft_mm_4b_184778/model/config.json:11-12,24`), and R's config travels with R's connector.
window 2026-11-17 → 2026-11-19 (until 2027-02-15) · 95 ticks · 11 wakes (2 rc≠0) · 24.5 GPU-h over 28 runs (SLURM only; off-cluster compute: expmon probe <jid> card_h=<hours>)
⚠ The campaign idled 20.0 h of its 47.0 h window (42%) — with work outstanding, that is the main operational fact.

## Where this leaves the question
**Settled**
- healthy: COMPLETED 0:0 24:26 homer-020; R s43 HotpotQA on today's code 1,500 rows, judge on all, 0 unparseable: .591 (recorded on old code .593) — job 191769, closed
- healthy: COMPLETED 0:0 22:15 homer-011; R s42 HotpotQA on today's code 1,500 rows, judge on all, 0 unparseable: .605 (recorded on old code .603) — job 191768, closed
- healthy: COMPLETED 0:0 22:55 homer-011; HotpotQA 1,500 rows, judge on all, 0 unparseable, no exceeds error; Q s43 judge .600; Q - R (191769) +0.9 pt [-0.5,+2.1] McNemar p .223, 55/42 discordant (reported, not gating); same md5 as the gating pairs — job 191767, closed
- healthy: COMPLETED 0:0 22:36 homer-011; HotpotQA 1,500 rows, judge on all, 0 unparseable, no exceeds error; Q s42 judge .600; Q - R (191768) -0.5 pt [-1.9,+0.9] McNemar p .562, 50/57 discordant (reported, not gating); md5 eval_mm_4b b3d1cda1 / eval dfe32aa0 / judge 02bc8748 / paired a3cc13b2 / model 994cff3d — job 191766, closed
- arm Q VALIDATED s43: Q-R judge, n=1500 paired: documents -0.1 [-1.1,+0.9] p.896; photos -0.5 [-1.7,+0.7] p.434; MRQA +0.3 [-0.8,+1.5] p.661. Control Q-F docs +3.3 [+1.1,+5.7] p.003. Module move 191627: compressor/connector 0.0%, decoder lora_B 66.9% — job 191530, best
- healthy: F s43 documents today's code judge .580 (recorded .580), 1500 judged, 0 unparseable, px 1003520, homer-020 — job 191678, closed
- … 22 more

**Open, and running**
- (none)

## Arms
| arm | job | state | GPU-h | verdict | headline (last verdict, else latest probe) |
|---|---|---|---|---|---|
| kind=build_init seeds=42 decoder_from=pt_mm_5M_spread | 191450 | COMPLETED | 0.0 | closed | arm B build OK: init_s42 / init_s43 built (exit 0, 3:07, homer-012). md5: decode |
| kind=check_module_move init=ft_mm_4b_184778 trained=qafreeze | 191451 | COMPLETED | 0.0 | closed | feasibility gate (1) PASS: R s42 (184778) -> init_s42: every compressor.* (lora_ |
| kind=check_module_move init=pt_mm_5M_spread trained=qafreeze | 191452 | COMPLETED | 0.0 | closed | feasibility gate (2) PASS: stage-2 (pt_mm_5M_spread) -> init_s42: every decoder. |
| kind=check_module_move init=ft_mm_4b_184780 trained=qafreeze | 191453 | COMPLETED | 0.0 | closed | arm B seed-43 init OK: R s43 (184780) -> init_s43: compressor.* and connector.*  |
| kind=ft_mm arm=qa_frozen seed=42 smoke=20 init=qafreeze_init_s42 samples=28000 bs=1 accum=64 lr=2e-5 max_pixels=1003520 | 191493 | COMPLETED | 0.5 | closed | plumbing PASS (Qs, s42, 20 steps = 1,280 rows = .046 epoch; COMPLETED 0:0, homer |
| kind=check_module_move init=qafreeze trained=ft_frozen_s42_smoke | 191494 | COMPLETED | 0.0 | closed | plumbing PASS: init_s42 -> ft_frozen_s42_smoke/model: every compressor.* (lora_A |
| kind=eval_mm mode=pisco ckpt=ref_s42 data=mm_docqa_1500_1m limit=all image_rate=ckpt topk=4 max_pixels=1003520 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 PX=1003520 | 191495 | COMPLETED | 0.5 | closed | healthy (COMPLETED 0:0, homer-017, 27:13, one attempt; 1,500 rows, judge on all, |
| kind=eval_mm mode=pisco ckpt=ref_s43 data=mm_docqa_1500_1m limit=all image_rate=ckpt topk=4 max_pixels=1003520 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 PX=1003520 | 191496 | COMPLETED | 0.5 | closed | healthy (COMPLETED 0:0, homer-019, 28:04, one attempt; 1,500 rows, judge on all, |
| kind=ft_mm arm=qa_frozen seed=42 smoke=0 init=qafreeze_init_s42 samples=28000 bs=1 accum=64 lr=2e-5 max_pixels=1003520 | 191527 | COMPLETED | 9.0 | best | arm Q VALIDATED s42: QA-trained compressor+connector frozen, decoder LoRA only,  |
| kind=ft_mm arm=qa_frozen seed=43 smoke=0 init=qafreeze_init_s43 samples=28000 bs=1 accum=64 lr=2e-5 max_pixels=1003520 | 191530 | COMPLETED | 9.3 | best | arm Q VALIDATED s43: Q-R judge, n=1500 paired: documents -0.1 [-1.1,+0.9] p.896; |
| kind=eval_mm mode=pisco ckpt=ref_s42 data=mm_naturalqa_1500 limit=all image_rate=ckpt topk=4 max_pixels=1003520 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 PX=1003520 | 191531 | COMPLETED | 0.3 | closed | healthy (COMPLETED 0:0, homer-018, 18:59, one attempt; 1,500 rows, judge on all, |
| kind=eval_mm mode=pisco ckpt=ref_s43 data=mm_naturalqa_1500 limit=all image_rate=ckpt topk=4 max_pixels=1003520 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 PX=1003520 | 191532 | COMPLETED | 0.3 | closed | healthy (COMPLETED 0:0, homer-017, 18:10, one attempt; 1,500 rows, judge on all, |
| kind=check_module_move init=qafreeze trained=ft_frozen_s42 | 191625 | COMPLETED | 0.0 | closed | health PASS: init_s42 -> ft_frozen_s42/model: every compressor.* (lora_A, lora_B |
| kind=eval_mm mode=pisco ckpt=ft_frozen_s42 data=mm_docqa_1500_1m limit=all image_rate=ckpt topk=4 max_pixels=1003520 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 PX=1003520 | 191626 | COMPLETED | 0.5 | closed | healthy: COMPLETED 0:0 homer-017 28:51 one attempt; 1,500 rows, judge on all, un |
| kind=check_module_move init=qafreeze trained=ft_frozen_s43 | 191627 | COMPLETED | 0.0 | closed | health PASS: init_s43 -> ft_frozen_s43/model: every compressor.* and connector.* |
| kind=eval_mm mode=pisco ckpt=ft_frozen_s43 data=mm_docqa_1500_1m limit=all image_rate=ckpt topk=4 max_pixels=1003520 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 PX=1003520 | 191628 | COMPLETED | 0.5 | closed | healthy: COMPLETED 0:0 homer-017 28:42 one attempt; 1,500 rows, judge on all, un |
| kind=eval_mm mode=pisco ckpt=ft_frozen_s42 data=mm_naturalqa_1500 limit=all image_rate=ckpt topk=4 max_pixels=1003520 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 PX=1003520 | 191659 | COMPLETED | 0.3 | closed | healthy: Q s42 photos judge .677, 1500 rows judged, 0 unparseable, px 1003520, h |
| kind=eval_mm mode=pisco ckpt=ft_frozen_s43 data=mm_naturalqa_1500 limit=all image_rate=ckpt topk=4 max_pixels=1003520 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 PX=1003520 | 191660 | COMPLETED | 0.3 | closed | healthy: Q s43 photos judge .679, 1500 judged, 0 unparseable, px 1003520, homer- |
| kind=eval_mm mode=pisco ckpt=ft_frozen_s42 data=text_mrqa_ood_1500 limit=all image_rate=ckpt topk=4 max_pixels=1003520 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 CML=4096 PX=1003520 | 191661 | COMPLETED | 0.3 | closed | healthy: Q s42 MRQA judge .726, 1500 judged, 0 unparseable, CML 4096, homer-020; |
| kind=eval_mm mode=pisco ckpt=ft_frozen_s43 data=text_mrqa_ood_1500 limit=all image_rate=ckpt topk=4 max_pixels=1003520 merged=0 samples_per_q=0 raw_policy=none answer_style=brief gold_only=0 mem_quant=none extra_distractors=0 rank=none drop=0 keep_ranked=0 CML=4096 PX=1003520 | 191662 | COMPLETED | 0.3 | closed | healthy: Q s43 MRQA judge .727, 1500 judged, 0 unparseable, CML 4096, homer-010; |
| … 8 more | | | | | |

## Findings (from verdicts, not prose)
- healthy: COMPLETED 0:0 24:26 homer-020; R s43 HotpotQA on today's code 1,500 rows, judge on all, 0 unparseable: .591 (recorded on old code .593) (job 191769, closed)
- healthy: COMPLETED 0:0 22:15 homer-011; R s42 HotpotQA on today's code 1,500 rows, judge on all, 0 unparseable: .605 (recorded on old code .603) (job 191768, closed)
- healthy: COMPLETED 0:0 22:55 homer-011; HotpotQA 1,500 rows, judge on all, 0 unparseable, no exceeds error; Q s43 judge .600; Q - R (191769) +0.9 pt [-0.5,+2.1] McNemar p .223, 55/42 discordant (reported, not gating); same md5 as the gating pairs (job 191767, closed)
- healthy: COMPLETED 0:0 22:36 homer-011; HotpotQA 1,500 rows, judge on all, 0 unparseable, no exceeds error; Q s42 judge .600; Q - R (191768) -0.5 pt [-1.9,+0.9] McNemar p .562, 50/57 discordant (reported, not gating); md5 eval_mm_4b b3d1cda1 / eval dfe32aa0 / judge 02bc8748 / paired a3cc13b2 / model 994cff3d (job 191766, closed)
- arm Q VALIDATED s43: Q-R judge, n=1500 paired: documents -0.1 [-1.1,+0.9] p.896; photos -0.5 [-1.7,+0.7] p.434; MRQA +0.3 [-0.8,+1.5] p.661. Control Q-F docs +3.3 [+1.1,+5.7] p.003. Module move 191627: compressor/connector 0.0%, decoder lora_B 66.9% (job 191530, best)

## What went wrong
- The campaign idled 20.0 h of its 47.0 h window (42%) — with work outstanding, that is the main operational fact.
- 2 of 11 LLM wakes returned rc≠0.
- Alert raised: last push FAILED (push_rc=, behind ) — git -C /beegfs/scratc|2026-11-18.
- 6 tick(s) could not push the campaign state.
- BLOCKED.md is standing: BLOCKED — campaign qafreeze   (owner checkpoint; one question).

## Left, and what it costs
- 0 arm(s) without a verdict
- window: until 2027-02-15

## Artifacts
- /beegfs/scratch/user/hdejean/pisco/expQ/191530
- /beegfs/scratch/user/hdejean/pisco/expQ/191527
- campaign dir: /beegfs/scratch/user/hdejean/pisco/campaign_qafreeze (STATE.md, BLOCKED.md)
- ledger: 46 entries of this campaign — expmon ledger (linked, never pasted)
