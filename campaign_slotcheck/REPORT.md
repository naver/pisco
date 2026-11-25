# Campaign report — slotcheck (2026-11-25 19:04)
Successor of campaign 'longdoc' (its report: /beegfs/scratch/user/hdejean/pisco/campaign_longdoc/REPORT.md). Open at hand-off: rejudge.sbatch — job 191629 COMPLETED, no verdict; CML=1024 PX=1003520 TOPK=1000 — job 191634 COMPLETED, no verdict; CML=1024 PX=1003520 TOPK=1000 — job 191635 COMPLETED, no verdict; CML=1024 DML=5612 PX=1003520 TOPK=1000 — job 191636 COMPLETED, no verdict. **Those four jobs belong to the LIVE sibling longdoc: never verdict, relaunch, kill or re-use them here.** **Question (owner, QUESTION.md):** "no progress has been done after many tries to improve the models. double check
window 2026-11-18 → 2026-11-25 (until 2027-02-16) · 339 ticks · 16 wakes (5 rc≠0) · 17.3 GPU-h over 31 runs (SLURM only; off-cluster compute: expmon probe <jid> card_h=<hours>)
⚠ The campaign idled 140.8 h of its 169.0 h window (83%) — with work outstanding, that is the main operational fact.

## Where this leaves the question
**Settled**
- check 3 repeat, n=500 new questions (disjoint ids), R s42, homer-016, health OK (3 x 500 rows, judge on all, 0 unparseable, 0 empty, 0 exceeds; image cell = recorded R s42 on these 500, 500/500 same answers): image .632 / text .478 / image+text .686 / uncompressed .832. image+text - image +5.4 pt [+1.8,+8.8] McNemar 56/29 p .005, no source CI below 0 -> PASSES the repeat; text - image -15.4 [-20.2,-10.6] -> worse again. Pooled 1,000 Q: +5.0 [+2.5,+7.6] p 1.2e-4 / -13.7 [-17.1,-10.3]. model.py f8ae6efe (one-row pad drop; no-op here) — job 192239, closed
- round-2 build healthy: COMPLETED 0:0 4:39 homer-017; 500 rows x 3 sets (img/text/imgtext _500_r2), empty 0, cut 0, slots per question 431.9 / 222.6 / 654.5, answer in transcription 454/500, ok=True; round-1 re-audit decodes its one 'cut' row as a chunk-boundary split (cut_boundary 1 = cut 1, ok=True); build_text_sets.py 6b438662 — job 192238, closed
- round-2 transcription healthy: COMPLETED 0:0 4:13:18 homer-017, one attempt; 1,252/1,252 pages, empty 0, 121 at the 1,024-token cap, median 9.4 s/page; transcribe_pages.py 03867f94 — job 192237, closed
- plumbing passed, decides nothing: COMPLETED 0:0 5:14 homer-017; round=2 reached the job; 500 questions, 2,000 page slots, 1,252 unique pages; 10/10 pages, empty 0, 3 at the 1,024 cap, median 25.2 s/page (round-1 smoke 13.6) — job 192236, closed
- check 3 first read, n=500 paired, homer-014, health OK (500/500 judged, 0 unparseable, 0 empty, 0 exceeds): img .602 / text .482 / imgtext .648 (uncompressed .790). text-img -12.0 [-16.6,-7.6] p<.001 = worse; imgtext-img +4.6 [+1.0,+8.0] p .013 = PROMISING, repeat 192236-192239 on new questions — job 192223, closed
- plumbing passed, decides nothing: COMPLETED 0:0 8:54 homer-014, one attempt; 30/30 rows in each of img / text / imgtext, empty predictions 0/30 each, no 'exceeds' error; mems_used 11,404 / 2,412 / 13,816 (img + text = imgtext +2); prompt max 494 / 269 / 673; JUDGE=0 by design — job 192220, closed
- … 23 more

**Open, and running**
- (none)

## Arms
| arm | job | state | GPU-h | verdict | headline (last verdict, else latest probe) |
|---|---|---|---|---|---|
| kind=audit arm=C n_control=50 topk_docs=4 sets=docpages_20k_1m_teacher,natural_8k_teacher | 191663 | COMPLETED | 0.0 | closed | check 1 decided, not a cause: COMPLETED 0:0 56 s homer-017; feasibility gate PAS |
| pad_probe.sbatch | 191671 | COMPLETED | 0.0 | closed | plumbing OK: COMPLETED 0:0 homer-011, 6 pages + 2 four-page questions, controls  |
| pad_probe.sbatch | 191672 | COMPLETED | 0.1 | closed | check 2a NOT INERT (n = 120 pages x widths 17/63 + 30 four-page questions): COMP |
| pad_split.sbatch | 191781 | COMPLETED | 0.2 | closed | split decided (40 pages x widths 17/63, bf16 + fp32, homer-011, COMPLETED 0:0 9: |
| JUDGE=0 | 191782 | COMPLETED | 0.3 | closed | plumbing PASS (decides nothing): COMPLETED 0:0 19:42 homer-020, R s42 300 Q, JUD |
| kind=eval_pad arm=P2b cell=p2b_s42 steps=none,64 limit=all judge=1 model=/beegfs/scratch/user/hdejean/pisco/expQ/ft_mm_4b_184778/model set=mm_docqa_1500_1m px=1003520 topk=4 mnt=32 | 191783 | COMPLETED | 0.9 | closed | check 2b s42 NOT A CAUSE: COMPLETED 0:0 55:10 homer-020; 1,500 rows x 2 files, j |
| eval_pad.sbatch | 191784 | COMPLETED | 1.3 | closed | check 2b s43 NOT A CAUSE: COMPLETED 0:0 1:17:41 homer-019; 1,500 rows x 2 files, |
| build_dense100.sbatch | 191890 | COMPLETED | 0.0 | closed | plumbing OK (decides nothing): COMPLETED 0:0 1:01 homer-010, MaxRSS 1.2 GB; scan |
| kind=build arm=O_build scan=60000 tag=none n=100 max_tokens=760 | 191891 | COMPLETED | 0.0 | closed | build OK: COMPLETED 0:0 2:46 homer-010, MaxRSS 1.9 GB; 60,000 scanned rows -> 10 |
| pt_overfit.sbatch | 191892 | COMPLETED | 0.3 | closed | smoke OK (decides nothing): COMPLETED 0:0 16:07 homer-016, MaxRSS 8.35 GB, 1.03  |
| recall.sbatch | 191893 | COMPLETED | 2.6 | closed | baseline OK: COMPLETED 0:0 2:37:33 homer-016, MaxRSS 4.8 GB; stride 8 printed, 4 |
| pt_overfit.sbatch | 192002 | FAILED | 0.0 | closed | healthy, trained to the end: attempt 1 PREEMPTED at 1:32 (no checkpoint, save_st |
| recall.sbatch | 192003 | CANCELLED | 0.0 | closed | never ran (0 of 2 recall passes): its afterok parent 192002 ended FAILED exit 3  |
| transcribe_pages.sbatch | 192009 | COMPLETED | 0.1 | closed | plumbing PASS (decides nothing): COMPLETED 0:0 4:25 homer-011, MaxRSS 9.0 GiB; 1 |
| kind=transcribe arm=X_transcribe shard=0 n_shards=4 limit=all set=mm_docqa_1500_1m questions=500 topk=4 px=1003520 max_new_tokens=1024 | 192010 | COMPLETED | 1.1 | closed | healthy: COMPLETED 0:0 1:04:14 homer-011, one attempt, MaxRSS 9.0 GiB; shard 0 = |
| kind=probe_recall arm=O_recall passes=o_bf16_dense100:/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/o_bf16/model:slotcheck_dense100 px=1003520 | 192209 | COMPLETED | 1.4 | closed | check 4 DECIDED, capacity is not the limit: healthy (COMPLETED 0:0 1:26:45 homer |
| transcribe_pages.sbatch | 192210 | COMPLETED | 1.1 | closed | healthy: COMPLETED 0:0 1:05:50 homer-011, one attempt; shard 1 = 317/317 pages,  |
| transcribe_pages.sbatch | 192211 | COMPLETED | 1.1 | closed | healthy: COMPLETED 0:0 1:05:10 homer-018, one attempt; shard 2 = 317/317 pages,  |
| transcribe_pages.sbatch | 192212 | COMPLETED | 1.1 | closed | healthy: COMPLETED 0:0 1:07:10 homer-018, one attempt; shard 3 = 317/317 pages,  |
| kind=build arm=X_build limit=30 tag=_smoke questions=500 sets=img,text,imgtext cml=256 chunk=1 | 192214 | FAILED | 0.0 | closed | plumbing FAILED its own health check by a bug in the check (exit 4, 1:11, homer- |
| … 11 more | | | | | |

## Findings (from verdicts, not prose)
- check 3 repeat, n=500 new questions (disjoint ids), R s42, homer-016, health OK (3 x 500 rows, judge on all, 0 unparseable, 0 empty, 0 exceeds; image cell = recorded R s42 on these 500, 500/500 same answers): image .632 / text .478 / image+text .686 / uncompressed .832. image+text - image +5.4 pt [+1.8,+8.8] McNemar 56/29 p .005, no source CI below 0 -> PASSES the repeat; text - image -15.4 [-20.2,-10.6] -> worse again. Pooled 1,000 Q: +5.0 [+2.5,+7.6] p 1.2e-4 / -13.7 [-17.1,-10.3]. model.py f8ae6efe (one-row pad drop; no-op here) (job 192239, closed)
- round-2 build healthy: COMPLETED 0:0 4:39 homer-017; 500 rows x 3 sets (img/text/imgtext _500_r2), empty 0, cut 0, slots per question 431.9 / 222.6 / 654.5, answer in transcription 454/500, ok=True; round-1 re-audit decodes its one 'cut' row as a chunk-boundary split (cut_boundary 1 = cut 1, ok=True); build_text_sets.py 6b438662 (job 192238, closed)
- round-2 transcription healthy: COMPLETED 0:0 4:13:18 homer-017, one attempt; 1,252/1,252 pages, empty 0, 121 at the 1,024-token cap, median 9.4 s/page; transcribe_pages.py 03867f94 (job 192237, closed)
- plumbing passed, decides nothing: COMPLETED 0:0 5:14 homer-017; round=2 reached the job; 500 questions, 2,000 page slots, 1,252 unique pages; 10/10 pages, empty 0, 3 at the 1,024 cap, median 25.2 s/page (round-1 smoke 13.6) (job 192236, closed)
- check 3 first read, n=500 paired, homer-014, health OK (500/500 judged, 0 unparseable, 0 empty, 0 exceeds): img .602 / text .482 / imgtext .648 (uncompressed .790). text-img -12.0 [-16.6,-7.6] p<.001 = worse; imgtext-img +4.6 [+1.0,+8.0] p .013 = PROMISING, repeat 192236-192239 on new questions (job 192223, closed)

## What went wrong
- The campaign idled 140.8 h of its 169.0 h window (83%) — with work outstanding, that is the main operational fact.
- 5 of 16 LLM wakes returned rc≠0.
- 3 run(s) ended FAILED/TIMEOUT/NODE_FAIL: 192002 192214 192221.

## Left, and what it costs
- 0 arm(s) without a verdict
- window: until 2027-02-16

## Artifacts
- (no best/runner-up run yet)
- campaign dir: /beegfs/scratch/user/hdejean/pisco/campaign_slotcheck (STATE.md)
- ledger: 54 entries of this campaign — expmon ledger (linked, never pasted)
