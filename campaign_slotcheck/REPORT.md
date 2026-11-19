# Campaign report — slotcheck (2026-11-19 16:42)
Successor of campaign 'longdoc' (its report: /beegfs/scratch/user/hdejean/pisco/campaign_longdoc/REPORT.md). Open at hand-off: rejudge.sbatch — job 191629 COMPLETED, no verdict; CML=1024 PX=1003520 TOPK=1000 — job 191634 COMPLETED, no verdict; CML=1024 PX=1003520 TOPK=1000 — job 191635 COMPLETED, no verdict; CML=1024 DML=5612 PX=1003520 TOPK=1000 — job 191636 COMPLETED, no verdict. **Those four jobs belong to the LIVE sibling longdoc: never verdict, relaunch, kill or re-use them here.** **Question (owner, QUESTION.md):** "no progress has been done after many tries to improve the models. double check
window 2026-11-18 → 2026-11-19 (until 2027-02-16) · 46 ticks · 12 wakes (4 rc≠0) · 12.4 GPU-h over 31 runs (SLURM only; off-cluster compute: expmon probe <jid> card_h=<hours>)
⚠ The campaign idled 4.7 h of its 22.5 h window (20%) — with work outstanding, that is the main operational fact.

## Where this leaves the question
**Settled**
- check 3 first read, n=500 paired, homer-014, health OK (500/500 judged, 0 unparseable, 0 empty, 0 exceeds): img .602 / text .482 / imgtext .648 (uncompressed .790). text-img -12.0 [-16.6,-7.6] p<.001 = worse; imgtext-img +4.6 [+1.0,+8.0] p .013 = PROMISING, repeat 192236-192239 on new questions — job 192223, closed
- plumbing passed, decides nothing: COMPLETED 0:0 8:54 homer-014, one attempt; 30/30 rows in each of img / text / imgtext, empty predictions 0/30 each, no 'exceeds' error; mems_used 11,404 / 2,412 / 13,816 (img + text = imgtext +2); prompt max 494 / 269 / 673; JUDGE=0 by design — job 192220, closed
- never ran (0 of 3 sets): afterok parent 192221 exited 4 on its cut rule although the sets are complete; killed, relaunched after the eval smoke — job 192222, closed
- sets complete and usable: FAILED exit 4 in 2:19 on homer-014 on its cut rule only. 500 rows x 3 sets written (stats.json + dml.txt: DML img 839 / text 862 / imgtext 1361); empty 0; mems per question img 430.4 (215,202 = the 1,500-Q set's 430.7/Q) / text 222.8 / imgtext 653.2 (= img + text); prompt max 583 / 606 / 1105; answer string in the transcription 424/500. The 1 'cut' row (37710.jpeg-9, answer 2016, in text and imgtext) is NOT truncation: n_max_chunks None, and the chunk sizes of its 4 pages (4x256 | 256+207 | 4x256 | 256+256+103 text tokens) equal the transcriptions' own generated token counts (1024 / 463 / 1024 / 615), so every token reaches the compressor; the audit joins decoded chunks with a space, so a digit string split at the 512-token boundary reads as missing (inferred from position ~char 1068 of 1416, not decoded). 1/500 = 0.2 pt at most, bar 5 pt — job 192221, closed
- never ran: afterok parent 192214 failed (a bug in its health check, patched 1c110e22); killed, relaunch next wake (submission cap 4 used) — jobs 192215-192217 (3 cells), closed
- plumbing FAILED its own health check by a bug in the check (exit 4, 1:11, homer-014): chunk length incl. <MEM>s compared to CML + 1, and imgtext's image chunks counted as text. Data passed: 30 Q x 3 sets, empty 0, cut 0, mems mean 380.1 / 80.4 / 460.5. Patched build_text_sets.py 1c110e22; decides nothing — job 192214, closed
- … 19 more

**Open, and running**
- ROUND=2 — job 192236 RUNNING 2:25
- ROUND=2 — job 192237 PENDING 0:00
- ROUND=2 — job 192238 PENDING 0:00
- eval_x.sbatch — job 192239 PENDING 0:00

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
- check 3 first read, n=500 paired, homer-014, health OK (500/500 judged, 0 unparseable, 0 empty, 0 exceeds): img .602 / text .482 / imgtext .648 (uncompressed .790). text-img -12.0 [-16.6,-7.6] p<.001 = worse; imgtext-img +4.6 [+1.0,+8.0] p .013 = PROMISING, repeat 192236-192239 on new questions (job 192223, closed)
- plumbing passed, decides nothing: COMPLETED 0:0 8:54 homer-014, one attempt; 30/30 rows in each of img / text / imgtext, empty predictions 0/30 each, no 'exceeds' error; mems_used 11,404 / 2,412 / 13,816 (img + text = imgtext +2); prompt max 494 / 269 / 673; JUDGE=0 by design (job 192220, closed)
- never ran (0 of 3 sets): afterok parent 192221 exited 4 on its cut rule although the sets are complete; killed, relaunched after the eval smoke (job 192222, closed)
- sets complete and usable: FAILED exit 4 in 2:19 on homer-014 on its cut rule only. 500 rows x 3 sets written (stats.json + dml.txt: DML img 839 / text 862 / imgtext 1361); empty 0; mems per question img 430.4 (215,202 = the 1,500-Q set's 430.7/Q) / text 222.8 / imgtext 653.2 (= img + text); prompt max 583 / 606 / 1105; answer string in the transcription 424/500. The 1 'cut' row (37710.jpeg-9, answer 2016, in text and imgtext) is NOT truncation: n_max_chunks None, and the chunk sizes of its 4 pages (4x256 | 256+207 | 4x256 | 256+256+103 text tokens) equal the transcriptions' own generated token counts (1024 / 463 / 1024 / 615), so every token reaches the compressor; the audit joins decoded chunks with a space, so a digit string split at the 512-token boundary reads as missing (inferred from position ~char 1068 of 1416, not decoded). 1/500 = 0.2 pt at most, bar 5 pt (job 192221, closed)
- never ran: afterok parent 192214 failed (a bug in its health check, patched 1c110e22); killed, relaunch next wake (submission cap 4 used) (jobs 192215-192217, closed)

## What went wrong
- The campaign idled 4.7 h of its 22.5 h window (20%) — with work outstanding, that is the main operational fact.
- 4 of 12 LLM wakes returned rc≠0.
- 3 run(s) ended FAILED/TIMEOUT/NODE_FAIL: 192002 192214 192221.

## Left, and what it costs
- 4 arm(s) without a verdict; at the median arm cost of 0.8 GPU-h that is ~3.1 GPU-h
- window: until 2027-02-16

## Artifacts
- (no best/runner-up run yet)
- campaign dir: /beegfs/scratch/user/hdejean/pisco/campaign_slotcheck (STATE.md)
- ledger: 50 entries of this campaign — expmon ledger (linked, never pasted)
