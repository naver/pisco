Owner's question (approved 2026-11-14 in the ~/.claude chat as hand-off 5b6e55; owner said "yes" to drafting it on 2026-11-17):

Successor to campaign freeze: test whether a QA-trained compressor can stay frozen (the case <MEM> caching needs).
Freeze R's compressor+connector (job 184778), train decoder LoRA only from the stage-2 decoder on the same 28k rows,
2 seeds, paired vs R. Pass: documents, photos, MRQA each >= -2 pt with no CI wholly below 0 at both seeds -> build the
<MEM> cache. Fail: <= -3 pt with CI below 0 at both seeds. ~9 GPU-h per seed.

Why: campaign freeze froze a stage-2 compressor that never saw QA training, so its gate 0 fail does not rule out caching
(campaign_freeze/STATE.md lists this as open). Evidence from freeze: connector finetune buys nothing (A1-F documents
-.001/-.004/+.006); photos tie under F; text loss concentrates in hotpot_bridge (-3.0/-2.8/-3.6, CI<0 at all seeds) and
MRQA RelationExtraction; F plateaus after ~0.23 epoch. Literature: ARC-Encoder trains its encoder at lr 2e-6 vs 3e-5
projector (a gentle nudge suffices); PISCO/COCOM/OSCAR lose 4-10 pt freezing the decoder, so the compressor side is the
cheaper one to freeze; LLoCO is the F setup without a joint baseline.
Optional second follow-up: F with decoder LoRA lr 1e-4 (current 2e-5 is a full-FT value) -- pass >= +2 pt over F on
documents. Third, only if the first fails: one slot-mixing transformer layer on cached <MEM>s.
Note: the freeze law forbids new compressor-freeze variants, which is why this is a new campaign. Arming is the owner's
decision.
