# QUESTION REVIEW -- 2026-11-25_1952   (a READ-ONLY reviewer that did not write the law; written by expmon, not by a wake)

Law reviewed: sha a34e1161c2e84b01. Editing CAMPAIGN.md voids this review: run `expmon campaign review-question` again.

**Verdict: CLEAR, the campaign may be armed.** The arms can settle the main question: keep today's stage-2 recipe, or train the image slots on more page transcription. Both gates have a matched control and floors for seed and code drift, and the recall test is known to move. Two caveats, neither blocking: the fp32 run uses a learning rate the project already found too high, so only a pass from it means something; and the cleaned-label finetune also trains on about a third fewer rows, so a tie there cannot separate the two effects.

## What decision is this campaign for?
Adopt a stage-2 recipe that gives document pages a larger share of page-transcription training, or keep today's recipe. Side decisions: whether window targets, fp32 master weights in stage 2, and finetune labels cleaned of teacher errors each earn a place in the recipe.

## Can an outcome of these arms change it?
**yes**. The main change and its control share one seed, one code date and one recipe; the only difference is the transcription share. Gate 1 reads page word recall on dense pages no stage 2 trained on. This metric is known to move: the recorded stage 2 scores .702 and the model trained on those pages scores .921, and the first check must reproduce both before any training starts. Gate 2 finetunes both models the same way and reads paired judge accuracy on documents, which is the number the adoption deci
> gate 1 = page word recall, change − control, paired per page; gate 2 = judge accuracy

## What is missing?
- Advice: State that a tie or loss of the fp32-master run closes only fp32 masters at the bf16 learning rate of 3e-5, not fp32 masters in stage 2, because the project already found fp32 at the bf16 rate too high in finetuning and settled on a rate 4 times lower. The fp32 run keeps lr 3e-5. The project's notes list fp32 masters at the bf16 rate as closed ('too high'). So a bad result from this run could come from the learning rate, not from the precision. Only a pass would be clean. This does not affect the main transcription-share decision.
  > fp32 masters at lr 2e-5 (too high once updates land) (docs/training_plan.md)
- Advice: Report how many rows and optimizer steps the cleaned-label finetune trained on, beside the reference finetune, and state that a tie does not separate the effect of cleaner labels from the effect of training on fewer rows. Dropping the rows the judge marks wrong also cuts the finetune's rows and steps. On the earlier label split only 68% of teacher labels were right, so about a third would go. A tie could therefore hide a cleaning gain offset by having less data. A pass would still read as a label effect.
  > raises the S+ share from 68% toward ~80% without changing style (docs/data_review.md)
