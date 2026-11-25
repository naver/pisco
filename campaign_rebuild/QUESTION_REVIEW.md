# QUESTION REVIEW -- 2026-11-25_2021   (a READ-ONLY reviewer that did not write the law; written by expmon, not by a wake)

Law reviewed: sha fd261ecd0eac119a. Editing CAMPAIGN.md voids this review: run `expmon campaign review-question` again.

**Verdict: CLEAR, the campaign may be armed.** Yes: this campaign can settle whether to adopt a rebuilt stage-2 recipe that makes the image slots keep the page's words. Each change is compared with a second seed of today's recipe on the same code. It is measured first on page word recall (the share of a page's words rebuilt from its slots, with a known-positive control), then on judge accuracy for document questions, with photos and text as guards. Nothing is missing. Expect a likely 'promising' result: the law forecasts a 1 to 3 point document gain against a 3-point bar, which would trigger the planned seed-43 repeat (about 24 GPU-h, owne

## What decision is this campaign for?
Adopt a rebuilt stage-2 recipe that teaches the image slots to keep the page's words, or keep today's stage-2 and finetune recipe. The main candidate is a larger share of page transcription on document pages. Three side options are tested too: transcription windows from anywhere on the page, fp32 master weights, and a finetune with the judged-wrong teacher labels dropped.

## Can an outcome of these arms change it?
**yes**. Each change is compared with a second seed of today's stage-2 recipe. That control runs on the same seed and the same code, and is finetuned the same way. The first test is page word recall: the share of a page's words the model can rebuild from its slots. It is read on untrained dense pages and on the eval documents' gold pages. It has a known-positive control (the overfit model, recall .921) and a self-pair check, and it is read through the same readout script slotcheck validated. The second t
> PASS = documents ≥ +3 pt with the CI above 0 AND above the finetune floor, and neither photos nor text loses ≥ 3 pt

## What is missing?
Nothing.
