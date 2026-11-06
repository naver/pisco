# extremewhenbench — paper card   (rewrite, never append; every number carries its page and its n)

source `/beegfs/scratch/user/hdejean/pisco/papers/extremewhenbench.pdf` · text `/beegfs/scratch/user/hdejean/pisco/papers/extremewhenbench.txt` (10 pages, 5829 words) · card started 2026-10-28 for project pisco

Seo & Kim (NAVER Cloud AI, KAIST AI), "Natural-Language Temporal Grounding in Hour-Long Videos is a Search Problem:
A Benchmark and Empirical Decomposition", arXiv 2606.12300v1, 10 Jun 2026. Benchmark: **ExtremeWhenBench**.
Tables were read from the text dump and checked against the prose that restates them. The PDF itself could not be
rendered: `pdftoppm` is missing on this host.

## Claim in one paragraph
The paper reports that for hour-long videos, finding *where* to look is the hard part of temporal grounding.
Temporal grounding here means returning the interval [ts, te] that a natural-language query refers to; recognising
the event once nearby is not the hard part. On its new benchmark (2,273 queries, 194 videos, mean 75.7 min, max 9 h,
p1), every open Video-LLM collapses 5–120× relative to Charades-STA (Table 1, p3). A plain frame-level CLIP retrieval
baseline (0.269 mIoU) beats every open and closed Video-LLM tested (best Gemini-3.5-flash 0.115, p3). 85% of
sampled failures are search failures (n = 100, p4). A training-free retrieve-then-ground hybrid is CLIP top-3
windows, then Qwen3.5-9B grounding inside 6 min of video; it reaches 0.354 mIoU, 6.7× the monolithic Video-LLM
(Table 2, p4).

## Method — what is actually new
- **Benchmark built from naturally long videos.** 195 source videos from LVBench, MLVU and VideoMME (raw videos
  only; their QA is not used). Queries are open-form, 8–18 words. The median GT event lasts 9 s, close to
  Charades-STA's 7.1 s, so the search space is ~153× larger at the same event grain (§2, p2).
- **Annotation pipeline (7 stages, ≈$200 of API, App. B, p7–9).** It starts from 1 fps Qwen3-VL-8B captions.
  Then gpt-5-mini groups events and gpt-5.1 verifies boundaries on 8×8 frame mosaics (79.4% verified / 20.6%
  refined). Then come within-video dedup, question generation, a gpt-5.4 rubric filter (validity, uniqueness,
  anti-CLIP difficulty; top 30%, score ≥ 7) and author review of 164 CLIP-flagged items (102 dropped).
- **Search vs localize decomposition (§3, p3–4).** Cropping to a GT-centred window of growing width changes search
  difficulty only. At ±15 s, both Video-LLMs reach their Charades ceiling (Qwen3.5-9B 0.572, Gemini-3.5-flash
  0.578). Over the full video, Qwen drops to 0.053 (Table 7, p10).
- **Retrieve-then-ground hybrid (Table 2 p4, Table 6 p9).** CLIP ViT-L/14-336 at 1 fps ranks windows, NMS 60 s.
  The top-K windows (±1 min) go to Qwen3.5-9B (N = 768 frames), which outputs the interval. Best at K = 3.

## Recipe — what a replication must match
| knob | paper (page) | this project (config / launcher) | same? |
|---|---|---|---|
| backbone / size | Qwen3.5-9B grounder (vLLM, N up to 2,048 frames), InternVL3.5-8B, LLaVA-OV-7B, LLaVA-NextVideo-7B; GPT-5.4, Gemini-2.5/3.5-flash; CLIP ViT-L/14-336 retriever (p3) | Qwen3.5-4B compressor + decoder (PISCO), checkpoints `expQ/ft_mm_4b_174908` (docs/photos, find-trained) and `expQ/ft_nav_4b` (robot frames) | same family, smaller |
| training data | none: the benchmark paper trains nothing | n/a (we would test zero-shot first) | — |
| steps or epochs · effective batch | not applicable | — | — |
| optimizer · lr · schedule · warmup | not applicable | — | — |
| loss / objective · negatives / sampling | not applicable | — | — |
| eval sets · metric · protocol | ExtremeWhenBench test, n = 2,273. mIoU over [ts, te], parse failure = IoU 0; also R@0.3/0.5/0.7 (App. C). Frame count N swept per model up to its context limit; best N reported (p3). Hybrid: CLIP top-3 ±1 min, NMS 60 s, Qwen3.5-9B N = 768 (p4) | `scripts/eval_mm_docqa.py` scores text answers (EM/F1/ANLS/judge), not intervals. Route probes: `--raw_policy route1/route2` | **no**: needs an interval metric and timestamps in the prompt |
| hardware · wall time · precision | not reported | A100 80 GB, bf16 | — |
"not reported" is a valid cell; a guess is not.

## Headline results — their numbers
| setting | metric | paper | strongest baseline | Δ | n | page |
|---|---|---|---|---|---|---|
| monolithic open Video-LLM, full video | mIoU | Qwen3.5-9B 0.110 (N = 2,048) | CLIP retrieval 0.269 | −0.159 | 2,273 Q / 194 videos, 1 run, no variance | p3 (Table 1, Fig 3) |
| monolithic, Qwen3.5-9B at N = 768 | mIoU | 0.053 | CLIP 0.269 | −0.216 | 2,273 Q | p4 (Table 2) |
| closed models | mIoU | Gemini-3.5-flash 0.115; Gemini-2.5-flash 0.053 (1k frames); GPT-5.4 0.013 (64 frames) | CLIP 0.269 | — | 2,273 Q | p3, p9 |
| retrieve-then-ground, CLIP top-3 ±1 min → Qwen3.5-9B | mIoU | 0.354 (6 min of Video-LLM context) | CLIP alone 0.269 | +0.085 (1.32×) | 2,273 Q | p4 (Table 2) |
| hybrid, retrieval breadth K | GT coverage / mIoU | top-1 ±2.5 min 62.8% / 0.302; top-3 ±1 min 81.6% / 0.354; top-10 ±1 min 96.7% / 0.295 | — | — | 2,273 Q | p9 (Table 6) |
| GT-centred window, Qwen3.5-9B N = 768 | mIoU | 30 s 0.572 · 5 min 0.466 · 20 min 0.265 · full 0.053 | CLIP 0.534 / 0.387 / 0.331 / 0.269 | — | 2,273 Q | p10 (Table 7) |
| failure taxonomy, Qwen3.5-9B N = 2,048, IoU < 0.05 | share | search 85%, localization 11%, uncertain 4% | — | — | 100 sampled of 1,817 failures, GPT-5.4 classifier, ±5 min split | p4 (Fig 5) |
The paper reports no seeds, CIs or variance. The repo README reports a reproduction of Qwen3.5-9B at N = 768 of
0.0469 (script) and 0.0485 (lmms-eval), against the paper's 0.053.

## Ablations that matter for us
- **Retrieval carries the gain.** At K = 3, coverage is 81.6% and mIoU 0.354. K = 10 raises coverage to 96.7% but
  drops mIoU to 0.295, because the grounder must search ~20 min again (Table 6, p9). Search and coverage trade
  against context length, the same trade-off as route1 vs route2 here.
- **The localizer is fine when the window is right.** Qwen3.5-9B gets 0.572 at ±15 s and 0.466 at ±2.5 min
  (Table 7, p10). So the ceiling of "pick the right 5 min, then ground raw" is about 0.47 with a 9B grounder.
- **Missing:** any retriever other than CLIP. The paper names a stronger, temporally aware, language-grounded
  retriever as the natural next step (p5). A Video-LLM routing over its own compressed context is exactly that, and
  is untested. Also missing: a sensitivity check of the ±5 min search/localize split (p5).

## Reproduction assets
- code: https://github.com/naver-ai/ExtremeWhenBench — Apache-2.0. Contains `evaluation/eval_qwen35_vllm.py` (vLLM
  server, `--num-frames 768`) and an lmms-eval task. The README says that task needs a patched OpenAI adapter:
  without it, mIoU drops ~10× because client-side frame extraction strips the absolute timestamps.
- annotations: https://huggingface.co/datasets/min1321/extreme-when-bench — CC-BY-4.0, **not gated**, one parquet,
  `test` split, 2,273 rows. Fields: `qid`, `video_id`, `source_corpus`, `question`, `correct_interval` [start_s,
  end_s], `duration_s`, `video_duration_s`, `event_summary`, `category` (7 classes), `youtube_url` (empty for MLVU).
  The paper says MIT for its added annotations (p9); the HF card says CC-BY-4.0.
- videos: **not redistributed.** VideoMME 89 and LVBench 67 are YouTube ids; MLVU 38 come from its repo
  (https://github.com/JUNJIE99/MLVU). No download script is provided. The upstream licences (LVBench, MLVU,
  VideoMME) apply. Availability of each YouTube video and download from this cluster are unchecked.
- checkpoints: none (training-free).
- prompts: per-stage table only (Table 4, p8); the prompt texts are not in the paper.

## Mapping to this project
- **What maps.** The paper's hybrid is retrieve-then-read with an external retriever (CLIP). Route-then-expand is
  the same shape with the retriever inside the model. Frames are compressed to `<MEM>`s; the model picks a window
  ("Document j") from the all-`<MEM>` context (`route_logprobs`, trie scoring, tested up to 20 candidates); that
  window goes raw. On page images the pick is right 87% of the time at 10 retrieved pages, and the paired gap to
  uncompressed is −.017 (ledger 2026-10-28).
- **Why it does not fit as is: context size** (we infer). The mean video is 4,542 s. At 1 fps and nav compression
  (25 `<MEM>`/frame at rate 16, ~440 visual tokens per frame), that is ~113,500 `<MEM>`s. At 0.1 fps it is still
  ~11,400. Our decoder runs have used 2k–8k tokens, so a whole-video all-`<MEM>` context fits only at ~0.05 fps
  (1 frame / 20 s, ≈5,700 `<MEM>`s). That spacing is coarser than the 9 s median event. So:
  - either a **hierarchy**: route among ~10 segments of ~8 min, then among ~8 windows inside the chosen ones;
  - or `<MEM>`s with fewer tokens per frame.
- **Why it does not fit as is: candidate count.** 1-min windows give ~76 candidates per video (up to ~540 for 9 h).
  Routing was trained (`find`) on ≤ 4 static pages and tested to 20. The current trie code also re-runs the full
  prefill once per branching node.
- **What is missing.**
  - an interval metric (mIoU, R@τ) in our eval;
  - absolute timestamps in the frame labels ("Document j (t = …)");
  - a grounder for step 2: the uncompressed Qwen3.5-4B on the picked window's raw frames, as the paper does with 9B;
  - the videos.
- **Champions.** For routing, `expQ/ft_mm_4b_174908` (find-trained; route1 = uncompressed on docs). For frame
  compression, `expQ/ft_nav_4b` (nav judge .806 vs fair ceiling .792). Its stage 2 had no find task (ledger
  172615: p_image_ae .6 / cont .2 / text .2), so it is not expected to route.
- **Smallest test of the claim on our side** (search only, no timestamps): candidate sets like our retr10 doc test.
  - per query: the GT minute plus 9 other 1-min windows of the same video;
  - 12 frames per window (1 per 5 s, ~2 frames inside a 9 s event), 10 × 12 × 25 ≈ 3,000 `<MEM>`s;
  - measure: route recall@1/@3 vs CLIP ViT-L/14-336 on the same candidates (CLIP must be re-measured by us: the
    paper's Table 6 coverage uses ±1/2.5 min windows with NMS, a different protocol).

## Proposed arms (CAMPAIGN.md Arms-table format; gates inside the fast-metric noise band)
No campaign is live and `CAMP_NOISE_PT` is unset. For a recall proportion near .6 at n = 500 queries, 1 SE ≈ 2.2 pt,
so the gates use 2 SE ≈ 4.4 pt.
| arm | recipe file (committed) | factors (what varies) | held constant | gate (metric ≥ X vs whom) | budget (GPU-h) |
|---|---|---|---|---|---|
| V0 feasibility: videos reachable | to write: `scripts/video/fetch_ewb.sh` (CPU) | none | — | ≥ 150 of 194 videos downloaded legally from the source corpora; otherwise stop | 0 (CPU) |
| V1 search-only routing, 10 windows | to write: `scripts/build_video_eval_ewb.py` + `eval_mm_docqa.py --raw_policy route1` on windows | router: PISCO 174908 self-routing vs CLIP ViT-L/14-336 on the same candidates | 500 Q, 10 × 1-min windows, 12 frames/window, 451,584 px, rate 16 | PISCO recall@1 ≥ CLIP recall@1 − 4.4 pt (else self-routing loses to a cheap index) | ~2 (≈60k frame compressions at ~75 ms + routing) |
| V2 route-then-ground | to write: interval output + mIoU scorer | grounder input: picked window raw (route1 / route2) vs CLIP top-1 / top-3 window raw | same 500 Q, uncompressed Qwen3.5-4B grounder, absolute timestamps per frame | mIoU ≥ CLIP-hybrid mIoU − 4.4 pt on the same Q | ~3 |
V1 alone can falsify the transfer: if self-routing over 10 minute-windows is clearly worse than CLIP, a hierarchy over
an hour will not save it.

## Risks / open questions
- **Video access** is the first blocker:
  - YouTube downloads from the cluster are untested, behind the corporate proxy;
  - some videos may be gone;
  - the upstream licences are research-use and were not checked here.
- **Event grain vs sampling.** A 9 s event needs ≥ 0.2 fps to be seen at all, while hour-long all-`<MEM>` contexts
  only fit at ~0.05 fps. Without a hierarchy or an external first-stage index, the router never sees most events.
- **Routing out of distribution:**
  - find trained on ≤ 4 static images; windows are frame sequences;
  - candidates go well past 20;
  - the queries describe motion ("jump from a boat … then snorkel"), and the anti-CLIP filter was built to need
    motion or sequence (p9).
- **No timestamps in our interface.** The README notes that losing absolute time costs ~10× mIoU. Step 2 needs time
  labels in the prompt, and our models were never trained to emit intervals.
- **The comparison is not like for like.** The paper's grounder is 9B and ours would be 4B. Every comparison must be
  re-run on our side with the same grounder, not read off Table 2.
