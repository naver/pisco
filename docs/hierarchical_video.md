# Hierarchical compression of hour-long video — literature review (2026-10-28)

Question: can a one-hour video be compressed into `<MEM>`s hierarchically, and does the hierarchy help the task
(search, grounding, QA)? Sub-question: does temporal information survive the coarse level? This review was done before
building anything. Each claim carries its source; "abstract only" marks papers read from the abstract page.

## 1. Where we start (our own numbers)

ExtremeWhenBench queries over LVBench videos (`papers/extremewhenbench.md`, `scripts/video/`, results in
`expQ/video/results/`), model 174908 at rate 16, no training. Recall@k = the picked minute overlaps the event
(higher is better).

| test (ledger 2026-10-28/29) | CLIP ViT-L/14 | PISCO 174908, rate 16 |
|---|---|---|
| 10 one-minute windows, recall@1 / @3 (500 Q) | .750 / .932 | .738 / .870 |
| static categories (scene, object, environment), recall@1 | .660 / .845 / .722 | .759 / .862 / .750 |
| dynamic categories (action, reaction), recall@1 | .802 / .745 | .716 / .596 |
| whole video (mean 81 min, chance@1 .015), coarse level 2 frames/min, recall@1 / @3 | .254 / .424 (same frames); .446 / .688 (12 frames/min) | .204 / .370 (3.9k `<MEM>`, 3.0 s/query) |
| whole video, zoom: top-3 minutes at 12 frames/min, recall@1 | - | own pick .284; **CLIP top-3 → model zoom .498** (+.052 over CLIP dense, CI +.018 to +.088) |
| pages made coarser without retraining, pick@1 at rate 8 / 32 / 64 | - | .87 / .66 / .55 |

LVBench multiple choice (1,026 Q, chance .25, question-only control .257), accuracy (higher is better): coarse level
only .401, fine level of the model's own top-3 .421, both levels in place .437, both with the gold minute in the
top-3 .492 (gold is in the model's top-3 for 33% of questions: the pick is the bottleneck). Speech transcripts
(59/67 videos) add nothing (coarse+ASR +.007, both+ASR +.002).

Per-frame `<MEM>`s match CLIP on static content and lose on motion. An untrained coarse level is worse than CLIP on
the same frames, and keeping 80 minutes of coarse `<MEM>`s in context distracts the fine pick (−.048). What works
without training: CLIP searches, the model zooms.

## 2. Four families of prior work

### 2a. Compression inside the Video-LLM

| method | hierarchy | tokens | temporal handling | key numbers |
|---|---|---|---|---|
| [VideoChat-Flash / HiCo](https://arxiv.org/abs/2501.00574) (ICLR 2026) | clip level: 4 frames → one spatio-temporal encoder (UMT-L) → similar-token merge → 64 tokens (16/frame). Video level: progressive token dropout inside the LLM (uniform in shallow layers, attention-based in deep layers) | 16/frame, ~1/50 | the clip encoder fuses frames; one text line "video lasts N s, T frames sampled". Removing that line: Charades mIoU 48.0 → 44.2 | 7B: MLVU 74.7, VideoMME 65.3, LVBench 48.2; NIAH 99.1% at 10k frames. Merge ratio 27% / 7% / 2% of tokens → +2.5 / +0.4 / −0.9 vs uncompressed; spatial pooling at 2% −3.2, resampler −8.3 |
| [Video-XL](https://arxiv.org/abs/2409.14485) (CVPR 2025) | one Visual Summarization Token per interval; the interval lives on as that token's KV | adaptive per interval | order kept by position of the VSTs; not ablated | "minimal loss at 16×", thousands of frames on one A100 (abstract only) |
| [Video-XL-2](https://arxiv.org/abs/2506.19225) | chunked prefill, then **bi-level KV decoding**: task-relevant chunks reloaded dense, the others sparse | — | — | >10k frames on one A100 (abstract only) |
| [MambaMia](https://arxiv.org/abs/2506.13564) (NAVER AI, AAAI 2026) | bidirectional SSM + learned queries inserted periodically; spatial and temporal downsampling | small fixed budget | SSM runs along time | "competitive at a much smaller budget" (abstract only) |
| [Tempo](https://arxiv.org/abs/2604.08120) (Apr 2026) | **small VLM compressor (Qwen3-VL-2B) → memory tokens → LLM decoder (Qwen3-4B)**, linear projector. 8-frame segments at 2 fps, up to 128 memory tokens each, cut per query to k_i | 0.5–16 per frame, 2.8–4.3 measured | text tag `<t=2.0s>` per segment; **every segment keeps ≥ 4 tokens as a temporal anchor** | LVBench 52.7 (VideoChat-Flash 48.2), VideoMME 67.8, MLVU 75.6, LongVideoBench 65.1. Ablations (LVB): uniform budget 61.9 vs query-aware 65.1; hard top-K 63.5; pruning irrelevant segments to zero 63.9 vs anchors 65.1; keep first k tokens 65.1 vs last k 61.9 |
| [Matryoshka MM](https://arxiv.org/abs/2405.17430) (ICLR 2025) | nested coarse-to-fine token sets per image, chosen at inference | 1 … 576 per frame | none | budget trade: 2,880 frames × 1 token or 5 frames × 576 |

Tempo is the nearest thing to multimodal PISCO: a small VLM compressor, soft memory tokens, a larger text decoder. Two
differences matter. Tempo's compressor sees the query, so a video must be recompressed for every question (no offline
index). And Tempo is one level: segments are never summarised again.

### 2b. Memory and streaming

[MovieChat](https://arxiv.org/abs/2307.16449), [MA-LMM](https://arxiv.org/abs/2404.05726), [Flash-VStream](https://arxiv.org/abs/2406.08085)
keep a bounded memory of merged frame features. [ReKV](https://www.researchgate.net/publication/389547995) keeps all
video KV off-GPU and retrieves by cosine similarity (cache grows linearly); [StreamMem](https://agenticlearning.ai/research/stream-mem/paper.pdf),
[CacheFlow](https://arxiv.org/abs/2511.13644), [StreamKV](https://arxiv.org/abs/2511.07278), [StreamMeCo](https://arxiv.org/abs/2604.09000)
compress that cache (CacheFlow: 70.7% of tokens dropped for −0.3 pt). These are flat memories: one level, merged by
similarity or attention score. They show the storage side is solved but say nothing about a coarse level for search.

### 2c. Search or retrieve, then read

| method | retriever | what the reader gets | finding |
|---|---|---|---|
| [ExtremeWhenBench](https://arxiv.org/abs/2606.12300) (`papers/extremewhenbench.md`) | CLIP frames, NMS 60 s | top-3 ±1 min raw to Qwen3.5-9B | hybrid 0.354 mIoU vs monolithic 0.110; top-10 (20 min) falls to 0.295: the reader searches again |
| [T*](https://arxiv.org/abs/2504.02259) (CVPR 2025) | the VLM itself, frames tiled into one image grid, iterative | 32 selected frames | LongVideoBench-XL: GPT-4o 50.5 → 53.1, LLaVA-OV-72B 55.5 → 62.4 |
| [OneClip-RAG](https://arxiv.org/abs/2512.08410) | query-guided chunking + retrieval in one pass | one clip | Qwen3-VL-8B on MLVU "to GPT-5 level" (abstract only) |
| [Rethinking RAG in long videos / CARVE](https://arxiv.org/abs/2606.13141) (2026) | Qwen3-VL-Embedding-2B, reranker; compares frame vs clip × visual vs text summary | top-5 chunks, each in its winning form | clip-level visual is the best single form (R@5 .507); mixing modalities always helps; **adding a second granularity within one modality does not** (frame+clip .497 < clip .507); the generator still misorders events after a correct retrieval |
| [MomentSeeker](https://arxiv.org/abs/2502.12558) | benchmark, 1,800 Q, mean 20 min | — | dual encoders (InternVideo2 R@1 19.7) beat MLLM embedders (E5V 14.3) and most generators; Gemini-2.5-Pro 29.6 |
| [VideoStir](https://arxiv.org/abs/2604.05418) | MLLM intent scorer, not embedding similarity | — | (abstract only) |

### 2d. Coarse-to-fine grounding and tree search

| method | coarse level | fine level | training | numbers |
|---|---|---|---|---|
| [ReVisionLLM](https://arxiv.org/abs/2411.14901) (CVPR 2025) | **one 768-d token per 125 s window** (MAD) or 500 s (VidChapters): CLIP CLS per frame, 250 frames, query cross-attention + a CLS-like token. 100 windows (~150 min) in one pass; answers "From s to e" or "Not present", ranked by answer confidence | dense per-frame CLIP features of the chosen windows, projected to the LLM; frame indexes as boundaries | hierarchical: LoRA on short clips with same-video negatives, then adapter, then LoRA on long videos | MAD R1@0.1 15.0 vs 12.4; VidChapters R1@0.7 21.8 vs 17.6; 57% of frames reach the LLM |
| [VideoTree](https://arxiv.org/abs/2405.19209), [VideoMiner](https://openaccess.thecvf.com/content/ICCV2025/papers/Cao_VideoMiner_Iteratively_Grounding_Key_Frames_of_Hour-Long_Videos_via_Tree-based_ICCV_2025_paper.pdf) | clustering or captions build a tree, LLM descends | key frames | training-free | — |
| [VideoTreeSearch](https://arxiv.org/abs/2607.16189) (Jul 2026) | tree from **scene boundaries** (CLIP cosine jump > mean + 1.5 sd), 3–8 children per node, leaves < 64 s; node = up to 64 frames at 224 px, each labelled with its timestamp; visited nodes shown as captions | actions zoom_in / **zoom_out** / shift / answer | Qwen3-VL-8B; SFT on 6.5k synthetic trajectories with deliberate wrong turns, then GRPO (format .5, IoU 1, answer .5) | CG-Bench mIoU 16.8 vs 12.1 uniform; LVBench 54.7 vs 47.6; removing zoom_out/shift −2.0 to −2.9 mIoU |
| [TimeSearch-R](https://arxiv.org/abs/2511.05489), [LongVideoAgent](https://arxiv.org/abs/2512.20618) | search interleaved with reasoning; multi-agent | — | RL / agents | Haystack-LVBench T-F1 8.1 (TimeSearch-R) |

Text-side analogues: recursive soft-token compression ([AutoCompressor](https://arxiv.org/abs/2305.14788),
[RMT](https://arxiv.org/abs/2207.06881)), gist tokens and [ICAE](https://arxiv.org/abs/2307.06945), recursive summary
trees ([RAPTOR](https://arxiv.org/abs/2401.18059)), hierarchical context merging ([HOMER](https://arxiv.org/abs/2404.10308)).
A 2025 ACL study ([gist compression limits](http://playbigdata.ruc.edu.cn/dou/publication/2025_ACL_Silver.pdf)) finds the
losses sit at segment boundaries and on low-surprise tokens. None of these tests a coarse level trained for *picking*.

## 3. Does temporal information survive compression?

- **Most benchmarks do not need order.** GPT-4o and Gemini keep their scores under frame shuffling
  ([2505.14321](https://arxiv.org/abs/2505.14321)); LLaVA-OV keeps > 82% ([2506.06928](https://arxiv.org/abs/2506.06928)).
  Order-dependent sets do react: EgoTempo sequence categories ([2503.13646](https://arxiv.org/abs/2503.13646)),
  TVBench and TempCompass by design. On [MotionBlind](https://www.alphaxiv.org/abs/2609.09528) (speed, direction),
  all open models fall to 0% with shuffled frames and are at chance with ordered ones; Gemini 3.1 Pro reaches 60%.
  So motion is barely read by any current model; order of events is read by some.
- **Two channels carry order, and models differ in which they use** ([2607.12304](https://arxiv.org/abs/2607.12304)):
  the pixels (what the frames show) and the positions (RoPE). Reverse the frames and keep RoPE forward: Molmo2 answers
  from positions, Qwen3-VL from the visual sequence. This is directly our B' vs B question: in B' the compressor can
  read order from pixels and positions; in B only positions remain.
- **Compression by similarity destroys temporal evidence.** Merging near-duplicate tokens across frames removes
  "the object stayed there" ([OTT-Vid](https://arxiv.org/abs/2605.11803)), and single-stage pruning causes "temporal
  disarray" ([DyCoke](https://arxiv.org/abs/2411.15024)); grounding tasks drop far more than QA under such methods.
  Learned compression (HiCo's clip encoder, Tempo's memory tokens) does not show this failure.
- **What keeps time in practice:** a text timestamp per segment (Tempo; Qwen3-VL replaced time-aligned RoPE by
  interleaved timestamp text, [report](https://arxiv.org/abs/2511.21631)); one duration line in the prompt (HiCo,
  −3.8 mIoU without it); never dropping a segment to zero tokens (Tempo, +1.2); frame timestamps on every node
  (VideoTreeSearch). Even then the reader can misorder events after a correct retrieval (CARVE).

## 4. What this says about our options

| our option | nearest prior work | what it found | implication |
|---|---|---|---|
| A. sampling pyramid (2 frames per minute, per-frame `<MEM>`s) | VideoTreeSearch nodes, T* | works as a tree *with* backtracking and RL; coarse frames miss short events | the running test gives the floor; without zoom_out an early miss is final |
| B'. joint segment (12 frames in one compressor chunk, high rate) | HiCo clip level (4 frames → 64 tokens through a spatio-temporal encoder); ReVisionLLM coarse token; Tempo segments | joint encoding beats pooling and resamplers at equal budget (HiCo); one token per 125 s is enough to rank windows (ReVisionLLM) but its absolute recall is low | the compressor must see frames together; 12 frames → ~40 `<MEM>` is in the range others use |
| B. recursive (fine `<MEM>`s → coarse `<MEM>`s) | AutoCompressor, RMT, RAPTOR | recursive soft tokens work for text; no video instance | nothing contradicts it; it is the only way to index 9 h once, but order is positions-only |
| route-then-expand across levels | Tempo anchors; Video-XL-2 bi-level KV; EWB K=3 vs K=10 | keep every segment at a few tokens and expand the relevant ones; expanding too much re-creates the search problem | coarse `<MEM>`s for all minutes + fine `<MEM>`s for top-3 is the pattern others converge on |
| self-routing vs an external index | MomentSeeker; EWB; our 10-window test | dual encoders beat MLLM embedders on moment retrieval; our pick ≈ CLIP at 10 candidates | the hierarchy must win on dynamic queries (where CLIP is per-frame) or it will not beat CLIP + reader |
| query-dependent coarse level (OSCAR mode) | Tempo (+3.2 over query-agnostic) | helps, but forces recompression per query | keep level 0 query-independent (offline index), consider query-dependence only at level 1 |

## 5. Gaps nobody has filled

1. A **two-level learned compressor** whose coarse level is trained to pick and whose fine level is the same vectors
   reused (index once, route many times). Tempo is one level and per query; ReVisionLLM's coarse token feeds a
   grounder, not a `<MEM>` decoder.
2. A **temporal-order probe on compressed segments** (same minute, frames reversed, as a distractor). Shuffle tests
   exist for raw frames only.
3. On ExtremeWhenBench, **any retriever other than CLIP**. The paper names a temporally aware, language-grounded
   retriever as the open item.

## 6. Design rules taken from the review

1. Build the coarse level from **all frames of the segment, jointly** (B'), with real consecutive frame pairs in the
   vision tower's temporal patch instead of a duplicated still.
2. **Segments by scene boundary**, not fixed minutes (VideoTreeSearch): CLIP cosine jumps are cheap and already computed.
3. **A text timestamp before every segment** in the compressor chunk and in the decoder label.
4. **Never drop a segment to zero**: all minutes stay as coarse `<MEM>`s while the picked ones are expanded (route2).
5. **Train with order-dependent queries** (X then Y vs Y then X; the second time X) and same-video negatives
   (ReVisionLLM); add the reversed-minute probe to the eval.
6. **Allow zoom_out**: after expanding, score the coarse level again (iterative routing, also needed for HotpotQA-style
   multi-evidence, ledger 2026-10-28).
7. Keep level 0 **query-independent** so the index is built once per video.
