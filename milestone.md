---
layout: default
title: Project Milestone Report
---

# 418 Final Project Milestone Report

**Sarah Huang (sarahh2), Gene Yang (geney)**

**November 23, 2026**

[← Back to Homepage](./)

---

## Summary

We will implement and evaluate a chunked algorithm for Recirculation, an LLM inference method that introduces heavy sequentialness. We want to profile and optimize it for the GPU we use, and benchmark its speed and prediction quality against vanilla inference and sequential recirculation. So far, we have measured and profiled the sequential baseline and run an initial perplexity evaluation. We have a preliminary plan for chunked recirculation and are in the process of implementing it.

## Background

Traditional transformers process all the tokens in an input prompt in parallel. Although each token attends to previous tokens, the computation at each layer can be done across all token positions at once. GPUs are particularly efficient at this process (prefill) because of the large amount of parallel computation.

Recirculation is a recently proposed technique for improving a transformer's state tracking ability. It takes the hidden state from a later layer, mixes it into an earlier layer at the same token position, and recomputes the subsequent layers. The updated information is stored in the KV cache, allowing later tokens to use it. However, this introduces dependencies between consecutive tokens, making prefill largely sequential. Existing implementations can partially address this by overlapping computation across tokens, but the sequential dependency remains.

Instead of processing each token individually, we propose processing a chunk of tokens in parallel, then recirculating the entire chunk before moving to the next chunk. Tokens within a chunk cannot fully benefit from the recirculation of earlier tokens in that chunk, making the computation approximate. We want to investigate whether this can significantly accelerate prefill without losing much of Recirculation's improvement in model quality.

## The Challenge

The main challenges from our proposal still apply:

1. We need to move the sequential dependencies from individual tokens to chunks while keeping the computation useful. Larger chunks give the GPU more work at once, but also change the information available to each token.
2. We need to manage the original and recirculated KV entries correctly. Replay must replace the right entries, and attention must not use future tokens or stale cache states.
3. We need to distinguish the cost of the algorithm from the cost of running many small GPU operations. Profiling itself adds overhead, so we use separate unprofiled runs for latency measurements.

Our remaining unknown is how much speedup chunking will give us and how much it will affect perplexity. We have not yet measured this tradeoff.

## Progress and Preliminary Results

### Sequential Baseline

We set up a Google Colab environment and measured vanilla inference and the starter repository's sequential recirculation path using Gemma 3 1B IT. The timing experiments used an NVIDIA L4 GPU, FP16, SDPA attention, and batch size one. Our recirculation settings were source layer 25, destination layer 20, α = 0.04, and no ramp.

We measured prefill and decode at context lengths of 128, 256, 512, 1024, and 2048 tokens. Timing excludes model loading and tokenization. Each point uses 11 paired trials after warmup. We also checked the sequential timing wrapper against the repository's original generate method at six prompt lengths: all eight generated tokens matched in each check, with zero recorded logit difference. This checks our wrapper against the starter code; it does not establish correctness for the proposed chunked approach.

At 2048 tokens, vanilla prefill took 0.111 seconds, while sequential recirculation took 111.585 seconds. Recirculation stayed near 18–19 prompt tokens per second as the prompt grew. Decode had a much smaller difference: 45.04 ms/token for vanilla versus 57.68 ms/token for recirculation, or about 1.28× slower. These measurements give us a baseline for evaluating chunking; the prefill ratio is not a speedup that we have achieved.

For decode, we supplied the same 16 continuation tokens to both methods and measured the average time per token. The continuation was tokenized before timing. Each step still processes one token through the model and updates the KV cache, but does not recompute earlier tokens. This keeps the workload the same even if the models would generate different text.

### What We Looked at in Profiling

We profiled GPU operations, the first pass and replay, gaps between operations, and memory use. In the detailed 32-token recirculation prefill trace, feed-forward layers took 55.66% of GPU kernel time, attention took 24.33%, normalization took 17.60%, and other operations took 2.41%. Feed-forward computation is therefore one of the main areas we want to focus on.

| Operation | Share of GPU kernel time |
|---|---:|
| Feed-forward | 55.66% |
| Attention | 24.33% |
| Normalization | 17.60% |
| Other / output projection | 2.41% |

*Where GPU kernel time goes in the detailed 32-token sequential recirculation prefill trace. Percentages show each operation's share of summed kernel time. Gaps between GPU operations are excluded and shown separately in the timeline.*

In an isolated MLP test, processing 16 or 32 independent vectors together was 10.57× or 20.10× faster than processing them one at a time. This gives us a reason to try batching work across tokens. It does not tell us the full-model speedup, since we still need to handle attention, cache updates, and dependencies between tokens.

We also saw many small launches and gaps between GPU operations. The detailed 32-token trace contained 85,367 kernels for recirculation, compared with 2,295 for vanilla. In the lighter trace, recorded GPU activity covered 357 ms out of 3,170 ms, with no observed first-pass/replay overlap. CPU work, launching kernels, and synchronization can contribute to these gaps. The detailed profiler made prefill 1.77× slower, so these traces help us identify things to investigate rather than measure normal execution time.

<iframe
  src="./figures/04_recirculation_timeline.pdf"
  title="GPU activity during sequential recirculation prefill"
  width="100%"
  height="480"
  style="border: 1px solid #ddd;">
</iframe>

[Open the GPU timeline figure](./figures/04_recirculation_timeline.pdf)

*GPU activity during a 32-token sequential recirculation prefill. The zoom shows the first pass and upper-layer replay. Blank intervals contain no recorded GPU activity for this process. This is a profiled run, not the latency used in our baseline timing results.*

At 2048 tokens, peak GPU allocation was 2.078 GiB for vanilla and 1.939 GiB for recirculation, including model weights. Both had about 19 MiB of KV data, but vanilla retained 52 MiB of underlying storage compared with 19 MiB for recirculation. A tensor slice can keep a larger allocation alive. We also looked at copies during cache updates, which are another possible optimization target; we have not established that memory bandwidth is the bottleneck.

### Perplexity Baseline

We ran a separate quality evaluation with Gemma 3 1B IT on an NVIDIA T4 GPU, using the same precision and recirculation settings. We used eight BookSum/Chapter test-set windows, each containing a BOS token and 1023 chapter tokens, for 8184 scored tokens total.

| Method | Perplexity |
|---|---:|
| Vanilla Gemma 3 1B IT | 37.2416 |
| Sequential recirculation | 37.0507 |

Recirculation reduced aggregate perplexity by 0.5125% and improved seven of the eight windows. We will use this sampled 1B baseline to evaluate the effect of chunking.

For comparison, Table 1 in the paper reports an 11.22% perplexity reduction for Gemma 3 4B IT on BookSum/Chapter, with a baseline perplexity of 26.63. The 0.19% reduction in the same row is for the pretrained 4B model; the instruction-tuned result is the more relevant comparison because we used an instruction-tuned model.

Our improvement is smaller, although this is not a matched comparison: we used a smaller model, different recirculation settings, and only 8184 scored tokens, compared with the paper's reported 6.9M tokens for BookSum/Chapter. These results give us an initial 1B baseline rather than a reproduction of the paper's 4B result. A matching 4B evaluation remains future work if time and GPU resources allow.

### Preliminary Plan for Chunked Recirculation

We have a preliminary plan for the chunked algorithm and are in the process of implementing it. The plan is to split a prompt into consecutive chunks of at most K tokens, compute a first pass using causal attention and the preceding chunks' cache, then apply feedback and replay the affected upper layers before advancing. For K > 1, this changes within-chunk dependencies. After implementing it, we will check that K = 1 matches sequential recirculation, including logits and KV-cache state, and test partial chunks before interpreting speed or quality results.

## Resources and Platform Choice

We are using the following resources:

1. [Recirculation starter code](https://github.com/ModelCloud/Recirculation)
2. [Recirculation paper](https://arxiv.org/pdf/2608.17981v3)
3. [BookSum/Chapter dataset](https://huggingface.co/datasets/kmfoda/booksum)
4. NVIDIA GPUs through Google Colab: L4 for our systems baseline and T4 for the separate perplexity evaluation.
5. PyTorch and the PyTorch profiler.

We are still targeting a single GPU, since the parallelism we want to exploit is across tokens and transformer layers within one inference request. This avoids introducing inter-GPU communication costs that are not central to our project. PyTorch lets us control the scheduling and cache management while using existing kernels for matrix multiplications and attention.

## Goals and Deliverables

### Plan to Achieve

Our main deliverables remain the same as in the proposal: a working chunked implementation, latency and memory measurements for at least four chunk sizes, and a comparison of perplexity against vanilla and sequential recirculation. We also want to explain the speedup and its limitations using profiling and synchronization analysis.

We have completed the sequential performance baseline and an initial quality evaluation, but have not yet established chunked K = 1 equivalence or measured K > 1. Implementation is in progress, while validation and benchmarking remain to be done. We have updated our timeline to reflect this. We still plan to produce the required deliverables, and will prioritize a working chunked implementation, correctness checks, and a matched speed–quality evaluation over additional low-level optimizations.

### Hope to Achieve

We still hope to achieve at least 2× prefill speedup over sequential recirculation with minimal increase in perplexity. If the initial implementation gives promising results, we will investigate a further optimization, such as reducing cache copies or overlapping independent work. If the speedup is small or comes at a high quality cost, we will explain the bottlenecks and the tradeoff instead.

For the poster session, we plan to show prefill latency across chunk sizes, a speed–perplexity comparison, and a GPU timeline explaining where time is spent. We will focus on reproducible benchmark results; a live demo is optional.

## Updated Timeline

1. **Progress through the milestone (Nov 23):** Environment setup, sequential profiling, and an initial quality evaluation are complete. We have a preliminary chunking plan and are in the process of implementing it; validation and benchmarking are still pending.

2. **Week 3 (Nov 23–29):** Submit the milestone report and retain the completed 1B quality baseline. Finish the initial chunked implementation and check K = 1 logits and KV-cache state, including partial chunks, then benchmark K = 1, 2, 4, 8, 16. Debug chunking and evaluate perplexity on the same BookSum windows.

3. **Week 4 (Nov 30–Dec 6):** Run and profile the chunked implementation, expand the quality evaluation if time permits, and compare latency, memory, and perplexity. Investigate one further optimization if the initial results justify it.

4. **Week 5 (Dec 7–10):** Finish the writeup and figures, create the poster, and practice the presentation. Submit the final report on Dec 10.

5. **Final Presentation (Dec 11, 5:30–8:30 PM):** Print the poster and present our results.

## Note on Proposal Feedback

Our understanding was that the proposal was graded out of 1.0. We initially received a 0, then requested a regrade because the TA had not seen all of our pages. After the regrade, our score changed to 0.5, but we did not receive any additional feedback beyond the original comment:

> The basics of the proposal are approved, but your proposal is missing the schedule, different deliverables that you plan vs hope to achieve, etc.

Our proposal included a schedule and separate "Plan to Achieve" and "Hope to Achieve" sections, so we were unsure whether there were other concerns we still needed to address. Could you clarify whether there are any remaining issues with the proposal or project scope that we should address moving forward?

---

[← Back to Homepage](./)