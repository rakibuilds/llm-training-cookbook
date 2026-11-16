# LLM Training Cheatsheet: Formulas, Numbers, Flashcards, Self-Quiz

> Companion to `llm-training-deep-dive.md`. Built for fast recall and for pasting into an LLM as context ("quiz me on section C", "explain flashcard 14 like I'm new").

---

## A. Formulas to memorize

| What | Formula | Notes |
|------|---------|-------|
| Training compute | `C ≈ 6·N·D` | N = **active** params, D = tokens |
| Inference compute per token | `≈ 2·N` | forward only |
| Memory/param (mixed-precision Adam) | `2+2+4+4+4 = 16 B` | BF16 w + BF16 g + FP32 master + Adam m + Adam v |
| ZeRO-1 per-GPU state | `(4 + 12/DP)·N` bytes (÷ TP·PP) | shard optimizer only |
| ZeRO-2 per-GPU state | `(2 + 14/DP)·N` bytes | + shard grads |
| ZeRO-3/FSDP per-GPU state | `16·N / DP` bytes | everything sharded |
| MFU | `achieved FLOPs/s ÷ peak FLOPs/s` | healthy large runs: 35 to 45% |
| Training time | `C / (GPUs · peak · MFU)` | add failure/restart overhead |
| Pipeline bubble (1F1B) | `(p−1)/(m+p−1)` | p = stages, m = micro-batches |
| Ring all-reduce traffic/GPU | `≈ 2·(n−1)/n · size` | ≈ 2× message size, nearly independent of n |
| Chinchilla optimal tokens | `D ≈ 20·N` | frontier labs over-train well beyond |
| Global batch (tokens) | `micro_batch · seq_len · grad_accum · DP` | |
| GPU-hour cost | `GPUs × hours × $/GPU-hr` | DeepSeek used $2/hr as an assumption |
| DP degree | `total_GPUs / (TP · PP · CP)` | (EP is carved out of the DP × TP space in MoE layouts) |

### Quick calc template
```
Given: N (active params), D (tokens), G (GPUs), peak (FLOPs/GPU), MFU
C      = 6 * N * D
time_s = C / (G * peak * MFU)
days   = time_s / 86400
```
Example: N=70e9, D=15e12, G=2048, peak=989e12, MFU=0.40 → ~90 days.

---

## B. Numbers worth knowing cold

| Fact | Value |
|------|-------|
| H100 BF16 dense peak | ~989 TFLOPs |
| H100 FP8 dense peak | ~1,979 TFLOPs |
| H100 HBM | 80 GB, ~3.35 TB/s |
| NVLink 3 / 4 / 5 (per GPU) | 600 GB/s / 900 GB/s / 1.8 TB/s |
| H800 NVLink | ~400 GB/s |
| InfiniBand per GPU (typical) | 400 Gb/s ≈ 50 GB/s |
| Llama 3 405B | 16,384 H100s, 15.6T tokens, ~3.8e25 FLOPs, 38 to 43% MFU |
| Llama 3 ops | 466 interruptions in 54 days; >90% effective training time |
| DeepSeek-V3 | 671B total / 37B active, 14.8T tokens, 2.788M H800 GPU-hrs ≈ $5.6M (final run, $2/hr assumed) |
| DeepSeek-V3 parallelism | 16-way PP, 64-way EP, ZeRO-1, **no TP**, FP8, DualPipe |
| DeepSeek-R1 RL spend | ~$294K total (R1-Zero + data gen + R1), on top of V3 |
| FP8 relative loss error (V3) | < 0.25% vs BF16 |
| 70B model, full training state | ~1.12 TB (16 B/param) |
| 70B BF16 weights only | ~140 GB |

---

## C. Glossary (alphabetical)

- **All-gather:** every rank collects every other rank's shard.
- **All-reduce:** sum (or average) a tensor across ranks; every rank gets the result.
- **All-to-all:** every rank sends a distinct chunk to every other rank (MoE dispatch).
- **Activation checkpointing:** discard activations in forward, recompute in backward to save memory.
- **BF16:** 16-bit float with FP32's exponent range; no loss scaling needed.
- **Bubble:** idle time in a pipeline while stages wait on each other.
- **Context parallelism (CP):** split a long sequence across GPUs for attention.
- **DDP:** DistributedDataParallel; full replica per GPU, all-reduce grads.
- **DualPipe:** DeepSeek's bidirectional pipeline schedule overlapping compute and communication.
- **EP:** expert parallelism; MoE experts distributed across GPUs.
- **FSDP:** PyTorch fully sharded data parallel (ZeRO-3 style).
- **GQA:** grouped-query attention; share KV heads across query heads to shrink the KV cache.
- **GRPO:** group relative policy optimization; PPO-like RL using group reward statistics instead of a critic.
- **HBM:** high-bandwidth memory on the GPU package.
- **KV cache:** stored keys/values from earlier tokens during generation.
- **MFU:** model FLOPs utilization.
- **MLA:** multi-head latent attention; compresses KV into a low-rank latent to shrink the cache dramatically.
- **MoE:** mixture of experts; sparse activation of a subset of FFN experts per token.
- **MTP:** multi-token prediction training objective.
- **Muon:** optimizer using orthogonalized momentum updates for matrix params.
- **NCCL:** NVIDIA collective communication library.
- **NVLink / NVSwitch:** high-bandwidth GPU-to-GPU links inside a node/domain.
- **PP:** pipeline parallelism.
- **RLAIF:** RL from AI feedback (Constitutional AI).
- **RLHF:** RL from human feedback.
- **RLVR:** RL with verifiable rewards (math/code checkers).
- **SDC:** silent data corruption; wrong results with no error.
- **TP:** tensor parallelism.
- **WSD:** warmup-stable-decay LR schedule.
- **ZeRO:** Zero Redundancy Optimizer; shards optimizer/grad/param state across DP ranks.

---

## D. Decision flowchart (text version)

```
Does the model + 16B/param state fit on one GPU?
 ├─ Yes → DDP. Done.
 └─ No
     ├─ Fits if sharded across DP ranks? → FSDP / ZeRO-3
     └─ No (layers too big or comm too heavy)
         ├─ Have fast NVLink domain? → add TP (within domain)
         ├─ Spanning nodes on slower network? → add PP
         ├─ Context ≥ ~128K? → add CP
         └─ MoE? → add EP (consider TP=1 + PP + EP like DeepSeek-V3)
Then: tune micro-batches (shrink bubble), overlap comm, recompute activations selectively.
```

---

## E. Failure-mode debugging table

| Symptom | Likely cause | First thing to try |
|---------|--------------|--------------------|
| Loss spike then recovery or divergence | LR too high, bad data batch, FP16 overflow, Adam ε/β2 | Rollback, lower LR, inspect batch, use BF16, QK-norm/z-loss |
| OOM at start | Static state too big | More sharding (FSDP/ZeRO), TP/PP, smaller micro-batch |
| OOM mid-run (sporadic) | Activation spikes, long sequences, fragmentation | Activation checkpointing, cap seq length, `expandable_segments` |
| MFU far below 30% | Comm-bound, small micro-batch, data loader stalls, pipeline bubble | Profile; overlap comm; raise micro-batches; check dataloader |
| Hang with no error | NCCL deadlock or dead rank | Enable NCCL flight recorder/timeouts; check node health |
| One rank slower than the others | Straggler (thermal, bad NIC) | Per-rank timing; evict node |
| Loss differs between DP configs | Non-deterministic reduction, seed/data order mismatch | Fix seeds; compare grad norms step by step |
| NaNs in FP8 run | Scale-factor overflow, outliers | Finer scaling granularity, keep sensitive ops in BF16 |
| MoE quality drops, throughput drops | Expert load imbalance/collapse | Check router stats; adjust balancing method |
| Eval metric flat despite lower loss | Contamination, wrong eval, data mix issue | Audit eval and data mix |

---

## F. Flashcards (Q → A)

**Fundamentals**
1. **Q:** Why is the intra- vs inter-node bandwidth gap important?
   **A:** NVLink (~900 GB/s) vs InfiniBand (~50 GB/s per GPU) is ~18×. Parallelism strategies place chatty traffic (TP) on fast links and tolerant traffic (PP, DP) on slow ones.
2. **Q:** What does `6ND` estimate?
   **A:** Total training FLOPs: 2 forward + 4 backward per active parameter per token.
3. **Q:** Memory per parameter in standard mixed-precision Adam?
   **A:** 16 bytes (2 w + 2 g + 4 master + 4 m + 4 v).
4. **Q:** What's MFU and what is a good value?
   **A:** Achieved/peak FLOPs; ~35 to 45% is strong at scale.
5. **Q:** Why does BF16 beat FP16 for LLMs?
   **A:** Same exponent range as FP32 → no loss scaling, far fewer overflow issues.

**Parallelism**
6. **Q:** ZeRO-1/2/3 shard what?
   **A:** 1 = optimizer states; 2 = + gradients; 3 = + parameters.
7. **Q:** Which parallelism must stay inside an NVLink domain, and why?
   **A:** Tensor parallelism: per-layer all-reduces are latency/bandwidth critical.
8. **Q:** Pipeline bubble fraction?
   **A:** `(p−1)/(m+p−1)`; shrink by raising micro-batches or using interleaved/zero-bubble/DualPipe schedules.
9. **Q:** Difference between Megatron sequence parallelism and context parallelism?
   **A:** SP shards LayerNorm/dropout activations along sequence *inside the TP group* to save memory; CP shards the whole sequence (including attention) across GPUs for long context.
10. **Q:** How do Ring Attention and DeepSpeed-Ulysses differ?
    **A:** Ring passes KV blocks around a ring overlapped with compute; Ulysses uses all-to-all to switch between sequence-split and head-split.
11. **Q:** What's hard about expert parallelism?
    **A:** All-to-all bandwidth and load imbalance across experts.
12. **Q:** Why did DeepSeek-V3 skip tensor parallelism?
    **A:** TP's heavy intra-layer communication is costly on H800's reduced NVLink; they used PP + EP + ZeRO-1 and recomputation to fit memory.

**Precision**
13. **Q:** Why does FP8 need fine-grained scaling?
    **A:** Tiny dynamic range; outliers break per-tensor scales. V3 uses 1×128 tiles (activations), 128×128 blocks (weights).
14. **Q:** Why promote FP8 GEMM partial sums to FP32?
    **A:** Tensor-core accumulation keeps only ~14 bits; promoting (every 128 elements in V3) prevents error buildup.
15. **Q:** What stays in high precision in V3's FP8 recipe?
    **A:** Embeddings, output head, MoE gating, norms, attention ops; plus FP32 master weights/grads.

**Systems/ops**
16. **Q:** Why overlap communication with computation?
    **A:** Hides comm latency so GPUs don't idle; main lever for MFU at scale.
17. **Q:** What is silent data corruption?
    **A:** A faulty GPU yields wrong numbers without error; detect via checksums/golden tests/loss anomalies.
18. **Q:** Effective-training-time figure from Llama 3?
    **A:** >90% despite 466 interruptions in 54 days.
19. **Q:** Why are optimizer states ~6× the BF16 weights in checkpoints?
    **A:** FP32 master (2×) + two FP32 moments (4×) relative to BF16 weights.

**Post-training**
20. **Q:** What does GRPO remove compared with PPO?
    **A:** The value network (critic); it uses group-relative rewards as the baseline.
21. **Q:** R1-Zero vs R1?
    **A:** R1-Zero = pure RL from base, no SFT (readability issues); R1 adds cold-start SFT, multi-stage RL, rejection sampling.
22. **Q:** What is RLAIF / Constitutional AI?
    **A:** AI-generated critiques and preference labels guided by written principles, used mainly for harmlessness.
23. **Q:** What is RLVR?
    **A:** RL where rewards come from automatic verifiers (math answers, unit tests).

**Interpretation / honesty**
24. **Q:** What does DeepSeek-V3's "$5.6M" include?
    **A:** Only the final training run's GPU-hours at an assumed $2/hr; not research, ablations, failed runs, or data.
25. **Q:** Which lab details are actually public?
    **A:** Meta (Llama 3 paper) and DeepSeek (V3/R1 reports) are detailed; OpenAI/Anthropic training internals largely are not.

---

## G. Self-quiz (answers at the bottom)

1. A 13B dense model is trained on 2T tokens on 256 H100s at 40% MFU (peak 989 TFLOPs). Roughly how many days?
2. Full mixed-precision Adam state for a 30B model, in TB?
3. You have 512 GPUs with TP=8, PP=4. What is DP?
4. With p=8 pipeline stages and m=32 micro-batches, what's the 1F1B bubble fraction?
5. Why might a training run have *worse* throughput after enabling ZeRO-3 compared to ZeRO-1?
6. Name two reasons a loss spike could happen and a first response to each.
7. Why is `N` in `6ND` the *active* parameter count for MoE?
8. Give one reason to prefer rented GPUs and LoRA fine-tuning over pretraining for a solo builder.
9. True/false: "DeepSeek-V3 cost $5.6M to build." Explain.
10. Which hardware platforms has Anthropic publicly said it trains on?

### Answers
1. C = 6 × 13e9 × 2e12 = 1.56e23. Throughput = 256 × 989e12 × 0.4 ≈ 1.01e17. Time ≈ 1.54e6 s ≈ **~18 days**.
2. 30e9 × 16 B = 480 GB ≈ **0.48 TB**.
3. 512 / (8 × 4) = **16**.
4. (8−1)/(32+8−1) = 7/39 ≈ **18%**.
5. ZeRO-3 all-gathers parameters every layer (more communication); if the network is slow or overlap is poor, comm dominates.
6. E.g., (a) LR too high → lower LR/rollback; (b) bad data batch → inspect/skip the batch; (c) FP16 overflow → switch to BF16.
7. Only the activated experts do FLOPs per token, so compute scales with active parameters, not total.
8. Cost/risk: pretraining needs huge compute and data; LoRA on rented GPUs gets domain gains for a tiny fraction, with faster iteration.
9. **False/misleading.** It's the final run's GPU-hour cost only; excludes research, ablations, failed runs, data, salaries, and hardware.
10. AWS Trainium, Google TPUs, and NVIDIA GPUs.

---

## H. Prompts to use with an LLM tutor

- "Quiz me on section F flashcards 6 to 12, one at a time, harder each time."
- "Give me a 2,048-GPU config for a 140B dense model with 16K context; show the memory math."
- "Explain DualPipe with a timeline diagram in ASCII."
- "Here's my profiler trace summary: [paste]. Is this comm-bound or compute-bound, and what should I change first?"
- "Challenge my claim: [your claim about frontier training]. Ask for the primary source."
