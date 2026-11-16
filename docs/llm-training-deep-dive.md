# LLM Training Deep Dive: How Frontier Labs Actually Train Models

> Corrected and expanded reference. Companion file: `llm-training-cheatsheet.md` (formulas, numbers to memorize, flashcards, self-quiz).
> Status labels used throughout: **[PUBLIC]** = documented in a paper/report, **[INFERRED]** = widely believed but not confirmed, **[UNKNOWN]** = labs have not disclosed.

---

## 0. How to use this document

**Three passes, not one.**

1. **Pass 1 (1 hour): Mental model.** Read sections 1, 2, 12. Goal: be able to explain *why* training is hard in two minutes without notes.
2. **Pass 2 (a week): Mechanics.** Read 3 to 9 slowly. For every concept, do the matching item in the Mastery Path (section 13).
3. **Pass 3 (ongoing): Case studies and papers.** Read section 11 next to the original papers. Re-derive the numbers yourself (the cheatsheet has the formulas).

**The first-principles rule:** every technique in this document exists to fight one of three physical limits:

| Limit | Question it answers | Techniques that fight it |
|-------|---------------------|--------------------------|
| **Compute** (FLOPs) | Can we do enough math in the time/budget we have? | Better GPUs, FP8, MoE (fewer active params), FlashAttention, high MFU |
| **Memory** (GB per GPU) | Does the model + optimizer + activations fit? | ZeRO/FSDP, TP, PP, activation recomputation, lower precision |
| **Bandwidth** (GB/s between GPUs) | Can GPUs exchange data fast enough that they don't sit idle? | Mapping parallelism to topology, comm/compute overlap, DualPipe, fewer all-to-alls |

When you meet a new technique, ask: *which limit is this fighting, and what does it cost in the other two?* Almost every technique trades one limit for another.

---

## 1. The Full Stack Overview

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 5: Data & Post-training Recipes                  │
│  (data mix, filtering, SFT, RLHF/RLAIF, GRPO, evals)    │
├─────────────────────────────────────────────────────────┤
│  LAYER 4: Model Architecture & Algorithms               │
│  (Transformer variants: MLA, MoE, GQA, attention tweaks)│
├─────────────────────────────────────────────────────────┤
│  LAYER 3: Training Framework & Orchestration            │
│  (PyTorch, Megatron-LM, DeepSpeed, FSDP, Ray, Triton)   │
├─────────────────────────────────────────────────────────┤
│  LAYER 2: GPU Compute & Communication                   │
│  (CUDA, cuBLAS, cuDNN, NCCL, NVLink, InfiniBand)        │
├─────────────────────────────────────────────────────────┤
│  LAYER 1: Hardware Cluster                              │
│  (GPUs: H100/H800/GB200, nodes, networking, storage)    │
└─────────────────────────────────────────────────────────┘
```

(The original had four layers. Data and post-training are added as a fifth because in practice they often matter more than any systems trick.)

The labs that win **co-design across all layers**: DeepSeek's architecture (MLA + MoE) was chosen *because* of their hardware constraint (H800 interconnect), and their precision choice (FP8) was chosen *because* of their cost constraint.

### 1.1 The three equations that run everything

**Compute cost:** `C ≈ 6 · N · D` FLOPs
(N = active parameters, D = training tokens. The 6 = 2 FLOPs/param forward + 4 backward.)

**Memory per parameter (standard mixed-precision Adam):** `≈ 16 bytes/param`
= 2 (BF16 weights) + 2 (BF16 grads) + 4 (FP32 master weights) + 4 (Adam m) + 4 (Adam v).
A 70B model needs ~1.12 TB of model state, before a single activation. No single GPU (80 GB) comes close, so sharding is mandatory.

**Efficiency:** `MFU = (achieved FLOPs/s) / (peak hardware FLOPs/s)`
Good large-scale runs land around **35 to 45% MFU** (Llama 3 405B reported 38 to 43% **[PUBLIC]**). If you ever hit 60%+ on a big cluster, double-check your accounting.

### 1.2 Worked example: how long does a 70B model take?

- Tokens: 15T. Compute: 6 × 70e9 × 15e12 = **6.3e24 FLOPs**
- H100 BF16 dense peak ≈ 989 TFLOPs. At 40% MFU → ~395 TFLOPs/GPU.
- 2,048 GPUs → 2,048 × 395e12 ≈ 8.1e17 FLOPs/s
- Time = 6.3e24 / 8.1e17 ≈ 7.8e6 s ≈ **~90 days** (assuming zero downtime).

This is the kind of back-of-envelope calc every lab does before committing millions of dollars. Learn to do it in your head.

---

## 2. Hardware Cluster Fundamentals

### 2.1 GPU types you'll encounter

| GPU | Memory | Intra-node interconnect | Notes |
|-----|--------|--------------------------|-------|
| A100 (40/80GB) | 40/80 GB HBM2e | NVLink 3 (600 GB/s) | Workhorse of 2020 to 2023 |
| H100 (80GB) | 80 GB HBM3 | NVLink 4 (900 GB/s) | Standard for 2023 to 2025 frontier training |
| H800 | 80 GB HBM3 | NVLink reduced (~400 GB/s) | China-export variant: same compute as H100, cut interconnect |
| H200 | 141 GB HBM3e | NVLink 4 (900 GB/s) | H100 compute, more/faster memory |
| B200 / GB200 (Blackwell) | up to ~192 GB HBM3e per GPU | NVLink 5 (1.8 TB/s) | GB200 NVL72 = 72 GPUs in one NVLink domain |
| TPU v5p | 95 GB HBM | ICI (custom torus) | Google-only hardware (also used by Anthropic under a Google deal) |
| TPU v6e "Trillium" | 32 GB HBM | ICI | Efficiency-oriented; much less memory per chip than v5p |
| AWS Trainium2 | 96 GB HBM | NeuronLink | Used by Anthropic in Project Rainier **[PUBLIC]** |

*(Fix from original: Trillium has 32 GB, not 95 GB; the 95 GB figure belongs to v5p only.)*

**Key insight:** H800 export-control limits hit **interconnect**, not FLOPS. That is *why* DeepSeek's tricks (no tensor parallelism, FP8, DualPipe overlap, node-limited expert routing) matter so much. They were engineered around the bandwidth wall.

**Blackwell changes the topology rules.** In NVL72, 72 GPUs share one NVLink domain, so "keep TP inside a node" becomes "keep TP inside the NVLink domain", which is much bigger. Expect parallelism recipes to shift toward wider TP/EP for this hardware.

### 2.2 Node and cluster topology

A typical H100 training node:
- **8 GPUs** per node (HGX/DGX)
- Intra-node: **NVLink/NVSwitch**, 900 GB/s per GPU (H100)
- Inter-node: **InfiniBand or RoCE**, typically 400 Gb/s per GPU ≈ **50 GB/s**
- Intra vs inter gap: ~**18× on H100**, ~**8× on H800**. **This gap is the single most important constraint in distributed training.** Every parallelism strategy exists to minimize slow-link communication.

Scale reality check:
- 2,048 GPUs = 256 nodes. 16,384 GPUs = 2,048 nodes.
- At 16K GPUs, Meta saw an interruption roughly every ~3 hours during Llama 3 training **[PUBLIC]**. Fault tolerance is a first-class engineering problem (section 9).

Network topologies you'll hear about: **fat-tree**, **rail-optimized** (GPU *i* in every node connects to the same switch "rail"), **dragonfly**. Rail-optimized layouts matter because collective ops between same-index GPUs across nodes are the dominant traffic pattern.

### 2.3 Storage and data pipeline

- **Checkpoint size (corrected):** For a 70B model, BF16 weights alone are ~140 GB. But a **full training checkpoint** also needs FP32 master weights (280 GB) + Adam m and v (560 GB) → roughly **~1 TB**, i.e. optimizer state adds ~6× the BF16 weight size, not 2 to 3×. (Original text understated this.)
- Parallel filesystems (Lustre, GPFS) or object storage (S3-compatible) hold checkpoints. Async checkpointing is required so training doesn't stall.
- **Data loading** must never starve GPUs: pre-tokenized, shuffled shards, streamed with prefetch; CPU preprocessing is never on the critical path.

---

## 3. Parallelism Strategies

You can't fit 70B+ parameters (plus optimizer state) on one GPU, so you split. There are **five** axes, and real runs combine them ("3D / 4D / 5D parallelism").

### 3.1 Data Parallelism (DP)
- **What:** Each GPU holds a full model replica, processes a different slice of the batch, gradients are averaged via all-reduce.
- **Pro:** Simple; scales throughput almost linearly.
- **Con:** Memory-limited, since every replica holds everything.
- **ZeRO (DeepSpeed) stages:**
  - ZeRO-1: shard **optimizer states** across DP ranks
  - ZeRO-2: also shard **gradients**
  - ZeRO-3: also shard **parameters** (≈ PyTorch **FSDP**)
- **Trade-off:** higher ZeRO stage = less memory, more communication (parameters must be all-gathered each layer).

### 3.2 Tensor Parallelism (TP)
- **What:** Split individual weight matrices across GPUs. Megatron pattern: **column-parallel** first linear, **row-parallel** second linear, one all-reduce per block (forward) and one per block (backward).
- **Pro:** Cuts per-GPU weight and activation memory.
- **Con:** Needs very frequent, high-bandwidth communication → stay **inside the NVLink domain**.
- **Typical degree:** 8 on H100 nodes (sometimes 2 to 4).
- **Megatron "sequence parallelism" (note the naming trap):** a companion optimization that splits LayerNorm/dropout activations along the sequence dimension *within the TP group* to save activation memory. It is **not** the same as context parallelism (3.4).

### 3.3 Pipeline Parallelism (PP)
- **What:** Split layers into stages. Micro-batches flow through the stages.
- **Pro:** Low communication (only activations between stages) → fine across nodes.
- **Con:** **Pipeline bubbles.** Idle fraction for plain 1F1B ≈ `(p − 1) / (m + p − 1)` where p = stages, m = micro-batches. To make bubbles small, you need m ≫ p.
- **Mitigations:** interleaved 1F1B (virtual stages), zero-bubble schedules, **DualPipe** (DeepSeek-V3's bidirectional schedule that also overlaps communication with compute).
- **Typical degree:** 4 to 16.

### 3.4 Context Parallelism (CP) (a.k.a. sequence-dimension parallelism for attention)
- **What:** Split the *sequence* across GPUs so very long contexts (128K+) fit. Attention is global, so GPUs must exchange information.
- **Variants (corrected, because they communicate differently):**
  - **Ring Attention / Megatron CP:** pass **KV blocks** around a ring while computing, overlapping comm and compute.
  - **DeepSpeed-Ulysses:** uses **all-to-all** to re-shard from sequence-split to head-split around the attention op (original text wrongly implied all variants pass KV blocks).
- Llama 3 used CP only in the long-context extension stage **[PUBLIC]**.

### 3.5 Expert Parallelism (EP)
- **What:** In MoE models, different experts live on different GPUs; tokens are dispatched to their routed experts via **all-to-all**, then combined back.
- **Cons:** all-to-all is bandwidth-hungry; **load imbalance** (hot experts) stalls everyone.
- **DeepSeek-V3's answers [PUBLIC]:** limit each token to experts on at most 4 nodes (caps cross-node traffic), auxiliary-loss-free load balancing (a per-expert bias adjusted online rather than a loss term that hurts quality), and custom all-to-all kernels overlapped with compute.

### 3.6 Choosing a configuration (decision table)

| Situation | Reach for |
|-----------|-----------|
| Model fits on one GPU with room to spare | **DDP** (plain DP) |
| Model fits only if state is sharded | **FSDP / ZeRO-3** |
| Model's layers too big even sharded, NVLink available | Add **TP within node** |
| Spanning many nodes, slow network | Add **PP across nodes** |
| Context ≥ 128K | Add **CP** |
| MoE model | Add **EP** (+ TP=1 and PP is a legit choice, as DeepSeek did) |

**Golden rule:** highest-bandwidth, highest-frequency communication (TP) goes **inside** the fast domain; lower-frequency, more latency-tolerant communication (PP, DP) goes **across** the slow network.

### 3.7 Worked example: 70B dense on 2,048 H100s

```
TP = 8   (inside a node)
PP = 8   (across nodes)
DP = 2048 / (8 × 8) = 32   (ZeRO-1 optimizer sharding across DP)
```

Memory sanity check:
- Model state with weights+grads in BF16 (4 B/param): 70e9 × 4 / 64 (TP×PP) ≈ **4.4 GB/GPU**
- FP32 master + Adam (12 B/param): 70e9 × 12 / 64 ≈ 13.1 GB; ZeRO-1 across DP=32 → **~0.4 GB/GPU**
- Total static ≈ 5 GB → ~75 GB of an 80 GB GPU left for activations. Very comfortable; in practice you'd tune TP/PP lower or use bigger micro-batches.

### 3.8 DeepSeek-V3's actual configuration **[PUBLIC]**
- **No tensor parallelism** (TP=1). 16-way PP, 64-way EP spanning 8 nodes, ZeRO-1 data parallelism.
- *Why no TP (corrected reasoning):* TP's per-layer all-reduces are costly on the H800's reduced NVLink bandwidth. To make memory fit without TP they recompute RMSNorm and MLA up-projections during backprop and keep the EMA of weights in CPU memory. (Original text gave a vaguer "changes the communication profile" explanation.)

---

## 4. Precision and Numerical Techniques

### 4.1 The precision ladder

| Format | Bits | Typical use |
|--------|------|-------------|
| FP32 | 32 | Master weights, optimizer states (classic recipe) |
| TF32 | 19 (1+8+10) | Tensor-core matmul mode on Ampere+. *Note: in PyTorch, TF32 for matmuls is **off by default** since 1.12; enable with `torch.set_float32_matmul_precision("high")`.* |
| BF16 | 16 (1+8+7) | **Standard training precision**. Same exponent range as FP32, so no loss scaling needed |
| FP16 | 16 (1+5+10) | Older; needs loss scaling because of small range |
| FP8 E4M3 / E5M2 | 8 | 2024+ frontier training (DeepSeek-V3). E4M3 = more precision, E5M2 = more range |
| FP4 / NVFP4 / MXFP4 | 4 | **Emerging for training** on Blackwell (NVIDIA published NVFP4 pretraining experiments in 2025); widely used for inference. *(Original said "inference mostly", which is now outdated.)* |

### 4.2 Standard mixed-precision recipe
- FP32 master weights; forward/backward in BF16 (autocast); FP32 Adam moments.
- This is the baseline from which you then optimize (FP8 GEMMs, BF16 moments, etc.).

### 4.3 FP8 training (DeepSeek-V3's key trick) **[PUBLIC]**

**Why it's hard:** FP8 has a tiny dynamic range, so a few outlier activations ruin per-tensor scaling.

**Their solution, fine-grained quantization:**
- **Activations:** scaled per **1×128 tile**. **Weights:** scaled per **128×128 block**. (Not per-tensor.)
- **Higher-precision accumulation:** raw tensor-core accumulation in FP8 GEMMs retains only ~14 bits, so partial sums are promoted to **FP32 on CUDA cores every 128 elements**.
- **E4M3 everywhere** (instead of the older hybrid E4M3 forward / E5M2 backward), made possible by the fine-grained scaling.
- **Selective precision:** embeddings, output head, MoE gating, normalization, attention operators stay in BF16/FP32; only the big linear-layer GEMMs go FP8. Master weights and gradients accumulate in FP32; Adam moments stored in BF16.
- **Result:** ~2× throughput/memory benefit on GEMMs, relative loss error vs BF16 stayed **below 0.25%** in their validation.

**Caveat:** the FP8 speedup you get in practice is far below 2× end-to-end, since non-GEMM ops, communication, and quantization overhead remain.

---

## 5. Optimizers, Schedules, Data, and Stability

- **Optimizer:** AdamW is the default (β1=0.9, β2=0.95 is common for LLMs, ε=1e-8). Alternatives: Adafactor (low memory), Lion, and **Muon** (orthogonalized-momentum optimizer; Moonshot's Kimi K2 trained a ~1T-parameter MoE with a Muon variant, MuonClip, reported in 2025). Treat Muon as a fast-moving area and verify details.
- **LR schedule:** linear warmup (~hundreds to thousands of steps) → cosine/linear decay. Increasingly **WSD** (warmup-stable-decay): hold LR constant, then a short decay on high-quality data. WSD lets you branch off checkpoints cheaply.
- **Batch size:** global batch in millions of tokens. Llama 3 405B went 4M → 8M → 16M tokens/step **[PUBLIC]**. Global batch = micro-batch × grad accumulation × DP.
- **Gradient clipping:** norm clipping at 1.0 is near-universal.
- **Weight decay:** ~0.1, excluded for norms/biases.
- **Loss spikes:** common at scale. Mitigations: lower LR, skip/inspect bad batches, rollback to a checkpoint, QK-norm / z-loss, BF16 over FP16, watch Adam β2/ε.

### 5.1 Scaling laws and data (the part people skip and shouldn't)
- **Chinchilla rule of thumb:** compute-optimal ≈ **20 tokens per parameter**. But labs now **over-train** small/medium models far beyond that because inference cost dominates (Llama 3 405B: ~15T tokens ≈ 37 tokens/param; small models go 100s of tokens/param).
- **Data quality > quantity.** Dedup, quality classifiers, domain mixing, synthetic data, and annealing on curated data at the end of training are major levers, and are exactly what labs disclose least.
- **MoE accounting:** use **active** parameters in `6·N·D`. DeepSeek-V3: N=37B, D=14.8T tokens.

---

## 6. Frameworks and Who Uses What

### 6.1 PyTorch
Base for model code and autograd at almost every lab except Google. Key pieces: `torch.compile`, FSDP2 (built on DTensor), `torch.distributed`, **torchtitan** (clean reference for 4D parallelism in native PyTorch).

### 6.2 Megatron-LM / Megatron-Core (NVIDIA)
Reference implementation of TP, PP, SP, CP, EP. Most frontier-scale open recipes either use it or copy its ideas.

### 6.3 DeepSpeed (Microsoft)
ZeRO stages 1 to 3, offload (CPU/NVMe), Ulysses sequence parallelism. Strong for fine-tuning and mid-scale; less dominant than Megatron-style stacks at the very frontier.

### 6.4 FSDP (PyTorch native)
PyTorch's ZeRO-3-style sharding. Default for many teams; composes with TP/PP/CP through DTensor.

### 6.5 Triton / CUTLASS / FlashAttention
- **Triton** (OpenAI, open source): Python-embedded GPU kernel language; also the backend of `torch.compile`'s Inductor.
- **FlashAttention 1/2/3:** IO-aware attention that avoids materializing the N×N matrix. Foundational for long context.
- **CUTLASS / CuTe:** NVIDIA templates for hand-tuned GEMMs; DeepGEMM (DeepSeek) is an FP8 GEMM library in this space.

### 6.6 Ray
Cluster orchestration, RL pipelines, data processing. OpenAI has publicly discussed using Ray **[PUBLIC]**.

### 6.7 Who uses what

| Lab | Stack | Confidence |
|-----|-------|------------|
| **Meta (Llama 3)** | PyTorch; 4D parallelism (FSDP-style DP + TP + PP + CP); BF16 | **[PUBLIC]** (Llama 3 paper) |
| **DeepSeek** | PyTorch-based custom framework (HAI-LLM); PP + EP + ZeRO-1, no TP; FP8; DualPipe | **[PUBLIC]** (V3 report) |
| **Google (Gemini)** | JAX + XLA/GSPMD + Pathways on TPUs | **[PUBLIC]** at a high level |
| **OpenAI** | PyTorch + CUDA + Triton; parallelism details not disclosed (Megatron-style is a reasonable guess) | Triton/Ray **[PUBLIC]**; rest **[INFERRED]** |
| **Anthropic** | Trains on **multiple hardware platforms: AWS Trainium, Google TPUs, and NVIDIA GPUs** **[PUBLIC]**. Framework/parallelism stack **[UNKNOWN]** | Hardware public; software unknown |

*(Fix from original: it listed Anthropic as "PyTorch, Megatron/DeepSpeed-style" as if confirmed. That isn't public, and Anthropic's multi-hardware setup means a pure PyTorch/CUDA story is not accurate.)*

---

## 7. Communication Primitives (NCCL and friends)

NCCL (NVIDIA Collective Communications Library) powers GPU collectives; on TPU/Trainium the equivalents are provided by XLA/ICI and the Neuron collectives.

| Collective | Used for |
|------------|----------|
| **All-reduce** | Gradient averaging in DDP; TP block outputs |
| **All-gather** | FSDP/ZeRO-3 parameter gather before a layer |
| **Reduce-scatter** | FSDP/ZeRO gradient sharding (the dual of all-gather) |
| **All-to-all** | MoE token dispatch/combine; Ulysses |
| **Send/recv (P2P)** | Pipeline stage boundaries; ring attention |

**Cost intuition:** a ring all-reduce moves ≈ `2 · (n−1)/n · message_size` per GPU, almost independent of n, which is why DP scales so well. All-to-all grows with how many peers each token must reach, which is why MoE routing is topology-constrained.

**Overlap is everything:** prefetch the next layer's all-gather while computing the current layer; start gradient reduce-scatter as soon as a layer's backward finishes. DualPipe takes this to the extreme by overlapping forward-chunk compute with backward-chunk communication.

---

## 8. Making It Efficient: Memory and Kernel Tricks

- **Activation (gradient) checkpointing/recomputation:** don't store activations, recompute in backward. Trades ~30% extra compute for big memory savings. *Selective* recomputation (only cheap-to-recompute but memory-heavy ops) is the modern default.
- **Operator fusion / `torch.compile`:** fewer kernel launches, fewer HBM round-trips.
- **FlashAttention:** attention is memory-bound; tiling in SRAM avoids HBM traffic.
- **Memory-bound vs compute-bound:** large GEMMs are compute-bound; elementwise ops, norms, softmax are memory-bound. Optimization = move work from memory-bound to compute-bound regimes (fusion, bigger tiles).
- **Roofline model:** attainable FLOPs = min(peak FLOPs, arithmetic intensity × memory bandwidth). Learn this one diagram; it explains most kernel optimization.

---

## 9. Fault Tolerance and Operations

At thousands of GPUs, failure is the normal state.

- **Real numbers [PUBLIC]:** Llama 3's 54-day snapshot on 16K H100s had **466 job interruptions** (419 unexpected), with GPU and HBM faults the largest category. Meta still reported **>90% effective training time** thanks to heavy automation.
- *(Fix: the original's "10 to 30% lost" is the cost of a **poorly engineered** setup, not a law of nature. Frame it as "unengineered runs can lose 10 to 30%; well-run ones lose under 10%".)*
- **Async checkpointing:** snapshot to CPU memory fast, flush to storage in the background. Frequency: every few hundred to few thousand steps; tuned against failure rate.
- **Elastic restart:** resume on a changed node count; requires resharding-aware checkpoint formats (e.g., PyTorch Distributed Checkpoint).
- **Silent data corruption (SDC):** a GPU returns wrong numbers without crashing. Detect with periodic checksum/golden tests and loss-curve anomaly alarms.
- **Stragglers:** one slow GPU stalls every collective. Track per-rank step times; evict slow nodes.
- **NCCL hangs/timeouts:** a top operational headache; flight recorders and watchdogs are standard tools.
- **Scheduling:** Slurm or Kubernetes with gang scheduling, plus hot spares.

---

## 10. Post-Training

Pretraining gives a base model; post-training turns it into an assistant or reasoner. Compute is much smaller, but this is where labs differentiate most.

### 10.1 SFT (Supervised Fine-Tuning)
Train on curated prompt→response pairs. Quality and diversity of data dominate. Often a **cold-start** stage before RL.

### 10.2 RLHF and RLAIF
- **RLHF:** train a **reward model** on human preference comparisons, then optimize the policy with PPO (or variants such as DPO-family methods that skip explicit RL) with a KL penalty to a reference model.
- **Constitutional AI (Anthropic) [PUBLIC]:** *RLAIF*. A written set of principles guides AI-generated critiques/revisions (supervised phase) and AI preference labels (RL phase) for **harmlessness**, reducing human labeling for that dimension. *(Fix: it is not a replacement for all human feedback; helpfulness data is still human-labeled in the original method.)*

### 10.3 RL for reasoning (R1, o-series style)
- **RLVR (RL with verifiable rewards):** math answers and code tests give objective rewards, no learned reward model needed.
- **GRPO** (DeepSeek): samples a *group* of responses per prompt and uses the group's mean/std reward as the baseline, so **no value network (critic)** is required. Cheaper and simpler than PPO.
- **R1-Zero:** pure RL from the base model, no SFT. Reasoning behaviors (long chains of thought, self-verification) emerged, but with readability and language-mixing problems.
- **R1:** cold-start SFT → reasoning RL → rejection-sampling SFT → final RL, fixing those problems.

### 10.4 Rejection sampling / best-of-N / distillation
Generate many samples, keep the verified/best ones, fine-tune on them. Also how reasoning ability gets **distilled** into smaller models (DeepSeek distilled R1 outputs into Qwen/Llama bases).

### 10.5 Evals
You can't improve what you can't measure: held-out benchmarks, contamination checks, human evals, and online A/B tests. Eval design is a core competency, not an afterthought.

---

## 11. Case Studies (corrected numbers)

### 11.1 DeepSeek-V3: the cost-efficiency benchmark **[PUBLIC]**
- **Hardware:** 2,048 H800 GPUs.
- **Model:** 671B total / 37B active parameters (DeepSeekMoE: 1 shared + 256 routed experts, top-8 routing), **MLA** attention, **multi-token prediction** objective. Trained on **14.8T tokens**.
- **Systems:** FP8 mixed precision, DualPipe, no TP, 16-way PP, 64-way EP, ZeRO-1.
- **Cost:** 2.788M H800 GPU-hours (≈2.664M pretraining + ~119K context extension + ~5K post-training) ≈ **$5.576M at an assumed $2/GPU-hour**.
- **Important caveat (missing from the original):** this is the cost of the **final run only**. It excludes prior research, ablations, failed runs, data, salaries, and hardware ownership. It's a valid *efficiency* benchmark, not the total cost of building the model.
- **Sanity check:** 6 × 37e9 × 14.8e12 ≈ 3.3e24 FLOPs over 2.664M GPU-hours → ~340 TFLOPs/GPU effective. That is roughly a third of H800 BF16 peak, a respectable result under a throttled interconnect.

### 11.2 DeepSeek-R1: RL on top of V3 **[PUBLIC]**
*(Original conflated stages; corrected here.)*
- R1 is trained **on top of the V3 base model** (the V3 cost is *not* included below).
- According to the figures DeepSeek published (Nature, 2025), the R1 RL work was on **512 H800s** and totaled **~$294K**, composed of: R1-Zero (~198 hours), SFT-data generation (~5K GPU-hours), and the final R1 RL run (~80 hours), not "512 GPUs for 80 hours = $294K" as a single run.
- **Lesson:** verifiable-reward RL on a strong base is cheap *relative to pretraining*. But it depends on having that strong base already.

### 11.3 Llama 3 405B (Meta's public recipe) **[PUBLIC]**
- Up to **16,384 H100s**, **15.6T tokens**, ~**3.8e25 FLOPs**, 405B dense parameters.
- 4D parallelism (TP + PP + CP + DP/FSDP-style); BF16 training; **38 to 43% MFU**.
- Batch ramp 4M→8M→16M tokens; context extended to 128K in later stages.
- The most detailed public account of failures, stability, and ops at this scale. **Read the paper's infrastructure section twice.**

### 11.4 OpenAI and Anthropic (what's actually known)
- Both operate at **tens of thousands of accelerators or more**; OpenAI's Stargate targets gigawatt-scale campuses **[PUBLIC]**. Anthropic has public multi-vendor compute deals spanning Trainium (Project Rainier), TPUs, and NVIDIA GPUs.
- Training software internals (parallelism, precision, kernels) are **[UNKNOWN]** for both. "PyTorch + Megatron-style" is a plausible inference for OpenAI, not a fact.
- Where the moat plausibly lies: **infrastructure reliability, data quality, and post-training/RL methodology**, which are also the least-disclosed parts.

### 11.5 Newer open reference points (verify details before relying on them)
- **Kimi K2** (Moonshot, 2025): ~1T-parameter MoE trained with MuonClip.
- **Qwen3** (Alibaba, 2025): dense + MoE family with open weights.
- **DeepSeek-V3.2** (late 2025): introduced sparse attention (DSA) to cut long-context cost.
- **OLMo 2/3 (AI2), SmolLM3 (Hugging Face):** fully open recipes, including data and training details. These are the best *reproducible* study material.

---

## 12. Public vs. Speculation: Know Which One You're Reading

| Claim | Status |
|-------|--------|
| DeepSeek-V3 used FP8 with 1×128/128×128 scaling, DualPipe, no TP | Public (technical report) |
| Llama 3: 16K H100s, 4D parallelism, 466 interruptions in 54 days | Public (paper) |
| Gemini trains on TPUs with JAX | Public (high level) |
| OpenAI uses Triton, Ray | Public |
| OpenAI trains with Megatron-style TP/PP | Inferred |
| Anthropic's training framework internals | Unknown |
| Exact frontier model sizes (GPT-4/5, Claude) | Unknown (rumors only) |

Rule: when someone states a frontier-lab detail confidently on social media, ask "where's the primary source?"

---

## 13. Mastery Path (do these in order)

### Stage A: Foundations (weeks 1 to 3, one consumer GPU or free Colab/Kaggle)
- [ ] Build GPT from scratch: Karpathy's *Let's build GPT* + **nanoGPT**; then read **llm.c** to see training without frameworks.
- [ ] Understand GPU memory hierarchy (registers → SRAM → HBM) and the roofline model.
- [ ] Write one kernel in **Triton** (softmax or layernorm) and benchmark vs PyTorch.
- [ ] Do the memory math on paper for a 7B model (16 bytes/param). Then verify with `torch.cuda.memory_summary()` on a small model.

### Stage B: Multi-GPU (weeks 4 to 7, rent 2 to 8 GPUs by the hour)
- [ ] Train with BF16 autocast, then add DDP.
- [ ] Implement/enable FSDP; measure memory savings vs DDP.
- [ ] Implement column-parallel + row-parallel linear by hand (TP).
- [ ] Implement a 1F1B pipeline schedule on a toy model.
- [ ] Profile a step with PyTorch Profiler/Nsight; separate comm time from compute time.
- [ ] Run **torchtitan** or **nanotron** with a 1 to 8B model and read their configs line by line.

### Stage C: Scale-realism (weeks 8 to 12)
- [ ] Reproduce a Megatron-style TP+PP run on 8 to 16 GPUs.
- [ ] FP8/low-precision GEMM with per-block scaling; measure error vs BF16.
- [ ] Async checkpoint + kill-a-process recovery test.
- [ ] Compute MFU for your own run and explain the gap to 100%.

### Stage D: Architecture and post-training (weeks 12+)
- [ ] Implement a small MoE layer (top-k routing + load-balancing).
- [ ] Implement MLA and compare KV-cache size vs GQA.
- [ ] Run GRPO on a small verifiable task (e.g., arithmetic/GSM8K-style) with a ~0.5 to 3B model.
- [ ] Run LoRA/QLoRA SFT on a domain dataset and evaluate honestly.

### Free/open study material
- **Stanford CS336: Language Modeling from Scratch** (lectures + assignments)
- **Hugging Face "Ultra-Scale Playbook"** (training on GPU clusters, parallelism explained with experiments)
- **"How to Scale Your Model"** (JAX/TPU scaling book; best treatment of the roofline and parallelism math)
- **Karpathy** lecture series, nanoGPT, llm.c
- **Primary papers:** Megatron-LM (2019, 2021), ZeRO (2020), FlashAttention 1/2/3, DeepSeek-V2/V3/R1, Llama 3 herd, Chinchilla, Constitutional AI, GRPO (DeepSeekMath)

---

## 14. Indie-Hacker Reality Check: What You Can Do Without a 2,048-GPU Cluster

First-principles take: **don't pretrain a foundation model. Compete where capital isn't the moat.**

| Goal | Realistic path | Rough hardware |
|------|----------------|----------------|
| Learn the mechanics | nanoGPT / small models from scratch | 1 consumer GPU or free Kaggle/Colab |
| Domain/language specialization | Continued pretraining + SFT of an open model (Qwen/Llama/Gemma-class) with LoRA/QLoRA | 1 rented 24 to 80 GB GPU |
| Reasoning for a narrow domain | GRPO/RLVR on a small open model with verifiable rewards | 1 to 8 rented GPUs |
| Product moat | Proprietary data, evals, UX, distribution | Mostly not GPUs |

Practical principles:
1. **Rent, don't buy.** Per-hour GPU clouds (spot/marketplace) beat owning hardware until utilization is very high.
2. **Evals first.** Build the eval set *before* training; otherwise you can't tell whether you improved anything.
3. **Data is the lever you control.** A cleaned, deduplicated, well-labeled dataset beats an extra week of GPU time.
4. **Budget with `6ND`** before you spend a rupee/dollar. If the math says weeks on a cluster, change the plan.
5. **Know when to use an API.** If a hosted model plus good prompting/RAG solves the problem, training is a distraction.

---

## 15. Corrections Log (what changed from the original draft)

| # | Original | Correction |
|---|----------|------------|
| 1 | TPU v5p / Trillium both "95 GB" | v5p = 95 GB; Trillium (v6e) = 32 GB |
| 2 | TF32 "A100+ default matmul precision" | Off by default for matmuls in PyTorch ≥1.12; must be enabled |
| 3 | FP4 "inference mostly" | Now actively explored for training (NVFP4) |
| 4 | Checkpoint optimizer state "2 to 3×" | ~6× BF16 weight size (FP32 master + 2 Adam moments) |
| 5 | SP/CP described as one thing; Ulysses passes KV blocks | Megatron SP ≠ CP; Ulysses uses all-to-all, ring attention passes KV |
| 6 | DeepSeek "TP=1 because MLA+MoE changes comm profile" | TP avoided due to reduced NVLink bandwidth; memory handled via recomputation + CPU EMA |
| 7 | FP8 detail incomplete | Added E4M3-everywhere, FP32 promotion every 128 elements, BF16 Adam moments |
| 8 | V3 cost "$5.6M" presented without caveat | Final-run only; excludes research, ablations, failed runs |
| 9 | R1: "512 GPUs, ~80 hours, $294K" as one run | $294K = R1-Zero + data gen + R1 stage; built on top of V3 |
| 10 | "10 to 30% lost to failures" as rule of thumb | That's for poor engineering; Llama 3 reported >90% effective time |
| 11 | Anthropic = PyTorch/Megatron/DeepSpeed-style | Not public; Anthropic uses Trainium, TPU, and NVIDIA hardware |
| 12 | OpenAI = Megatron-style TP/PP + FSDP | Inferred, not confirmed |
| 13 | Constitutional AI "RLHF variant replacing human labeling" | RLAIF mainly for harmlessness; helpfulness still human-labeled in the original method |
| 14 | DeepSpeed "FP8 tooling" | Removed; FP8 is mainly via Transformer Engine / custom kernels |
| 15 | Missing: scaling laws, MFU, memory/compute equations, Muon, MTP, kernel/roofline basics | Added (sections 1, 5, 8) |

---

## 16. One-Paragraph Summary

Frontier model training is **physics-constrained systems engineering**. You budget compute with `6ND`, fit memory by sharding the ~16 bytes/param of training state (ZeRO/FSDP, TP, PP, EP), keep the GPUs fed by mapping the chattiest communication onto the fastest links, squeeze efficiency with BF16→FP8, fused kernels, and comm/compute overlap, and survive thousands of hardware faults with async checkpoints and automation. Then post-training (SFT, RLHF/RLAIF, GRPO-style RL with verifiable rewards) turns a base model into a product. Meta published the most complete ops story (Llama 3), DeepSeek published the most striking efficiency co-design (MLA + MoE + FP8 + DualPipe under an H800 bandwidth cap), and OpenAI/Anthropic disclose the least. The framework is the least differentiating layer; the durable advantages are **data, reliability engineering, and post-training methodology**.
