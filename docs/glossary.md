# Glossary: Every Term in This Cookbook, in Plain English

> Companion to [`llm-training-deep-dive.md`](llm-training-deep-dive.md), [`llm-training-cheatsheet.md`](llm-training-cheatsheet.md), and [`indie-deep-dive.md`](indie-deep-dive.md).
> Every entry gives the **full form** (what the letters stand for) and a **plain-English meaning**. No prior knowledge assumed.

---

## How to use this glossary

- **Brand new?** Read the [Starter Pack](#1-starter-pack-the-15-terms-to-learn-first) first, then the [Common Confusions](#17-common-confusions-cleared-up).
- **Hit an unfamiliar word in a guide?** Use your browser's search (Ctrl+F) on this page.
- **Terms are grouped by topic**, not alphabetically, so related ideas sit together. Each table has three columns: **Term**, **Stands for** (full form, if it is an abbreviation), and **What it means**.
- A dash (-) in "Stands for" means the term is an ordinary word or a name, not an abbreviation.

### Reading big numbers

| Written as | Means | Example |
|-----------|-------|---------|
| K | thousand | 8K context = 8,000 tokens |
| M | million | 125M parameters = 125 million |
| B | billion | 7B model = 7 billion parameters |
| T | trillion | 15T tokens = 15 trillion tokens |
| KB / MB / GB / TB | kilo-, mega-, giga-, terabytes of data | an 80 GB GPU holds 80 gigabytes |
| e18, e24 | "times 10 to the power of" | 6e24 = 6 followed by 24 zeros |

---

## 1. Starter Pack: the 15 terms to learn first

| Term | Stands for | What it means |
|------|-----------|---------------|
| **LLM** | Large Language Model | A computer program trained on huge amounts of text that predicts the next piece of text. ChatGPT, Claude, Llama, and Qwen are LLMs. |
| **Token** | - | A chunk of text the model reads, roughly a word or part of a word. "unhappiness" might be 3 tokens. Models count and charge by tokens. |
| **Parameter** | - | One adjustable number inside the model. A "7B model" has 7 billion of them. More parameters usually means more capacity, but also more cost. |
| **Weights** | - | Another word for the parameters once they are learned. "Open weights" means the learned numbers are published for anyone to download. |
| **Training** | - | The process of adjusting the parameters by showing the model lots of examples until it gets good at predicting text. |
| **Inference** | - | Using a finished model to produce answers (what happens when you chat with it). Much cheaper than training. |
| **Pretraining** | - | The first, biggest training stage: learn from a giant pile of raw text. Produces a "base model". |
| **Post-training** | - | Stages after pretraining (like SFT and RLHF) that turn the base model into a helpful assistant. |
| **Fine-tuning** | - | Taking an already-trained model and training it a bit more on your own data to specialize it. |
| **GPU** | Graphics Processing Unit | A chip with thousands of small calculators working in parallel. The workhorse of AI training. |
| **VRAM** | Video Random Access Memory | The GPU's own memory. If the model does not fit in VRAM, you cannot run or train it on that GPU. |
| **Loss** | - | A score of how wrong the model currently is. Training tries to make it go down. |
| **FLOPs** | Floating-Point Operations | A count of math operations. Used to measure how much work training takes. |
| **LoRA / QLoRA** | Low-Rank Adaptation / Quantized LoRA | Cheap fine-tuning tricks that change only a small add-on instead of the whole model. This is how most solo builders fine-tune. |
| **Eval** | Evaluation | A fixed set of test questions used to measure whether your model is actually improving. |

---

## 2. Core Model Concepts

| Term | Stands for | What it means |
|------|-----------|---------------|
| **Model** | - | The program plus its learned numbers (parameters). |
| **Transformer** | - | The neural-network design behind nearly all modern LLMs. Its key trick is attention. |
| **Attention** | - | The mechanism that lets the model look back at earlier words and decide which ones matter for the next prediction. |
| **Layer** | - | One processing block in the model. A model stacks many layers (e.g., 32 or 80). |
| **Embedding** | - | A list of numbers representing a token so the model can do math on it. |
| **Context window** | - | How much text (in tokens) the model can consider at once. 128K context means about 128,000 tokens. |
| **Vocabulary** | - | The set of tokens the model knows. |
| **Tokenizer** | - | The tool that splits text into tokens and back. Different models use different tokenizers. |
| **BPE** | Byte-Pair Encoding | A common way to build a tokenizer by merging frequent character pairs into tokens. |
| **SentencePiece** | - | A popular tokenizer-building library. |
| **FFN** | Feed-Forward Network | The part of each layer that processes each token individually after attention. Holds most of the parameters. |
| **Residual connection** | - | A shortcut that adds a layer's input to its output, which makes deep networks easier to train. |
| **LayerNorm / RMSNorm** | Layer Normalization / Root Mean Square Normalization | Steps that keep numbers in a stable range so training does not blow up. |
| **Dense model** | - | A model where all parameters are used for every token. |
| **MoE** | Mixture of Experts | A model split into many "expert" sub-networks where only a few are used per token. Big total size, cheaper per token. |
| **Expert** | - | One of the sub-networks inside an MoE model. |
| **Router / gating** | - | The small component in an MoE that decides which experts handle each token. |
| **Total vs active parameters** | - | Total = every parameter stored. Active = parameters actually used per token. DeepSeek-V3: 671B total, 37B active. |
| **Load balancing** | - | Making sure tokens are spread evenly across experts so none is overloaded or ignored. |
| **GQA** | Grouped-Query Attention | A way to share key/value data across attention heads to shrink memory use. |
| **MLA** | Multi-head Latent Attention | DeepSeek's method that compresses the KV cache a lot more, saving memory for long text. |
| **KV cache** | Key-Value cache | Memory that stores earlier tokens' attention data during generation so it is not recomputed. Grows with context length. |
| **MTP** | Multi-Token Prediction | A training goal where the model predicts several future tokens at once, which gives a stronger learning signal. |
| **Base model** | - | A pretrained model that continues text but does not follow instructions well. |
| **Instruct / chat model** | - | A model further trained to follow instructions and hold conversations. |
| **Open weights** | - | The trained parameters are downloadable. This is **not** the same as open source (see Common Confusions). |

---

## 3. How Training Works (Basics)

| Term | Stands for | What it means |
|------|-----------|---------------|
| **Forward pass** | - | Feeding data through the model to get a prediction. |
| **Backward pass / backpropagation** | - | Working out how each parameter contributed to the error, so it can be adjusted. |
| **Gradient** | - | The direction and size in which to nudge a parameter to reduce the loss. |
| **Learning rate (LR)** | - | How big each nudge is. Too high: training explodes. Too low: it crawls. |
| **Step** | - | One update of the model's parameters. |
| **Batch** | - | A group of examples processed together for one step. |
| **Micro-batch** | - | A small slice of a batch that fits in memory; several are combined into one step. |
| **Global batch size** | - | Total tokens or examples across all GPUs and accumulations in one step. |
| **Gradient accumulation** | - | Adding up gradients from several micro-batches before updating, to imitate a bigger batch with less memory. |
| **Epoch** | - | One full pass through the training data. |
| **Checkpoint** | - | A saved snapshot of the training state (parameters and more) so you can resume after a crash. |
| **Overfitting** | - | The model memorizes training data and does worse on new data. Sign: training loss drops while validation loss rises. |
| **Underfitting** | - | The model has not learned enough yet. |
| **Training loss / validation loss** | - | Error on data it learns from / on held-out data it never trains on. |
| **Hyperparameter** | - | A setting you choose before training (learning rate, batch size, LoRA rank), as opposed to a learned parameter. |
| **Dataset / corpus** | - | The collection of text or examples used for training. |
| **Catastrophic forgetting** | - | The model loses old skills while learning something new. |
| **Ablation** | - | An experiment that changes one thing at a time to see what matters. |

---

## 4. Optimizers and Training Stability

| Term | Stands for | What it means |
|------|-----------|---------------|
| **Optimizer** | - | The algorithm that uses gradients to update parameters. |
| **Adam / AdamW** | Adaptive Moment Estimation / Adam with decoupled Weight decay | The standard LLM optimizer. Keeps two extra numbers (moments) per parameter, which is why training needs extra memory. |
| **Moments (m and v)** | - | Adam's running averages of gradients and squared gradients. |
| **Optimizer state** | - | Everything the optimizer stores (for Adam: moments, often plus master weights). A big memory cost. |
| **Adafactor / Lion** | - | Optimizers that use less memory or work differently from Adam. |
| **Muon / MuonClip** | - | A newer optimizer that orthogonalizes updates for matrix weights. MuonClip is Moonshot's stabilized variant used for Kimi K2. |
| **Weight decay** | - | A small penalty that discourages large weights and helps generalization. |
| **Warmup** | - | Starting with a tiny learning rate and ramping up, to avoid early instability. |
| **Cosine decay** | - | Lowering the learning rate along a smooth curve toward the end. |
| **WSD** | Warmup-Stable-Decay | A schedule: ramp up, stay flat for most of training, then drop quickly at the end. |
| **Gradient clipping** | - | Capping the size of gradients (commonly norm 1.0) to prevent sudden jumps. |
| **Loss spike** | - | A sudden jump in loss during training; may recover or diverge. |
| **Divergence** | - | Training goes off the rails and the loss blows up or becomes NaN. |
| **NaN** | Not a Number | An invalid number that poisons everything it touches; a sign training broke. |
| **QK-norm** | Query-Key normalization | A stabilizing step on attention inputs that reduces loss spikes. |
| **z-loss** | - | An extra penalty that keeps output scores from growing too large. |
| **Rollback** | - | Restarting from an earlier checkpoint after a failure or spike. |

---

## 5. Hardware

| Term | Stands for | What it means |
|------|-----------|---------------|
| **CPU** | Central Processing Unit | The general-purpose chip in every computer. Slow for deep-learning math compared with a GPU. |
| **GPU** | Graphics Processing Unit | See Starter Pack. |
| **TPU** | Tensor Processing Unit | Google's custom AI chip. Gemini is trained on TPUs. |
| **Trainium** | - | Amazon's custom AI training chip (AWS). Used in Anthropic's Project Rainier. |
| **CUDA** | Compute Unified Device Architecture | NVIDIA's software platform for programming its GPUs. |
| **HBM** | High-Bandwidth Memory | Very fast memory stacked next to the GPU chip. The GPU's "main memory". |
| **SRAM** | Static Random-Access Memory | Tiny, extremely fast memory on the GPU chip itself. FlashAttention exploits it. |
| **A100 / H100 / H200 / B200** | - | NVIDIA data-center GPU generations (Ampere, Hopper, Hopper+more memory, Blackwell). Newer = faster and more memory. |
| **H800** | - | An H100 variant sold in China with reduced interconnect speed (same compute). DeepSeek trained on it. |
| **GB200 / NVL72** | Grace-Blackwell 200 / NVLink 72 | A Blackwell rack where 72 GPUs share one ultra-fast NVLink network. |
| **Ampere / Hopper / Blackwell** | - | NVIDIA's architecture names (A100, H100, B200 respectively). |
| **Node** | - | One server, typically with 8 GPUs inside. |
| **Cluster** | - | Many nodes connected together to act as one big machine. |
| **HGX / DGX** | - | NVIDIA's 8-GPU server designs. |
| **NVLink / NVSwitch** | - | NVIDIA's very fast links between GPUs inside a node. Like hallways between rooms in one office. |
| **InfiniBand** | - | A fast network between nodes. Like roads between office buildings: slower than the hallways. |
| **RoCE** | RDMA over Converged Ethernet | An Ethernet-based alternative to InfiniBand. |
| **Interconnect** | - | Any link that moves data between chips or machines. |
| **Bandwidth** | - | How much data can move per second (GB/s). The key bottleneck in distributed training. |
| **Latency** | - | The delay before data starts arriving. |
| **Topology (fat-tree, rail-optimized, dragonfly)** | - | The wiring layout of a cluster network. Rail-optimized connects same-numbered GPUs across nodes to the same switches. |
| **Export controls** | - | Government limits on selling advanced chips to certain countries; the reason the H800 exists. |
| **Slurm** | Simple Linux Utility for Resource Management | A job scheduler for sharing a cluster among users. |
| **Kubernetes (K8s)** | - | A system for running and managing containers across many machines. |
| **Spot / interruptible instance** | - | Cheap rented GPU that the provider can take back at any moment. Needs frequent checkpoints. |
| **On-demand instance** | - | Rented GPU that stays yours until you stop it; costs more. |
| **Marketplace GPU** | - | Rented GPUs offered by many hosts (Vast.ai, RunPod community). Cheap but variable reliability. |
| **Burn-in** | - | Testing a new cluster (network, GPU health) before a long run. |

---

## 6. Memory and Compute Math

| Term | Stands for | What it means |
|------|-----------|---------------|
| **FLOP / FLOPs / FLOPS** | Floating-Point OPerations / (plural) / per Second | FLOPs (lowercase s) = a count of operations. FLOPS (all caps) = operations per second, a speed. Easy to mix up. |
| **TFLOPs** | Tera-FLOPs per second | Trillions of operations per second. An H100 does about 989 TFLOPs in BF16. |
| **6ND** | - | The rule of thumb for training cost: FLOPs = 6 x N (active parameters) x D (training tokens). |
| **N, D, C** | - | N = parameters, D = tokens (data), C = compute in FLOPs. |
| **16 bytes per parameter** | - | Typical memory to train with Adam in mixed precision: 2 (weights) + 2 (gradients) + 4 (master copy) + 4 + 4 (Adam's two moments). |
| **MFU** | Model FLOPs Utilization | The share of the hardware's peak speed you actually achieve. 35 to 45% is good at scale. |
| **Throughput** | - | How much work per second (e.g., tokens per second). |
| **Memory-bound / compute-bound** | - | Limited by moving data vs limited by doing math. Big matrix multiplies are compute-bound; many small operations are memory-bound. |
| **Arithmetic intensity** | - | How much math you do per byte of data moved. Higher = more compute-bound. |
| **Roofline model** | - | A chart showing whether a workload is limited by memory or by compute. |
| **GEMM** | General Matrix Multiply | The big matrix multiplication that makes up most of training compute. |
| **Activations** | - | Intermediate results saved during the forward pass so the backward pass can use them. Their memory scales with batch and sequence length. |
| **Activation checkpointing / recomputation** | - | Not saving activations and recomputing them later: less memory, about 30% more compute. |
| **Offload** | - | Moving data from GPU memory to CPU RAM or disk to save VRAM, at a speed cost. |
| **Compute-optimal** | - | The best split of a fixed compute budget between model size and data. |

---

## 7. Parallelism (Splitting a Model Across GPUs)

| Term | Stands for | What it means |
|------|-----------|---------------|
| **DP** | Data Parallelism | Every GPU holds a full copy of the model and trains on different data; gradients are averaged. |
| **DDP** | Distributed Data Parallel | PyTorch's standard DP implementation. |
| **TP** | Tensor Parallelism | Splitting each layer's big matrices across GPUs. Needs very fast links, so it stays inside a node. |
| **PP** | Pipeline Parallelism | Splitting the layers into stages on different GPUs, like an assembly line. |
| **CP** | Context Parallelism | Splitting a very long text sequence across GPUs. |
| **SP** | Sequence Parallelism | In Megatron: splits some activations along the sequence inside a TP group to save memory. Not the same as CP. |
| **EP** | Expert Parallelism | Placing different MoE experts on different GPUs. |
| **3D / 4D / 5D parallelism** | - | Combining several of the above at once. |
| **ZeRO** | Zero Redundancy Optimizer | DeepSpeed's way to avoid every GPU storing the same copy of state. Stage 1 shards optimizer state, 2 adds gradients, 3 adds parameters. |
| **FSDP** | Fully Sharded Data Parallel | PyTorch's ZeRO-3-style approach: shard everything, gather pieces when needed. |
| **Sharding** | - | Splitting data or state into pieces spread across GPUs. |
| **Pipeline bubble** | - | Idle time in a pipeline while GPUs wait for earlier stages. |
| **1F1B** | One-Forward-One-Backward | A pipeline schedule that alternates forward and backward micro-batches to cut memory and bubbles. |
| **Interleaved / zero-bubble schedule** | - | Smarter pipeline schedules that shrink idle time. |
| **DualPipe** | - | DeepSeek's bidirectional pipeline schedule that also overlaps communication with compute. |
| **Ring Attention** | - | Passing key/value blocks around a ring of GPUs while computing attention, for long contexts. |
| **DeepSpeed-Ulysses** | - | A sequence-parallel method using all-to-all to reshuffle between sequence-split and head-split. |
| **Overlap** | - | Doing communication and computation at the same time so GPUs do not sit idle. |
| **Straggler** | - | One slow GPU that makes all others wait. |
| **Rank** | - | The ID number of one process or GPU in a distributed job. (Different from LoRA rank.) |
| **World size** | - | Total number of processes/GPUs in the job. |

---

## 8. Communication (How GPUs Talk)

| Term | Stands for | What it means |
|------|-----------|---------------|
| **NCCL** | NVIDIA Collective Communications Library | The software that handles GPU-to-GPU data exchange. Pronounced "nickel". |
| **Collective** | - | A communication operation involving many GPUs at once. |
| **All-reduce** | - | Everyone contributes a number set; everyone receives the sum. Used to average gradients. |
| **All-gather** | - | Everyone shares their piece; everyone ends up with all the pieces. |
| **Reduce-scatter** | - | Sum across GPUs, then each GPU keeps just one slice of the result. |
| **All-to-all** | - | Every GPU sends a different chunk to every other GPU. Used in MoE routing. |
| **Send / recv** | - | Point-to-point transfer between two GPUs (pipeline stage handoff). |
| **torchrun** | - | PyTorch's launcher for multi-GPU or multi-node jobs. |
| **nccl-tests** | - | A benchmark that checks whether your cluster network delivers the advertised speed. |

---

## 9. Numbers, Precision, and Quantization

| Term | Stands for | What it means |
|------|-----------|---------------|
| **Precision** | - | How many bits are used to store each number. Fewer bits = less memory and faster math, but less accuracy. |
| **FP32** | 32-bit Floating Point | Full precision. Accurate but heavy. |
| **TF32** | TensorFloat-32 | A 19-bit format that NVIDIA tensor cores use to speed up FP32 math. |
| **BF16** | Brain Floating Point 16 | 16-bit format with a wide range. The standard for LLM training. |
| **FP16** | 16-bit Floating Point | Narrower range; needs loss scaling. |
| **FP8 (E4M3 / E5M2)** | 8-bit Floating Point | 8-bit formats. E4M3 = 4 exponent bits, 3 mantissa bits (more precision); E5M2 = more range. |
| **FP4 / NVFP4 / MXFP4** | 4-bit Floating Point | Emerging 4-bit formats for Blackwell-class hardware. |
| **INT8 / INT4** | 8- / 4-bit Integer | Whole-number formats common for compressing models. |
| **NF4** | NormalFloat 4-bit | The 4-bit format QLoRA uses, designed for neural-network weights. |
| **Mixed precision** | - | Using low precision for most math and high precision where accuracy matters. |
| **Master weights** | - | A high-precision (FP32) copy of the parameters kept for accurate updates. |
| **Loss scaling** | - | Multiplying the loss so small FP16 gradients do not vanish. Not needed with BF16. |
| **Quantization** | - | Storing numbers with fewer bits to shrink the model. |
| **Fine-grained quantization** | - | Using separate scale factors for small tiles or blocks (e.g., 1x128) instead of one per tensor, to handle outliers. |
| **Dynamic range** | - | The span between the smallest and largest numbers a format can represent. |
| **Accumulation** | - | Adding many products together. Done in higher precision to avoid error build-up. |
| **Autocast** | - | PyTorch feature that automatically picks lower precision for safe operations. |
| **GPTQ / AWQ** | - | Popular methods for compressing a trained model to 4-bit with little quality loss. |
| **GGUF** | GPT-Generated Unified Format | The file format used by llama.cpp and Ollama for running quantized models locally. |
| **Q4_K_M** | - | A common GGUF 4-bit setting balancing size and quality. |

---

## 10. Scaling Laws and Data

| Term | Stands for | What it means |
|------|-----------|---------------|
| **Scaling law** | - | A pattern showing how model quality improves predictably with more parameters, data, and compute. |
| **Chinchilla** | - | A 2022 DeepMind study suggesting about 20 training tokens per parameter is compute-optimal. |
| **Over-training** | - | Training a smaller model on far more tokens than Chinchilla suggests, to make it cheaper to run later. |
| **Deduplication (dedup)** | - | Removing repeated or near-identical documents from training data. |
| **MinHash** | - | A fast technique to find near-duplicate documents. |
| **Quality filtering** | - | Removing low-quality text using rules or a classifier. |
| **Decontamination** | - | Removing test-set content from training data so your scores are honest. |
| **Data mix** | - | The recipe of how much of each data type (web, code, books, math) goes into training. |
| **Annealing** | - | Ending training with a short phase on high-quality data. |
| **Synthetic data** | - | Training data generated by another model rather than written by humans. |
| **Replay data** | - | Old, general data mixed back in during new training to reduce forgetting. |
| **Perplexity** | - | A measure of how surprised the model is by text; lower is better. |
| **Tokenizer fertility** | - | Average tokens needed per word. High fertility (common for under-served languages) wastes context and compute. |
| **Vocabulary extension** | - | Adding new tokens (e.g., for a new language) to a model's tokenizer and embeddings. |
| **Low-resource language** | - | A language with little digital text available, so models handle it poorly. |
| **FineWeb / FineWeb-Edu** | - | Large open web datasets and their education-focused filter, from Hugging Face. |
| **Dolma** | - | AI2's open pretraining dataset. |
| **CulturaX / FineWeb-2 / MADLAD-400 / OSCAR** | - | Large multilingual text collections used for non-English training. |
| **TinyStories** | - | A dataset of simple synthetic stories showing tiny models can write coherent text. |

---

## 11. Post-Training and Reinforcement Learning

| Term | Stands for | What it means |
|------|-----------|---------------|
| **SFT** | Supervised Fine-Tuning | Training on example prompt-and-answer pairs so the model learns to respond the way you want. |
| **RL** | Reinforcement Learning | Learning by trying things and receiving rewards or penalties. |
| **RLHF** | Reinforcement Learning from Human Feedback | Training with a reward model built from human preference rankings. |
| **RLAIF** | Reinforcement Learning from AI Feedback | Like RLHF but the preference labels come from an AI. |
| **Constitutional AI** | - | Anthropic's method using a written set of principles so an AI critiques and ranks outputs, mainly for harmlessness. |
| **Reward model** | - | A model trained to score how good an answer is. |
| **PPO** | Proximal Policy Optimization | A classic RL algorithm used in RLHF. Needs a critic (value) network. |
| **DPO** | Direct Preference Optimization | A simpler method that learns from preference pairs without a separate reward model or RL loop. |
| **GRPO** | Group Relative Policy Optimization | DeepSeek's RL method: sample a group of answers, compare their rewards, no critic needed. |
| **RLVR** | Reinforcement Learning with Verifiable Rewards | RL where rewards come from automatic checks (correct math answer, passing tests). |
| **Policy** | - | The model being trained, viewed as a decision-maker. |
| **Reference model** | - | A frozen copy used to keep the trained model from drifting too far. |
| **KL penalty** | Kullback-Leibler penalty | A penalty for straying too far from the reference model. |
| **Critic / value network** | - | A second model that estimates future reward; GRPO removes it. |
| **Reward hacking** | - | The model finds loopholes that earn reward without truly solving the task. |
| **R1-Zero** | - | DeepSeek's model trained with pure RL from a base model, no SFT. |
| **R1** | - | The polished DeepSeek reasoning model: cold-start SFT, RL, rejection sampling, final RL. |
| **Cold start** | - | A small SFT stage before RL so the model begins with a sensible format. |
| **Rejection sampling / best-of-N** | - | Generate many answers, keep only the best ones, train on those. |
| **Distillation** | - | Training a smaller model to imitate a bigger one's outputs. |
| **Chain of thought (CoT)** | - | Showing step-by-step reasoning before the final answer. |
| **Alignment** | - | Making a model behave helpfully and safely according to human goals. |
| **Preference data** | - | Pairs of answers where humans or AI marked which is better. |
| **Chat template** | - | The exact formatting (special tokens, roles) a model expects for conversations. Mismatching it hurts quality. |
| **Assistant-only loss** | - | Computing loss only on the assistant's reply tokens, not on the prompt. |
| **Packing** | - | Combining several short examples into one sequence for efficient training. |

---

## 12. Fine-Tuning Methods and Tools

| Term | Stands for | What it means |
|------|-----------|---------------|
| **Full fine-tuning** | - | Updating every parameter. Powerful but memory-hungry. |
| **PEFT** | Parameter-Efficient Fine-Tuning | Family of methods that train only a small part of the model. Also the name of Hugging Face's library. |
| **LoRA** | Low-Rank Adaptation | Freeze the model and train small add-on matrices. |
| **QLoRA** | Quantized LoRA | LoRA on top of a 4-bit compressed base model, so big models fit on small GPUs. |
| **Adapter** | - | The small trained add-on produced by LoRA. Can be swapped or merged. |
| **LoRA rank (r)** | - | The size of the add-on. Higher rank = more capacity and memory. (Unrelated to a GPU's rank.) |
| **LoRA alpha** | - | A scaling factor for the add-on; commonly set to about 2x the rank. |
| **target_modules** | - | Which layers receive LoRA adapters. `all-linear` means all linear layers. |
| **modules_to_save** | - | Layers trained fully (not just adapted), e.g., embeddings when adding a new language. |
| **Merge (merge_and_unload)** | - | Folding the adapter back into the base model to get one standalone model. |
| **CPT** | Continued Pretraining | Further training of a base model on raw domain or language text to add knowledge. |
| **Hugging Face (HF)** | - | The main hub for models, datasets, and open-source libraries. |
| **Transformers** | - | Hugging Face's core library for loading and running models. |
| **TRL** | Transformer Reinforcement Learning | Hugging Face library for SFT, DPO, PPO, and GRPO. |
| **bitsandbytes** | - | Library that provides 4-bit and 8-bit quantization for training. |
| **Unsloth** | - | Library with optimized kernels that makes fine-tuning faster and lighter on memory. |
| **Axolotl / LLaMA-Factory** | - | Config-file-driven tools for fine-tuning many models. |
| **torchtune** | - | PyTorch's native fine-tuning recipes. |
| **MLX / MLX-LM** | - | Apple's machine-learning framework for Apple Silicon Macs, and its LLM toolkit. |
| **Accelerate** | - | Hugging Face library that makes multi-GPU launches simple. |

---

## 13. Frameworks, Kernels, and Infrastructure Software

| Term | Stands for | What it means |
|------|-----------|---------------|
| **PyTorch** | - | The dominant deep-learning framework; most labs build on it. |
| **JAX** | - | Google's numerical-computing framework, used with TPUs (Gemini). |
| **XLA / GSPMD / Pathways** | Accelerated Linear Algebra / General and Scalable Parallelization for ML Computation Graphs / - | Google's compiler, auto-parallelization system, and large-scale runtime. |
| **Megatron-LM / Megatron-Core** | - | NVIDIA's reference library for tensor, pipeline, and expert parallelism. |
| **DeepSpeed** | - | Microsoft's library featuring ZeRO and offload. |
| **torchtitan** | - | A clean PyTorch-native example for multi-dimensional parallel training. |
| **nanotron** | - | Hugging Face's minimal 3D-parallel pretraining library. |
| **OLMo-core** | - | AI2's open pretraining code. |
| **Triton** | - | A Python-like language for writing fast GPU kernels. (Not NVIDIA's "Triton Inference Server".) |
| **CUTLASS / CuTe** | CUDA Templates for Linear Algebra Subroutines | NVIDIA's building blocks for hand-tuned matrix-multiply kernels. |
| **DeepGEMM** | - | DeepSeek's library for fast FP8 matrix multiplication. |
| **Kernel** | - | A small program that runs on the GPU to do one operation. |
| **Operator fusion** | - | Combining several operations into one kernel to avoid extra memory trips. |
| **FlashAttention** | - | A memory-efficient way to compute attention by working in on-chip SRAM. |
| **torch.compile / Inductor** | - | PyTorch's compiler that optimizes and fuses operations automatically. |
| **Ray** | - | A framework for running distributed Python jobs and pipelines. |
| **SkyPilot** | - | A tool that launches jobs on whichever cloud has the cheapest GPUs. |
| **Weights & Biases (W&B) / TensorBoard / MLflow** | - | Tools that log and chart your training runs. |
| **uv** | - | A fast Python package and environment manager. |
| **WSL2** | Windows Subsystem for Linux 2 | Lets Windows run a real Linux environment; the easiest way to use CUDA training tools on Windows. |
| **tmux / screen** | - | Terminal tools that keep a job running after you disconnect. |

---

## 14. Reliability and Operations

| Term | Stands for | What it means |
|------|-----------|---------------|
| **Fault tolerance** | - | Surviving hardware failures without losing the run. |
| **Async checkpointing** | Asynchronous checkpointing | Saving checkpoints in the background so training does not pause. |
| **Elastic training** | - | Resuming on a different number of nodes after a failure. |
| **SDC** | Silent Data Corruption | A faulty chip silently gives wrong results with no error message. |
| **Gang scheduling** | - | Starting all GPUs of a job at the same moment, or not at all. |
| **Hot spare** | - | An idle ready-to-use node kept to replace a failed one. |
| **Watchdog / flight recorder** | - | Tools that detect and log hangs in GPU communication. |
| **Effective training time** | - | The share of wall-clock time the run was actually progressing. Llama 3 reported over 90%. |
| **Stop-loss** | - | A pre-set rule for when to halt a costly run that is not working. |

---

## 15. Evaluation

| Term | Stands for | What it means |
|------|-----------|---------------|
| **Eval set** | - | Fixed test questions with reference answers, never used for training. |
| **Baseline** | - | The score of the starting model (or a simple alternative) that you must beat. |
| **Benchmark** | - | A standard public test (MMLU, GSM8K, etc.) for comparing models. |
| **Held-out data** | - | Data deliberately kept out of training so it can test generalization. |
| **Contamination / leakage** | - | Test or evaluation material accidentally included in training data, making scores fake. |
| **LLM-as-judge** | - | Using a strong model to grade answers. Should be checked against human labels. |
| **Regression test** | - | A check that new training did not break something that used to work. |
| **lm-evaluation-harness** | - | EleutherAI's standard toolkit for running benchmarks. |
| **lighteval / Inspect** | - | Other evaluation frameworks (Hugging Face / UK AI Security Institute). |

---

## 16. Deployment and Serving

| Term | Stands for | What it means |
|------|-----------|---------------|
| **llama.cpp** | - | Lightweight program for running quantized models on ordinary CPUs and GPUs. |
| **Ollama** | - | A user-friendly app for running models locally, built on llama.cpp. |
| **vLLM / SGLang** | - | High-throughput servers for hosting models for many users. |
| **Batching** | - | Handling many requests together to use the GPU efficiently. |
| **Serverless GPU** | - | Renting a GPU only while code runs, billed per second (Modal, RunPod serverless). |
| **Endpoint** | - | A web address that accepts requests and returns model output. |
| **Cost per 1M tokens** | - | The standard price unit for comparing hosting options and APIs. |
| **Model card** | - | A document describing a model's data, limits, and intended use. |

---

## 17. Common Confusions Cleared Up

| Confusion | Clear answer |
|-----------|--------------|
| **Parameters vs tokens** | Parameters = the model's size (how many learned numbers). Tokens = the amount of text it reads. A 7B model trained on 1T tokens has 7 billion parameters and saw 1 trillion tokens. |
| **Training vs inference** | Training builds or changes the model (expensive, needs lots of memory). Inference just uses it (cheap by comparison). |
| **Pretraining vs fine-tuning** | Pretraining learns general language from scratch on huge data. Fine-tuning adjusts an already-trained model for your task. |
| **FLOPs vs FLOPS** | FLOPs = a total count of operations. FLOPS = operations per second (speed). |
| **GB (memory) vs Gb (network)** | Capital B = bytes; lowercase b = bits. 400 Gb/s network = 50 GB/s. |
| **Rank (LoRA) vs rank (distributed)** | LoRA rank = size of the adapter. Distributed rank = a GPU's ID number in the job. Same word, unrelated. |
| **Sequence parallelism vs context parallelism** | Megatron's SP saves activation memory inside a TP group. CP splits whole long sequences, including attention, across GPUs. |
| **Open weights vs open source** | Open weights = you can download the trained model. Open source = code, data, and recipes are also released (like OLMo). Many "open" models are only open weights. |
| **Total vs active parameters (MoE)** | You must store all parameters but only compute with the active ones per token. Cost of compute follows active; memory follows total. |
| **MoE vs dense** | Dense uses everything every time. MoE routes each token to a few experts. |
| **Quantization vs LoRA** | Quantization shrinks stored numbers (smaller model). LoRA limits what gets trained (smaller training cost). QLoRA uses both. |
| **Loss dropping vs model improving** | Lower loss does not guarantee a better product. Always check outputs and your eval set. |
| **"$5.6M to train DeepSeek-V3"** | That is the final run's GPU rental only, not research, failed runs, data, or staff. |
| **Triton (kernel language) vs Triton (inference server)** | Same name, different products. This cookbook means the GPU kernel language from OpenAI. |
| **Frontier lab claims** | Always ask: is it **[PUBLIC]**, **[INFERRED]**, or **[UNKNOWN]**? |

---

## 18. Organizations, Projects, and People Mentioned

| Name | What it is |
|------|-----------|
| **Anthropic** | AI lab behind Claude and Constitutional AI. Trains on Trainium, TPUs, and NVIDIA GPUs. |
| **OpenAI** | AI lab behind GPT models and the Triton kernel language. |
| **Meta (Llama)** | Released Llama models and the detailed Llama 3 paper. |
| **DeepSeek** | Chinese lab behind V3 and R1, known for efficiency and openly published methods. |
| **Google / DeepMind (Gemini, Gemma)** | Trains Gemini on TPUs with JAX; releases the open Gemma models. |
| **Alibaba (Qwen)** | Open-weight Qwen model families, popular fine-tuning bases. |
| **Mistral AI** | French lab with several open-weight models. |
| **Moonshot AI (Kimi)** | Lab behind Kimi K2 and the MuonClip optimizer. |
| **Microsoft (Phi, DeepSpeed, TinyStories)** | Small-model research and the DeepSpeed library. |
| **NVIDIA** | Makes GPUs, CUDA, NCCL, Megatron, NeMo, and Nemotron models. |
| **AI2 (Allen Institute for AI)** | Nonprofit lab behind fully open OLMo, Dolma, and Tulu. |
| **Hugging Face** | Model and dataset hub; makers of Transformers, TRL, PEFT, SmolLM, FineWeb. |
| **EleutherAI** | Open research collective behind Pythia, GPT-NeoX, and lm-evaluation-harness. |
| **Andrej Karpathy (nanoGPT, llm.c, nanochat)** | Educator whose small repos teach the full training stack. |
| **Stanford CS336** | Course on building a language model from scratch. |
| **AI4Bharat** | IIT Madras group building datasets and models for Indian languages. |
| **Sarvam AI / Krutrim / BharatGen** | Indian efforts building Indic-language models. |
| **IndiaAI Mission** | Indian government program including subsidized compute access (verify eligibility). |
| **Stargate** | OpenAI's announced gigawatt-scale datacenter project. |
| **Project Rainier** | Amazon's large Trainium cluster used by Anthropic. |
| **Thinking Machines Lab (Tinker)** | AI lab offering a LoRA fine-tuning API (verify availability). |
| **Prime Intellect / Nous Research** | Groups working on open models and decentralized training and compute. |
| **Vast.ai, RunPod, Lambda, Nebius, Crusoe, Modal, Together AI, SF Compute** | GPU rental and cloud providers covered in the indie guide. |

---

## 19. Licenses and Legal Terms

| Term | Stands for | What it means |
|------|-----------|---------------|
| **MIT** | - | A very permissive software license: use it for almost anything, keep the copyright notice. Used here for code. |
| **Apache-2.0** | - | Permissive license that also includes a patent grant. Common for open models. |
| **CC BY 4.0** | Creative Commons Attribution 4.0 | Anyone may reuse and adapt the content if they credit you. Used here for the docs. |
| **Llama / Gemma license** | - | Custom model licenses with extra conditions (acceptable-use rules, attribution, size thresholds). Read before commercial use. |
| **ToS** | Terms of Service | A provider's rules. Many AI APIs forbid using their outputs to train competing models. |
| **robots.txt** | - | A website file stating which pages automated crawlers may visit. |
| **Personal data / PII** | Personally Identifiable Information | Information that identifies a person. Needs consent and care under privacy law. |
| **Credits / grants** | - | Free or discounted cloud compute from startup programs (AWS Activate, Google for Startups, NVIDIA Inception) or research programs (Google TRC). |
| **TRC** | TPU Research Cloud | Google's program giving free TPU access to approved researchers. |

---

## 20. Labels Used in These Guides

| Label | Meaning |
|-------|---------|
| **[PUBLIC]** | Documented in a paper, technical report, or official announcement. |
| **[INFERRED]** | Widely believed or logically implied, but not confirmed by the lab. |
| **[UNKNOWN]** | Not disclosed. Treat any confident claim about it with suspicion. |
| **(verify)** | A fact that changes quickly (prices, quotas, library APIs, availability). Check the provider's page before relying on it. |
| **Approximate / rough** | A planning estimate, not a measured value. Real results depend on your setup. |

---

## Still stuck on a word?

1. Search this page (Ctrl+F) for the word or its abbreviation.
2. Ask an LLM: "Explain **[term]** like I'm new to machine learning, with a one-line analogy."
3. Open an issue on the repo with the term you could not find so it can be added.