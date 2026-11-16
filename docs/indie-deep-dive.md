# Indie Deep Dive: Train Your Own Model as a Solo Builder
### Laptop → rented GPU → small cluster, plus the labs, tools, and open recipes worth building on

> Companion to `llm-training-deep-dive.md` and `llm-training-cheatsheet.md`.
> **Honesty note:** hardware specs, algorithms, and code patterns here are stable. **Prices, free tiers, credit programs, and library versions change fast.** Anything marked **(verify)** was true to the best of my knowledge but should be checked on the provider's site before you spend money. Libraries (TRL, PEFT, Unsloth, MLX) change APIs often, so if a snippet errors, check the version in the docs.

---

## 0. TL;DR: the 10 rules

1. **You will not pretrain a frontier model. You can still build real, valuable models.** Compete on data, domain, language, and product, not raw compute.
2. **Climb the ladder only as far as needed:** prompt → RAG → fine-tune (LoRA) → continued pretraining → train from scratch. Each rung costs ~10× more time and money.
3. **Fine-tuning an open model (QLoRA) is the 90% path.** A single 8 to 24 GB GPU is enough.
4. **Training from scratch on a laptop is for learning**, with models of roughly 1M to 125M parameters. Do it once; it teaches more than any course.
5. **Data quality is the real lever.** 1,000 excellent examples beat 100,000 sloppy ones.
6. **Build your eval set before you train.** If you can't measure it, you can't improve it.
7. **Do the math first:** `C = 6·N·D`. It tells you in 30 seconds whether a plan is feasible.
8. **Rent, don't buy.** Buy hardware only when you're running GPUs most hours of the month.
9. **Debug on 1 GPU, scale to 8, and only then think about multi-node.** Most solo builders never need a cluster.
10. **Check licenses.** Both for the base model and for any data (including AI-generated data).

---

## 1. Honest reality table: what can one person do at each hardware tier?

*(Approximate; depends on sequence length, batch size, and library. Treat as planning ranges.)*

| Tier | Typical hardware | What's realistic |
|------|------------------|------------------|
| **T0: CPU only** | Any laptop, 16 GB RAM | Train toy models (1 to 10M params, char-level); run 1 to 3B models quantized (GGUF) for inference; prepare data, tokenizers, evals |
| **T1: Laptop NVIDIA GPU** | RTX 3050/4050/4060 laptop (4 to 8 GB VRAM) | QLoRA on 0.5 to 3B models; 7B only with tricks (short context, Unsloth); from-scratch 10 to 50M params |
| **T2: Apple Silicon** | M-series, 16 to 64+ GB unified memory | MLX LoRA/QLoRA on 1 to 8B (bigger with 64 GB+); from-scratch small models; excellent for inference |
| **T3: 24 GB desktop GPU** | RTX 3090/4090 (or rented) | QLoRA up to ~30B (tight); LoRA on 7 to 8B in BF16; from-scratch up to ~0.5 to 1B (slow, token-limited) |
| **T4: 1× 80 GB GPU (rented)** | A100/H100 80 GB | LoRA/QLoRA up to 70B; full fine-tune ~7B with memory tricks; pretrain 100M to 1B for small token budgets |
| **T5: 8× H100 node (rented)** | One DGX/HGX-class node | Pretrain 1 to 3B models in days; nanochat-style full pipeline; full fine-tune 70B-class with FSDP/DeepSpeed |
| **T6: multi-node cluster** | 16 to 256+ GPUs with InfiniBand | Pretrain 7B+ on 1T+ tokens. Out of solo-budget range ($50K+); needed only for serious pretraining |

**Where most people should live:** T1/T2 for learning and prototyping, T3/T4 (rented by the hour) for the real training run.

---

## 2. Pick your path (decision tree)

```
Can a good prompt or RAG already solve it?
 ├─ Yes → Don't train. Ship it.
 └─ No → What's missing?
      ├─ Style / format / task behavior ........ SFT with LoRA/QLoRA (Path B)
      ├─ Domain or language knowledge ........... Continued pretraining + SFT (Path C)
      ├─ Reasoning on verifiable tasks .......... RL with GRPO/RLVR (Path D)
      ├─ Need small/fast/cheap model ............ Distill a big model into a small one (Path E)
      └─ I want to understand everything ........ Train from scratch, small (Path A)
```

### The five paths in one table

| Path | What | Data needed | Compute | Output |
|------|------|-------------|---------|--------|
| **A. From scratch (tiny)** | Train a small GPT on your own corpus | 10 MB to ~10 GB of text | Laptop to 1 GPU | A toy/learning model; occasionally useful for narrow tasks |
| **B. SFT + LoRA** | Teach an open model a task/style | 500 to 50K examples | 1 GPU, 1 to 8 hrs | A production-usable specialized model |
| **C. Continued pretraining (CPT)** | Feed an open model lots of raw domain/language text | 100M to 10B+ tokens | 1 to 8 GPUs, days | A model that "knows" your domain/language; follow with SFT |
| **D. RL (GRPO/RLVR)** | Reward correct answers; model learns to reason | Prompts with verifiable answers | 1 to 8 GPUs | Better reasoning on narrow tasks |
| **E. Distillation** | Train small model on outputs of a big model | 10K to 1M+ generated examples | API cost + 1 GPU | Cheap fast model; **check API terms of service** |

---

## 3. Set up your laptop

### 3.1 Windows (NVIDIA GPU): use WSL2 for training

Native Windows works for many things, but **WSL2 (Ubuntu) is the least painful path** for CUDA training libraries.

```powershell
# PowerShell (Admin)
wsl --install -d Ubuntu
# reboot, create a Linux user, then inside Ubuntu:
```

```bash
# Inside WSL2 Ubuntu
nvidia-smi                      # should show your GPU. Do NOT install a Linux NVIDIA driver in WSL;
                                # the Windows driver is used.
sudo apt update && sudo apt install -y build-essential git python3-venv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/venvs/train && source ~/venvs/train/bin/activate
# Pick the right torch command from https://pytorch.org/get-started/locally/ (CUDA build)
uv pip install torch --index-url https://download.pytorch.org/whl/cu124   # example; match your CUDA
uv pip install transformers datasets peft trl accelerate bitsandbytes sentencepiece wandb
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Tips:
- Keep project files **inside the Linux filesystem** (`~/projects`), not `/mnt/c/...`, because I/O is far faster.
- 8 GB VRAM laptop? Close browsers/Discord (they eat VRAM). Check with `nvidia-smi`.
- Laptop thermals matter. Training is a sustained load; use a cooling pad and plug in. Watch for throttling.

### 3.2 Mac (Apple Silicon): use MLX

```bash
python3 -m venv ~/venvs/mlx && source ~/venvs/mlx/bin/activate
pip install mlx-lm
# LoRA fine-tune a 4-bit model (data folder needs train.jsonl and valid.jsonl)
mlx_lm.lora --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --train --data ./data --iters 600 --batch-size 2
mlx_lm.generate --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --adapter-path ./adapters --prompt "Hello"
```
Rule of thumb: you need unified memory ≈ 1.2 to 1.5× the quantized model size plus activations. 16 GB handles ~3B comfortably; 32 to 64 GB opens 7 to 14B.

### 3.3 CPU-only laptop
Still useful: Path A (toy models), data work, tokenizers, evals, and running quantized models with **llama.cpp / Ollama**. Do heavy training on rented GPUs (section 9).

---

## 4. Path A: train a tiny GPT from scratch on your laptop (do this first)

**Goal:** understand every part of the loop. Not to beat GPT-anything.

### 4.1 Run the classic: nanoGPT on Shakespeare (CPU-friendly)

```bash
git clone https://github.com/karpathy/nanoGPT && cd nanoGPT
pip install torch numpy transformers datasets tiktoken wandb tqdm
python data/shakespeare_char/prepare.py
# CPU-friendly tiny config (~minutes):
python train.py config/train_shakespeare_char.py --device=cpu --compile=False \
  --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 \
  --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
python sample.py --out_dir=out-shakespeare-char --device=cpu
```
(Windows native: keep `--compile=False`. torch.compile support there is limited.)

### 4.2 What you should learn from it
- Watch **train vs val loss**. When val rises while train falls, you're overfitting (a tiny corpus does this fast).
- Change one knob at a time: `n_layer`, `n_embd`, `block_size`, `lr`. Record what happens. This is your intuition-building loop.
- Compute the parameter count by hand and match it to what the script prints.

### 4.3 Level up: models that actually say something coherent
- **TinyStories** (Microsoft Research, 2023): showed models of only ~1M to 30M params trained on simple synthetic stories can produce fluent, coherent text. Lesson: *narrow, clean data beats big, messy data at small scale.* Reproducible on a single consumer GPU.
- **Train your own tokenizer** (BPE via Hugging Face `tokenizers` or SentencePiece) on your corpus. Compare token counts vs GPT-2's tokenizer.
- **Karpathy's `llm.c`** and **nanochat** (full ChatGPT-style pipeline: tokenizer → pretrain → SFT → RL → web UI; documented as ~$100 of 8×H100 time, ~4 hours **(verify current README)**). Read nanochat end to end, since it's the best "whole stack in one small repo" reference.

### 4.4 Expected sizes on a laptop (rough)

| Params | Tokens you can afford on a laptop GPU in hours | Typical result |
|--------|-----------------------------------------------|----------------|
| 1 to 10M | 10 to 500M | Char/word-level toy generation |
| 10 to 50M | 100M to 1B | Coherent short text on a narrow domain |
| 100 to 125M (GPT-2-small class) | needs ~1 to 10B tokens → rent a GPU | Useful for experiments, not products |

---

## 5. Memory math for your GPU (do this before every run)

### 5.1 Inference / loading
`weights memory ≈ params × bytes_per_param`

| Format | Bytes/param | 7B model |
|--------|-------------|----------|
| FP32 | 4 | 28 GB |
| BF16/FP16 | 2 | 14 GB |
| INT8 | 1 | 7 GB |
| 4-bit (NF4/Q4) | ~0.5 | ~3.5 to 4.5 GB |

### 5.2 Training (rough per-method)

| Method | Memory per param (approx.) | 7B model |
|--------|---------------------------|----------|
| Full fine-tune, AdamW, mixed precision | ~16 B | ~112 GB (+ activations) |
| Full fine-tune, 8-bit optimizer + tricks | ~6 to 8 B | ~45 to 60 GB |
| LoRA (BF16 base) | ~2 B + small adapters + activations | ~16 to 24 GB |
| **QLoRA (4-bit base)** | ~0.5 B + adapters + activations | **~6 to 12 GB** depending on context length and batch |

**Activations scale with batch size × sequence length.** If you OOM: lower `max_length`, lower micro-batch to 1, raise gradient accumulation, turn on gradient checkpointing, or use Unsloth's memory-optimized kernels.

### 5.3 Quick "can I run it?" guide for QLoRA (rough)

| VRAM | Comfortable model size | Stretch |
|------|------------------------|---------|
| 4 GB | ≤1B | 1.5B, short context |
| 6 to 8 GB | 1.5 to 3B | 7B with Unsloth + short context |
| 12 GB | 7B | 13B tight |
| 16 to 24 GB | 7 to 14B | ~30B tight (24 GB) |
| 48 GB | 30 to 34B | 70B tight |
| 80 GB | 70B | bigger with offload |

---

## 6. Path B: fine-tune an open model with QLoRA (the money path)

### 6.1 Choose the base model
Pick the **smallest model that can do the job**. Larger = slower, costlier, harder to deploy.
- Start: **Qwen2.5/Qwen3 (0.5 to 8B), Llama 3.x (1 to 8B), Gemma 2/3, Phi, SmolLM3**, whichever has the best *base capability in your language/domain* and a license you can live with (section 12).
- Use the **instruct** variant if you're teaching a task/style; use the **base** variant if you plan heavy continued pretraining.

### 6.2 Data format (chat JSONL, one example per line)

```json
{"messages": [
  {"role": "system", "content": "You are a helpful tutor."},
  {"role": "user", "content": "Explain photosynthesis in simple terms."},
  {"role": "assistant", "content": "Plants use sunlight to turn water and CO2 into sugar..."}
]}
```

### 6.3 QLoRA training script (TRL + PEFT + bitsandbytes)

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"       # start small; scale up after it works

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,  # use float16 on GPUs without BF16 (e.g. T4)
)
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")

ds = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules="all-linear", task_type="CAUSAL_LM")

cfg = SFTConfig(
    output_dir="out", num_train_epochs=2, learning_rate=2e-4,
    per_device_train_batch_size=2, gradient_accumulation_steps=8,   # effective batch 16
    gradient_checkpointing=True, bf16=True,                          # fp16=True on older GPUs
    max_length=2048,            # older TRL versions call this max_seq_length
    logging_steps=10, eval_strategy="steps", eval_steps=100, save_steps=200,
    report_to="none",           # or "wandb"
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"],
                     eval_dataset=ds["validation"], peft_config=lora, processing_class=tok)
trainer.train()
trainer.save_model("out/lora")
```

### 6.4 Hyperparameter starting points

| Knob | Starting value | Notes |
|------|----------------|-------|
| LoRA rank `r` | 16 (8 to 64) | Higher for harder tasks/new knowledge |
| `lora_alpha` | 2×r | Common heuristic |
| Learning rate (LoRA) | 1e-4 to 2e-4 | Full fine-tune: 1e-5 to 2e-5 |
| Epochs | 1 to 3 | More epochs on small data → overfitting |
| Effective batch size | 16 to 64 | Use grad accumulation |
| Max length | Cover 95% of your examples | Longer costs memory quadratically-ish |
| Warmup | 3 to 5% of steps | |

### 6.5 The mistakes that waste weeks
- **Wrong chat template:** training with one template and serving with another silently ruins quality. Always use `tokenizer.apply_chat_template`.
- **Training on the prompt too:** for SFT, usually compute loss on assistant tokens only (TRL supports assistant-only loss for supported templates **(verify per version)**).
- **Data leakage:** val examples near-duplicate of train → fake great metrics.
- **Catastrophic forgetting:** too many epochs/high LR on narrow data. Mix in some general instruction data (5 to 20%).
- **No baseline:** always evaluate the *untuned* base model on your eval set first.

### 6.5b Faster/cheaper tooling
- **Unsloth:** optimized kernels, large VRAM savings, great free-Colab notebooks.
- **Axolotl / LLaMA-Factory:** YAML-driven, supports many models/methods and multi-GPU.
- **torchtune:** PyTorch-native recipes.
- **MLX-LM:** Apple Silicon.

### 6.6 Merge and export
```python
# merge LoRA into base (do this on a machine with enough RAM)
from peft import AutoPeftModelForCausalLM
m = AutoPeftModelForCausalLM.from_pretrained("out/lora", torch_dtype="auto")
m = m.merge_and_unload(); m.save_pretrained("merged"); tok.save_pretrained("merged")
```
Then convert to **GGUF** with llama.cpp (`convert_hf_to_gguf.py`, then `llama-quantize`) to run in Ollama/llama.cpp on any laptop.

---

## 7. Path C: continued pretraining (teach knowledge or a new language)

Use when SFT alone can't add enough *knowledge* (a niche domain, or a language the base model handles poorly).

### 7.1 Recipe
1. Collect **raw text** (hundreds of MB to tens of GB), then clean and dedup it.
2. **Stage 1 (CPT):** train the base model on raw text (next-token prediction). Mix in **replay data** (10 to 30% general-domain/English) to limit forgetting.
3. **Stage 2 (SFT):** instruction-tune on high-quality Q&A/chat data.
4. Evaluate against baseline at each stage.

### 7.2 Low-resource languages (big one for Indic and many other languages)
- **Tokenizer fertility** is the hidden cost: many LLM tokenizers split non-Latin-script words into many tokens, so you burn context, memory, and compute (and quality suffers). **Measure it:** tokens per word for your language vs English on a sample.
- If fertility is terrible, options: (a) pick a base model whose tokenizer already handles your script well; (b) **extend the vocabulary** (train a tokenizer on your corpus, merge new tokens, initialize new embeddings from the mean of the sub-token embeddings they replace), then do CPT with embeddings and LM-head trainable (`modules_to_save`) before LoRA SFT.
- Data sources: Wikipedia dumps, CommonCrawl-derived multilingual sets (CulturaX, FineWeb-2, MADLAD-400, OSCAR **(verify current availability and licenses)**), government/news/literature with permission, community-built corpora, AI4Bharat datasets for Indic languages.
- Evaluation is the hard part: build your own native-speaker eval (100 to 500 items) because translated English benchmarks miss the culture and idioms.

### 7.3 Compute estimate
CPT on 1B tokens of a 7B model: `6 × 7e9 × 1e9 = 4.2e19 FLOPs`. On one H100 at ~40% MFU (~395 TFLOPs): ~1.06e5 s ≈ **~30 hours**. With LoRA you cut optimizer/memory cost but not forward/backward FLOPs much. Hence small models (1 to 3B) are the sweet spot for solo CPT.

---

## 8. Path D and E: RL and distillation

### 8.1 RL with verifiable rewards (GRPO)
Works when you can *check* answers automatically (math, code with tests, structured extraction, format compliance).

```python
# Sketch with TRL's GRPOTrainer (API evolves; check the docs for your version)
from trl import GRPOConfig, GRPOTrainer

def reward_exact(completions, answer, **kwargs):
    # completions: list of generated strings (or message lists in conversational mode)
    # extra dataset columns (like `answer`) arrive as kwargs
    return [1.0 if str(a).strip() in c else 0.0 for c, a in zip(completions, answer)]

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=reward_exact,
    args=GRPOConfig(output_dir="grpo-out", num_generations=8, learning_rate=1e-6,
                    per_device_train_batch_size=8, max_completion_length=512, bf16=True),
    train_dataset=ds_with_prompt_and_answer,   # needs a "prompt" column + your answer column
)
trainer.train()
```
Tips: start from an SFT'd model that already formats answers; keep rewards simple; watch for reward hacking; log sample outputs every N steps. Unsloth and TRL both have GRPO notebooks that fit on 16 to 24 GB GPUs for small models.

### 8.2 Distillation
Generate training data from a stronger model, then SFT a smaller one.
- Use **open-weight teachers** (Qwen, DeepSeek, Llama, etc.) whenever possible: fewer legal questions.
- **Read the terms of service of any hosted API** you use for generating training data. Many providers (including Anthropic and OpenAI) restrict using outputs to build competing models. Don't assume it's allowed.
- Filter the generated data hard (correctness checks, dedup, diversity). Garbage-in applies doubly here.

---

## 9. Renting GPUs: where to go, how to choose

### 9.1 Free/cheap starting points

| Option | What you get | Notes |
|--------|--------------|-------|
| **Google Colab** | Free T4-class GPUs (limits vary); paid tiers for A100/H100-class | Sessions time out; fine for Unsloth notebooks and small runs |
| **Kaggle Notebooks** | Free weekly GPU hours (T4/P100) and TPU access **(verify current quotas)** | Great for 1 to 3B QLoRA experiments |
| **Lightning AI Studios** | Persistent dev environments with a free monthly credit allotment **(verify)** | Nice VS Code-like experience |
| **Hugging Face** | Free Spaces CPU; GPU grants for community projects **(verify)** | Good for demos |
| **Google TPU Research Cloud (TRC)** | Free TPU access for researchers who apply | JAX-oriented; strong if you learn JAX |

### 9.2 Pay-per-hour GPU clouds (typical choices for solo builders)

| Provider type | Examples | Good for | Watch-outs |
|---------------|----------|----------|------------|
| **Marketplaces** | **Vast.ai**, **RunPod (community cloud)**, TensorDock | Cheapest $/hr, consumer GPUs (3090/4090) and datacenter GPUs | Variable reliability/network; verify host quality; interruptible instances can vanish |
| **Reliable GPU clouds** | **RunPod (secure)**, **Lambda**, **Nebius**, **Crusoe**, **Hyperstack**, **Vultr**, **Paperspace (DigitalOcean)**, **Fluidstack**, **Voltage Park** | A100/H100 on demand, 1 to 8 GPU nodes, clean images | Capacity can be tight; check quotas |
| **Serverless GPU / app-style** | **Modal**, **RunPod serverless**, **Replicate** | Run a script on a GPU, pay per second; great for batch jobs/inference | Cold starts; less control for long multi-GPU jobs |
| **Hyperscalers** | **AWS, GCP, Azure** | Credits via startup programs; managed services | Often pricier per GPU-hr; quota requests needed; egress fees |
| **Managed fine-tuning APIs** | **Together AI, Fireworks, Predibase, Mistral fine-tuning, OpenAI fine-tuning API, Thinking Machines' Tinker (LoRA training API) (verify availability/terms)** | You upload data, they train; no infra work | Less control; model choices limited; vendor lock-in; check data/IP terms |
| **Short-term cluster rental** | **SF Compute**, Lambda 1-Click Clusters, Together GPU Clusters, CoreWeave, Nebius, Crusoe **(verify)** | Hours-to-days of multi-node H100/B200 | Minimum commitments; need ops skill (section 11) |
| **Indian GPU clouds** | **E2E Networks**, **Yotta (Shakti Cloud)**; **IndiaAI Mission** subsidized compute for eligible startups/researchers **(verify eligibility, current rates, application process)** | Data residency, INR billing, possible subsidies | Smaller catalog; availability varies |

**Approximate hourly price ranges (2025-era; verify, since prices dropped steadily and vary by provider):**
- RTX 3090/4090 (24 GB): roughly **$0.2 to $0.8/hr** on marketplaces
- A100 80 GB: roughly **$1 to $2.5/hr**
- H100 80 GB: roughly **$2 to $4/hr** (cheaper on marketplaces/spot; more on hyperscalers)
- H200/B200: higher; check current availability

### 9.3 How to pick (decision rules)
- **Learning/experiments, short sessions:** Colab/Kaggle/Lightning → then RunPod/Vast 4090.
- **A real 1 to 3 day fine-tune on 7 to 30B:** 1× A100/H100 on a reliable cloud. Don't save $0.30/hr and risk a vanishing host.
- **8-GPU single-node pretraining or 70B work:** Lambda/Nebius/Crusoe/RunPod/Hyperstack/Together on-demand 8× H100/H200.
- **Spot/interruptible instances:** 30 to 70% cheaper, but **only** if your training script **checkpoints often and resumes automatically** (section 9.5).
- **Compare total cost:** GPU-hr price **+ storage + egress + idle time** (billing continues while you debug!).

### 9.4 Rental checklist (save money and sanity)
1. **Prepare everything locally first:** code tested on a tiny dataset, data preprocessed and uploaded to cloud storage.
2. **Use a container/template** with PyTorch + CUDA preinstalled; pin your versions in `requirements.txt`.
3. **Tmux/screen** (or a job runner) so your run survives SSH disconnects.
4. **Persistent volume** for checkpoints. Instance disks vanish when instances are terminated.
5. **Log to W&B/TensorBoard** so you can watch remotely.
6. **Set a budget alarm and an auto-shutdown**: `python train.py && shutdown -h now` (or provider API to stop the pod).
7. **Dry-run 100 steps** and measure tokens/sec before launching a long job. Extrapolate time and cost.
8. **Download results and terminate**. Idle GPUs are the #1 cause of surprise bills.

### 9.5 Surviving interruptions (spot instances)
- Save a checkpoint every N minutes (model, optimizer, scheduler, RNG state, data-loader position).
- On startup, auto-detect the latest checkpoint and resume.
- Keep checkpoints on a persistent network volume or object storage (S3-compatible).

### 9.6 Credits and grants worth applying for (all **verify**)
- **AWS Activate, Google for Startups Cloud Program, Microsoft for Startups (Azure), NVIDIA Inception**: cloud credits for early-stage startups.
- **Lambda / Together / Modal / RunPod startup or research credits**: often by application or via accelerators.
- **Hugging Face community grants**, **Google TRC (TPUs)**, **IndiaAI Mission** compute (India), and university/non-profit research programs.
- Tip: a clear one-page project description, an open-source repo, and a measurable goal improve acceptance odds.

---

## 10. Cost estimation: real numbers using 6ND

Formula: `GPU-hours = 6·N·D / (peak_FLOPs × MFU × 3600)`. H100 BF16 peak ≈ 989 TFLOPs; assume MFU 40% → ~395 TFLOPs ≈ 1.42e18 FLOPs per GPU-hour.

| Job | N | D | FLOPs | H100-hours (≈) | Rough cost at $2 to $4/hr |
|-----|---|---|-------|----------------|---------------------------|
| Toy model | 125M | 10B | 7.5e18 | ~5 | **~$10 to $20** |
| Small LM | 1B | 20B (Chinchilla-ish) | 1.2e20 | ~85 | **~$170 to $340** |
| 1B, overtrained | 1B | 100B | 6e20 | ~420 | **~$850 to $1,700** |
| 3B | 3B | 100B | 1.8e21 | ~1,270 | **~$2.5K to $5K** |
| 7B (pretrain) | 7B | 1T | 4.2e22 | ~29,600 | **~$60K to $120K** |
| QLoRA SFT, 7B, 20K examples × 1K tokens × 2 epochs | 7B | 40M | ~1.7e18 | ~1 to 3 (QLoRA overheads) | **~$3 to $15** |
| CPT, 7B, 1B tokens | 7B | 1B | 4.2e19 | ~30 | **~$60 to $120** |

Notes:
- Real runs add overhead: failed runs, ablations, evals, data prep. **Multiply the "clean" estimate by 2 to 3× for planning.**
- MoE changes the picture: use *active* params in `6ND`, but you still need memory for *all* params.
- If your quoted MFU is much lower on a small setup (it often is, 20 to 30%), scale the hours up proportionally.

### Budget ladders

| Budget | What you can do |
|--------|-----------------|
| **$0** | Toy models on laptop/Colab; QLoRA ≤3B on Kaggle/Colab free tiers; learn the full pipeline |
| **~$50** | Several QLoRA runs on rented 24 GB GPUs; a decent 1 to 7B specialization experiment |
| **~$500** | Serious SFT/RL experiments up to ~30B; CPT on a small model; a "nanochat-style" small pretrain experiment |
| **~$5K** | A 1 to 3B pretrain from scratch on ~100B tokens; heavy CPT; many ablations |
| **$50K+** | 7B-class pretraining; this is startup/grant territory |

---

## 11. When you actually need a cluster (and how to do it without pain)

### 11.1 Do you really need one?
You need multi-GPU/multi-node only if:
- The model + optimizer state doesn't fit on one node (e.g., full fine-tuning ≥70B, pretraining ≥ ~7B), or
- Wall-clock time is prohibitive on one node (the 6ND math says weeks), and your timeline matters.

**Try first:** gradient checkpointing, QLoRA, 8-bit optimizers, FSDP on one 8-GPU node. A single 8× H100 node handles an enormous range of solo projects.

### 11.2 Escalation ladder
1. **1 GPU** → get the pipeline right; measure tokens/sec.
2. **Multi-GPU, single node (2 to 8 GPUs)** → DDP, then FSDP/DeepSpeed ZeRO. Use `torchrun --nproc_per_node=8 train.py`.
3. **Multi-node (2 to 16 nodes)** → needs InfiniBand/RoCE; use `torchrun --nnodes=... --rdzv_backend=c10d` or Slurm; test networking first.
4. **Large clusters (64+ GPUs)** → needs dedicated infra skills: Slurm/K8s, health checks, NCCL tuning, auto-resume. Realistically a team job.

### 11.3 Tools for multi-GPU/multi-node training
| Tool | Use |
|------|-----|
| **PyTorch DDP/FSDP2** | Native; start here |
| **Hugging Face Accelerate** | Easy launcher for DDP/FSDP/DeepSpeed |
| **DeepSpeed (ZeRO)** | Memory-saving sharding; offload to CPU/NVMe |
| **Axolotl / LLaMA-Factory** | Config-driven multi-GPU fine-tuning |
| **torchtitan** | Clean PyTorch-native 4D parallelism reference/training |
| **nanotron** (Hugging Face) | Minimal 3D-parallel pretraining library |
| **Megatron-LM / Megatron-Core** | The reference for TP/PP/EP at scale; steeper learning curve |
| **OLMo-core / OLMo codebase** (AI2) | Fully open pretraining codebase with documented recipes |

### 11.4 Before you pay for a cluster: the burn-in checklist
- [ ] Run **`nccl-tests`** (`all_reduce_perf`) across nodes; compare to the provider's advertised bandwidth.
- [ ] Verify GPU health (`nvidia-smi`, DCGM diagnostics) and identical driver/CUDA/NCCL versions on all nodes.
- [ ] Put **data on fast shared storage** (parallel FS or well-sharded object storage), and pre-tokenize it.
- [ ] Prove **checkpoint → kill → resume** works at scale.
- [ ] Run a **short real job** (30 to 60 min), compute MFU; fix bottlenecks before the long run.
- [ ] Monitor: loss, grad norm, tokens/sec per GPU, GPU utilization, temperature, NCCL errors.
- [ ] Write down a **stop-loss**: "If MFU < X% or loss diverges by step Y, stop and debug."

---

## 12. Top labs, open recipes, and models for indie builders to learn from and build on

*"Top-notch" for an indie hacker = **open weights + open recipes + permissive licenses + good docs.** Licenses change; always read the current one **(verify)**.*

### 12.1 Fully/mostly open training recipes (best for learning *how*)

| Org | What to use | Why it matters for you |
|-----|-------------|------------------------|
| **AI2 (Allen Institute): OLMo family, Dolma/Tülu** | Open weights, data, training code, logs, and post-training recipes | The most complete reproducible reference for end-to-end LLM training |
| **Hugging Face: SmolLM family, FineWeb/FineWeb-Edu, Ultra-Scale Playbook, nanotron, TRL, PEFT, Transformers** | Small, strong models; open data pipelines; training handbooks | Best ecosystem + docs for solo builders |
| **Andrej Karpathy: nanoGPT, llm.c, nanochat** | Minimal full-stack code | Learn every moving part in a few thousand lines |
| **EleutherAI: Pythia, GPT-NeoX, lm-evaluation-harness** | Research-grade open models with checkpoints; standard eval harness | Study training dynamics; evaluate consistently |
| **Stanford CS336** | Course: build an LLM from scratch | Structured curriculum with assignments |
| **Microsoft Research: TinyStories, Phi series** | Small-model, data-quality-first research | Proves clean, curated data lets small models punch above their size |

### 12.2 Strong open-weight base models (best to fine-tune)

| Lab | Models | Notes |
|-----|--------|-------|
| **Alibaba (Qwen)** | Qwen2.5 / Qwen3 (dense + MoE, many sizes) | Very popular fine-tuning base; strong multilingual coverage; check license per size |
| **Meta (Llama)** | Llama 3.x family | Huge ecosystem; custom community license with conditions |
| **Google (Gemma)** | Gemma 2/3 | Strong small models; Gemma terms of use apply |
| **Mistral AI** | Mistral/Mixtral/Ministral variants | Several Apache-licensed options |
| **DeepSeek** | V3/R1 and distilled R1 variants (Qwen/Llama-based) | Reasoning-focused; open reports with deep technical detail |
| **Moonshot (Kimi), Zhipu (GLM), others** | Large open MoE models | Great to study; too big to fine-tune on a laptop, but distilled/smaller variants may exist **(verify)** |
| **NVIDIA** | Nemotron models, NeMo/Megatron tooling | Tooling plus open models/datasets |
| **Microsoft** | Phi small models | Small, data-quality-focused |
| **TII (Falcon), Liquid AI, others** | Alternative architectures/small models | Worth trying for edge/on-device use |

### 12.3 India-focused (useful if you build for Indian languages)
- **AI4Bharat (IIT Madras):** IndicTrans, IndicBERT, Indic datasets and benchmarks.
- **Sarvam AI:** Indic-language models and releases **(verify what's open)**.
- **Krutrim, BharatGen, IndiaAI Mission initiatives:** public efforts and compute programs **(verify)**.
- Community datasets on Hugging Face for Assamese, Bengali, Hindi, etc. Search for language-specific corpora and evals.

### 12.4 Compute & platform companies useful to solo builders
- **Hugging Face** (models/datasets/Spaces/AutoTrain), **Together AI** (fine-tuning + clusters), **Modal**, **RunPod**, **Lambda**, **Vast.ai**, **Nebius**, **Crusoe**, **SF Compute**, **Prime Intellect** (decentralized training/compute and RL environments; **verify offerings**), **Nous Research** (open models and decentralized-training experiments), **Thinking Machines Lab** (Tinker training API; **verify**).

---

## 13. Tooling map

| Need | Tools |
|------|-------|
| Training (SFT/LoRA) | TRL, PEFT, Unsloth, Axolotl, LLaMA-Factory, torchtune, MLX-LM |
| RL | TRL (GRPO/PPO/DPO), Unsloth GRPO, verl, OpenRLHF **(verify maturity)** |
| Pretraining | nanoGPT, nanotron, torchtitan, Megatron-LM, OLMo-core |
| Data processing | `datasets`, **datatrove** (HF), `datasketch` (MinHash dedup), fastText language ID, trafilatura/justext (web text extraction) |
| Tokenizers | HF `tokenizers`, SentencePiece, tiktoken |
| Evaluation | **lm-evaluation-harness**, lighteval, Inspect, custom task evals, human review |
| Tracking | Weights & Biases, TensorBoard, MLflow |
| Quantization | bitsandbytes, GPTQ/AWQ, GGUF (llama.cpp) |
| Inference/serving | **llama.cpp, Ollama, vLLM, SGLang, MLX**, Text Generation Inference |
| Orchestration | tmux, Slurm, SkyPilot (multi-cloud launcher), Ray |

---

## 14. Data: the part that decides whether you win

### 14.1 Pipeline
1. **Collect** (with rights!) → 2. **Extract text** (HTML/PDF/OCR) → 3. **Clean** (boilerplate, encoding, language ID) → 4. **Deduplicate** (exact + near-dup MinHash) → 5. **Filter quality** (heuristics, classifier, perplexity filter) → 6. **Decontaminate** vs your evals → 7. **Shard & tokenize** → 8. **Version it** (hashes, manifest).

### 14.2 For SFT data
- Aim for **diversity and correctness** over volume. Start with 500 to 2,000 curated examples; measure; add where the model fails.
- Write a **style guide** for your data (tone, format, refusal behavior) so examples are consistent.
- **Error-driven loop:** run model on eval → collect failures → write targeted examples → retrain → repeat. This beats blind scaling.
- Synthetic data: have a strong model draft, then **verify with code/rules/humans**. Include negative examples ("I don't know") to curb hallucination.

### 14.3 Evals (build these first)
- 100 to 500 realistic prompts with reference answers or rubrics, **frozen** and never trained on.
- Mix automatic checks (exact match, unit tests, regex) with **LLM-as-judge (calibrated against human labels)** and periodic human review.
- Track baseline (untuned model) vs tuned vs a strong API model, so you know your value-add.
- Include **regression tests** to catch forgetting.

---

## 15. Shipping: deployment for a solo builder

| Target | How |
|--------|-----|
| **Local/offline app** | GGUF + llama.cpp/Ollama; Q4_K_M quantization is a common quality/size balance |
| **Cheap API** | vLLM or SGLang on a rented GPU; or serverless GPU (Modal/RunPod) to pay only when used |
| **Edge/mobile** | Small models (≤3B) quantized; MLX (Apple), llama.cpp, ExecuTorch/ONNX |
| **Web demo** | Hugging Face Spaces or a small FastAPI + hosted inference endpoint |

Serving cost intuition: a 7B 4-bit model can run on a single 12 to 24 GB GPU with good throughput via vLLM. Batch requests to multiply throughput. Measure **cost per 1M tokens** and compare to API prices to confirm your model is worth hosting.

---

## 16. 12-week solo plan

| Week | Goal | Output |
|------|------|--------|
| 1 | Set up env (WSL2/MLX), run nanoGPT Shakespeare | Understand the training loop |
| 2 | Train a tokenizer; train a ~10 to 50M model on your own text | A tiny domain model + loss curves |
| 3 | Define your **real problem**; write 100 to 300 eval items | Frozen eval set + baseline scores |
| 4 | Collect/clean data (500 to 2,000 SFT examples) | Versioned dataset |
| 5 | First QLoRA run on a 1 to 3B model (laptop or Colab/Kaggle) | Fine-tuned adapter + eval gains |
| 6 | Error analysis → targeted data → retrain | Measured improvement |
| 7 | Rent a 24 to 80 GB GPU; scale to 7 to 8B | Better model; learn the rental workflow |
| 8 | (If needed) CPT on domain/language text + SFT | Knowledge-expanded model |
| 9 | (If verifiable task) GRPO experiment | Reasoning improvement |
| 10 | Quantize to GGUF; serve with Ollama/vLLM | Working demo |
| 11 | Harden: safety, regression tests, cost per request | Reliable system |
| 12 | Publish: model card, open dataset (if allowed), demo, write-up | Portfolio/community traction |

---

## 17. Pitfalls and legal/ethical checks

**Technical**
- Training before building evals; changing many variables at once; ignoring chat templates; evaluating on training data; forgetting to checkpoint; leaving GPUs running.
- Believing a loss drop = better product. Always check outputs and task metrics.
- Over-trusting LLM-as-judge without human calibration.

**Legal/ethical**
- **Data rights:** scraping isn't automatically permitted. Check licenses/ToS/robots rules, privacy law (personal data), and copyright in your jurisdiction. When in doubt, use permissively licensed or self-created data, or get permission.
- **Model licenses:** Apache-2.0/MIT are permissive; Llama and Gemma have custom terms (acceptable-use policies, usage thresholds, naming/attribution requirements). Read before commercializing.
- **API output restrictions:** many hosted-model ToS restrict training competing models on outputs.
- **Safety:** if your model gives advice (medical, legal, financial), add guardrails, disclaimers, and human-in-the-loop review. Test for harmful outputs and bias, especially in under-served languages where errors are less visible.
- **Privacy:** don't train on users' private data without consent; consider memorization risks.
- **Attribution/transparency:** publish a model card (data sources, limitations, eval results).

---

## 18. Study resources (free unless noted)

- **Courses/lectures:** Stanford CS336; Karpathy's *Neural Networks: Zero to Hero* and *Let's build GPT*; Hugging Face LLM course.
- **Books/guides:** Hugging Face *Ultra-Scale Playbook*; *How to Scale Your Model* (JAX/TPU scaling book); Unsloth docs/notebooks; TRL docs.
- **Repos to read line by line:** nanoGPT, llm.c, nanochat, torchtitan, nanotron, OLMo, Axolotl examples.
- **Papers:** LoRA, QLoRA, Chinchilla, TinyStories, Llama 3 herd, DeepSeek-V3/R1, Tülu 3 (open post-training recipe), FineWeb (data pipeline), DeepSeekMath (GRPO).
- **Communities:** Hugging Face forums/Discord, r/LocalLLaMA, Unsloth/Axolotl Discords, EleutherAI Discord, AI4Bharat community.

---

## 19. Prompts to use with an LLM tutor

- "My GPU is [model] with [X] GB VRAM. Which base model and QLoRA settings should I start with for [task]? Show the memory math."
- "Here is my training loss/val loss log: [paste]. Is this overfitting, underfitting, or a data issue?"
- "Review my SFT dataset sample for consistency problems and suggest 10 targeted examples to add."
- "Estimate cost and time to continue-pretrain [model] on [N] tokens using 6ND, and compare 1× H100 vs 8× H100."
- "Design an eval set for [task] with 5 failure categories and how to score each."
- "Write a `torchrun` + FSDP launch script for 8 GPUs with auto-resume from the latest checkpoint."

---

## 20. One-paragraph summary

As a solo builder, think of training as a **ladder of cost and control**. Start on your laptop with a tiny from-scratch model to understand the loop, then do the work that actually creates value: **build an eval set, curate data, and QLoRA-fine-tune the smallest open model that works**, using Unsloth/TRL/MLX on a laptop or a rented 24 to 80 GB GPU for a few dollars an hour. Use continued pretraining only when you must add domain or language knowledge, RL (GRPO) only when answers are verifiable, and clusters only when the `6ND` math and the 8-GPU single-node limit truly demand it. Learn from the open labs (AI2, Hugging Face, Karpathy, EleutherAI, Qwen, DeepSeek, Meta, Google, Mistral), respect licenses and data rights, and remember: **the moat for an indie hacker is data, evals, and distribution, not FLOPs.**
