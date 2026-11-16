<p align="center">
  <img src="assets/logo-horizontal.svg" alt="LLM Training Cookbook" width="560">
</p>

<p align="center">
  <b>From a laptop to a GPU cluster: everything a solo builder needs to train, fine-tune, and ship their own LLM.</b>
</p>
A practical, honest guide to how large language models are trained, and how one person with a laptop and a small budget can build real models of their own.

> Last verified: November 2026. Prices, free tiers, and library APIs change fast. Items marked **(verify)** should be checked before you rely on them.

---

## What is this?

This repo is a set of deep-dive guides that explain, in plain language with real numbers, how modern LLMs are trained. It covers the hardware, parallelism, precision tricks, and post-training methods used by labs like Meta, DeepSeek, and others, then turns that knowledge into a practical path for indie builders.

It separates three kinds of claims throughout:

- **[PUBLIC]**: documented in a paper or technical report
- **[INFERRED]**: widely believed but not confirmed
- **[UNKNOWN]**: not disclosed by the labs

## Who is this for?

- **Indie hackers and solo builders** who want to train or fine-tune their own models without a big budget
- **Developers** who use LLM APIs and want to understand what happens underneath
- **Students and self-learners** building a mental model of distributed training
- **Builders working on low-resource languages or niche domains** where off-the-shelf models fall short

You should be comfortable with Python and basic machine learning ideas. You do **not** need a GPU cluster, a PhD, or a big budget.

## What's inside

| Guide | What you get |
|-------|--------------|
| [`docs/llm-training-deep-dive.md`](docs/llm-training-deep-dive.md) | How frontier labs train models: hardware, parallelism (DP/TP/PP/CP/EP), FP8 and mixed precision, optimizers, fault tolerance, post-training (SFT, RLHF, GRPO), and case studies of Llama 3 and DeepSeek-V3/R1. |
| [`docs/llm-training-cheatsheet.md`](docs/llm-training-cheatsheet.md) | Formulas, numbers to memorize, glossary, a debugging table, 25 flashcards, and a self-quiz. |
| [`docs/indie-deep-dive.md`](docs/indie-deep-dive.md) | The solo-builder path: what fits on your laptop, QLoRA fine-tuning, continued pretraining, GRPO, renting GPUs, cost estimates, when you need a cluster, and which open labs and models to build on. |
| [`docs/glossary.md`](docs/glossary.md) | Plain-English glossary of every term used in the guides, with full forms and common confusions cleared up. |

## Why this exists

Most LLM training content is either marketing hype or dense research papers. This guide aims to sit in between: **accurate numbers, honest uncertainty, and a practical path you can follow.** It also corrects common misconceptions, such as treating DeepSeek-V3's "$5.6M" as the total cost of building the model.

## What you can expect

**You will learn:**
- How to estimate training cost and time with `C = 6 × N × D`
- Why 16 bytes per parameter decides whether a model fits on your GPU
- How tensor, pipeline, data, and expert parallelism map onto real hardware
- How to fine-tune a model with QLoRA on a laptop or a rented GPU
- When you actually need a cluster (usually later than you think)
- Which open labs, models, and tools are worth building on

**You will not get:**
- Secret details of closed models. Training internals for OpenAI and Anthropic are largely not public, and this guide says so.
- A promise that you can match frontier models on a budget. The goal is useful, specialized models.
- Guaranteed up-to-date prices. Always check provider pages.

## Suggested learning path

1. **Skim** the cheat sheet to see the vocabulary and formulas.
2. **Read** the deep dive, sections 0 to 5, for the mental model.
3. **Follow** the indie guide: run nanoGPT on your laptop, then your first QLoRA fine-tune.
4. **Build** your own eval set before training anything serious.
5. **Revisit** the deep dive's case studies and the primary papers.

## Roadmap

- [ ] Runnable recipes: nanoGPT on a laptop, QLoRA SFT, GRPO, multi-GPU FSDP, GGUF export
- [ ] Memory calculator script (16 B/param, QLoRA fit)
- [ ] Cost calculator script (`6ND` to GPU-hours to dollars)
- [ ] Eval templates
- [ ] More languages and region-specific compute options

## Contributing

Corrections and improvements are very welcome, especially:
- Outdated prices, quotas, or library APIs
- Errors in numbers or claims (please link a primary source)
- New recipes, tools, or provider notes

Open an issue or a pull request. For factual corrections, include the source.

## Disclaimer

- This is an independent educational project. It is **not affiliated with or endorsed by** Anthropic, OpenAI, Meta, DeepSeek, Google, NVIDIA, or any other company mentioned.
- All trademarks belong to their owners.
- Parts of this guide were drafted with AI assistance and then edited for accuracy. Verify anything important against primary sources.
- Not legal, financial, or professional advice. Check data rights and model licenses before commercial use.

## License

- **Documentation** (`README.md` and everything in `docs/`): [CC BY 4.0](LICENSE-DOCS)
- **Code** (scripts and recipes, when added): [MIT](LICENSE)

If this helped you, star the repo and share it with another builder.
