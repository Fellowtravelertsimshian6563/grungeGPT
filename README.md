# grungeGPT

[![Apache 2.0](https://img.shields.io/github/license/arpanpathak/grungeGPT)](LICENSE)
[![Rust 2024](https://img.shields.io/badge/rust-2024-orange.svg)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-edition-field)
[![CI](https://github.com/arpanpathak/grungeGPT/actions/workflows/ci.yml/badge.svg)](https://github.com/arpanpathak/grungeGPT/actions/workflows/ci.yml)

A from-scratch, decoder-only GPT written in Rust and trained on song lyrics plus
plain English text. It is a complete educational implementation of a language
model: tokenizer, dataset, transformer, training loop, sampling, and GGUF export
for Ollama.

The goal is to make modern deep learning understandable. Every piece of the
pipeline is written in plain Rust using the Candle tensor library, so you can
read every line instead of treating the model as a black box.

---

## Table of contents

- [What you will learn](#what-you-will-learn)
- [How a GPT works](#how-a-gpt-works)
- [Project layout](#project-layout)
- [Quickstart](#quickstart)
- [Download data](#download-data)
- [Train the tokenizer](#train-the-tokenizer)
- [Train the model](#train-the-model)
- [Continue training from a checkpoint](#continue-training-from-a-checkpoint)
- [Generate lyrics](#generate-lyrics)
- [Export to Ollama](#export-to-ollama)
- [CUDA on Jetson and desktops](#cuda-on-jetson-and-desktops)
- [References](#references)
- [License](#license)

---

## What you will learn

- How text is converted to tokens with byte pair encoding.
- How a causal transformer predicts the next token.
- How cross-entropy loss drives training.
- How temperature and top-k sampling generate text.
- How model weights are saved, loaded, and exported to GGUF.
- How to run a custom model in local Ollama.

## How a GPT works

A GPT is a language model. It learns to predict the next token given the tokens
before it. The pipeline has five stages:

1. **Tokenization**: text becomes a sequence of token ids.
2. **Embedding**: each token id becomes a vector.
3. **Transformer**: the vectors pass through causal self-attention blocks.
4. **Prediction**: the final vector is projected to a probability distribution
   over the vocabulary.
5. **Training**: the probabilities are compared with the real next token using
   cross-entropy, and AdamW updates the weights.

For a gentle introduction, watch Andrej Karpathy's
[Let's build GPT: from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) and
3Blue1Brown's [neural network series](https://www.3blue1brown.com/topics/neural-networks).

## Project layout

```text
src/
  lib.rs        crate-level docs and module list
  tokenizer.rs  GPT-2 byte level BPE tokenizer
  dataset.rs    text loading and fixed-length training sequences
  model.rs      decoder-only transformer and checkpoints
  trainer.rs    AdamW training loop
  sampler.rs    temperature and top-k generation
  fetcher.rs    lyrics.ovh downloader
  gguf.rs       GGUF exporter for Ollama and llama.cpp
  main.rs       CLI
data/
  lyrics/       plain text lyric files (not committed)
  text/         plain text English corpus (not committed)
checkpoints/    trained models (not committed)
```

Every module has detailed doc comments with links to papers, videos, and
courses. Run `cargo doc --open` to read them as a website.

## Quickstart

```bash
cargo run --release -- go
```

This downloads lyrics if the data directory is empty, trains a small model, and
prints generated lyrics.

## Download data

The project reads every `.txt` file recursively from `--data`.

Add lyrics:

```bash
mkdir -p data/lyrics
cp my_songs/*.txt data/lyrics/
```

Add English text such as public domain books:

```bash
mkdir -p data/text
cp my_books/*.txt data/text/
```

Then use `--data data` so both directories are included:

```bash
cargo run --release -- train --data data --out-dir checkpoints/grungegpt
```

Large public domain corpora are available from
[Project Gutenberg](https://www.gutenberg.org). The
[Gutendex API](https://gutendex.com) is convenient for finding popular books.

## Train the tokenizer

```bash
cargo run --release -- tokenizer \
  --data data \
  --merges 1024 \
  --output tokenizer.json
```

The tokenizer is GPT-2 compatible byte level BPE. It maps every byte to a
unicode character, so any UTF-8 text can be tokenized without unknown tokens.
The saved JSON can be reused across training runs.

## Train the model

```bash
cargo run --release -- train \
  --data data \
  --out-dir checkpoints/grungegpt \
  --steps 5000 \
  --batch-size 8 \
  --block-size 128 \
  --n-layer 4 \
  --n-embd 192 \
  --n-head 8 \
  --learning-rate 0.001 \
  --eval-every 500
```

Output files:

- `model.safetensors` model weights
- `config.json` model hyperparameters
- `tokenizer.json` tokenizer vocabulary and merges

Use `--device cuda` when built with the `cuda` feature.

## Continue training from a checkpoint

To continue from an existing checkpoint, pass the old tokenizer and the old
weights. The model loads the saved weights and keeps training:

```bash
cargo run --release -- train \
  --data data \
  --tokenizer checkpoints/grungegpt/tokenizer.json \
  --resume checkpoints/grungegpt/model.safetensors \
  --out-dir checkpoints/grungegpt-v2 \
  --steps 5000 \
  --batch-size 8 \
  --block-size 128 \
  --n-layer 4 \
  --n-embd 192 \
  --n-head 8 \
  --learning-rate 0.0005
```

This is how you can append new songs or books to `data/lyrics` or `data/text`
and continue improving an existing model instead of starting from zero.

The optimizer state starts fresh, but the model weights are preserved.

## Generate lyrics

```bash
cargo run --release -- generate \
  --config checkpoints/grungegpt/config.json \
  --checkpoint checkpoints/grungegpt/model.safetensors \
  --tokenizer checkpoints/grungegpt/tokenizer.json \
  --prompt "Bones in the river" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-k 40
```

- `--temperature 0.0` makes output greedy and deterministic.
- Higher temperature makes output more random.
- Top-k limits sampling to the `k` most likely tokens.

Generation stops when the model emits `<eos>` or reaches `--max-tokens`.

## Export to Ollama

Export the model to GGUF:

```bash
cargo run --release -- export \
  --config checkpoints/grungegpt/config.json \
  --checkpoint checkpoints/grungegpt/model.safetensors \
  --tokenizer checkpoints/grungegpt/tokenizer.json \
  --output grungegpt.gguf \
  --context 8192
```

Create a Modelfile:

```text
FROM /absolute/path/to/grungegpt.gguf
PARAMETER num_predict 128
PARAMETER temperature 0.8
```

Create and run the Ollama model:

```bash
ollama create grungegpt -f Modelfile
ollama run grungegpt "Bones in the river"
```

## CUDA on Jetson and desktops

The default build runs on CPU. To use an NVIDIA GPU:

```bash
cargo build --release --features cuda
cargo run --release --features cuda -- train --device cuda
```

On Jetson aarch64, Candle CPU kernels need the FP16 target feature. This repo
ships a `.cargo/config.toml` that enables `+fp16` automatically.

## References

Papers:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
- [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)

Videos:

- [3Blue1Brown Neural Networks](https://www.3blue1brown.com/topics/neural-networks)
- [Karpathy: Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Karpathy: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)

Courses:

- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
- [Stanford CS224n](https://web.stanford.edu/class/cs224n/)
- [Fast.ai Practical Deep Learning](https://course.fast.ai)

Tools:

- [Candle](https://github.com/huggingface/candle)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Ollama](https://docs.ollama.com)
- [Project Gutenberg](https://www.gutenberg.org)

## License

Apache 2.0. See [LICENSE](LICENSE).
