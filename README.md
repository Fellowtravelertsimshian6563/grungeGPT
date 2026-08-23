# grungeGPT

[![Apache 2.0](https://img.shields.io/github/license/arpanpathak/grungeGPT)](LICENSE)
[![Rust 2024](https://img.shields.io/badge/rust-2024-orange.svg)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-edition-field)
[![CI](https://github.com/arpanpathak/grungeGPT/actions/workflows/ci.yml/badge.svg)](https://github.com/arpanpathak/grungeGPT/actions/workflows/ci.yml)

A small decoder-only GPT written in Rust and trained on song lyrics. It ships with a pure Rust byte pair encoding tokenizer, a lyrics dataset loader, a Candle-based transformer, a training loop, and a sampling CLI.

The project is designed for experimentation on laptops, desktops, and Jetson devices. The default build runs on CPU. Enable the `cuda` feature to use an NVIDIA GPU.

## Features

- BPE tokenizer with word end markers, based on the reference in `coding_standards.md`
- Recursive directory loader for plain text lyrics
- Causal decoder-only transformer with configurable depth, width, and heads
- AdamW training with random mini-batches
- Safetensors checkpoints with JSON configuration
- Temperature and top-k sampling for generation
- CLI for tokenizer training, model training, and lyric generation

## Quickstart

```bash
cargo run --release -- tokenizer --data data/lyrics --output tokenizer.json
cargo run --release -- train --data data/lyrics --out-dir checkpoints/grungegpt
cargo run --release -- generate \
  --config checkpoints/grungegpt/config.json \
  --checkpoint checkpoints/grungegpt/model.safetensors \
  --tokenizer checkpoints/grungegpt/tokenizer.json \
  --prompt "Bones in the river" \
  --max-tokens 200
```

## Add your own lyrics

Put `.txt` files in `data/lyrics`. Each file is one training document. The repo includes `data/lyrics/sample_lyrics.txt`, an original demo corpus, so the commands above work before you add any data.

The loader reads `.txt` files recursively, normalizes line endings, and preserves newlines as tokens. See `data/lyrics/README.md` for the expected layout.

## Tokenizer

The tokenizer is a byte pair encoder trained on the words in the lyrics corpus. It keeps `</w>` word end markers, a `<s>` start token, and an `<unk>` fallback token. Train it directly:

```bash
cargo run --release -- tokenizer --data data/lyrics --merges 512 --output tokenizer.json
```

The output is a JSON file that can be reused across training runs.

## Train

```bash
cargo run --release -- train \
  --data data/lyrics \
  --out-dir checkpoints/grungegpt \
  --steps 1000 \
  --batch-size 8 \
  --block-size 64 \
  --n-layer 2 \
  --n-embd 64 \
  --n-head 4 \
  --learning-rate 0.001
```

The training command writes three files to the output directory:

- `model.safetensors` for the model weights
- `config.json` for the model configuration
- `tokenizer.json` for the tokenizer

## Generate

```bash
cargo run --release -- generate \
  --config checkpoints/grungegpt/config.json \
  --checkpoint checkpoints/grungegpt/model.safetensors \
  --tokenizer checkpoints/grungegpt/tokenizer.json \
  --prompt "Thunder in the canyon" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-k 40
```

Use `--temperature 0` for greedy decoding. Use `--device cuda` when the binary was built with CUDA support.

## CUDA support

The Jetson and desktop builds can use CUDA when the `cuda` feature is enabled:

```bash
cargo build --release --features cuda
cargo run --release --features cuda -- train --device cuda
```

CUDA requires a compatible NVIDIA driver and CUDA toolkit. The default build has no CUDA dependency.

## Jetson note

The Candle CPU kernels use half precision SIMD instructions on aarch64. On Jetson boards that support FP16, build with the matching target feature:

```bash
RUSTFLAGS="-C target-feature=+fp16" cargo build --release
```

Use `+fullfp16` on older Rust toolchains that do not recognize `+fp16`.

## Project layout

```text
src/
  tokenizer.rs   BPE tokenizer
  dataset.rs     lyrics loading and training sequences
  model.rs       GPT model and checkpoints
  trainer.rs     training loop
  sampler.rs     text generation
  main.rs        CLI
data/
  lyrics/        plain text lyric files
```

## License

Apache 2.0. See [LICENSE](LICENSE).
