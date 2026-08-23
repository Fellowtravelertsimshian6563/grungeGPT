//! # grungeGPT
//!
//! A small, from-scratch decoder-only GPT trained on song lyrics and plain
//! English text, written in idiomatic Rust with the Candle tensor library.
//!
//! This crate is designed as a learning project. Every stage of a modern
//! language model is implemented here in one place:
//!
//! 1. **Tokenization** with a GPT-2 compatible byte level BPE tokenizer.
//! 2. **Dataset construction** from plain text files.
//! 3. **A decoder-only transformer** with causal self-attention.
//! 4. **AdamW training** on CPU or CUDA.
//! 5. **Sampling** with temperature and top-k.
//! 6. **GGUF export** so the trained model can run in Ollama or llama.cpp.
//!
//! # Educational references
//!
//! - Transformer paper: <https://arxiv.org/abs/1706.03762>
//! - GPT-2 paper: <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>
//! - GPT-3 paper: <https://arxiv.org/abs/2005.14165>
//! - BPE paper: <https://arxiv.org/abs/1508.07909>
//! - Adam optimizer paper: <https://arxiv.org/abs/1412.6980>
//! - AdamW paper: <https://arxiv.org/abs/1711.05101>
//! - Layer Normalization paper: <https://arxiv.org/abs/1607.06450>
//! - Attention Is All You Need explainer: <https://arxiv.org/abs/1706.03762>
//! - Candle docs: <https://github.com/huggingface/candle>
//! - llama.cpp GGUF docs: <https://github.com/ggerganov/llama.cpp>
//! - Ollama docs: <https://docs.ollama.com>
//!
//! # Video and course references
//!
//! - 3Blue1Brown, "Neural networks" and "Transformers": <https://www.3blue1brown.com/topics/neural-networks>
//! - Andrej Karpathy, "Let's build GPT: from scratch": <https://www.youtube.com/watch?v=kCc8FmEb1nY>
//! - Andrej Karpathy, "Let's build the GPT Tokenizer": <https://www.youtube.com/watch?v=zduSFxRajkE>
//! - Hugging Face NLP course: <https://huggingface.co/learn/nlp-course>
//! - Stanford CS224n: <https://web.stanford.edu/class/cs224n/>
//! - Fast.ai Practical Deep Learning: <https://course.fast.ai>
//!
//! # Quick start
//!
//! ```bash
//! cargo run -- go
//! ```
//!
//! This downloads lyrics if needed, trains a small model, and generates a
//! sample. See the README for the full tutorial.

pub mod dataset;
pub mod fetcher;
pub mod gguf;
pub mod model;
pub mod sampler;
pub mod tokenizer;
pub mod trainer;
