//! # grungeGPT
//!
//! Most people interact with large language models through an API and never
//! see what is inside. This crate exists to open the box. It is a small,
//! from-scratch decoder-only GPT written in idiomatic Rust with the Candle
//! tensor library, trained on song lyrics and plain English text. Every stage
//! of a modern language model is implemented here, in the open, so you can read
//! it and change it.
//!
//! The journey from raw text to generated lyrics passes through six stations:
//!
//! 1. **Tokenization**: a GPT-2 compatible byte level BPE tokenizer turns text
//!    into integer ids.
//! 2. **Dataset construction**: plain text files become fixed-length training
//!    sequences with `<s>` and `<eos>` markers.
//! 3. **The model**: a decoder-only transformer with causal self-attention
//!    predicts the next token.
//! 4. **Training**: AdamW minimizes cross-entropy on CPU or CUDA.
//! 5. **Sampling**: temperature and top-k turn probabilities into text.
//! 6. **GGUF export**: the trained weights are written into the format used by
//!    llama.cpp and Ollama, so the model can run locally.
//!
//! Start with `cargo run -- go` to see the whole pipeline in action, then read
//! the module docs in order. The references below are the best places to go
//! deeper.
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
