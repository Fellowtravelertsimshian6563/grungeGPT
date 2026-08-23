//! # Sampler
//!
//! A trained model is not a text generator by itself. It is a probability
//! machine: give it a sequence of tokens and it returns a score for every token
//! that could come next. The sampler turns those scores into actual words, one
//! token at a time, and that is the moment a language model finally "speaks".
//!
//! The hard part is choosing how to speak. If we always take the most likely
//! token, the model quickly falls into a loop and repeats itself. If we sample
//! uniformly from the whole vocabulary, every word is equally likely and the
//! output becomes pure noise. Real language lives somewhere in between: we want
//! to stay close to the model's best guesses, but keep enough randomness to
//! avoid boring repetition.
//!
//! Two simple but powerful ideas solve this:
//!
//! - **Temperature** reshapes the probability distribution. Mathematically, we
//!   divide the raw logits by a temperature `T` before softmax:
//!
//!   ```text
//!   p(i) = exp(logit(i) / T) / sum_j exp(logit(j) / T)
//!   ```
//!
//!   With `T < 1` the distribution becomes sharper and the model is more
//!   decisive. With `T > 1` it becomes flatter and the model is more playful.
//!   At `T = 0` (or any value below zero) we give up on randomness and greedily
//!   pick the argmax.
//!
//! - **Top-k** cuts off the long tail of improbable tokens. We keep only the
//!   `k` highest scoring tokens and redistribute probability among them. This
//!   prevents the model from occasionally producing a bizarre token that it
//!   only barely considered.
//!
//! Generation runs in a loop: sample one token, append it to the context, feed
//! the context back into the model, and repeat. The loop ends when the model
//! emits the special `<eos>` token or when we hit the requested token limit.
//!
//! ## References
//!
//! - "The Curious Case of Neural Text Degeneration": <https://arxiv.org/abs/1904.09751>
//! - Karpathy, "Let's build GPT: from scratch": <https://www.youtube.com/watch?v=kCc8FmEb1nY>
//! - Hugging Face generation strategies: <https://huggingface.co/docs/transformers/generation_strategies>

use crate::model::Gpt;
use crate::tokenizer::Bpe;
use anyhow::Result;
use candle_core::{Device, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Sampling options for lyric generation.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Maximum tokens to generate before stopping.
    pub max_tokens: usize,
    /// Softmax temperature. Higher is more random, lower is more greedy.
    pub temperature: f64,
    /// Keep only the top-k most likely tokens when sampling.
    pub top_k: Option<usize>,
    /// Random seed for reproducible sampling.
    pub seed: u64,
}

/// Generate text from a trained model using the configured sampler.
///
/// This is the moment everything comes together. The tokenizer turns your
/// prompt into ids, the model scores every possible next token, the sampler
/// picks one with a little randomness, and the loop feeds the new token back
/// in. Each iteration extends the story by one word, and the loop keeps going
/// until the model says `<eos>` or the token budget runs out.
///
/// # Parameters
///
/// - `model`: trained GPT model.
/// - `tokenizer`: tokenizer used for encoding and decoding.
/// - `prompt`: seed text for generation.
/// - `config`: sampling options.
/// - `device`: device for the model tensors.
///
/// # Returns
///
/// Generated text including the prompt.
///
/// # Errors
///
/// Returns an error when tensors cannot be created or the forward pass fails.
///
/// # Behavior
///
/// The prompt is tokenized and repeatedly fed to the model. At each step the
/// next token is sampled with temperature and top-k. Generation stops when the
/// `<eos>` token is sampled or `max_tokens` is reached.
/// See <https://arxiv.org/abs/1904.09751> for the degeneracy problems that
/// temperature and top-k are designed to mitigate.
pub fn generate(
    model: &Gpt,
    tokenizer: &Bpe,
    prompt: &str,
    config: &GenerateConfig,
    device: &Device,
) -> Result<String> {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let block_size = model.block_size();
    let bos = tokenizer.bos_id();
    let eos = tokenizer.eos_id();

    let mut context = tokenizer.encode_ids(prompt);
    if context.is_empty() {
        context.push(bos);
    }
    let mut output = tokenizer.encode_ids(prompt);

    for _ in 0..config.max_tokens {
        let start = context.len().saturating_sub(block_size);
        let window = &context[start..];
        let mut input = vec![bos; block_size - window.len()];
        input.extend_from_slice(window);

        let input_tensor = Tensor::new(input.as_slice(), device)?.unsqueeze(0)?;
        let logits = model.forward(&input_tensor, false)?;
        let last_logits = logits
            .narrow(1, block_size - 1, 1)?
            .squeeze(1)?
            .squeeze(0)?;
        let next = sample_token(&last_logits, config.temperature, config.top_k, &mut rng)?;

        context.push(next);
        output.push(next);
        if next == eos {
            break;
        }
    }

    Ok(tokenizer.decode_ids(&output))
}

/// Sample the next token id from logits with temperature and top-k.
///
/// # Parameters
///
/// - `logits`: vector of raw logits for the last token.
/// - `temperature`: softmax temperature. `<= 0.0` selects greedy argmax.
/// - `top_k`: optional number of highest-scoring tokens to keep.
/// - `rng`: random number generator for sampling.
///
/// # Returns
///
/// The sampled token id.
///
/// # Errors
///
/// Returns an error when logits cannot be read from the tensor.
///
/// ## References
///
/// - Text degeneration paper: <https://arxiv.org/abs/1904.09751>
fn sample_token(
    logits: &Tensor,
    temperature: f64,
    top_k: Option<usize>,
    rng: &mut StdRng,
) -> Result<u32> {
    let logits = logits.to_vec1::<f32>()?;
    if temperature <= 0.0 {
        return Ok(argmax(&logits) as u32);
    }

    let mut scores: Vec<f32> = logits
        .iter()
        .map(|logit| logit / temperature as f32)
        .collect();

    if let Some(k) = top_k.filter(|k| *k > 0 && *k < scores.len()) {
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let cutoff = sorted[k - 1];
        scores.iter_mut().for_each(|score| {
            if *score < cutoff {
                *score = f32::NEG_INFINITY;
            }
        });
    }

    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = Vec::with_capacity(scores.len());
    let mut sum = 0.0f32;
    for score in &scores {
        let probability = (score - max).exp();
        sum += probability;
        probabilities.push(probability);
    }

    let sample: f32 = rng.gen_range(0.0..1.0);
    let mut cumulative = 0.0f32;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += probability / sum;
        if sample <= cumulative {
            return Ok(index as u32);
        }
    }

    Ok(argmax(&logits) as u32)
}

/// Return the index of the largest value.
///
/// # Parameters
///
/// - `values`: slice of float scores.
///
/// # Returns
///
/// Index of the maximum value, or `0` for an empty slice.
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0)
}
