use crate::model::Gpt;
use crate::tokenizer::Bpe;
use anyhow::Result;
use candle_core::{Device, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub seed: u64,
}

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
    }

    Ok(tokenizer.decode_ids(&output))
}

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

    if let Some(k) = top_k {
        if k > 0 && k < scores.len() {
            let mut sorted = scores.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let cutoff = sorted[k - 1];
            for score in &mut scores {
                if *score < cutoff {
                    *score = f32::NEG_INFINITY;
                }
            }
        }
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

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0)
}
