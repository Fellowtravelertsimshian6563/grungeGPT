//! # Trainer
//!
//! Runs the AdamW optimizer over the language modeling objective.
//!
//! ## What the model learns
//!
//! For every input token the model outputs a probability distribution over the
//! vocabulary. The loss is cross-entropy between those distributions and the
//! actual next tokens. Minimizing this loss makes the model assign high
//! probability to real text, which is how it learns language patterns.
//!
//! AdamW is a popular optimizer because it uses per-parameter adaptive learning
//! rates and decouples weight decay from the gradient update.
//!
//! ## References
//!
//! - Adam paper: <https://arxiv.org/abs/1412.6980>
//! - AdamW paper: <https://arxiv.org/abs/1711.05101>
//! - Cross-entropy: <https://en.wikipedia.org/wiki/Cross_entropy>
//! - Karpathy, "Let's build GPT: from scratch": <https://www.youtube.com/watch?v=kCc8FmEb1nY>
//! - Candle optimizer docs: <https://docs.rs/candle-nn>

use crate::dataset::Dataset;
use crate::model::{Gpt, GptConfig};
use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

/// Training loop configuration.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Sequences per optimizer step.
    pub batch_size: usize,
    /// Total optimizer steps in this run.
    pub steps: usize,
    /// AdamW learning rate. See <https://arxiv.org/abs/1412.6980>.
    pub learning_rate: f64,
    /// Print loss every N steps.
    pub eval_every: usize,
    /// Random seed for batch sampling and reproducibility.
    pub seed: u64,
}

/// Train a GPT model on a dataset and return the weights and model.
///
/// # Parameters
///
/// - `dataset`: training dataset with fixed-length sequences.
/// - `config`: model hyperparameters.
/// - `train_config`: optimizer and loop settings.
/// - `device`: device for model and tensors.
/// - `initial_varmap`: optional existing weights to continue training from.
///
/// # Returns
///
/// `(varmap, model)` after `train_config.steps` optimizer steps.
///
/// # Errors
///
/// Returns an error when the model cannot be built, a batch cannot be sampled,
/// or a forward/backward step fails.
///
/// # Behavior
///
/// For every step, a random batch is sampled, logits are computed, cross-entropy
/// loss is measured against the shifted targets, and AdamW updates the weights.
/// When `initial_varmap` is provided, the model weights are preserved but the
/// optimizer state starts fresh.
pub fn train_model(
    dataset: &Dataset,
    config: &GptConfig,
    train_config: &TrainConfig,
    device: &Device,
    initial_varmap: Option<&VarMap>,
) -> Result<(VarMap, Gpt)> {
    let varmap = match initial_varmap {
        Some(varmap) => varmap.clone(),
        None => VarMap::new(),
    };
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = Gpt::new(vb, config)?;

    let params = ParamsAdamW {
        lr: train_config.learning_rate,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;
    let mut rng = StdRng::seed_from_u64(train_config.seed);
    let started = Instant::now();

    for step in 1..=train_config.steps {
        let (input, target) = dataset.sample_batch(&mut rng, train_config.batch_size, device)?;
        let logits = model.forward(&input, true)?;
        let (batch, seq, vocab) = logits.dims3()?;
        let logits = logits.reshape((batch * seq, vocab))?;
        let target = target.flatten_all()?;
        let loss = candle_nn::loss::cross_entropy(&logits, &target)?;
        optimizer.backward_step(&loss)?;

        if step % train_config.eval_every == 0 {
            let loss_value = loss.to_scalar::<f32>()?;
            let elapsed = started.elapsed().as_secs_f64();
            println!(
                "step {step:>6}/{} loss {loss_value:8.4} elapsed {elapsed:7.2}s",
                train_config.steps
            );
        }
    }

    Ok((varmap, model))
}
