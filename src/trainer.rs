//! # Trainer
//!
//! A random transformer is just a fancy random-number generator. The trainer is
//! what turns it into a language model by repeatedly answering one question:
//! how wrong was the model's prediction, and how should every weight change to
//! be less wrong?
//!
//! For every input token the model outputs a probability distribution over the
//! vocabulary. The loss is cross-entropy between that distribution and the one
//! hot truth of the actual next token. Cross-entropy is low when the model is
//! confident and correct, high when it is confident and wrong, and moderate
//! when it is unsure. Minimizing it across millions of real sentences pushes
//! the model to assign high probability to the kind of language that actually
//! appears in the data.
//!
//! The gradients from that loss are handed to AdamW. Adam keeps a running
//! estimate of the first and second moments of each gradient, which gives every
//! parameter an adaptive learning rate, and its weight decay is decoupled from
//! the gradient update. That small design choice, described in the AdamW paper,
//! makes training more stable and generalizes better than plain Adam.
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
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;

/// A single training loss sample for visualization.
#[derive(Debug, Serialize)]
pub struct LossEntry {
    /// Optimizer step number.
    pub step: usize,
    /// Cross-entropy loss value.
    pub loss: f32,
    /// Wall-clock seconds since training started.
    pub elapsed: f64,
}

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
    /// Optional path to write loss history as JSON for plotting.
    pub loss_log: Option<PathBuf>,
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
///
/// ## References
///
/// - AdamW paper: <https://arxiv.org/abs/1711.05101>
/// - Karpathy training lecture: <https://www.youtube.com/watch?v=kCc8FmEb1nY>
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
    let mut loss_history: Vec<LossEntry> = Vec::new();
    let tokens_per_step = train_config.batch_size * dataset.block_size();

    for step in 1..=train_config.steps {
        let (input, target) = dataset.sample_batch(&mut rng, train_config.batch_size, device)?;
        let logits = model.forward(&input, true)?;
        let (batch, seq, vocab) = logits.dims3()?;
        let logits = logits.reshape((batch * seq, vocab))?;
        let target = target.flatten_all()?;
        let loss = candle_nn::loss::cross_entropy(&logits, &target)?;
        optimizer.backward_step(&loss)?;

        if step % train_config.eval_every == 0 || step == train_config.steps {
            let loss_value = loss.to_scalar::<f32>()?;
            let elapsed = started.elapsed().as_secs_f64();
            let tokens_per_sec = (step * tokens_per_step) as f64 / elapsed;
            println!(
                "step {step:>6}/{} loss {loss_value:8.4} elapsed {elapsed:7.2}s ({tokens_per_sec:.0} tok/s)",
                train_config.steps
            );
            loss_history.push(LossEntry {
                step,
                loss: loss_value,
                elapsed,
            });
        }
    }

    if let Some(ref path) = train_config.loss_log {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create loss log dir {}", parent.display()))?;
        }
        let file = std::fs::File::create(path)
            .with_context(|| format!("failed to create loss log {}", path.display()))?;
        serde_json::to_writer_pretty(file, &loss_history)
            .with_context(|| format!("failed to write loss log {}", path.display()))?;
        println!("saved {} loss entries to {}", loss_history.len(), path.display());
    }

    Ok((varmap, model))
}
