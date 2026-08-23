use crate::dataset::Dataset;
use crate::model::{Gpt, GptConfig};
use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub batch_size: usize,
    pub steps: usize,
    pub learning_rate: f64,
    pub eval_every: usize,
    pub seed: u64,
}

pub fn train_model(
    dataset: &Dataset,
    config: &GptConfig,
    train_config: &TrainConfig,
    device: &Device,
) -> Result<(VarMap, Gpt)> {
    let varmap = VarMap::new();
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
