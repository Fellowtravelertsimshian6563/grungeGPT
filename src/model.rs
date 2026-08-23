//! # Model
//!
//! A decoder-only transformer in the style of GPT-2.
//!
//! ## Architecture
//!
//! 1. Token embeddings map token ids to vectors.
//! 2. Position embeddings add the position of each token.
//! 3. A stack of transformer blocks processes the sequence.
//! 4. A final layer norm and linear head predict the next token.
//!
//! Each transformer block contains:
//!
//! - Causal multi-head self-attention, so a token can only attend to itself and
//!   previous tokens.
//! - A two-layer MLP with GELU activation.
//! - Pre-norm LayerNorm and residual connections, which make deep networks
//!   easier to optimize.
//!
//! ## References
//!
//! - Transformer paper: <https://arxiv.org/abs/1706.03762>
//! - GPT-2 paper: <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>
//! - Layer Normalization paper: <https://arxiv.org/abs/1607.06450>
//! - 3Blue1Brown, "But what is a GPT?": <https://www.3blue1brown.com/topics/neural-networks>
//! - Karpathy, "Let's build GPT: from scratch": <https://www.youtube.com/watch?v=kCc8FmEb1nY>
//! - Candle examples: <https://github.com/huggingface/candle/tree/main/candle-examples>

use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Hyperparameters for the decoder-only transformer.
///
/// - `vocab_size`: number of tokens in the vocabulary.
/// - `block_size`: maximum number of tokens the model can attend to.
/// - `n_layer`: number of transformer blocks.
/// - `n_embd`: embedding width.
/// - `n_head`: number of attention heads.
/// - `dropout`: regularization probability, applied during training.
///
/// See the GPT-2 paper <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>
/// for the original architecture choices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GptConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub dropout: f32,
}

impl GptConfig {
    pub fn new(
        vocab_size: usize,
        block_size: usize,
        n_layer: usize,
        n_embd: usize,
        n_head: usize,
        dropout: f32,
    ) -> Self {
        Self {
            vocab_size,
            block_size,
            n_layer,
            n_embd,
            n_head,
            dropout,
        }
    }
}

struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_embd: usize,
    dropout: f32,
}

impl CausalSelfAttention {
    fn new(vb: VarBuilder, config: &GptConfig) -> CandleResult<Self> {
        let c_attn = candle_nn::linear(config.n_embd, 3 * config.n_embd, vb.pp("c_attn"))?;
        let c_proj = candle_nn::linear(config.n_embd, config.n_embd, vb.pp("c_proj"))?;
        Ok(Self {
            c_attn,
            c_proj,
            n_head: config.n_head,
            n_embd: config.n_embd,
            dropout: config.dropout,
        })
    }

    fn forward(&self, x: &Tensor, train: bool) -> CandleResult<Tensor> {
        let (batch, seq, _) = x.dims3()?;
        let head_dim = self.n_embd / self.n_head;
        let qkv = self.c_attn.forward(x)?;

        let q = qkv.narrow(2, 0, self.n_embd)?;
        let k = qkv.narrow(2, self.n_embd, self.n_embd)?;
        let v = qkv.narrow(2, 2 * self.n_embd, self.n_embd)?;

        let q = q
            .reshape((batch, seq, self.n_head, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq, self.n_head, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq, self.n_head, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let attention = (q.matmul(&k.t()?)? * scale)?;
        let mask = causal_mask(seq, x.device())?;
        let attention = attention.broadcast_add(&mask)?;
        let attention = candle_nn::ops::softmax(&attention, D::Minus1)?;
        let attention = if train && self.dropout > 0.0 {
            candle_nn::ops::dropout(&attention, self.dropout)?
        } else {
            attention
        };

        let y = attention.matmul(&v)?;
        let y = y
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq, self.n_embd))?;
        let y = self.c_proj.forward(&y)?;
        if train && self.dropout > 0.0 {
            candle_nn::ops::dropout(&y, self.dropout)
        } else {
            Ok(y)
        }
    }
}

struct Mlp {
    c_fc: Linear,
    c_proj: Linear,
    dropout: f32,
}

impl Mlp {
    fn new(vb: VarBuilder, config: &GptConfig) -> CandleResult<Self> {
        let c_fc = candle_nn::linear(config.n_embd, 4 * config.n_embd, vb.pp("c_fc"))?;
        let c_proj = candle_nn::linear(4 * config.n_embd, config.n_embd, vb.pp("c_proj"))?;
        Ok(Self {
            c_fc,
            c_proj,
            dropout: config.dropout,
        })
    }

    fn forward(&self, x: &Tensor, train: bool) -> CandleResult<Tensor> {
        let x = self.c_fc.forward(x)?;
        let x = x.gelu_erf()?;
        let x = self.c_proj.forward(&x)?;
        if train && self.dropout > 0.0 {
            candle_nn::ops::dropout(&x, self.dropout)
        } else {
            Ok(x)
        }
    }
}

struct Block {
    ln1: LayerNorm,
    attn: CausalSelfAttention,
    ln2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    fn new(vb: VarBuilder, config: &GptConfig) -> CandleResult<Self> {
        let ln1 = candle_nn::layer_norm(config.n_embd, 1e-5, vb.pp("ln1"))?;
        let attn = CausalSelfAttention::new(vb.pp("attn"), config)?;
        let ln2 = candle_nn::layer_norm(config.n_embd, 1e-5, vb.pp("ln2"))?;
        let mlp = Mlp::new(vb.pp("mlp"), config)?;
        Ok(Self {
            ln1,
            attn,
            ln2,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor, train: bool) -> CandleResult<Tensor> {
        let residual = x.clone();
        let x = self.ln1.forward(x)?;
        let x = self.attn.forward(&x, train)?;
        let x = (residual + x)?;

        let residual = x.clone();
        let x = self.ln2.forward(&x)?;
        let x = self.mlp.forward(&x, train)?;
        residual + x
    }
}

/// A decoder-only transformer for next token prediction.
pub struct Gpt {
    config: GptConfig,
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
}

impl Gpt {
    pub fn new(vb: VarBuilder, config: &GptConfig) -> CandleResult<Self> {
        let wte = candle_nn::embedding(config.vocab_size, config.n_embd, vb.pp("wte"))?;
        let wpe = candle_nn::embedding(config.block_size, config.n_embd, vb.pp("wpe"))?;

        let mut blocks = Vec::with_capacity(config.n_layer);
        for index in 0..config.n_layer {
            blocks.push(Block::new(vb.pp(format!("h.{index}")), config)?);
        }

        let ln_f = candle_nn::layer_norm(config.n_embd, 1e-5, vb.pp("ln_f"))?;
        let lm_head = candle_nn::linear(config.n_embd, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            config: config.clone(),
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head,
        })
    }

    pub fn forward(&self, input: &Tensor, train: bool) -> CandleResult<Tensor> {
        let (batch, seq) = input.dims2()?;
        let positions = Tensor::arange(0u32, seq as u32, input.device())?
            .reshape((1, seq))?
            .broadcast_as((batch, seq))?;

        let token_emb = self.wte.forward(input)?;
        let position_emb = self.wpe.forward(&positions)?;
        let mut x = (token_emb + position_emb)?;

        if train && self.config.dropout > 0.0 {
            x = candle_nn::ops::dropout(&x, self.config.dropout)?;
        }

        for block in &self.blocks {
            x = block.forward(&x, train)?;
        }

        let x = self.ln_f.forward(&x)?;
        self.lm_head.forward(&x)
    }

    pub fn config(&self) -> &GptConfig {
        &self.config
    }

    pub fn block_size(&self) -> usize {
        self.config.block_size
    }
}

/// Save a model together with its configuration.
pub fn save_checkpoint(varmap: &VarMap, config: &GptConfig, out_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create {}", out_dir.display()))?;
    let model_path = out_dir.join("model.safetensors");
    varmap
        .save(&model_path)
        .with_context(|| format!("failed to save {}", model_path.display()))?;

    let config_path = out_dir.join("config.json");
    let file = std::fs::File::create(&config_path)
        .with_context(|| format!("failed to create {}", config_path.display()))?;
    serde_json::to_writer_pretty(file, config)
        .with_context(|| format!("failed to write {}", config_path.display()))?;
    Ok(())
}

/// Load a model from a safetensors checkpoint and a configuration file.
pub fn load_checkpoint(config_path: &Path, model_path: &Path, device: &Device) -> Result<Gpt> {
    let file = std::fs::File::open(config_path)
        .with_context(|| format!("failed to open {}", config_path.display()))?;
    let config: GptConfig = serde_json::from_reader(file)
        .with_context(|| format!("failed to read {}", config_path.display()))?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let _model = Gpt::new(vb, &config)?;
    varmap
        .load(model_path)
        .with_context(|| format!("failed to load {}", model_path.display()))?;
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    Ok(Gpt::new(vb, &config)?)
}

fn causal_mask(seq: usize, device: &Device) -> CandleResult<Tensor> {
    let row = Tensor::arange(0u32, seq as u32, device)?
        .reshape((seq, 1))?
        .broadcast_as((seq, seq))?;
    let col = Tensor::arange(0u32, seq as u32, device)?
        .reshape((1, seq))?
        .broadcast_as((seq, seq))?;
    let mask = row.lt(&col)?;
    let negative_inf = Tensor::full(f32::NEG_INFINITY, (seq, seq), device)?;
    let zero = Tensor::zeros((seq, seq), DType::F32, device)?;
    let mask = mask.where_cond(&negative_inf, &zero)?;
    mask.unsqueeze(0)?.unsqueeze(0)
}
