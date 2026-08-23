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
    /// Number of tokens in the vocabulary.
    pub vocab_size: usize,
    /// Maximum number of tokens the model can attend to.
    pub block_size: usize,
    /// Number of transformer blocks.
    pub n_layer: usize,
    /// Embedding width.
    pub n_embd: usize,
    /// Number of attention heads.
    pub n_head: usize,
    /// Dropout probability applied during training.
    pub dropout: f32,
}

impl GptConfig {
    /// Create a new model configuration.
    ///
    /// # Parameters
    ///
    /// - `vocab_size`: vocabulary size from the tokenizer.
    /// - `block_size`: maximum context length in tokens.
    /// - `n_layer`: number of transformer blocks.
    /// - `n_embd`: embedding width.
    /// - `n_head`: number of attention heads. Must divide `n_embd`.
    /// - `dropout`: regularization probability between 0 and 1.
    ///
    /// # Returns
    ///
    /// A [`GptConfig`] ready to build a [`Gpt`].
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

/// Causal multi-head self-attention layer.
struct CausalSelfAttention {
    /// Projects input to query, key, and value vectors.
    c_attn: Linear,
    /// Projects attention output back to the embedding width.
    c_proj: Linear,
    /// Number of attention heads.
    n_head: usize,
    /// Embedding width.
    n_embd: usize,
    /// Dropout probability for attention weights and output.
    dropout: f32,
}

impl CausalSelfAttention {
    /// Build the attention layer with random initial weights.
    ///
    /// # Parameters
    ///
    /// - `vb`: variable builder that owns the model parameters.
    /// - `config`: model hyperparameters.
    ///
    /// # Errors
    ///
    /// Returns an error when the linear layers cannot be created.
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

    /// Run causal multi-head self-attention.
    ///
    /// # Parameters
    ///
    /// - `x`: input tensor shaped `(batch, seq, n_embd)`.
    /// - `train`: whether to apply dropout.
    ///
    /// # Returns
    ///
    /// Output tensor shaped `(batch, seq, n_embd)`.
    ///
    /// # Errors
    ///
    /// Returns an error on invalid shapes or device failures.
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

/// Two-layer feed-forward network with GELU activation.
struct Mlp {
    /// First linear layer expanding `n_embd` to `4 * n_embd`.
    c_fc: Linear,
    /// Second linear layer projecting back to `n_embd`.
    c_proj: Linear,
    /// Dropout probability applied to the output.
    dropout: f32,
}

impl Mlp {
    /// Build the MLP with random initial weights.
    ///
    /// # Parameters
    ///
    /// - `vb`: variable builder that owns the parameters.
    /// - `config`: model hyperparameters.
    ///
    /// # Errors
    ///
    /// Returns an error when the linear layers cannot be created.
    fn new(vb: VarBuilder, config: &GptConfig) -> CandleResult<Self> {
        let c_fc = candle_nn::linear(config.n_embd, 4 * config.n_embd, vb.pp("c_fc"))?;
        let c_proj = candle_nn::linear(4 * config.n_embd, config.n_embd, vb.pp("c_proj"))?;
        Ok(Self {
            c_fc,
            c_proj,
            dropout: config.dropout,
        })
    }

    /// Run the feed-forward network.
    ///
    /// # Parameters
    ///
    /// - `x`: input tensor shaped `(batch, seq, n_embd)`.
    /// - `train`: whether to apply dropout.
    ///
    /// # Returns
    ///
    /// Output tensor shaped `(batch, seq, n_embd)`.
    ///
    /// # Errors
    ///
    /// Returns an error on shape or device failures.
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

/// One transformer block: attention followed by MLP with pre-norm residuals.
struct Block {
    /// First layer norm before attention.
    ln1: LayerNorm,
    /// Causal multi-head self-attention.
    attn: CausalSelfAttention,
    /// Second layer norm before the MLP.
    ln2: LayerNorm,
    /// Feed-forward network.
    mlp: Mlp,
}

impl Block {
    /// Build a transformer block with random initial weights.
    ///
    /// # Parameters
    ///
    /// - `vb`: variable builder scoped to this block.
    /// - `config`: model hyperparameters.
    ///
    /// # Errors
    ///
    /// Returns an error when any submodule cannot be created.
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

    /// Run the block with residual connections.
    ///
    /// # Parameters
    ///
    /// - `x`: input tensor shaped `(batch, seq, n_embd)`.
    /// - `train`: whether to apply dropout.
    ///
    /// # Returns
    ///
    /// Output tensor shaped `(batch, seq, n_embd)`.
    ///
    /// # Errors
    ///
    /// Returns an error on shape or device failures.
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
///
/// # Fields
///
/// - `wte`: token embedding table.
/// - `wpe`: position embedding table.
/// - `blocks`: stack of transformer blocks.
/// - `ln_f`: final layer norm.
/// - `lm_head`: output projection to vocabulary logits.
pub struct Gpt {
    config: GptConfig,
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
}

impl Gpt {
    /// Build a GPT model with random initial weights.
    ///
    /// # Parameters
    ///
    /// - `vb`: variable builder that owns all model parameters.
    /// - `config`: model hyperparameters.
    ///
    /// # Returns
    ///
    /// A [`Gpt`] ready for training or inference.
    ///
    /// # Errors
    ///
    /// Returns an error when any submodule cannot be created.
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

    /// Run the full transformer forward pass.
    ///
    /// # Parameters
    ///
    /// - `input`: token id tensor shaped `(batch, seq)`.
    /// - `train`: whether to apply dropout.
    ///
    /// # Returns
    ///
    /// Logits tensor shaped `(batch, seq, vocab_size)`.
    ///
    /// # Errors
    ///
    /// Returns an error on invalid shapes or device failures.
    ///
    /// ## References
    ///
    /// - Transformer forward pass: <https://arxiv.org/abs/1706.03762>
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

    /// Return the model configuration.
    ///
    /// # Returns
    ///
    /// A reference to the [`GptConfig`] used to build this model.
    pub fn config(&self) -> &GptConfig {
        &self.config
    }

    /// Return the maximum context length in tokens.
    ///
    /// # Returns
    ///
    /// The `block_size` from the model configuration.
    pub fn block_size(&self) -> usize {
        self.config.block_size
    }
}

/// Save a model together with its configuration.
///
/// # Parameters
///
/// - `varmap`: variable map containing the trained weights.
/// - `config`: model hyperparameters.
/// - `out_dir`: destination directory. Created if missing.
///
/// # Errors
///
/// Returns an error when the directory cannot be created, weights cannot be
/// saved, or config cannot be written.
///
/// ## References
///
/// - Safetensors format: <https://github.com/huggingface/safetensors>
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
///
/// # Parameters
///
/// - `config_path`: path to `config.json`.
/// - `model_path`: path to `model.safetensors`.
/// - `device`: device to place the model on.
///
/// # Returns
///
/// A [`Gpt`] with the loaded weights.
///
/// # Errors
///
/// Returns an error when config or weights cannot be read.
///
/// ## References
///
/// - Safetensors format: <https://github.com/huggingface/safetensors>
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

/// Build a causal mask that prevents attending to future tokens.
///
/// # Parameters
///
/// - `seq`: sequence length.
/// - `device`: device for the mask tensor.
///
/// # Returns
///
/// A tensor shaped `(1, 1, seq, seq)` with `0.0` for allowed positions and
/// `-inf` for future positions.
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
