//! # grungeGPT CLI
//!
//! A library is a collection of parts, but a project only feels alive when the
//! parts can be driven from a terminal. This binary is the control room for the
//! whole pipeline: download lyrics, train a tokenizer, train the model,
//! generate text, and export GGUF files for Ollama.
//!
//! The commands are arranged in the order you would use them:
//!
//! - `go`: the one-shot journey. If lyrics are missing it downloads them, then
//!   trains a model and prints a sample.
//! - `fetch-lyrics`: fetch lyrics from lyrics.ovh for the configured artists.
//! - `tokenizer`: train and save a BPE tokenizer from your text corpus.
//! - `train`: train the GPT model, optionally continuing from a checkpoint with
//!   `--resume`.
//! - `generate` / `gen`: load a checkpoint and sample lyrics.
//! - `export` / `export-gguf`: write a GGUF file that Ollama can serve.
//!
//! Run `grungegpt --help` for all options.

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use clap::{Args, Parser, Subcommand};
use grungegpt::dataset::{Dataset, load_lyrics_dir, lyrics_file_count};
use grungegpt::fetcher::fetch_lyrics;
use grungegpt::gguf::export_gguf;
use grungegpt::model::{Gpt, GptConfig, load_checkpoint, save_checkpoint};
use grungegpt::sampler::{GenerateConfig, generate};
use grungegpt::tokenizer::Bpe;
use grungegpt::trainer::{TrainConfig, train_model};
use std::path::PathBuf;

/// Default hyperparameters for the `go` quick-start pipeline.
///
/// These are intentionally small so the demo finishes quickly.
/// For serious training, use the `train` command with explicit flags.
const GO_MERGES: usize = 512;
const GO_BATCH_SIZE: usize = 8;
const GO_BLOCK_SIZE: usize = 64;
const GO_N_LAYER: usize = 2;
const GO_N_EMBD: usize = 64;
const GO_N_HEAD: usize = 4;
const GO_DROPOUT: f32 = 0.1;
const GO_LEARNING_RATE: f64 = 0.001;
const GO_EVAL_EVERY: usize = 100;
const GO_SEED: u64 = 42;
const GO_TEMPERATURE: f64 = 0.8;
const GO_TOP_K: usize = 40;

/// Device selection for training and generation.
///
/// # Variants
///
/// - `Cpu`: always use the CPU.
/// - `Cuda`: always use CUDA device 0.
/// - `Auto`: use CUDA when available, otherwise CPU.
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
enum DeviceChoice {
    Cpu,
    Cuda,
    #[default]
    Auto,
}

impl DeviceChoice {
    /// Resolve the choice into a Candle [`Device`].
    ///
    /// # Returns
    ///
    /// The selected device.
    ///
    /// # Errors
    ///
    /// Returns an error when CUDA is requested but unavailable.
    fn resolve(self) -> Result<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Cuda => Ok(Device::new_cuda(0)?),
            Self::Auto => Ok(Device::cuda_if_available(0)?),
        }
    }
}

#[derive(Parser)]
#[command(
    name = "grungegpt",
    version,
    about = "Train and sample a small GPT on song lyrics"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the full pipeline: fetch lyrics, train, and generate.
    Go(GoArgs),
    /// Train a BPE tokenizer from a lyrics directory.
    Tokenizer(TokenizerArgs),
    /// Train a GPT model on tokenized lyrics.
    Train(TrainArgs),
    /// Generate lyrics from a trained checkpoint.
    #[command(alias = "gen")]
    Generate(GenerateArgs),
    /// Download lyrics for the configured artists.
    #[command(alias = "fetch-lyrics")]
    Fetch(FetchLyricsArgs),
    /// Export a trained checkpoint to GGUF for Ollama.
    #[command(alias = "export-gguf")]
    Export(ExportGgufArgs),
}

/// Arguments for the `tokenizer` command.
#[derive(Args)]
struct TokenizerArgs {
    /// Directory containing .txt lyric files.
    #[arg(long, default_value = "data/lyrics")]
    data: PathBuf,

    /// Number of BPE merges to learn.
    #[arg(long, default_value_t = 512)]
    merges: usize,

    /// Output tokenizer file.
    #[arg(long, default_value = "tokenizer.json")]
    output: PathBuf,
}

/// Arguments for the `go` command.
#[derive(Args)]
struct GoArgs {
    /// Directory containing .txt lyric files.
    #[arg(long, default_value = "data/lyrics")]
    data: PathBuf,

    /// Directory for checkpoints.
    #[arg(long, default_value = "checkpoints/grungegpt")]
    out_dir: PathBuf,

    /// Number of optimizer steps.
    #[arg(long, default_value_t = 500)]
    steps: usize,

    /// Prompt used for the final generation test.
    #[arg(long, default_value = "Bones in the river")]
    prompt: String,

    /// Maximum tokens to generate in the final test.
    #[arg(long, default_value_t = 100)]
    max_tokens: usize,

    /// Maximum songs to fetch per artist when lyrics are missing.
    #[arg(long, default_value_t = 120)]
    max_songs: usize,

    /// Device to use: cpu, cuda, or auto.
    #[arg(long, value_enum, default_value_t = DeviceChoice::Auto)]
    device: DeviceChoice,
}

/// Arguments for the `fetch-lyrics` command.
#[derive(Args)]
struct FetchLyricsArgs {
    /// Directory where artist lyric folders are written.
    #[arg(long, default_value = "data/lyrics")]
    out_dir: PathBuf,

    /// Maximum songs to keep per artist.
    #[arg(long, default_value_t = 120)]
    max_songs: usize,
}

/// Arguments for the `export-gguf` command.
#[derive(Args)]
struct ExportGgufArgs {
    /// Model configuration file.
    #[arg(long, default_value = "checkpoints/grungegpt/config.json")]
    config: PathBuf,

    /// Model weights in safetensors format.
    #[arg(long, default_value = "checkpoints/grungegpt/model.safetensors")]
    checkpoint: PathBuf,

    /// Tokenizer file.
    #[arg(long, default_value = "checkpoints/grungegpt/tokenizer.json")]
    tokenizer: PathBuf,

    /// Output GGUF file.
    #[arg(long, default_value = "grungegpt.gguf")]
    output: PathBuf,

    /// Context length advertised in the GGUF. Position embeddings are padded to this size.
    #[arg(long, default_value_t = 8192)]
    context: usize,
}

/// Arguments for the `train` command.
#[derive(Args)]
struct TrainArgs {
    /// Directory containing .txt lyric files.
    #[arg(long, default_value = "data/lyrics")]
    data: PathBuf,

    /// Path to an existing tokenizer file. When omitted, one is trained from --data.
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Path to an existing model.safetensors to continue training from.
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Number of BPE merges to learn when no tokenizer is provided.
    #[arg(long, default_value_t = 512)]
    merges: usize,

    /// Directory for checkpoints, config, and the copied tokenizer.
    #[arg(long, default_value = "checkpoints/grungegpt")]
    out_dir: PathBuf,

    /// Number of optimizer steps.
    #[arg(long, default_value_t = 500)]
    steps: usize,

    /// Sequences per optimizer step.
    #[arg(long, default_value_t = 8)]
    batch_size: usize,

    /// Tokens per training sequence.
    #[arg(long, default_value_t = 64)]
    block_size: usize,

    /// Number of transformer blocks.
    #[arg(long, default_value_t = 2)]
    n_layer: usize,

    /// Embedding width.
    #[arg(long, default_value_t = 64)]
    n_embd: usize,

    /// Number of attention heads.
    #[arg(long, default_value_t = 4)]
    n_head: usize,

    /// Dropout probability for attention and MLP layers.
    #[arg(long, default_value_t = 0.1)]
    dropout: f32,

    /// AdamW learning rate.
    #[arg(long, default_value_t = 0.001)]
    learning_rate: f64,

    /// Print loss every N steps.
    #[arg(long, default_value_t = 100)]
    eval_every: usize,

    /// Random seed for batching and initialization.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Write loss history as JSON for plotting.
    #[arg(long)]
    loss_log: Option<PathBuf>,

    /// Linear warmup steps before cosine decay begins.
    #[arg(long, default_value_t = 500)]
    warmup_steps: usize,

    /// Save a checkpoint every N steps (0 to disable).
    #[arg(long, default_value_t = 5000)]
    save_every: usize,

    /// Device to use: cpu, cuda, or auto.
    #[arg(long, value_enum, default_value_t = DeviceChoice::Auto)]
    device: DeviceChoice,
}

/// Arguments for the `generate` command.
#[derive(Args)]
struct GenerateArgs {
    /// Model configuration file.
    #[arg(long, default_value = "checkpoints/grungegpt/config.json")]
    config: PathBuf,

    /// Model weights in safetensors format.
    #[arg(long, default_value = "checkpoints/grungegpt/model.safetensors")]
    checkpoint: PathBuf,

    /// Tokenizer file.
    #[arg(long, default_value = "tokenizer.json")]
    tokenizer: PathBuf,

    /// Optional prompt text.
    #[arg(long, default_value = "")]
    prompt: String,

    /// Number of tokens to generate.
    #[arg(long, default_value_t = 200)]
    max_tokens: usize,

    /// Sampling temperature. Values below or equal to zero use greedy decoding.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Keep only the top K most likely tokens at each step.
    #[arg(long)]
    top_k: Option<usize>,

    /// Random seed for sampling.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Device to use: cpu, cuda, or auto.
    #[arg(long, value_enum, default_value_t = DeviceChoice::Auto)]
    device: DeviceChoice,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Go(args) => run_go(args),
        Command::Tokenizer(args) => run_tokenizer(args),
        Command::Train(args) => run_train(args),
        Command::Generate(args) => run_generate(args),
        Command::Fetch(args) => run_fetch_lyrics(args),
        Command::Export(args) => run_export_gguf(args),
    }
}

/// Run the `go` pipeline: fetch lyrics if needed, train, and generate.
fn run_go(args: GoArgs) -> Result<()> {
    if lyrics_file_count(&args.data)? <= 1 {
        println!("no lyrics found, downloading now");
        fetch_lyrics(&args.data, args.max_songs)?;
    }

    let train_args = TrainArgs {
        data: args.data.clone(),
        tokenizer: None,
        resume: None,
        merges: GO_MERGES,
        out_dir: args.out_dir.clone(),
        steps: args.steps,
        batch_size: GO_BATCH_SIZE,
        block_size: GO_BLOCK_SIZE,
        n_layer: GO_N_LAYER,
        n_embd: GO_N_EMBD,
        n_head: GO_N_HEAD,
        dropout: GO_DROPOUT,
        learning_rate: GO_LEARNING_RATE,
        eval_every: GO_EVAL_EVERY,
        seed: GO_SEED,
        loss_log: None,
        warmup_steps: 100,
        save_every: 0,
        device: args.device,
    };
    run_train(train_args)?;

    let generate_args = GenerateArgs {
        config: args.out_dir.join("config.json"),
        checkpoint: args.out_dir.join("model.safetensors"),
        tokenizer: args.out_dir.join("tokenizer.json"),
        prompt: args.prompt,
        max_tokens: args.max_tokens,
        temperature: GO_TEMPERATURE,
        top_k: Some(GO_TOP_K),
        seed: GO_SEED,
        device: args.device,
    };
    run_generate(generate_args)?;
    println!("done");
    Ok(())
}

/// Export a checkpoint to GGUF for Ollama.
fn run_export_gguf(args: ExportGgufArgs) -> Result<()> {
    let tokenizer = Bpe::load(&args.tokenizer)?;
    export_gguf(
        &args.config,
        &args.checkpoint,
        &tokenizer,
        &args.output,
        args.context,
    )?;
    println!("exported GGUF model to {}", args.output.display());
    Ok(())
}

/// Download lyrics for all configured artists.
fn run_fetch_lyrics(args: FetchLyricsArgs) -> Result<()> {
    let written = fetch_lyrics(&args.out_dir, args.max_songs)?;
    println!("downloaded lyrics for {} artists", written.len());
    Ok(())
}

/// Train and save a BPE tokenizer.
fn run_tokenizer(args: TokenizerArgs) -> Result<()> {
    let texts = load_lyrics_dir(&args.data)?;
    let tokenizer = Bpe::train(&texts, args.merges);
    tokenizer.save(&args.output)?;
    println!(
        "trained tokenizer with {} tokens and saved it to {}",
        tokenizer.vocab_size(),
        args.output.display()
    );
    Ok(())
}

/// Train a GPT model, optionally resuming from an existing checkpoint.
fn run_train(args: TrainArgs) -> Result<()> {
    let texts = load_lyrics_dir(&args.data)?;
    let tokenizer = match &args.tokenizer {
        Some(path) => Bpe::load(path)?,
        None => Bpe::train(&texts, args.merges),
    };

    let dataset = Dataset::from_texts(&texts, &tokenizer, args.block_size)?;
    let config = GptConfig::new(
        tokenizer.vocab_size(),
        args.block_size,
        args.n_layer,
        args.n_embd,
        args.n_head,
        args.dropout,
    );

    let device = args.device.resolve()?;
    let train_config = TrainConfig {
        batch_size: args.batch_size,
        steps: args.steps,
        learning_rate: args.learning_rate,
        eval_every: args.eval_every,
        seed: args.seed,
        loss_log: args.loss_log.clone(),
        warmup_steps: args.warmup_steps,
        save_every: args.save_every,
        out_dir: Some(args.out_dir.clone()),
    };

    println!(
        "training on {} sequences, vocab {}, device {:?}",
        dataset.sequence_count(),
        tokenizer.vocab_size(),
        device
    );

    let initial_varmap = match &args.resume {
        Some(path) => {
            let mut varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let _model = Gpt::new(vb, &config)?;
            varmap
                .load(path)
                .with_context(|| format!("failed to load checkpoint {}", path.display()))?;
            Some(varmap)
        }
        None => None,
    };

    let (varmap, _model) = train_model(
        &dataset,
        &config,
        &train_config,
        &device,
        initial_varmap.as_ref(),
    )?;
    save_checkpoint(&varmap, &config, &args.out_dir)?;

    let tokenizer_path = args.out_dir.join("tokenizer.json");
    tokenizer.save(&tokenizer_path)?;

    println!("saved checkpoint and config to {}", args.out_dir.display());
    println!("saved tokenizer to {}", tokenizer_path.display());
    Ok(())
}

/// Generate text from a trained checkpoint.
fn run_generate(args: GenerateArgs) -> Result<()> {
    let tokenizer = Bpe::load(&args.tokenizer)?;
    let device = args.device.resolve()?;
    let model = load_checkpoint(&args.config, &args.checkpoint, &device)?;
    let config = GenerateConfig {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        seed: args.seed,
    };

    let text = generate(&model, &tokenizer, &args.prompt, &config, &device)?;
    println!("{text}");
    Ok(())
}
