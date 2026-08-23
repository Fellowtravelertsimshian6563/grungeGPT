use anyhow::Result;
use candle_core::Device;
use clap::{Args, Parser, Subcommand};
use grungegpt::dataset::{Dataset, load_lyrics_dir};
use grungegpt::model::{GptConfig, load_checkpoint, save_checkpoint};
use grungegpt::sampler::{GenerateConfig, generate};
use grungegpt::tokenizer::Bpe;
use grungegpt::trainer::{TrainConfig, train_model};
use std::path::PathBuf;

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
    /// Train a BPE tokenizer from a lyrics directory.
    Tokenizer(TokenizerArgs),
    /// Train a GPT model on tokenized lyrics.
    Train(TrainArgs),
    /// Generate lyrics from a trained checkpoint.
    Generate(GenerateArgs),
}

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

#[derive(Args)]
struct TrainArgs {
    /// Directory containing .txt lyric files.
    #[arg(long, default_value = "data/lyrics")]
    data: PathBuf,

    /// Path to an existing tokenizer file. When omitted, one is trained from --data.
    #[arg(long)]
    tokenizer: Option<PathBuf>,

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

    /// Device to use: cpu, cuda, or auto.
    #[arg(long, default_value = "auto")]
    device: String,
}

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
    #[arg(long, default_value = "auto")]
    device: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Tokenizer(args) => run_tokenizer(args),
        Command::Train(args) => run_train(args),
        Command::Generate(args) => run_generate(args),
    }
}

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

    let device = resolve_device(&args.device)?;
    let train_config = TrainConfig {
        batch_size: args.batch_size,
        steps: args.steps,
        learning_rate: args.learning_rate,
        eval_every: args.eval_every,
        seed: args.seed,
    };

    println!(
        "training on {} sequences, vocab {}, device {:?}",
        dataset.sequence_count(),
        tokenizer.vocab_size(),
        device
    );

    let (varmap, _model) = train_model(&dataset, &config, &train_config, &device)?;
    save_checkpoint(&varmap, &config, &args.out_dir)?;

    let tokenizer_path = args.out_dir.join("tokenizer.json");
    tokenizer.save(&tokenizer_path)?;

    println!("saved checkpoint and config to {}", args.out_dir.display());
    println!("saved tokenizer to {}", tokenizer_path.display());
    Ok(())
}

fn run_generate(args: GenerateArgs) -> Result<()> {
    let tokenizer = Bpe::load(&args.tokenizer)?;
    let device = resolve_device(&args.device)?;
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

fn resolve_device(name: &str) -> Result<Device> {
    match name {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::new_cuda(0)?),
        "auto" => Ok(Device::cuda_if_available(0)?),
        other => anyhow::bail!("unsupported device {other}, use cpu, cuda, or auto"),
    }
}
