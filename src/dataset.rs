//! # Dataset
//!
//! Converts raw text files into fixed-length token sequences for language
//! modeling.
//!
//! ## How language model data works
//!
//! A decoder-only GPT is trained to predict the next token. For every sequence
//! of `block_size + 1` tokens, the first `block_size` tokens are the input and
//! the remaining tokens are the targets. The model sees only past tokens thanks
//! to causal masking.
//!
//! Each document is prefixed with `<s>` and suffixed with `<eos>`, so the model
//! learns where a song or chapter starts and where it ends. Short chunks are
//! padded with `<eos>`.
//!
//! ## References
//!
//! - Transformer paper: <https://arxiv.org/abs/1706.03762>
//! - GPT-2 paper: <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>
//! - Karpathy, "Let's build GPT: from scratch": <https://www.youtube.com/watch?v=kCc8FmEb1nY>
//! - Hugging Face course, "Datasets": <https://huggingface.co/learn/nlp-course>

use crate::tokenizer::Bpe;
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use rand::Rng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::path::Path;

/// Fixed-length training sequences built from tokenized lyric files.
///
/// # Fields
///
/// - `sequences`: each entry is `block_size + 1` token ids. The first
///   `block_size` tokens are the model input and the shifted-by-one window is
///   the training target.
/// - `block_size`: number of input tokens per training step.
#[derive(Debug, Clone)]
pub struct Dataset {
    sequences: Vec<Vec<u32>>,
    block_size: usize,
}

impl Dataset {
    /// Build fixed-length sequences from texts, prefixing each with BOS and appending EOS.
    ///
    /// # Parameters
    ///
    /// - `texts`: raw documents to tokenize.
    /// - `tokenizer`: tokenizer used to convert text to ids.
    /// - `block_size`: number of input tokens per sequence.
    ///
    /// # Returns
    ///
    /// A [`Dataset`] containing at least one sequence.
    ///
    /// # Errors
    ///
    /// Returns an error when no training sequences can be built.
    ///
    /// # Behavior
    ///
    /// Each document is tokenized and an `<eos>` id is appended. The ids are
    /// split into `block_size` chunks, each prefixed with `<s>` and padded with
    /// `<eos>` when the final chunk is short.
    pub fn from_texts(texts: &[String], tokenizer: &Bpe, block_size: usize) -> Result<Self> {
        let seq_len = block_size + 1;
        let bos = tokenizer.bos_id();
        let eos = tokenizer.eos_id();
        let mut sequences = Vec::new();

        for text in texts {
            let mut ids = tokenizer.encode_ids(text);
            ids.push(eos);
            for chunk in ids.chunks(block_size) {
                let mut sequence = Vec::with_capacity(seq_len);
                sequence.push(bos);
                sequence.extend_from_slice(chunk);
                sequence.resize(seq_len, eos);
                sequences.push(sequence);
            }
        }

        if sequences.is_empty() {
            anyhow::bail!("no training sequences could be built from the provided lyrics");
        }

        Ok(Self {
            sequences,
            block_size,
        })
    }

    /// Number of tokens in each training sequence.
    ///
    /// # Returns
    ///
    /// The `block_size` used when the dataset was built.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of fixed-length sequences in the dataset.
    ///
    /// # Returns
    ///
    /// Total sequence count available for sampling.
    pub fn sequence_count(&self) -> usize {
        self.sequences.len()
    }

    /// Sample a random batch of input and target tensors on the given device.
    ///
    /// # Parameters
    ///
    /// - `rng`: random number generator used to pick sequences.
    /// - `batch_size`: number of sequences to sample.
    /// - `device`: target device for the tensors.
    ///
    /// # Returns
    ///
    /// `(input, target)` tensors, each shaped `(batch_size, block_size)`.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensors cannot be created on the device.
    ///
    /// # Behavior
    ///
    /// For every sampled sequence, the first `block_size` ids become the input
    /// and the ids shifted by one become the target. This shift is what makes
    /// the model learn next-token prediction.
    pub fn sample_batch(
        &self,
        rng: &mut StdRng,
        batch_size: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = self.block_size + 1;
        let mut inputs = Vec::with_capacity(batch_size * self.block_size);
        let mut targets = Vec::with_capacity(batch_size * self.block_size);

        for _ in 0..batch_size {
            let sequence = &self.sequences[rng.gen_range(0..self.sequences.len())];
            debug_assert_eq!(sequence.len(), seq_len);
            inputs.extend_from_slice(&sequence[..self.block_size]);
            targets.extend_from_slice(&sequence[1..]);
        }

        let input =
            Tensor::new(inputs.as_slice(), device)?.reshape((batch_size, self.block_size))?;
        let target =
            Tensor::new(targets.as_slice(), device)?.reshape((batch_size, self.block_size))?;
        Ok((input, target))
    }
}

/// Read every `.txt` file under a directory and normalize the text.
///
/// # Parameters
///
/// - `path`: root directory to scan recursively.
///
/// # Returns
///
/// A vector of normalized document strings, one per `.txt` file.
///
/// # Errors
///
/// Returns an error when the directory is unreadable or contains no `.txt`
/// files.
pub fn load_lyrics_dir(path: &Path) -> Result<Vec<String>> {
    let mut files = Vec::new();
    collect_txt_files(path, &mut files)?;

    if files.is_empty() {
        anyhow::bail!("no .txt files found under {}", path.display());
    }

    let texts = files
        .par_iter()
        .map(|file| {
            let raw = std::fs::read_to_string(file)
                .with_context(|| format!("failed to read {}", file.display()))?;
            Ok::<_, anyhow::Error>(normalize_text(&raw))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(texts)
}

/// Count `.txt` files under a directory, recursively.
///
/// # Parameters
///
/// - `path`: root directory to scan.
///
/// # Returns
///
/// The number of `.txt` files found.
///
/// # Errors
///
/// Returns an error when the directory cannot be read.
pub fn lyrics_file_count(path: &Path) -> Result<usize> {
    let mut files = Vec::new();
    collect_txt_files(path, &mut files)?;
    Ok(files.len())
}

/// Recursively collect `.txt` file paths under a directory.
///
/// # Parameters
///
/// - `path`: directory to scan.
/// - `files`: mutable vector that receives matching file paths.
///
/// # Errors
///
/// Returns an error when a directory entry cannot be read.
fn collect_txt_files(path: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<()> {
    let entries = std::fs::read_dir(path)
        .with_context(|| format!("failed to read directory {}", path.display()))?;

    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        match (file_type.is_dir(), file_type.is_file()) {
            (true, _) => collect_txt_files(&path, files)?,
            (false, true)
                if path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("txt")) =>
            {
                files.push(path);
            }
            _ => {}
        }
    }
    Ok(())
}

/// Normalize line endings and trim trailing whitespace.
///
/// # Parameters
///
/// - `raw`: raw file content.
///
/// # Returns
///
/// Normalized text with `\r\n` and `\r` converted to `\n`, trailing spaces
/// removed per line, and surrounding blank lines trimmed.
fn normalize_text(raw: &str) -> String {
    raw.replace("\r\n", "\n")
        .replace('\r', "\n")
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Bpe;
    use rand::SeedableRng;

    #[test]
    fn builds_sequences_with_bos_prefix() {
        let texts = vec!["one".to_string()];
        let tokenizer = Bpe::train(&texts, 8);
        let dataset = Dataset::from_texts(&texts, &tokenizer, 4).unwrap();
        assert_eq!(dataset.sequence_count(), 1);
        let sequence = &dataset.sequences[0];
        assert_eq!(sequence.len(), 5);
        assert_eq!(sequence[0], tokenizer.bos_id());
    }

    #[test]
    fn sample_batch_has_correct_shapes() {
        let texts = vec!["one two three four five six".to_string()];
        let tokenizer = Bpe::train(&texts, 8);
        let dataset = Dataset::from_texts(&texts, &tokenizer, 4).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let (input, target) = dataset.sample_batch(&mut rng, 2, &Device::Cpu).unwrap();
        assert_eq!(input.dims(), &[2, 4]);
        assert_eq!(target.dims(), &[2, 4]);
    }
}
