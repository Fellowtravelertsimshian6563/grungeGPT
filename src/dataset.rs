use crate::tokenizer::Bpe;
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use rand::Rng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::path::Path;

/// Fixed-length training sequences built from tokenized lyric files.
#[derive(Debug, Clone)]
pub struct Dataset {
    sequences: Vec<Vec<u32>>,
    block_size: usize,
}

impl Dataset {
    pub fn from_texts(texts: &[String], tokenizer: &Bpe, block_size: usize) -> Result<Self> {
        let seq_len = block_size + 1;
        let bos = tokenizer.bos_id();
        let mut sequences = Vec::new();

        for text in texts {
            let ids = tokenizer.encode_ids(text);
            for chunk in ids.chunks(block_size) {
                let mut sequence = Vec::with_capacity(seq_len);
                sequence.push(bos);
                sequence.extend_from_slice(chunk);
                sequence.resize(seq_len, bos);
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

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn sequence_count(&self) -> usize {
        self.sequences.len()
    }

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

fn collect_txt_files(path: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<()> {
    let entries = std::fs::read_dir(path)
        .with_context(|| format!("failed to read directory {}", path.display()))?;

    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_dir() {
            collect_txt_files(&path, files)?;
        } else if file_type.is_file()
            && path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("txt"))
        {
            files.push(path);
        }
    }
    Ok(())
}

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
