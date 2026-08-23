use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;

pub const BOS_TOKEN: &str = "<s>";
pub const UNK_TOKEN: &str = "<unk>";

/// A single BPE merge rule: `(left, right)` becomes `merged`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BpeTrio {
    pub left: String,
    pub right: String,
    pub merged: String,
}

/// GPT-2 byte-to-unicode encoder.
///
/// Printable ASCII and Latin-1 bytes map to themselves. Every other byte maps to
/// a private use codepoint, which lets BPE operate on bytes without ambiguity.
static BYTE_ENCODER: LazyLock<Vec<char>> = LazyLock::new(|| {
    let mut bytes = Vec::new();
    let mut chars = Vec::new();

    for byte in b'!'..=b'~' {
        bytes.push(byte);
        chars.push(byte as char);
    }
    for byte in 0xA1..=0xAC {
        bytes.push(byte);
        chars.push(byte as char);
    }
    for byte in 0xAE..=0xFF {
        bytes.push(byte);
        chars.push(byte as char);
    }

    let mut extra = 0u32;
    for byte in 0..=255u8 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            chars.push(char::from_u32(0x0100 + extra).expect("valid private use codepoint"));
            extra += 1;
        }
    }

    let mut table = vec!['\0'; 256];
    for (byte, ch) in bytes.iter().zip(chars.iter()) {
        table[*byte as usize] = *ch;
    }
    table
});

static BYTE_DECODER: LazyLock<HashMap<char, u8>> = LazyLock::new(|| {
    BYTE_ENCODER
        .iter()
        .enumerate()
        .map(|(byte, ch)| (*ch, byte as u8))
        .collect()
});

/// A byte pair encoding tokenizer compatible with llama.cpp GPT-2 tokenizers.
///
/// Text is byte-encoded first, split into words with GPT-2 style leading spaces,
/// and then merged with learned BPE rules. There are no word end markers, so the
/// same vocab and merges can be written into a GGUF file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Bpe {
    merges: Vec<BpeTrio>,
    vocab: Vec<String>,
    #[serde(skip)]
    ids: HashMap<String, u32>,
}

impl Bpe {
    /// Train a byte level BPE tokenizer from a set of texts.
    pub fn train(texts: &[String], num_merges: usize) -> Self {
        let words: Vec<String> = texts
            .iter()
            .flat_map(|text| Self::pretokenize(text))
            .collect();

        let mut vocab: Vec<String> = vec![BOS_TOKEN.to_string(), UNK_TOKEN.to_string()];
        let mut seen = vocab
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        let mut word_tokens: Vec<Vec<String>> = Vec::with_capacity(words.len());

        for word in &words {
            let tokens = Self::tokenize(word);
            for token in &tokens {
                if seen.insert(token.clone()) {
                    vocab.push(token.clone());
                }
            }
            word_tokens.push(tokens);
        }

        let mut merges = Vec::with_capacity(num_merges);
        for _ in 0..num_merges {
            let Some((left, right)) = Self::find_best_pair(&word_tokens) else {
                break;
            };
            let merged = format!("{left}{right}");
            if seen.insert(merged.clone()) {
                vocab.push(merged.clone());
            }
            merges.push(BpeTrio {
                left: left.clone(),
                right: right.clone(),
                merged: merged.clone(),
            });
            word_tokens = Self::merge_corpus(&word_tokens, &left, &right, &merged);
        }

        let mut tokenizer = Self {
            merges,
            vocab,
            ids: HashMap::new(),
        };
        tokenizer.rebuild_index();
        tokenizer
    }

    /// Encode text into byte level token strings.
    pub fn encode(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        for word in Self::pretokenize(text) {
            let mut word_tokens = Self::tokenize(&word);
            // Destructuring the struct directly in the loop pattern.
            for BpeTrio {
                left,
                right,
                merged,
            } in &self.merges
            {
                word_tokens = Self::merge_tokens(&word_tokens, left, right, merged);
            }
            tokens.extend(word_tokens);
        }
        tokens
    }

    /// Encode text into token ids.
    pub fn encode_ids(&self, text: &str) -> Vec<u32> {
        let unk = self.id(UNK_TOKEN).unwrap_or(0);
        self.encode(text)
            .iter()
            .map(|token| self.id(token).unwrap_or(unk))
            .collect()
    }

    /// Decode byte level token strings back into text.
    pub fn decode(&self, tokens: &[String]) -> String {
        let encoded: String = tokens.concat();
        byte_decode(&encoded)
    }

    /// Decode token ids back into text.
    pub fn decode_ids(&self, ids: &[u32]) -> String {
        let tokens: Vec<String> = ids
            .iter()
            .map(|id| self.token(*id).unwrap_or(UNK_TOKEN).to_string())
            .collect();
        self.decode(&tokens)
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// All vocabulary tokens in id order.
    pub fn tokens(&self) -> &[String] {
        &self.vocab
    }

    /// Learned BPE merge rules.
    pub fn merges(&self) -> &[BpeTrio] {
        &self.merges
    }

    /// Id of the beginning of sequence token.
    pub fn bos_id(&self) -> u32 {
        self.id(BOS_TOKEN).expect("BOS token must be in vocab")
    }

    /// Id of the unknown token.
    pub fn unk_id(&self) -> u32 {
        self.id(UNK_TOKEN).expect("UNK token must be in vocab")
    }

    /// Look up the token string for an id.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.token(id)
    }

    /// Save the tokenizer as JSON.
    pub fn save(&self, path: &Path) -> Result<()> {
        let file = std::fs::File::create(path)
            .with_context(|| format!("failed to create tokenizer file {}", path.display()))?;
        serde_json::to_writer_pretty(file, self)
            .with_context(|| format!("failed to write tokenizer file {}", path.display()))?;
        Ok(())
    }

    /// Load a tokenizer from JSON.
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("failed to open tokenizer file {}", path.display()))?;
        let mut tokenizer: Self = serde_json::from_reader(file)
            .with_context(|| format!("failed to read tokenizer file {}", path.display()))?;
        tokenizer.rebuild_index();
        Ok(tokenizer)
    }

    fn rebuild_index(&mut self) {
        self.ids = self
            .vocab
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect();
    }

    fn id(&self, token: &str) -> Option<u32> {
        self.ids.get(token).copied()
    }

    fn token(&self, id: u32) -> Option<&str> {
        self.vocab.get(id as usize).map(String::as_str)
    }

    /// Split text into byte encoded words.
    ///
    /// A single space is kept as the prefix of every word that follows whitespace,
    /// matching the GPT-2 pre-tokenizer. Newlines become standalone words.
    fn pretokenize(text: &str) -> Vec<String> {
        let mut words = Vec::new();
        let mut current = String::new();
        let mut needs_space = false;

        for ch in text.chars() {
            match ch {
                '\n' => {
                    if !current.is_empty() {
                        words.push(byte_encode(&current));
                        current.clear();
                    }
                    words.push(byte_encode("\n"));
                    needs_space = false;
                }
                c if c.is_whitespace() => {
                    if !current.is_empty() {
                        words.push(byte_encode(&current));
                        current.clear();
                    }
                    needs_space = true;
                }
                c => {
                    if current.is_empty() && needs_space {
                        current.push(' ');
                        needs_space = false;
                    }
                    current.push(c);
                }
            }
        }

        if !current.is_empty() {
            words.push(byte_encode(&current));
        }
        words
    }

    fn tokenize(word: &str) -> Vec<String> {
        word.chars().map(|c| c.to_string()).collect()
    }

    fn pair_counts(words: &[Vec<String>]) -> HashMap<(String, String), usize> {
        words
            .iter()
            .flat_map(|word| word.windows(2))
            .fold(HashMap::new(), |mut acc, pair| {
                let key = (pair[0].clone(), pair[1].clone());
                *acc.entry(key).or_insert(0) += 1;
                acc
            })
    }

    fn find_best_pair(words: &[Vec<String>]) -> Option<(String, String)> {
        Self::pair_counts(words)
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|((left, right), _)| (left, right))
    }

    fn merge_corpus(
        words: &[Vec<String>],
        left: &str,
        right: &str,
        merged: &str,
    ) -> Vec<Vec<String>> {
        words
            .iter()
            .map(|word| Self::merge_tokens(word, left, right, merged))
            .collect()
    }

    /// Recursive slice pattern matching with zero `if`/`else` branches.
    fn merge_tokens(tokens: &[String], left: &str, right: &str, merged: &str) -> Vec<String> {
        match tokens {
            [head, second, tail @ ..] if head == left && second == right => {
                let mut result = vec![merged.to_string()];
                result.extend(Self::merge_tokens(tail, left, right, merged));
                result
            }

            [head, tail @ ..] => {
                let mut result = vec![head.clone()];
                result.extend(Self::merge_tokens(tail, left, right, merged));
                result
            }

            [] => vec![],
        }
    }
}

fn byte_encode(text: &str) -> String {
    text.as_bytes()
        .iter()
        .map(|byte| BYTE_ENCODER[*byte as usize])
        .collect()
}

fn byte_decode(encoded: &str) -> String {
    let bytes: Vec<u8> = encoded
        .chars()
        .filter_map(|ch| BYTE_DECODER.get(&ch).copied())
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_texts() -> Vec<String> {
        vec![
            "low low low\nlowest lower".to_string(),
            "new newer newest".to_string(),
        ]
    }

    #[test]
    fn round_trip_preserves_text() {
        let text = "lowest newer\nlow low";
        let tokenizer = Bpe::train(&sample_texts(), 20);
        let ids = tokenizer.encode_ids(text);
        assert_eq!(tokenizer.decode_ids(&ids), text);
    }

    #[test]
    fn learns_merges() {
        let tokenizer = Bpe::train(&sample_texts(), 20);
        assert!(!tokenizer.merges.is_empty());
        assert!(tokenizer.vocab_size() >= tokenizer.merges.len() + 2);
    }

    #[test]
    fn encode_uses_known_ids() {
        let tokenizer = Bpe::train(&sample_texts(), 20);
        let ids = tokenizer.encode_ids("low");
        assert!(ids.iter().all(|id| *id < tokenizer.vocab_size() as u32));
    }

    #[test]
    fn byte_round_trip() {
        let text = "hello world\nfoo bar";
        let encoded = byte_encode(text);
        assert_eq!(byte_decode(&encoded), text);
    }
}
