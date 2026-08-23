//! # Tokenizer
//!
//! A byte pair encoding (BPE) tokenizer compatible with llama.cpp GPT-2
//! tokenizers.
//!
//! ## What is BPE?
//!
//! BPE starts with single characters and repeatedly merges the most frequent
//! adjacent pair of symbols into a new symbol. The result is a vocabulary that
//! contains common subwords like `"ing"` or `" the"`, so the model does not
//! have to learn every word from raw characters.
//!
//! ## Why byte level?
//!
//! GPT-2 maps every possible byte to a unicode character before BPE. This
//! guarantees that any UTF-8 text can be tokenized without unknown tokens.
//! Spaces become `Ġ`, newlines become `Ċ`, and all other bytes map to
//! printable unicode. The same mapping is used by llama.cpp, which is why a
//! model trained with this tokenizer can be exported to GGUF and run in Ollama.
//!
//! ## References
//!
//! - BPE paper: <https://arxiv.org/abs/1508.07909>
//! - GPT-2 paper: <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>
//! - Karpathy, "Let's build the GPT Tokenizer": <https://www.youtube.com/watch?v=zduSFxRajkE>
//! - Hugging Face tokenizer docs: <https://huggingface.co/docs/tokenizers>
//!
//! ## Files
//!
//! - `Bpe::train` learns merge rules from raw text.
//! - `Bpe::encode` converts text to token strings.
//! - `Bpe::decode` converts token strings back to text.
//! - `Bpe::save` and `Bpe::load` persist the vocabulary as JSON.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;

/// Beginning-of-sequence token id 0 in the vocabulary.
pub const BOS_TOKEN: &str = "<s>";

/// End-of-sequence token id 1 in the vocabulary.
///
/// The model is trained to emit this token at the end of every document.
/// Generation stops when this token is sampled.
pub const EOS_TOKEN: &str = "<eos>";

/// Unknown token fallback id 2 in the vocabulary.
///
/// Any byte sequence that is not in the vocabulary is mapped to this token
/// during encoding.
pub const UNK_TOKEN: &str = "<unk>";

/// A single BPE merge rule: `(left, right)` becomes `merged`.
///
/// # Fields
///
/// - `left`: first symbol in the pair.
/// - `right`: second symbol in the pair.
/// - `merged`: the new symbol created by concatenating `left` and `right`.
///
/// # Example
///
/// `("a", "b") -> "ab"` means that every adjacent `"a" "b"` in a word is
/// replaced with the single token `"ab"` during encoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BpeTrio {
    /// First symbol of the merge pair.
    pub left: String,
    /// Second symbol of the merge pair.
    pub right: String,
    /// Concatenated replacement symbol.
    pub merged: String,
}

/// GPT-2 byte-to-unicode encoder.
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

struct SymbolWord {
    symbols: Vec<String>,
    freq: usize,
}

/// A byte pair encoding tokenizer compatible with llama.cpp GPT-2 tokenizers.
///
/// # Design
///
/// - `merges` is the ordered list of learned BPE merge rules.
/// - `vocab` maps token ids to token strings in insertion order.
/// - `ids` is a reverse index from token string to id, rebuilt after loading.
/// - `bpe_ranks` maps each merge pair to its rank, so encoding can merge the
///   lowest-rank pair first.
///
/// The `ids` and `bpe_ranks` maps are not serialized; they are rebuilt from the
/// serialized `vocab` and `merges` in [`Bpe::rebuild_index`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Bpe {
    /// Ordered BPE merge rules learned during training.
    merges: Vec<BpeTrio>,
    /// Vocabulary tokens in id order.
    vocab: Vec<String>,
    /// Reverse lookup from token string to id.
    #[serde(skip)]
    ids: HashMap<String, u32>,
    /// Merge pair to rank for fast encoding.
    #[serde(skip)]
    bpe_ranks: HashMap<(String, String), usize>,
}

impl Bpe {
    /// Train a byte level BPE tokenizer from a set of texts.
    ///
    /// # Parameters
    ///
    /// - `texts`: slice of raw documents used to count word frequencies.
    /// - `num_merges`: maximum number of BPE merge rules to learn.
    ///
    /// # Returns
    ///
    /// A fully built [`Bpe`] with vocab and merge ranks ready for encoding.
    ///
    /// # Algorithm
    ///
    /// 1. Pre-tokenize every document into GPT-2 style byte encoded words.
    /// 2. Count how often each unique word appears.
    /// 3. Seed the vocabulary with `<s>`, `<eos>`, `<unk>`, and every character
    ///    that appears in the corpus.
    /// 4. Count adjacent symbol pairs weighted by word frequency.
    /// 5. Repeatedly find the most frequent pair, add it to the vocabulary, and
    ///    merge every non-overlapping occurrence in the affected words.
    ///
    /// See the BPE paper <https://arxiv.org/abs/1508.07909>.
    pub fn train(texts: &[String], num_merges: usize) -> Self {
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        for text in texts {
            for word in Self::pretokenize(text) {
                *word_freq.entry(word).or_insert(0) += 1;
            }
        }

        let mut words: Vec<SymbolWord> = word_freq
            .into_iter()
            .map(|(word, freq)| SymbolWord {
                symbols: Self::tokenize(&word),
                freq,
            })
            .collect();

        let mut vocab: Vec<String> = vec![
            BOS_TOKEN.to_string(),
            EOS_TOKEN.to_string(),
            UNK_TOKEN.to_string(),
        ];
        let mut seen = vocab
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>();

        for word in &words {
            for symbol in &word.symbols {
                if seen.insert(symbol.clone()) {
                    vocab.push(symbol.clone());
                }
            }
        }

        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
        let mut pair_to_words: HashMap<(String, String), Vec<usize>> = HashMap::new();
        for (index, word) in words.iter().enumerate() {
            for pair in word.symbols.windows(2) {
                let key = (pair[0].clone(), pair[1].clone());
                *pair_counts.entry(key.clone()).or_insert(0) += word.freq;
                pair_to_words.entry(key).or_default().push(index);
            }
        }

        let mut merges = Vec::with_capacity(num_merges);
        let mut processed = vec![false; words.len()];

        for _ in 0..num_merges {
            processed.fill(false);
            let Some((left, right)) = pair_counts
                .iter()
                .max_by_key(|(_, count)| **count)
                .map(|((left, right), _)| (left.clone(), right.clone()))
            else {
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

            let Some(indices) = pair_to_words.get(&(left.clone(), right.clone())).cloned() else {
                break;
            };

            for index in indices {
                if processed[index] {
                    continue;
                }
                processed[index] = true;

                let word = &mut words[index];
                if !word
                    .symbols
                    .windows(2)
                    .any(|pair| pair[0] == left && pair[1] == right)
                {
                    continue;
                }

                for pair in word.symbols.windows(2) {
                    let key = (pair[0].clone(), pair[1].clone());
                    let count = pair_counts.entry(key.clone()).or_insert(0);
                    *count = count.saturating_sub(word.freq);
                    if *count == 0 {
                        pair_counts.remove(&key);
                    }
                }

                let mut new_symbols = Vec::with_capacity(word.symbols.len());
                let mut index_in_word = 0;
                while index_in_word < word.symbols.len() {
                    if index_in_word + 1 < word.symbols.len()
                        && word.symbols[index_in_word] == left
                        && word.symbols[index_in_word + 1] == right
                    {
                        new_symbols.push(merged.clone());
                        index_in_word += 2;
                    } else {
                        new_symbols.push(word.symbols[index_in_word].clone());
                        index_in_word += 1;
                    }
                }

                for pair in new_symbols.windows(2) {
                    let key = (pair[0].clone(), pair[1].clone());
                    *pair_counts.entry(key.clone()).or_insert(0) += word.freq;
                    pair_to_words.entry(key).or_default().push(index);
                }
                word.symbols = new_symbols;
            }
        }

        let mut tokenizer = Self {
            merges,
            vocab,
            ids: HashMap::new(),
            bpe_ranks: HashMap::new(),
        };
        tokenizer.rebuild_index();
        tokenizer
    }

    /// Encode text into byte level token strings.
    ///
    /// # Parameters
    ///
    /// - `text`: raw UTF-8 input text.
    ///
    /// # Returns
    ///
    /// A vector of token strings from the vocabulary.
    ///
    /// # Behavior
    ///
    /// Each pre-tokenized word is split into characters and repeatedly merged
    /// using the lowest-rank BPE pair first, matching the standard GPT-2 BPE
    /// decoding order. See Karpathy's tokenizer video
    /// <https://www.youtube.com/watch?v=zduSFxRajkE>.
    pub fn encode(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        for word in Self::pretokenize(text) {
            let mut symbols = Self::tokenize(&word);
            while symbols.len() > 1 {
                let mut best: Option<(usize, usize)> = None;
                for index in 0..symbols.len() - 1 {
                    let key = (symbols[index].clone(), symbols[index + 1].clone());
                    if let Some(rank) = self.bpe_ranks.get(&key) {
                        if best.is_none_or(|(_, best_rank)| *rank < best_rank) {
                            best = Some((index, *rank));
                        }
                    }
                }
                let Some((index, _)) = best else {
                    break;
                };
                let merged = format!("{}{}", symbols[index], symbols[index + 1]);
                symbols[index] = merged;
                symbols.remove(index + 1);
            }
            tokens.extend(symbols);
        }
        tokens
    }

    /// Encode text into token ids.
    ///
    /// # Parameters
    ///
    /// - `text`: raw UTF-8 input text.
    ///
    /// # Returns
    ///
    /// A vector of vocabulary ids. Unknown tokens are replaced with the
    /// `<unk>` id.
    pub fn encode_ids(&self, text: &str) -> Vec<u32> {
        let unk = self.id(UNK_TOKEN).unwrap_or(0);
        self.encode(text)
            .iter()
            .map(|token| self.id(token).unwrap_or(unk))
            .collect()
    }

    /// Decode byte level token strings back into text.
    ///
    /// # Parameters
    ///
    /// - `tokens`: slice of token strings produced by [`Bpe::encode`].
    ///
    /// # Returns
    ///
    /// The original UTF-8 text with spaces and newlines restored.
    ///
    /// # Behavior
    ///
    /// Tokens are concatenated in order, then every byte-encoded unicode
    /// character is mapped back to its original byte. See the GPT-2 paper's
    /// byte encoding appendix <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>.
    pub fn decode(&self, tokens: &[String]) -> String {
        let encoded: String = tokens.concat();
        byte_decode(&encoded)
    }

    /// Decode token ids back into text.
    ///
    /// # Parameters
    ///
    /// - `ids`: slice of vocabulary ids produced by [`Bpe::encode_ids`].
    ///
    /// # Returns
    ///
    /// The decoded UTF-8 text. Unknown ids are rendered as `<unk>`.
    pub fn decode_ids(&self, ids: &[u32]) -> String {
        let tokens: Vec<String> = ids
            .iter()
            .map(|id| self.token(*id).unwrap_or(UNK_TOKEN).to_string())
            .collect();
        self.decode(&tokens)
    }

    /// Number of tokens in the vocabulary.
    ///
    /// # Returns
    ///
    /// The total count of special tokens, characters, and merged BPE symbols.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// All vocabulary tokens in id order.
    ///
    /// # Returns
    ///
    /// A slice where `index` is the token id and the value is the token string.
    pub fn tokens(&self) -> &[String] {
        &self.vocab
    }

    /// Learned BPE merge rules.
    ///
    /// # Returns
    ///
    /// The ordered list of merge rules used by the tokenizer.
    pub fn merges(&self) -> &[BpeTrio] {
        &self.merges
    }

    /// Id of the beginning of sequence token.
    ///
    /// # Panics
    ///
    /// Panics if the `<s>` token is missing from the vocabulary.
    pub fn bos_id(&self) -> u32 {
        self.id(BOS_TOKEN).expect("BOS token must be in vocab")
    }

    /// Id of the end of sequence token.
    ///
    /// # Panics
    ///
    /// Panics if the `<eos>` token is missing from the vocabulary.
    pub fn eos_id(&self) -> u32 {
        self.id(EOS_TOKEN).expect("EOS token must be in vocab")
    }

    /// Id of the unknown token.
    ///
    /// # Panics
    ///
    /// Panics if the `<unk>` token is missing from the vocabulary.
    pub fn unk_id(&self) -> u32 {
        self.id(UNK_TOKEN).expect("UNK token must be in vocab")
    }

    /// Look up the token string for an id.
    ///
    /// # Parameters
    ///
    /// - `id`: vocabulary id.
    ///
    /// # Returns
    ///
    /// `Some(token)` when the id is valid, `None` when it is out of range.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.token(id)
    }

    /// Save the tokenizer as JSON.
    ///
    /// # Parameters
    ///
    /// - `path`: destination file path.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be created or written.
    pub fn save(&self, path: &Path) -> Result<()> {
        let file = std::fs::File::create(path)
            .with_context(|| format!("failed to create tokenizer file {}", path.display()))?;
        serde_json::to_writer_pretty(file, self)
            .with_context(|| format!("failed to write tokenizer file {}", path.display()))?;
        Ok(())
    }

    /// Load a tokenizer from JSON.
    ///
    /// # Parameters
    ///
    /// - `path`: source file path written by [`Bpe::save`].
    ///
    /// # Returns
    ///
    /// A tokenizer with rebuilt id and rank indexes.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be opened or parsed.
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("failed to open tokenizer file {}", path.display()))?;
        let mut tokenizer: Self = serde_json::from_reader(file)
            .with_context(|| format!("failed to read tokenizer file {}", path.display()))?;
        tokenizer.rebuild_index();
        Ok(tokenizer)
    }

    /// Rebuild the in-memory id and rank indexes after deserialization.
    ///
    /// `ids` maps token strings to ids. `bpe_ranks` maps merge pairs to their
    /// position in the merge list so encoding can always merge the lowest rank
    /// pair first.
    fn rebuild_index(&mut self) {
        self.ids = self
            .vocab
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect();
        self.bpe_ranks = self
            .merges
            .iter()
            .enumerate()
            .map(|(rank, rule)| ((rule.left.clone(), rule.right.clone()), rank))
            .collect();
    }

    /// Look up the vocabulary id for a token string.
    ///
    /// # Parameters
    ///
    /// - `token`: token string.
    ///
    /// # Returns
    ///
    /// `Some(id)` when the token is in the vocabulary, `None` otherwise.
    fn id(&self, token: &str) -> Option<u32> {
        self.ids.get(token).copied()
    }

    /// Look up the token string for a vocabulary id.
    ///
    /// # Parameters
    ///
    /// - `id`: vocabulary id.
    ///
    /// # Returns
    ///
    /// `Some(token)` when the id is valid, `None` when it is out of range.
    fn token(&self, id: u32) -> Option<&str> {
        self.vocab.get(id as usize).map(String::as_str)
    }

    /// Split text into byte encoded words.
    ///
    /// A single space is kept as the prefix of every word that follows
    /// whitespace, matching the GPT-2 pre-tokenizer. Newlines become standalone
    /// words.
    ///
    /// # Parameters
    ///
    /// - `text`: raw UTF-8 input text.
    ///
    /// # Returns
    ///
    /// A vector of byte encoded word strings suitable for BPE tokenization.
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

    /// Split a byte encoded word into single-character symbols.
    ///
    /// # Parameters
    ///
    /// - `word`: byte encoded word string.
    ///
    /// # Returns
    ///
    /// A vector of single-character token strings ready for BPE merging.
    fn tokenize(word: &str) -> Vec<String> {
        word.chars().map(|c| c.to_string()).collect()
    }
}

/// Encode a UTF-8 string into GPT-2 byte encoded unicode.
///
/// # Parameters
///
/// - `text`: raw UTF-8 input text.
///
/// # Returns
///
/// A string where every byte is represented by its byte-to-unicode character.
fn byte_encode(text: &str) -> String {
    text.as_bytes()
        .iter()
        .map(|byte| BYTE_ENCODER[*byte as usize])
        .collect()
}

/// Decode a GPT-2 byte encoded unicode string back to UTF-8.
///
/// # Parameters
///
/// - `encoded`: byte encoded string produced by [`byte_encode`].
///
/// # Returns
///
/// The original UTF-8 text. Invalid byte sequences are replaced lossily.
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
        assert!(tokenizer.vocab_size() >= tokenizer.merges.len() + 3);
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
