//! # Tokenizer
//!
//! Neural networks cannot read letters. They consume numbers, and ideally a
//! small number of numbers per word, because every token the model has to think
//! about costs memory and compute. The tokenizer is the bridge between human
//! text and the integer ids a transformer can process.
//!
//! The simplest approach would be one id per character, but that makes every
//! word many tokens long. The opposite extreme, one id per word, explodes the
//! vocabulary and makes the model unable to handle words it has never seen.
//! Byte pair encoding (BPE) finds a middle path: start with characters, then
//! repeatedly merge the most frequent adjacent pair into a new symbol. After
//! training, common pieces of words like `"ing"` or `" the"` become single
//! tokens while rare words stay broken into smaller pieces. The model gets a
//! compact vocabulary and still never sees an unknown word.
//!
//! There is one more subtlety. Text contains every byte imaginable, and a
//! tokenizer that only knows ASCII would choke on emoji or accented letters.
//! GPT-2 solved this with byte level encoding: map every possible byte to a
//! unicode character before BPE. Spaces become `Ġ`, newlines become `Ċ`, and
//! every other byte maps to a printable character. This guarantees that any
//! UTF-8 text can be tokenized without unknown tokens, and it is exactly the
//! scheme llama.cpp uses, which is why a model trained with this tokenizer can
//! be exported to GGUF and run in Ollama.
//!
//! ## References
//!
//! - BPE paper: <https://arxiv.org/abs/1508.07909>
//! - GPT-2 paper: <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>
//! - Karpathy, "Let's build the GPT Tokenizer": <https://www.youtube.com/watch?v=zduSFxRajkE>
//! - Hugging Face tokenizer docs: <https://huggingface.co/docs/tokenizers>

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
    /// Training a tokenizer is like learning the alphabet of a language, except
    /// the alphabet is not fixed: it grows as frequent letter pairs become
    /// subwords. The trainer counts how often each word appears, then
    /// repeatedly fuses the most common adjacent pair into a new token. After
    /// enough merges, the vocabulary contains useful pieces like `"ing"` and
    /// `" the"` that make the model's job much easier.
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
        let word_freq = Self::count_word_frequencies(texts);
        let mut words: Vec<SymbolWord> = word_freq
            .into_iter()
            .map(|(word, freq)| SymbolWord {
                symbols: Self::tokenize(&word),
                freq,
            })
            .collect();

        let mut vocab = Self::seed_vocabulary(&words);
        let mut pair_counts = HashMap::new();
        let mut pair_to_words = HashMap::new();
        Self::count_pair_stats(&words, &mut pair_counts, &mut pair_to_words);

        let mut merges = Vec::with_capacity(num_merges);

        for _ in 0..num_merges {
            let (left, right) = match Self::most_frequent_pair(&pair_counts) {
                Some(pair) => pair,
                None => break,
            };

            let merged = format!("{left}{right}");
            vocab.push(merged.clone());
            merges.push(BpeTrio {
                left: left.clone(),
                right: right.clone(),
                merged: merged.clone(),
            });

            let affected = pair_to_words
                .get(&(left.clone(), right.clone()))
                .cloned()
                .unwrap_or_default();

            Self::apply_merge_to_words(
                &mut words,
                &affected,
                &left,
                &right,
                &merged,
                &mut pair_counts,
                &mut pair_to_words,
            );
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

    /// Count how often each pre-tokenized word appears across all texts.
    fn count_word_frequencies(texts: &[String]) -> HashMap<String, usize> {
        texts
            .iter()
            .flat_map(|text| Self::pretokenize(text))
            .fold(HashMap::new(), |mut acc, word| {
                *acc.entry(word).or_insert(0) += 1;
                acc
            })
    }

    /// Build the initial vocabulary from special tokens and all character symbols.
    fn seed_vocabulary(words: &[SymbolWord]) -> Vec<String> {
        let mut vocab = vec![
            BOS_TOKEN.to_string(),
            EOS_TOKEN.to_string(),
            UNK_TOKEN.to_string(),
        ];
        let mut seen: std::collections::HashSet<String> = vocab.iter().cloned().collect();

        // Seed all 256 byte-encoded characters so the vocabulary can represent
        // any UTF-8 byte, even those absent from the training sample. Without
        // this, encoding text with unseen bytes produces `<unk>` tokens.
        for byte in 0u8..=255 {
            let encoded = BYTE_ENCODER[byte as usize].to_string();
            if seen.insert(encoded.clone()) {
                vocab.push(encoded);
            }
        }

        for word in words {
            for symbol in &word.symbols {
                if seen.insert(symbol.clone()) {
                    vocab.push(symbol.clone());
                }
            }
        }
        vocab
    }

    /// Count weighted adjacent symbol pairs and map each pair to affected word indices.
    fn count_pair_stats(
        words: &[SymbolWord],
        pair_counts: &mut HashMap<(String, String), usize>,
        pair_to_words: &mut HashMap<(String, String), Vec<usize>>,
    ) {
        for (index, word) in words.iter().enumerate() {
            for pair in word.symbols.windows(2) {
                let key = (pair[0].clone(), pair[1].clone());
                *pair_counts.entry(key.clone()).or_insert(0) += word.freq;
                pair_to_words.entry(key).or_default().push(index);
            }
        }
    }

    /// Find the most frequent symbol pair, or `None` when all counts are zero.
    fn most_frequent_pair(
        pair_counts: &HashMap<(String, String), usize>,
    ) -> Option<(String, String)> {
        pair_counts
            .iter()
            .filter(|(_, count)| **count > 0)
            .max_by_key(|(_, count)| **count)
            .map(|((left, right), _)| (left.clone(), right.clone()))
    }

    /// Apply a merge rule to all affected words, updating pair statistics in place.
    fn apply_merge_to_words(
        words: &mut [SymbolWord],
        affected: &[usize],
        left: &str,
        right: &str,
        merged: &str,
        pair_counts: &mut HashMap<(String, String), usize>,
        pair_to_words: &mut HashMap<(String, String), Vec<usize>>,
    ) {
        let mut seen = std::collections::HashSet::new();

        for &index in affected {
            if !seen.insert(index) {
                continue;
            }

            let has_pair = words[index]
                .symbols
                .windows(2)
                .any(|pair| pair[0] == left && pair[1] == right);

            if !has_pair {
                continue;
            }

            let freq = words[index].freq;
            Self::decrement_old_pairs(&words[index].symbols, freq, pair_counts);

            let new_symbols = Self::merge_pair_in_word(&words[index].symbols, left, right, merged);
            Self::increment_new_pairs(&new_symbols, freq, index, pair_counts, pair_to_words);
            words[index].symbols = new_symbols;
        }
    }

    /// Subtract word frequency from all adjacent pair counts, removing zeroed entries.
    fn decrement_old_pairs(
        symbols: &[String],
        freq: usize,
        pair_counts: &mut HashMap<(String, String), usize>,
    ) {
        for pair in symbols.windows(2) {
            let key = (pair[0].clone(), pair[1].clone());
            let count = pair_counts.entry(key.clone()).or_insert(0);
            *count = count.saturating_sub(freq);
            if *count == 0 {
                pair_counts.remove(&key);
            }
        }
    }

    /// Add word frequency to pair counts for all new adjacent pairs after merging.
    fn increment_new_pairs(
        symbols: &[String],
        freq: usize,
        word_index: usize,
        pair_counts: &mut HashMap<(String, String), usize>,
        pair_to_words: &mut HashMap<(String, String), Vec<usize>>,
    ) {
        for pair in symbols.windows(2) {
            let key = (pair[0].clone(), pair[1].clone());
            *pair_counts.entry(key.clone()).or_insert(0) += freq;
            pair_to_words.entry(key).or_default().push(word_index);
        }
    }

    /// Replace every non-overlapping `(left, right)` in a symbol slice with `merged`.
    ///
    /// Uses recursive slice pattern matching — zero `if`/`else`, zero loops.
    /// See the coding standards example for the canonical BPE merge pattern.
    fn merge_pair_in_word(symbols: &[String], left: &str, right: &str, merged: &str) -> Vec<String> {
        match symbols {
            [head, second, tail @ ..] if head == left && second == right => {
                let mut result = vec![merged.to_string()];
                result.extend(Self::merge_pair_in_word(tail, left, right, merged));
                result
            }
            [head, tail @ ..] => {
                let mut result = vec![head.clone()];
                result.extend(Self::merge_pair_in_word(tail, left, right, merged));
                result
            }
            [] => vec![],
        }
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
    ///
    /// ## References
    ///
    /// - GPT-2 tokenization: <https://huggingface.co/docs/transformers/tokenizer_summary>
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
    ///
    /// ## References
    ///
    /// - Decoding strategies: <https://huggingface.co/docs/transformers/generation_strategies>
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
    ///
    /// ## References
    ///
    /// - Serde JSON: <https://serde.rs>
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
    ///
    /// ## References
    ///
    /// - Serde JSON: <https://serde.rs>
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
