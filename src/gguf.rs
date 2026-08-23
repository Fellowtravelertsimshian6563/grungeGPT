//! # GGUF export
//!
//! Writes the trained Candle model into the GGUF format used by llama.cpp and
//! Ollama.
//!
//! ## What is GGUF?
//!
//! GGUF is a binary format for neural network weights and metadata. It stores:
//!
//! 1. A magic number and format version.
//! 2. Metadata key-value pairs such as architecture, context length, and the
//!    tokenizer.
//! 3. Tensor names, shapes, types, and offsets.
//! 4. Aligned tensor data.
//!
//! This exporter uses the `gpt2` architecture, so llama.cpp can load the model
//! with its standard GPT-2 implementation.
//!
//! ## References
//!
//! - llama.cpp GGUF specification: <https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/gguf_reader.py>
//! - Ollama Modelfile docs: <https://docs.ollama.com/modelfile>
//! - Hugging Face safetensors format: <https://github.com/huggingface/safetensors>

use crate::model::{Gpt, GptConfig};
use crate::tokenizer::Bpe;
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use std::io::{Seek, Write};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION: u32 = 3;
const ALIGNMENT: usize = 32;

const TYPE_UINT32: u32 = 4;
const TYPE_FLOAT32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_F32_TENSOR: u32 = 0;

/// One tensor in the GGUF file.
struct TensorEntry {
    /// GGUF tensor name, e.g. `token_embd.weight`.
    name: String,
    /// Tensor dimensions in GGUF order.
    shape: Vec<u64>,
    /// Flattened f32 weights.
    data: Vec<f32>,
}

/// Export a trained checkpoint and tokenizer to a GPT-2 architecture GGUF file.
///
/// # Parameters
///
/// - `config_path`: path to `config.json`.
/// - `model_path`: path to `model.safetensors`.
/// - `tokenizer`: tokenizer used to write GGUF tokenizer metadata.
/// - `output_path`: destination `.gguf` file.
/// - `context_length`: advertised context length. Position embeddings are padded
///   when this is larger than the training `block_size`.
///
/// # Returns
///
/// `Ok(())` when the file is written successfully.
///
/// # Errors
///
/// Returns an error when config or weights cannot be read, or when the GGUF
/// cannot be written.
///
/// The exported file can be loaded by llama.cpp or served by Ollama. See the
/// GGUF reference in llama.cpp <https://github.com/ggerganov/llama.cpp>.
pub fn export_gguf(
    config_path: &Path,
    model_path: &Path,
    tokenizer: &Bpe,
    output_path: &Path,
    context_length: usize,
) -> Result<()> {
    let file = std::fs::File::open(config_path)
        .with_context(|| format!("failed to open {}", config_path.display()))?;
    let config: GptConfig = serde_json::from_reader(file)
        .with_context(|| format!("failed to read {}", config_path.display()))?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let _model = Gpt::new(vb, &config)?;
    varmap
        .load(model_path)
        .with_context(|| format!("failed to load {}", model_path.display()))?;

    let entries = collect_tensors(&varmap, &config, context_length)?;

    let mut output = std::fs::File::create(output_path)
        .with_context(|| format!("failed to create {}", output_path.display()))?;
    write_gguf(&mut output, &config, tokenizer, &entries, context_length)?;
    Ok(())
}

/// Collect all model weights into GGUF tensor entries with mapped names.
///
/// # Parameters
///
/// - `varmap`: trained variable map.
/// - `config`: model hyperparameters.
/// - `context_length`: context length for padded position embeddings.
///
/// # Returns
///
/// A sorted vector of [`TensorEntry`] for every weight accepted by llama.cpp.
fn collect_tensors(
    varmap: &VarMap,
    config: &GptConfig,
    context_length: usize,
) -> Result<Vec<TensorEntry>> {
    let mut entries = Vec::new();
    let data = varmap.data().lock().unwrap();
    for (name, var) in data.iter() {
        let Some((gguf_name, shape)) = map_tensor_name(name, config, context_length) else {
            continue;
        };
        let tensor = var.as_tensor();
        let mut values = tensor
            .to_dtype(DType::F32)
            .with_context(|| format!("failed to cast tensor {name} to f32"))?
            .flatten_all()
            .with_context(|| format!("failed to flatten tensor {name}"))?
            .to_vec1::<f32>()
            .with_context(|| format!("failed to read tensor {name}"))?;

        if name == "wpe.weight" && context_length > config.block_size {
            pad_position_embeddings(&mut values, config, context_length);
        }

        entries.push(TensorEntry {
            name: gguf_name,
            shape,
            data: values,
        });
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}

/// Repeat the last learned position embedding row to reach `context_length`.
///
/// # Parameters
///
/// - `values`: flattened position embedding weights.
/// - `config`: model hyperparameters.
/// - `context_length`: target number of position rows.
fn pad_position_embeddings(values: &mut Vec<f32>, config: &GptConfig, context_length: usize) {
    let n_embd = config.n_embd;
    let last_row = values[values.len() - n_embd..].to_vec();
    for _ in 0..(context_length - config.block_size) {
        values.extend_from_slice(&last_row);
    }
}

/// Map a Candle parameter name to the llama.cpp GPT-2 GGUF name and shape.
///
/// # Parameters
///
/// - `name`: Candle variable name from the VarMap.
/// - `config`: model hyperparameters.
/// - `context_length`: context length used for position embeddings.
///
/// # Returns
///
/// `Some((gguf_name, shape))` when the weight should be exported, `None` when
/// it should be skipped.
fn map_tensor_name(
    name: &str,
    config: &GptConfig,
    context_length: usize,
) -> Option<(String, Vec<u64>)> {
    let n_embd = config.n_embd as u64;
    let n_ff = (4 * config.n_embd) as u64;
    let n_vocab = config.vocab_size as u64;
    let n_ctx = context_length as u64;

    match name {
        "wte.weight" => Some(("token_embd.weight".into(), vec![n_vocab, n_embd])),
        "wpe.weight" => Some(("position_embd.weight".into(), vec![n_ctx, n_embd])),
        "ln_f.weight" => Some(("output_norm.weight".into(), vec![n_embd])),
        "ln_f.bias" => Some(("output_norm.bias".into(), vec![n_embd])),
        "lm_head.weight" => Some(("output.weight".into(), vec![n_vocab, n_embd])),
        "lm_head.bias" => None,
        _ => {
            let rest = name.strip_prefix("h.")?;
            let (index_str, rest) = rest.split_once('.')?;
            let index: usize = index_str.parse().ok()?;
            let prefix = format!("blk.{index}.");
            match rest {
                "ln1.weight" => Some((format!("{prefix}attn_norm.weight"), vec![n_embd])),
                "ln1.bias" => Some((format!("{prefix}attn_norm.bias"), vec![n_embd])),
                "attn.c_attn.weight" => {
                    Some((format!("{prefix}attn_qkv.weight"), vec![3 * n_embd, n_embd]))
                }
                "attn.c_attn.bias" => Some((format!("{prefix}attn_qkv.bias"), vec![3 * n_embd])),
                "attn.c_proj.weight" => {
                    Some((format!("{prefix}attn_output.weight"), vec![n_embd, n_embd]))
                }
                "attn.c_proj.bias" => Some((format!("{prefix}attn_output.bias"), vec![n_embd])),
                "ln2.weight" => Some((format!("{prefix}ffn_norm.weight"), vec![n_embd])),
                "ln2.bias" => Some((format!("{prefix}ffn_norm.bias"), vec![n_embd])),
                "mlp.c_fc.weight" => Some((format!("{prefix}ffn_up.weight"), vec![n_ff, n_embd])),
                "mlp.c_fc.bias" => Some((format!("{prefix}ffn_up.bias"), vec![n_ff])),
                "mlp.c_proj.weight" => {
                    Some((format!("{prefix}ffn_down.weight"), vec![n_embd, n_ff]))
                }
                "mlp.c_proj.bias" => Some((format!("{prefix}ffn_down.bias"), vec![n_embd])),
                _ => None,
            }
        }
    }
}

/// Write the complete GGUF header, metadata, tensor infos, and tensor data.
///
/// # Parameters
///
/// - `writer`: seekable output writer.
/// - `config`: model hyperparameters.
/// - `tokenizer`: tokenizer metadata source.
/// - `entries`: tensor entries to serialize.
/// - `context_length`: context length advertised in metadata.
///
/// # Errors
///
/// Returns an error on any write failure.
fn write_gguf<W: Write + Seek>(
    writer: &mut W,
    config: &GptConfig,
    tokenizer: &Bpe,
    entries: &[TensorEntry],
    context_length: usize,
) -> Result<()> {
    let kv_count = metadata_kv_count();
    writer.write_all(&GGUF_MAGIC.to_le_bytes())?;
    writer.write_all(&GGUF_VERSION.to_le_bytes())?;
    writer.write_all(&(entries.len() as u64).to_le_bytes())?;
    writer.write_all(&(kv_count as u64).to_le_bytes())?;

    write_metadata(writer, config, tokenizer, context_length)?;
    write_tensor_infos(writer, entries)?;
    write_tensor_data(writer, entries)?;
    Ok(())
}

fn metadata_kv_count() -> usize {
    18
}

/// Write GGUF metadata key-value pairs for a GPT-2 architecture model.
///
/// # Parameters
///
/// - `writer`: output writer.
/// - `config`: model hyperparameters.
/// - `tokenizer`: tokenizer used for token and merge metadata.
/// - `context_length`: context length advertised to llama.cpp.
///
/// # Errors
///
/// Returns an error on any write failure.
fn write_metadata<W: Write>(
    writer: &mut W,
    config: &GptConfig,
    tokenizer: &Bpe,
    context_length: usize,
) -> Result<()> {
    write_string_value(writer, "general.architecture", "gpt2")?;
    write_string_value(writer, "general.name", "grungegpt")?;
    write_u32_value(writer, "general.file_type", 0)?;
    write_u32_value(writer, "gpt2.context_length", context_length as u32)?;
    write_u32_value(writer, "gpt2.embedding_length", config.n_embd as u32)?;
    write_u32_value(writer, "gpt2.block_count", config.n_layer as u32)?;
    write_u32_value(
        writer,
        "gpt2.feed_forward_length",
        (4 * config.n_embd) as u32,
    )?;
    write_u32_value(writer, "gpt2.attention.head_count", config.n_head as u32)?;
    write_f32_value(writer, "gpt2.attention.layer_norm_epsilon", 1e-5)?;
    write_bool_value(writer, "gpt2.attention.causal", true)?;
    write_string_value(writer, "tokenizer.ggml.model", "gpt2")?;
    write_string_value(writer, "tokenizer.ggml.pre", "gpt-2")?;
    write_string_array(writer, "tokenizer.ggml.tokens", tokenizer.tokens())?;

    let merges: Vec<String> = tokenizer
        .merges()
        .iter()
        .map(|rule| format!("{} {}", rule.left, rule.right))
        .collect();
    write_string_array(writer, "tokenizer.ggml.merges", &merges)?;

    let bos = tokenizer.bos_id();
    let eos = tokenizer.eos_id();
    write_u32_value(writer, "tokenizer.ggml.bos_token_id", bos)?;
    write_u32_value(writer, "tokenizer.ggml.eos_token_id", eos)?;
    write_bool_value(writer, "tokenizer.ggml.add_bos_token", false)?;
    write_bool_value(writer, "tokenizer.ggml.add_eos_token", false)?;
    Ok(())
}

/// Write the tensor info table used to locate tensor data.
///
/// # Parameters
///
/// - `writer`: output writer.
/// - `entries`: tensor entries with names, shapes, and data lengths.
///
/// # Errors
///
/// Returns an error on any write failure.
fn write_tensor_infos<W: Write>(writer: &mut W, entries: &[TensorEntry]) -> Result<()> {
    let mut offset = 0u64;
    for entry in entries {
        write_string(writer, &entry.name)?;
        writer.write_all(&(entry.shape.len() as u32).to_le_bytes())?;
        for dim in entry.shape.iter().rev() {
            writer.write_all(&dim.to_le_bytes())?;
        }
        writer.write_all(&TYPE_F32_TENSOR.to_le_bytes())?;
        writer.write_all(&offset.to_le_bytes())?;
        offset += aligned_size(entry.data.len() * 4) as u64;
    }
    Ok(())
}

/// Write aligned tensor data after the metadata section.
///
/// # Parameters
///
/// - `writer`: seekable output writer.
/// - `entries`: tensor entries whose data will be serialized.
///
/// # Errors
///
/// Returns an error on any write failure.
fn write_tensor_data<W: Write + Seek>(writer: &mut W, entries: &[TensorEntry]) -> Result<()> {
    let position = writer.stream_position()? as usize;
    let padding = aligned_size(position) - position;
    for _ in 0..padding {
        writer.write_all(&[0u8])?;
    }

    for entry in entries {
        for value in &entry.data {
            writer.write_all(&value.to_le_bytes())?;
        }
        let written = entry.data.len() * 4;
        let padding = aligned_size(written) - written;
        for _ in 0..padding {
            writer.write_all(&[0u8])?;
        }
    }
    Ok(())
}

/// Round a byte size up to the GGUF alignment boundary.
///
/// # Parameters
///
/// - `size`: raw byte count.
///
/// # Returns
///
/// The next multiple of `ALIGNMENT` at or after `size`.
fn aligned_size(size: usize) -> usize {
    size.div_ceil(ALIGNMENT) * ALIGNMENT
}

/// Write a string typed GGUF metadata key-value pair.
///
/// # Parameters
///
/// - `writer`: output writer.
/// - `key`: metadata key.
/// - `value`: string value.
///
/// # Errors
///
/// Returns an error on any write failure.
fn write_string_value<W: Write>(writer: &mut W, key: &str, value: &str) -> Result<()> {
    write_string(writer, key)?;
    writer.write_all(&TYPE_STRING.to_le_bytes())?;
    write_string(writer, value)?;
    Ok(())
}

/// Write a uint32 typed GGUF metadata key-value pair.
///
/// # Parameters
///
/// - `writer`: output writer.
/// - `key`: metadata key.
/// - `value`: uint32 value.
///
/// # Errors
///
/// Returns an error on any write failure.
fn write_u32_value<W: Write>(writer: &mut W, key: &str, value: u32) -> Result<()> {
    write_string(writer, key)?;
    writer.write_all(&TYPE_UINT32.to_le_bytes())?;
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

/// Write a float32 typed GGUF metadata key-value pair.
///
/// # Parameters
///
/// - `writer`: output writer.
/// - `key`: metadata key.
/// - `value`: float32 value.
///
/// # Errors
///
/// Returns an error on any write failure.
fn write_f32_value<W: Write>(writer: &mut W, key: &str, value: f32) -> Result<()> {
    write_string(writer, key)?;
    writer.write_all(&TYPE_FLOAT32.to_le_bytes())?;
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

/// Write a boolean typed GGUF metadata key-value pair.
///
/// # Parameters
///
/// - `writer`: output writer.
/// - `key`: metadata key.
/// - `value`: boolean value.
///
/// # Errors
///
/// Returns an error on any write failure.
fn write_bool_value<W: Write>(writer: &mut W, key: &str, value: bool) -> Result<()> {
    write_string(writer, key)?;
    writer.write_all(&TYPE_BOOL.to_le_bytes())?;
    writer.write_all(&[u8::from(value)])?;
    Ok(())
}

/// Write a string array typed GGUF metadata key-value pair.
///
/// # Parameters
///
/// - `writer`: output writer.
/// - `key`: metadata key.
/// - `values`: array of string values.
///
/// # Errors
///
/// Returns an error on any write failure.
fn write_string_array<W: Write>(writer: &mut W, key: &str, values: &[String]) -> Result<()> {
    write_string(writer, key)?;
    writer.write_all(&TYPE_ARRAY.to_le_bytes())?;
    writer.write_all(&TYPE_STRING.to_le_bytes())?;
    writer.write_all(&(values.len() as u64).to_le_bytes())?;
    for value in values {
        write_string(writer, value)?;
    }
    Ok(())
}

/// Write a length-prefixed string in GGUF encoding.
///
/// # Parameters
///
/// - `writer`: output writer.
/// - `value`: string to write.
///
/// # Errors
///
/// Returns an error on any write failure.
fn write_string<W: Write>(writer: &mut W, value: &str) -> Result<()> {
    writer.write_all(&(value.len() as u64).to_le_bytes())?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}
