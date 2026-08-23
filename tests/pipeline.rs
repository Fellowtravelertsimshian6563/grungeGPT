use candle_core::Device;
use grungegpt::dataset::Dataset;
use grungegpt::model::{GptConfig, load_checkpoint, save_checkpoint};
use grungegpt::sampler::{GenerateConfig, generate};
use grungegpt::tokenizer::Bpe;
use grungegpt::trainer::{TrainConfig, train_model};

fn sample_texts() -> Vec<String> {
    vec![
        "bones in the river\nmud on the floor".to_string(),
        "sirens in the distance\nsmoke across the sky".to_string(),
        "dust on the window\nrain on the glass".to_string(),
    ]
}

#[test]
fn trains_saves_loads_and_generates() {
    let texts = sample_texts();
    let tokenizer = Bpe::train(&texts, 16);
    let dataset = Dataset::from_texts(&texts, &tokenizer, 8).unwrap();
    let config = GptConfig::new(tokenizer.vocab_size(), 8, 1, 8, 2, 0.0);
    let device = Device::Cpu;
    let train_config = TrainConfig {
        batch_size: 2,
        steps: 3,
        learning_rate: 0.01,
        eval_every: 2,
        seed: 42,
    };

    let (varmap, _model) = train_model(&dataset, &config, &train_config, &device, None).unwrap();

    let out_dir = std::env::temp_dir().join(format!(
        "grungegpt-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    save_checkpoint(&varmap, &config, &out_dir).unwrap();

    let model = load_checkpoint(
        &out_dir.join("config.json"),
        &out_dir.join("model.safetensors"),
        &device,
    )
    .unwrap();
    let generate_config = GenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: None,
        seed: 1,
    };
    let text = generate(&model, &tokenizer, "bones", &generate_config, &device).unwrap();
    assert!(!text.is_empty());

    let _ = std::fs::remove_dir_all(&out_dir);
}
