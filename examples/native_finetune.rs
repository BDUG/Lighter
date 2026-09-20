//! Supervised fine-tuning, LoRA, AdamW, and reward-model example.

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use candlelighter::native::{NativeError, NativeResult};
    use candlelighter::native_training::*;
    use candlelighter::{native::HuggingFaceArtifacts, NativeLlama};

    // With a snapshot argument, run a real LM-head LoRA update through the
    // native transformer. Without one, continue with the small inspectable
    // examples below.
    if let Some(directory) = std::env::args().nth(1) {
        let artifacts = HuggingFaceArtifacts::from_dir(directory)?;
        let tokenizer = tokenizers::Tokenizer::from_file(&artifacts.tokenizer)?;
        let encoded = tokenizer.encode("Fine-tuning a native transformer", true)?;
        let ids = encoded.get_ids();
        if ids.len() < 2 {
            return Err("tokenized fine-tuning text must contain two tokens".into());
        }
        let mut model = NativeLlama::load(&artifacts)?;
        model.enable_lm_head_lora(
            LoraConfig {
                rank: 8,
                alpha: 16.0,
                dropout: 0.0,
            },
            42,
        )?;
        let labels: Vec<i64> = ids[1..].iter().map(|id| i64::from(*id)).collect();
        let loss =
            supervised_fine_tune_step(&mut model, &ids[..ids.len() - 1], &labels, -100, 0.0, 1e-4)?;
        println!("NativeLlama LM-head LoRA step loss={loss}");
        return Ok(());
    }

    let mut adapter = LoraAdapter::new(
        3,
        2,
        LoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.05,
        },
        42,
    )?;
    let input = [1.0, 0.5, -0.5];
    let delta = adapter.forward(&input, true, 7)?;
    let (gradients, input_gradient) = adapter.backward(&input, &[0.2, -0.1])?;
    let mut optimizer = AdamW::new(AdamWConfig::default())?;
    adapter.apply_gradients(&gradients, &mut optimizer)?;
    println!("LoRA delta={delta:?}, input gradient={input_gradient:?}");

    struct TinyStudent {
        logits: Vec<Vec<f32>>,
        last_gradient: Vec<f32>,
    }
    impl FineTunableTransformer for TinyStudent {
        fn token_logits(&mut self, _: &[u32]) -> NativeResult<Vec<Vec<f32>>> {
            Ok(self.logits.clone())
        }
        fn apply_token_gradients(
            &mut self,
            gradients: &[f32],
            learning_rate: f32,
        ) -> NativeResult<()> {
            if gradients.iter().any(|value| !value.is_finite()) {
                return Err(NativeError("non-finite training gradient".into()));
            }
            self.last_gradient = gradients
                .iter()
                .map(|value| value * learning_rate)
                .collect();
            Ok(())
        }
    }
    let mut student = TinyStudent {
        logits: vec![vec![2.0, 0.0, -1.0], vec![0.0, 2.0, -1.0]],
        last_gradient: vec![],
    };
    let loss = supervised_fine_tune_step(&mut student, &[10, 11], &[0, 1], -100, 0.1, 1e-4)?;
    println!(
        "SFT loss={loss}, gradient elements={}",
        student.last_gradient.len()
    );

    let mut reward = RewardHead::new(3)?;
    let pair_loss =
        reward.apply_pairwise_update(&[1.0, 0.5, 0.0], &[0.0, 0.5, 1.0], &mut optimizer)?;
    println!("reward-model pairwise loss={pair_loss}");
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("enable this example with --no-default-features --features native");
}
