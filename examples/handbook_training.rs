//! Explicit gradient accumulation, optimizer hooks, token loss and cache operations.
#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::native::NativeResult;
    use candlelighter::native_advanced::*;
    use candlelighter::native_training::*;

    let mut adapter = LoraAdapter::new(
        3,
        2,
        LoraConfig {
            rank: 2,
            alpha: 4.,
            dropout: 0.,
        },
        42,
    )?;
    let inputs = [[1f32, 0.5, -0.5], [0.5, 1., 0.]];
    let before = adapter.forward(&inputs[0], false, 42)?;
    let mut accumulated = adapter.zero_gradients();
    for input in &inputs {
        let (gradient, input_gradient) = adapter.backward(input, &[0.2, -0.1])?;
        assert_eq!(input_gradient.len(), 3);
        LoraAdapter::accumulate(&mut accumulated, &gradient)?;
    }
    for gradient in accumulated.a.iter_mut().chain(&mut accumulated.b) {
        *gradient /= inputs.len() as f32;
    }
    let mut optimizer = AdamW::new(AdamWConfig {
        learning_rate: 1e-3,
        max_gradient_norm: Some(0.5),
        ..Default::default()
    })?;
    adapter.apply_gradients(&accumulated, &mut optimizer)?;
    assert_ne!(before, adapter.forward(&inputs[0], false, 42)?);
    adapter.apply_sgd(&accumulated, 1e-3)?;
    let (a, b) = adapter.weights();
    let restored = LoraAdapter::from_weights(
        3,
        2,
        LoraConfig {
            rank: 2,
            alpha: 4.,
            dropout: 0.,
        },
        a.to_vec(),
        b.to_vec(),
    )?;
    assert_eq!(
        adapter.forward(&inputs[0], false, 1)?,
        restored.forward(&inputs[0], false, 1)?
    );

    let loss = causal_lm_loss(
        &[vec![2., 0., -1.], vec![0., 2., -1.]],
        &[0, -100],
        -100,
        0.1,
    )?;
    assert!(loss.loss.is_finite());
    assert_eq!(&loss.gradients[3..], &[0., 0., 0.]);
    let mut reward = RewardHead::new(3)?;
    assert!(reward
        .apply_pairwise_update(&[1., 0., 0.], &[0., 0., 1.], &mut optimizer)?
        .is_finite());
    assert!(reward.score(&[1., 0., 0.])?.is_finite());

    // These parameters represent objective outputs, not transformer weights.
    // A real backend must propagate output gradients through its model.
    struct OutputParameters(Vec<f32>);
    impl OptimizationTarget for OutputParameters {
        fn apply_output_gradients(&mut self, gradient: &[f32], rate: f32) -> NativeResult<()> {
            assert_eq!(self.0.len(), gradient.len());
            for (parameter, derivative) in self.0.iter_mut().zip(gradient) {
                *parameter -= rate * derivative;
            }
            Ok(())
        }
    }
    let (advantages, returns) =
        generalized_advantage_estimate(&[1., 0.5], &[0.2, 0.3, 0.], &[false, true], 0.99, 0.95)?;
    let batch = PpoBatch {
        old_log_probs: vec![-0.8, -0.7],
        new_log_probs: vec![-0.75, -0.8],
        advantages,
        old_values: vec![0.2, 0.3],
        new_values: vec![0.25, 0.35],
        returns,
        entropies: vec![0.5, 0.4],
    };
    let mut policy = OutputParameters(batch.new_log_probs.clone());
    let ppo = apply_ppo_update(&mut policy, &batch, &PpoConfig::default(), 0.01)?;
    assert!(ppo.loss.is_finite());
    let mut preference = OutputParameters(vec![0.4, 0.3]);
    assert!(
        apply_dpo_update(&mut preference, &[0.4, 0.3], &[-0.2, -0.1], 0.1, 0., 0.01)?.is_finite()
    );
    let mut student = OutputParameters(vec![1., 2., 0.]);
    let distilled = apply_distillation_update(
        &mut student,
        &[4., 1., 0.],
        &[1., 2., 0.],
        Some(0),
        &DistillationConfig::default(),
        0.01,
    )?;
    assert!(distilled.loss.is_finite());

    for dtype in [QuantizationType::Int8, QuantizationType::Int4] {
        let matrix = QuantizedMatrix::quantize(
            2,
            2,
            &[1., -1., 0.5, 0.25],
            QuantizationConfig {
                dtype,
                group_size: 2,
            },
        )?;
        assert_eq!(matrix.dequantize().len(), 4);
        assert_eq!(matrix.matvec(&[2., 1.])?.len(), 2);
        println!(
            "quantized matrix: {}x{}, {} bytes",
            matrix.rows(),
            matrix.cols(),
            matrix.storage_bytes()
        );
    }
    let mut cache = PagedKvCache::new(16, 2)?;
    cache.append("prompt", &[1., 2.], &[3., 4.])?;
    cache.fork("prompt", "branch")?;
    cache.append("branch", &[5., 6.], &[7., 8.])?;
    cache.truncate_left("branch", 1)?;
    assert_eq!(cache.tokens("prompt"), 1);
    assert_eq!(cache.read("branch")?, (vec![5., 6.], vec![7., 8.]));
    assert!(cache.remove("branch"));
    println!("gradient accumulation, loss/update hooks, quantization and cache checks completed");
    Ok(())
}
#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("enable the native feature");
}
