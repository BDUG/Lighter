//! Exhaustive tour of native quantization, paged KV caching, reinforcement
//! learning objectives, and teacher/student distillation.

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::native_advanced::*;

    // Quantize a row-major projection and execute it without dequantizing the
    // complete matrix. Int8 is also available via QuantizationType::Int8.
    let weights = [1.0, -1.0, 0.5, 0.25];
    let projection = QuantizedMatrix::quantize(
        2,
        2,
        &weights,
        QuantizationConfig {
            dtype: QuantizationType::Int4,
            group_size: 2,
        },
    )?;
    println!(
        "quantized projection: {:?}",
        projection.matvec(&[2.0, 1.0])?
    );

    // Pages are shared across forks until one branch writes to its tail. This
    // is suitable for prompt-prefix reuse and beam-search branches.
    let mut cache = PagedKvCache::new(16, 2)?;
    cache.append("prompt", &[1.0, 2.0], &[3.0, 4.0])?;
    cache.fork("prompt", "beam-2")?;
    cache.append("beam-2", &[5.0, 6.0], &[7.0, 8.0])?;
    println!("KV cache: {:?}", cache.stats());

    // Compute terminal-aware GAE, followed by the clipped PPO objective.
    let (advantages, returns) =
        generalized_advantage_estimate(&[1.0, 0.5], &[0.2, 0.3, 0.0], &[false, true], 0.99, 0.95)?;
    let ppo = ppo_objective(
        &PpoBatch {
            old_log_probs: vec![-0.8, -0.7],
            new_log_probs: vec![-0.75, -0.8],
            advantages,
            old_values: vec![0.2, 0.3],
            new_values: vec![0.25, 0.35],
            returns,
            entropies: vec![0.5, 0.4],
        },
        &PpoConfig::default(),
    )?;
    println!("PPO loss: {}", ppo.loss);

    // DPO consumes policy-vs-reference log-ratios for chosen/rejected answers.
    let (dpo, dpo_gradients) = dpo_loss(&[0.4, 0.3], &[-0.2, -0.1], 0.1, 0.0)?;
    println!("DPO loss: {dpo}, gradients: {dpo_gradients:?}");

    // Distillation combines temperature-scaled teacher KL with an optional
    // hard-label cross entropy term and returns student-logit gradients.
    let distilled = distillation_loss(
        &[4.0, 1.0, 0.0],
        &[1.0, 2.0, 0.0],
        Some(0),
        &DistillationConfig::default(),
    )?;
    println!("distillation loss: {}", distilled.loss);
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("enable this example with --no-default-features --features native");
}
