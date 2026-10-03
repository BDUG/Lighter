//! The three common VecAttention examples: global self-attention, local point
//! attention, and masked cross/sequence attention.
//! Inspired by <https://github.com/anminliu/VecAttention>.

use candlelighter::vec_attention::{causal_mask, Neighborhood, VecAttention, VecAttentionConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let features = vec![
        vec![1.0, 0.0, 0.5, 0.2],
        vec![0.0, 1.0, 0.5, 0.4],
        vec![0.8, 0.2, 0.0, 1.0],
        vec![0.1, 0.9, 1.0, 0.0],
    ];
    let points = vec![
        vec![0., 0., 0.],
        vec![1., 0., 0.],
        vec![0., 1., 0.],
        vec![3., 3., 0.],
    ];

    let mut config = VecAttentionConfig::new(4, 8);
    config.position_dim = 3;
    config.share_planes = 2;
    config.seed = 7;

    println!("Global vector self-attention:");
    let global = VecAttention::new(config.clone())?;
    let result = global.forward(&features, &points, None)?;
    for (i, row) in result.values.iter().enumerate() {
        println!("  point {i}: {row:.4?}");
    }

    println!("\nLocal point attention (2 nearest neighbours):");
    config.neighborhood = Neighborhood::KNearest(2);
    let local = VecAttention::new(config.clone())?;
    let result = local.forward(&features, &points, None)?;
    for (i, weights) in result.weights.iter().enumerate() {
        let active: Vec<_> = weights
            .iter()
            .enumerate()
            .filter(|(_, w)| w[0] > 0.0)
            .map(|(j, _)| j)
            .collect();
        println!("  point {i} attends to {active:?}");
    }

    println!("\nCausal vector attention:");
    config.neighborhood = Neighborhood::Global;
    let sequence = VecAttention::new(config)?;
    let result = sequence.forward(&features, &points, Some(&causal_mask(features.len())))?;
    for (i, weights) in result.weights.iter().enumerate() {
        let sum: f32 = weights.iter().map(|w| w[0]).sum();
        println!("  token {i}: visible={}, weight sum={sum:.3}", i + 1);
    }
    Ok(())
}
