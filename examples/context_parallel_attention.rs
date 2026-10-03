//! Exact decode-context parallel attention on three simulated workers.
use candlelighter::context_parallel::{
    decode_context_parallel, shard_kv, ContextPlan, DecodeConfig, PartitionStrategy,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let keys: Vec<_> = (0..12).map(|i| vec![vec![i as f32 / 12.0, 1.0]]).collect();
    let values: Vec<_> = (0..12)
        .map(|i| vec![vec![i as f32, (i * i) as f32]])
        .collect();
    let plan = ContextPlan::new(12, 3, PartitionStrategy::Contiguous)?;
    let shards = shard_kv(&keys, &values, &plan)?;
    println!("DCP assignments: {:?}", plan.assignments);

    // One decode token; KV cache remains distributed and only three small
    // online-softmax states need to be reduced.
    let query = vec![vec![0.5, 0.75]];
    let output = decode_context_parallel(&query, 11, &shards, DecodeConfig::for_head_dim(2))?;
    println!("decode output: {:.4?}", output[0]);
    Ok(())
}
