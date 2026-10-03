# Lighter attention usage book

This book documents Lighter's backend-independent vector-valued attention and
exact context-parallel attention APIs. It includes tensor shapes, an example
and an explicit next step for every capability.

## Quick start

```toml
[dependencies]
candlelighter = { version = "0.4", default-features = false }
```

```bash
cargo run --example vec_attention --no-default-features
cargo run --example context_parallel_attention --no-default-features
```

Use `VecAttention` when attention needs one learned weight per channel group,
particularly for sets and point clouds. Use `context_parallel` when a normal
Q/K/V attention cache must be split across workers without approximating the
softmax. Both are correctness-oriented CPU references built on nested Rust
vectors, not hardware-specific fused kernels.

## Tensor conventions

| API | Shape |
|---|---|
| Vector features | `[item][input_dim]` |
| Vector positions | `[item][position_dim]` |
| Vector pair mask | `[query][key]` (`false` blocks a pair) |
| Vector weights | `[query][key][hidden_dim / share_planes]` |
| Distributed queries | `[query_head][head_dim]` |
| Sharded keys | `[local_token][kv_head][head_dim]` |
| Sharded values | `[local_token][kv_head][value_dim]` |
| Prefill output | `[query][query_head][value_dim]` |

All inference calls return `Result`; callers should propagate shape and
configuration failures rather than substituting an empty attention result.

# Part I: vector attention

Vector attention implements the point-transformer-style relation

```text
y_i = sum_j softmax_j(gamma(q_i - k_j + delta_ij)) * (v_j + delta_ij)
```

where softmax is independently normalized across visible keys for each channel
group.

## Global self-attention

**Capability.** Every item attends to every item. The output width defaults to
the input width and can be changed through `output_dim`.

```rust
use candlelighter::vec_attention::{VecAttention, VecAttentionConfig};
let x = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
let p = vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
let layer = VecAttention::new(VecAttentionConfig::new(2, 8))?;
let result = layer.forward(&x, &p, None)?;
assert_eq!(result.values.len(), 2);
# Ok::<(), Box<dyn std::error::Error>>(())
```

**Next:** add Candle/native tensor adapters and fused Q/K/V projections to
avoid row-by-row allocation.

## Local k-nearest attention

**Capability.** `KNearest(k)` selects keys by squared Euclidean position before
attention. Equal distances are resolved by key index.

```rust
use candlelighter::vec_attention::{Neighborhood, VecAttention, VecAttentionConfig};
let mut c = VecAttentionConfig::new(2, 8);
c.position_dim = 2;
c.neighborhood = Neighborhood::KNearest(2);
let layer = VecAttention::new(c)?;
let x = vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]];
let p = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![9.0, 9.0]];
let result = layer.forward(&x, &p, None)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

**Next:** accept caller-supplied graph edges and spatial/GPU k-NN providers
instead of sorting all candidate points.

## Grouped vector weights

**Capability.** Adjacent hidden channels share a logit when `share_planes > 1`.
The value must divide `hidden_dim`.

```rust
use candlelighter::vec_attention::{VecAttention, VecAttentionConfig};
let mut c = VecAttentionConfig::new(16, 64);
c.share_planes = 8; // eight independently normalized groups
let layer = VecAttention::new(c)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

**Next:** learned, non-contiguous and per-head channel grouping.

## Relative-position encoding

**Capability.** Query-minus-key coordinates pass through a learned two-layer
projection. Its result affects relation logits and values. `position_dim` can
represent 1D time, 2D images, 3D points or custom coordinates.

```rust
use candlelighter::vec_attention::{VecAttention, VecAttentionConfig};
let mut c = VecAttentionConfig::new(4, 8);
c.position_dim = 2;
let image_attention = VecAttention::new(c)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

**Next:** pluggable rotary, bucketed and Fourier position encoders.

## Pair and causal masks

**Capability.** Arbitrary boolean pair masks and a lower-triangular causal mask
are supported.

```rust
use candlelighter::vec_attention::{causal_mask, VecAttention, VecAttentionConfig};
let x = vec![vec![1.0], vec![2.0], vec![3.0]];
let p = vec![vec![0.0], vec![1.0], vec![2.0]];
let mut c = VecAttentionConfig::new(1, 4); c.position_dim = 1;
let output = VecAttention::new(c)?.forward(&x, &p, Some(&causal_mask(3)))?;
assert_eq!(output.weights[0][2][0], 0.0);
# Ok::<(), Box<dyn std::error::Error>>(())
```

**Next:** bit-packed and block-sparse masks that avoid quadratic boolean
storage.

## Cross-attention

**Capability.** Queries may attend to a separate key/value feature set.

```rust
use candlelighter::vec_attention::{VecAttention, VecAttentionConfig};
let layer = VecAttention::new(VecAttentionConfig::new(2, 4))?;
let q = vec![vec![1.0, 0.0]]; let qp = vec![vec![0.0, 0.0, 0.0]];
let memory = vec![vec![0.0, 1.0], vec![1.0, 1.0]];
let mp = vec![vec![1.0, 0.0, 0.0], vec![2.0, 0.0, 0.0]];
let result = layer.forward_cross(&q, &qp, &memory, &mp, None)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

**Next:** separate key and value inputs with independent query/KV widths.

## Variable-size batches

**Capability.** `forward_batch` independently evaluates sets of different
lengths.

```rust
use candlelighter::vec_attention::{VecAttention, VecAttentionConfig};
let layer = VecAttention::new(VecAttentionConfig::new(1, 2))?;
let x = vec![vec![vec![1.0]], vec![vec![2.0], vec![3.0]]];
let p = vec![vec![vec![0.0,0.0,0.0]], vec![vec![0.0,0.0,0.0],vec![1.0,0.0,0.0]]];
assert_eq!(layer.forward_batch(&x, &p)?[1].len(), 2);
# Ok::<(), Box<dyn std::error::Error>>(())
```

**Next:** packed offset-based batches, per-item masks and parallel execution.

## Reproducibility and persistence

**Capability.** `seed` controls deterministic initialization. The complete
layer and projections implement Serde.

```rust
use candlelighter::vec_attention::{VecAttention, VecAttentionConfig};
let layer = VecAttention::new(VecAttentionConfig::new(4, 8))?;
let json = serde_json::to_string(&layer)?;
let restored: VecAttention = serde_json::from_str(&json)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

**Next:** safetensors import/export and autograd/optimizer integration. The
current parameters are public but the layer does not train itself.

# Part II: context-parallel attention

Keys and values remain on their owner ranks. Each rank calculates a stable
`(maximum, exponential sum, weighted value)` state. Associatively merging rank
states yields the exact dense result without gathering the KV cache. The
reference simulates ranks in-process; it does not provide networking.

## Context layouts

**Capability.** Contiguous ownership favors paged decode locality, striped
ownership round-robins tokens, and blockwise zig-zag ownership balances
triangular causal work.

```rust
use candlelighter::context_parallel::{ContextPlan, PartitionStrategy};
let decode = ContextPlan::new(1_000, 4, PartitionStrategy::Contiguous)?;
let prefill = ContextPlan::new(1_000, 4, PartitionStrategy::Striped)?;
let balanced = ContextPlan::new(1_000, 4, PartitionStrategy::ZigZag { block_size: 16 })?;
assert_eq!(decode.owner(0), Some(0));
# Ok::<(), Box<dyn std::error::Error>>(())
```

**Next:** topology-, page- and load-aware planning with live request migration.

## KV-cache sharding

**Capability.** `shard_kv` converts dense K/V tensors into checked rank-local
`KvShard`s.

```rust
use candlelighter::context_parallel::{shard_kv, ContextPlan, PartitionStrategy};
let k = vec![vec![vec![1.0,0.0]], vec![vec![0.0,1.0]]];
let v = vec![vec![vec![10.0]], vec![vec![20.0]]];
let plan = ContextPlan::new(2, 2, PartitionStrategy::Contiguous)?;
let shards = shard_kv(&k, &v, &plan)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

**Next:** direct integration with paged, shared-prefix, migrated and quantized
KV-cache pages in the native runtime.

## Rank-local decode

**Capability.** `local_decode` performs only local QK/value work, enforces
absolute-position causality and returns an unnormalized reduction payload.

```rust
use candlelighter::context_parallel::{local_decode, DecodeConfig, KvShard};
let shard = KvShard { rank: 0, positions: vec![0], keys: vec![vec![vec![1.0,0.0]]], values: vec![vec![vec![3.0]]] };
let state = local_decode(&[vec![1.0,0.0]], 0, &shard, DecodeConfig::for_head_dim(2))?;
assert_eq!(state.visited_tokens, 1);
# Ok::<(), Box<dyn std::error::Error>>(())
```

**Next:** fused SIMD/CUDA/Metal/ROCm kernels and communication overlap.

## Exact online-softmax reduction

**Capability.** `push` remains stable for large logits and `merge` is
associative, so a backend may use ring, tree or all-reduce collectives.

```rust
use candlelighter::context_parallel::SoftmaxState;
let mut a = SoftmaxState::empty(1); a.push(10_000.0, &[2.0])?;
let mut b = SoftmaxState::empty(1); b.push(9_999.0, &[4.0])?;
a.merge(&b)?; let exact = a.finish()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Never average normalized rank-local outputs: their softmax denominators differ.

**Next:** collective transport traits and low-precision wire formats with
explicit error bounds.

## End-to-end decode context parallelism

**Capability.** `decode_context_parallel` computes and merges all rank states.

```rust
use candlelighter::context_parallel::{decode_context_parallel, shard_kv, ContextPlan, DecodeConfig, PartitionStrategy};
let k = vec![vec![vec![1.0,0.0]], vec![vec![0.0,1.0]]];
let v = vec![vec![vec![1.0]], vec![vec![3.0]]];
let shards = shard_kv(&k, &v, &ContextPlan::new(2,2,PartitionStrategy::Contiguous)?)?;
let output = decode_context_parallel(&[vec![0.5,0.5]], 1, &shards, DecodeConfig::for_head_dim(2))?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

**Next:** rank discovery, NCCL/Gloo transport, failure propagation and
multi-node scheduling.

## MHA, GQA and MQA

**Capability.** Query heads may equal or be an integer multiple of KV heads.
Query head `qh` maps to `qh * kv_heads / query_heads`, covering MHA, GQA and a
single-KV-head MQA cache.

```rust
let queries = vec![vec![0.1, 0.2]; 4]; // four Q heads
let keys_for_one_token = vec![vec![0.2, 0.1]; 2]; // two KV heads: GQA
```

**Next:** combined tensor/context parallel ownership and explicit non-uniform
head maps.

## Sliding windows and logit soft caps

**Capability.** Sliding windows omit old absolute positions; a soft cap applies
`cap * tanh(logit / cap)` before softmax.

```rust
use candlelighter::context_parallel::DecodeConfig;
let mut config = DecodeConfig::for_head_dim(128);
config.sliding_window = Some(4_096);
config.soft_cap = Some(50.0);
```

**Next:** attention sinks, global/prefix tokens, ALiBi and composable masks.

## Causal prefill

**Capability.** `causal_prefill` evaluates multiple query positions against one
sharded cache and returns `[query][query_head][value_dim]`.

```rust
use candlelighter::context_parallel::{causal_prefill, DecodeConfig, KvShard};
let queries = vec![vec![vec![0.5,0.5]], vec![vec![0.4,0.6]]];
// causal_prefill(&queries, &[10,11], &populated_shards,
//     DecodeConfig::for_head_dim(2))?;
```

**Next:** fused triangular tiles, sequence packing and blockwise/ring prefill
instead of invoking the reference decode path per query.

## Independent value widths and serialization

**Capability.** Key width equals query width, while value width may differ.
`DecodeConfig` and `KvShard` implement Serde.

```rust
use candlelighter::context_parallel::DecodeConfig;
let json = serde_json::to_string(&DecodeConfig::for_head_dim(64))?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

**Next:** versioned wire schemas, zero-copy views and f16/bf16/int8 storage.

# Combining the APIs

The modules solve different problems. `VecAttention` owns learned projections
and channel-wise relation weights. `context_parallel` distributes conventional
scalar-softmax Q/K/V attention. A model can use local vector-attention blocks
to encode geometry, project those outputs to Q/K/V, and use context parallelism
for long-range prefill and decode.

**Next:** generalize distributed reduction to vector-valued group logits so a
single `VecAttention` operation can itself span ranks.

# Validation and errors

The APIs reject zero dimensions, invalid grouping, `KNearest(0)`, empty sets,
malformed masks, feature/position width mismatches, ragged Q/K/V, non-finite
scales, non-positive soft caps, incompatible GQA head counts, invalid partition
coverage, and attention with no visible keys. Handle `VecAttentionError` and
`ContextParallelError` at the application boundary.

# Research references

- [VecAttention reference repository](https://github.com/anminliu/VecAttention).
- [arXiv:2402.01763](https://arxiv.org/pdf/2402.01763): blockwise and
  context-partitioned attention foundations.
- [vLLM Decode Context Parallelism](https://vllm.ai/blog/2026-08-07-decode-context-parallelism):
  decode-time sharded KV ownership and rank-local attention reduction.
- [arXiv:2505.02922v3](https://arxiv.org/html/2505.02922v3): balanced
  context-parallel execution and prefill considerations.
- [arXiv:2508.02184v2](https://arxiv.org/html/2508.02184v2): long-context
  inference and context-parallel decode considerations.

These are algorithmic references. The portable Rust implementation does not
claim to reproduce a particular CUDA kernel, network stack or benchmark result.

# Capability roadmap

| Capability | Now | Next |
|---|---:|---|
| Global/local vector attention | Yes | Fused kernels and spatial indices |
| Group weights / relative positions | Yes | Learned groups and custom encoders |
| Pair/causal masks and cross-attention | Yes | Sparse masks and independent K/V |
| Variable batches / JSON persistence | Yes | Packed batches and safetensors |
| Context layout and KV sharding | Yes | Dynamic topology-aware paged cache |
| Stable exact distributed softmax | Yes | Collective transport backend |
| DCP with MHA/GQA/MQA | Yes | Multi-process GPU and tensor parallelism |
| Sliding window / soft cap | Yes | Sinks, global tokens and ALiBi |
| Causal prefill reference | Yes | Fused ring/blockwise prefill |
| f32 CPU reference | Yes | f16/bf16/int8 accelerator kernels |
