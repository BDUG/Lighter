# Lighter context-parallel attention usage book

This book documents Lighter's backend-independent exact context-parallel
attention API. It includes tensor shapes, examples, and an explicit next step
for every capability.

## Quick start

```toml
[dependencies]
candlelighter = { version = "0.4", default-features = false }
```

```bash
cargo run --example context_parallel_attention --no-default-features
```

Use `context_parallel` when a Q/K/V attention cache must be split across
workers without approximating the softmax. It is a correctness-oriented CPU
reference built on nested Rust vectors, not a hardware-specific fused kernel.

## Tensor conventions

| API | Shape |
|---|---|
| Distributed queries | `[query_head][head_dim]` |
| Sharded keys | `[local_token][kv_head][head_dim]` |
| Sharded values | `[local_token][kv_head][value_dim]` |
| Prefill output | `[query][query_head][value_dim]` |

All inference calls return `Result`; callers should propagate shape and
configuration failures rather than substituting an empty attention result.

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

# Validation and errors

The API rejects zero dimensions, ragged Q/K/V, non-finite scales, non-positive
soft caps, incompatible GQA head counts, invalid partition coverage, and
attention with no visible keys. Handle `ContextParallelError` at the
application boundary.

# Research references

- [arXiv:2402.01763](https://arxiv.org/pdf/2402.01763): blockwise and
  context-partitioned attention foundations.
- [vLLM Decode Context Parallelism](https://vllm.ai/blog/2026-08-07-decode-context-parallelism):
  decode-time sharded KV ownership and rank-local attention reduction.
- [arXiv:2505.02922v3](https://arxiv.org/html/2505.02922v3): balanced
  context-parallel execution and prefill considerations.
- [arXiv:2508.02184v2](https://arxiv.org/html/2508.02184v2): long-context
  inference and context-parallel decode considerations.

These are algorithmic references. The portable Rust implementation does not
claim to reproduce a particular CUDA kernel, network stack, or benchmark
result.

# Capability roadmap

| Capability | Now | Next |
|---|---:|---|
| Context layout and KV sharding | Yes | Dynamic topology-aware paged cache |
| Stable exact distributed softmax | Yes | Collective transport backend |
| DCP with MHA/GQA/MQA | Yes | Multi-process GPU and tensor parallelism |
| Sliding window / soft cap | Yes | Sinks, global tokens and ALiBi |
| Causal prefill reference | Yes | Fused ring/blockwise prefill |
| f32 CPU reference | Yes | f16/bf16/int8 accelerator kernels |
