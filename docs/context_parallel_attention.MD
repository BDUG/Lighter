# Context-parallel attention

> For complete examples and a capability-by-capability roadmap, read the
> [Lighter attention usage book](attention_usage_book.MD). This page remains a
> concise context-parallel architecture overview.

Lighter provides backend-independent building blocks for exact, distributed
long-context attention in `context_parallel`. The implementation covers the
common algorithmic scope of the following resources:

- [arXiv:2402.01763](https://arxiv.org/pdf/2402.01763): context partitioning and exact blockwise/online softmax composition.
- [vLLM decode context parallelism](https://vllm.ai/blog/2026-08-07-decode-context-parallelism): keep the paged KV cache sharded during decode, compute rank-local attention, and reduce only `(max, sum, weighted value)` state.
- [arXiv:2505.02922v3](https://arxiv.org/html/2505.02922v3): balanced context layouts and context-parallel prefill.
- [arXiv:2508.02184v2](https://arxiv.org/html/2508.02184v2): context-parallel inference with grouped-query/multi-query attention and long-context decode.

The module is a portable reference rather than a networking runtime. Each
`KvShard` represents a rank's local paged-KV payload. `local_decode` performs
only local QK and value work. `reduce_decode` implements the associative,
numerically stable reduction that a production backend maps to collectives.
`decode_context_parallel` runs both stages in-process and is useful for CPU
inference, correctness testing, and backend development.

## Features

- contiguous decode partitioning, striped prefill partitioning, and blockwise
  zig-zag layouts for balancing triangular causal work;
- exact online softmax without gathering the KV cache;
- causal masking by absolute position, optional sliding windows and logit soft
  caps;
- multi-head, grouped-query, and multi-query attention;
- different key and value dimensions;
- causal prefill over a batch of query positions;
- serializable configuration and KV shards, plus strict ragged-input checks.

Run the standalone demonstration:

```bash
cargo run --example context_parallel_attention --no-default-features
```

For a distributed backend, execute `local_decode` on every owner rank and
combine the returned `SoftmaxState`s using the formula in `merge`. Do not
average rank-local normalized outputs: softmax denominators differ by rank and
that shortcut is not exact.
