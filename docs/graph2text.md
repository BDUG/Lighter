# Graph-to-text

Lighter implements property-graph ingestion, validation, reproducible content planning,
exact realization and optional native language-model realization. The core runs without
Candle, a tokenizer, network access or model weights. It supports directed relations,
parallel edges, cycles, self-loops, isolated entities and arbitrary JSON-valued properties.
This is a graph-grounded generation pipeline; it does not include a pretrained graph
neural encoder or a graph-to-sequence training algorithm.

## Run a complete example

```bash
cargo run --no-default-features --example graph2text
cargo run --no-default-features --example graph2text -- examples/data/graph2text.json /tmp/description.json
```

The second command writes `text` plus a `plan` containing every selected fact and its
provenance ID. Output includes literal values, entity IDs, edge IDs and direction:

```text
Entity "ada" has label "Ada Lovelace".
"Ada Lovelace" ("ada") has "birth_year" = 1815.
Entity "engine" has label "Analytical Engine".
"Analytical Engine" ("engine") has "type" = "mechanical computer".
Entity "babbage" has label "Charles Babbage".
"Charles Babbage" ("babbage") — "designed" → "Analytical Engine" ("engine") [edge "design"].
"Ada Lovelace" ("ada") — "wrote notes about" → "Analytical Engine" ("engine") [edge "notes"; attributes: "year" = 1843].
```

## Input contract

[Complete input fixture](../examples/data/graph2text.json). Each node requires a unique,
nonblank `id` and nonblank `label`. Each edge requires its own unique nonblank `id`,
existing `source` and `target` node IDs, and a nonblank `relation`. `properties` defaults
to an empty object; its keys must be nonblank. JSON scalars, arrays, nested objects and
null values remain literal data. Labels need not be unique. Node and edge IDs occupy
separate namespaces. Unknown fields fail JSON decoding; empty graphs fail validation.

Graph arrays are public for ergonomic construction; every processing entry point validates
them. `Graph::from_json` and `to_json` validate too. `GraphError` implements `Error`.
Property and label strings are JSON-escaped in output so newlines or quotes cannot
break fact boundaries.

## Rust construction, selection and provenance

```rust
use candlelighter::graph2text::{Edge, Graph, Node, Options, plan, verbalize};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ada = Node::new("ada", "Ada Lovelace");
    ada.properties.insert("birth_year".into(), json!(1815));
    let graph = Graph {
        nodes: vec![ada, Node::new("engine", "Analytical Engine")],
        edges: vec![Edge::new("notes", "ada", "wrote notes about", "engine")],
    };
    let options = Options {
        root: Some("ada".into()), radius: Some(1), max_facts: Some(100),
    };
    let content = plan(&graph, &options)?;
    assert!(content.omitted_fact_ids.is_empty());
    println!("{}", verbalize(&graph, &options)?.text);
    let json = graph.to_json()?;
    assert_eq!(Graph::from_json(&json)?, graph);
    Ok(())
}
```

Planning is breadth-first, traversing both incoming and outgoing edges while preserving
original direction in every relation fact. Neighbors and component seeds use ID order;
edges and properties have stable ordering. Reordering input arrays leaves output unchanged.
With no radius, all disconnected components are included, even with a root. A radius
requires a root and includes only its undirected neighborhood; radius zero retains that
node, its properties and self-loops. An edge is selected only if both endpoints are selected.

Each node contributes one identity fact plus one fact per property. Each edge contributes
one relation fact with all edge properties. `max_facts` is a positive fact budget,
not a token budget. Every excluded fact is listed in `omitted_fact_ids`, whether excluded
by neighborhood selection or budget. Budgeting can omit node identity facts while retaining
others; callers needing full coverage should leave it unset and check omissions.

## Knowledge triples

```rust
use candlelighter::graph2text::{Graph, Options, Triple, verbalize};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph = Graph::from_triples(&[
        Triple { subject: "Paris".into(), predicate: "is capital of".into(), object: "France".into() },
        Triple { subject: "France".into(), predicate: "is in".into(), object: "Europe".into() },
    ])?;
    println!("{}", verbalize(&graph, &Options::default())?.text);
    Ok(())
}
```

Triple ingestion merges exact entity labels, allocates deterministic `n0`, `n1`, … IDs
in lexical label order, and assigns `e0`, `e1`, … IDs in triple order. Duplicate triples
retain separate edges. Use explicit property graphs to distinguish equal-label entities.

## Native language-model realization

```bash
cargo run --no-default-features --features native --example graph2text_native -- /path/to/hf-model examples/data/graph2text.json
```

[Complete native example](../examples/graph2text_native.rs) loads local Hugging Face
artifacts once, constructs a native engine and calls `graph2text::generate`. It returns
both the content plan and the runtime's full `GenerateResponse` (text, tokens, usage and
finish reason). Reuse the idle engine for subsequent graphs. The helper rejects busy
engines; runtime failures abort its requests and clear their caches. The sample uses
256 output tokens and greedy decoding. Choose output limits to cover the graph and a
model whose context can accommodate the serialized facts. The native example uses a
plain completion prompt; use the HTTP endpoint with a chat template for instruction models.

`graph2text::prompt` serializes selected facts as JSON and instructs the model to preserve
identity, direction and literal values. Graph strings are untrusted data. Prompt instructions
reduce, but cannot guarantee protection from prompt injection or factual errors. Generated
prose may omit, distort or invent facts. The returned plan records supplied facts, not proof
that generated prose expresses them. Use `verbalize` when exact coverage is required.
No model download is necessary for exact realization; language-model realization needs
compatible weights and tokenizers as described in the [model host guide](model_host.md).

## Hosted graph generation

```bash
cargo run --no-default-features --features server --bin lighter-serve -- --model /path/to/hf-model --chat-template llama3
python3 - <<'PY'
import json, urllib.request
with open("examples/data/graph2text.json") as source:
    graph = json.load(source)
payload = {"model": "lighter", "graph": graph,
           "options": {"root": "ada", "radius": 1},
           "max_tokens": 256, "temperature": 0}
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/graph/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request) as response:
    result = json.load(response)
print(result["choices"][0].get("message", {}).get("content",
      result["choices"][0].get("text", "")))
PY
```

`POST /v1/graph/completions` accepts `graph`, optional `options` and the
[host's completion generation fields](model_host.md). `prompt` and `messages` are
rejected. Invalid graphs/options return 400 before admission. A configured chat template
wraps the graph prompt as a user message and returns chat completion fields; otherwise
it returns text completion fields. `stream: true` uses the same SSE format, queue,
authentication, token/context limits, cancellation and timeout handling as other endpoints.
This is a Lighter extension; OpenAI clients can call it through raw HTTP. Hosted replies
contain standard completion usage, not the plan; compute `plan` locally for provenance.

## Keras comparison

Keras has no built-in property-graph-to-text layer. A learned encoder/decoder requires a
chosen graph architecture, vocabulary, paired training data and decoding policy. Calling
a pretrained language model with serialized graph facts implements the same realization
approach as Lighter's native/hosted path. Exact realization corresponds to a deterministic
Python data transformation, independent of Keras:

```python
import json

def exact_graph_text(graph):
    nodes = {n["id"]: n for n in graph["nodes"]}
    if not nodes or len(nodes) != len(graph["nodes"]):
        raise ValueError("empty graph or duplicate node IDs")
    q = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"))
    lines = []
    for node in sorted(nodes.values(), key=lambda n: n["id"]):
        lines.append(f'Entity {q(node["id"])} has label {q(node["label"])}.')
        for key, value in sorted(node.get("properties", {}).items()):
            lines.append(f'{q(node["label"])} ({q(node["id"])}) has {q(key)} = {q(value)}.')
    for edge in sorted(graph["edges"], key=lambda e: e["id"]):
        src, dst = nodes[edge["source"]], nodes[edge["target"]]
        attributes = ", ".join(f"{q(k)} = {q(v)}" for k, v in sorted(edge.get("properties", {}).items()))
        suffix = f"; attributes: {attributes}" if attributes else ""
        lines.append(f'{q(src["label"])} ({q(src["id"])}) — {q(edge["relation"])} → {q(dst["label"])} ({q(dst["id"])}) [edge {q(edge["id"])}{suffix}].')
    return "\n".join(lines)

with open("examples/data/graph2text.json") as source:
    print(exact_graph_text(json.load(source)))
```

This Python comparison uses ID ordering rather than rooted traversal and illustrates
realization only; Lighter additionally validates the full schema, plans neighborhoods and
reports omissions. See the [general Keras comparison](keras_comparison.md) for trainable
layers and inference differences.

## Verification

```bash
cargo test --no-default-features --test test_graph2text
cargo test --no-default-features --features server --test test_graph2text
cargo test --no-default-features --features server --lib graph_api_tests
```

Tests cover graph serialization, missing endpoints, duplicate identities, bad options,
cycles, self-loops, disconnected components, incoming edges, budgets, parallel edges,
Unicode/escaping, arbitrary properties, native decoding/cache cleanup and HTTP validation.
