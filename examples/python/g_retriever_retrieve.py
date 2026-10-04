"""Run actual PCST retrieval without a language model."""
import argparse
import json
from g_retriever import HashEncoder, SentenceEncoder, retrieve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="examples/data/graph2text.json")
    parser.add_argument("--question", default="Who wrote notes about the Analytical Engine?")
    parser.add_argument("--encoder", help="HF/local sentence encoder; omit for offline hash demo")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--topk-edges", type=int, default=3)
    parser.add_argument("--edge-cost", type=float, default=0.5)
    parser.add_argument("--output", help="write retrieved property graph JSON (replaces existing file)")
    args = parser.parse_args()
    with open(args.graph) as source:
        graph = json.load(source)
    encoder = SentenceEncoder(args.encoder) if args.encoder else HashEncoder()
    result = retrieve(graph, args.question, encoder, args.topk, args.topk_edges, args.edge_cost)
    if args.output:
        with open(args.output, "w") as output:
            json.dump(result.property_graph, output, indent=2)
    print(result.description)
    print(json.dumps(result.metadata(), indent=2))


if __name__ == "__main__":
    main()
