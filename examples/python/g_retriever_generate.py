"""PCST retrieval -> upstream GNN -> learned soft prompt -> causal language model."""
import argparse
import json
from g_retriever import GraphRetriever, HashEncoder, SentenceEncoder, retrieve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="examples/data/graph2text.json")
    parser.add_argument("--question", default="Who wrote notes about the Analytical Engine?")
    parser.add_argument("--model", help="HF/local causal LM; omit for offline random-model demo")
    parser.add_argument("--encoder", help="HF/local sentence encoder; omit for hash demo")
    parser.add_argument("--adapter", help="trained GNN/projector checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    with open(args.graph) as source:
        graph = json.load(source)
    encoder = SentenceEncoder(args.encoder) if args.encoder else HashEncoder()
    result = retrieve(graph, args.question, encoder)
    if args.model:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        from g_retriever.demo import tiny_model
        model, tokenizer = tiny_model()
        print("Offline random-model demo: output is not a trained QA answer.")
    retriever = GraphRetriever(model, tokenizer, encoder.dimension, hidden_dim=args.hidden_dim,
        max_new_tokens=args.max_new_tokens)
    if args.adapter:
        retriever.load_adapter(args.adapter)
    else:
        print("GNN/projector are untrained; train or load an adapter for meaningful graph conditioning.")
    print(json.dumps(result.metadata(), indent=2))
    print("Answer:", retriever.generate([result], [args.question])[0])


if __name__ == "__main__":
    main()
