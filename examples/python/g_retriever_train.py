"""Train graph encoder/projector on JSONL graph/question/answer records; frozen LM."""
import argparse
import json
import torch
from g_retriever import GraphRetriever, HashEncoder, SentenceEncoder, retrieve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="examples/data/g_retriever_train.jsonl")
    parser.add_argument("--model", help="HF/local causal LM; omit for offline random-model demo")
    parser.add_argument("--encoder", help="HF/local sentence encoder; omit for hash demo")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--resume", help="adapter checkpoint to continue training")
    parser.add_argument("--output", required=True, help="output adapter checkpoint (replaces existing file)")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.learning_rate < float("inf"):
        parser.error("epochs, batch size and finite learning rate must be positive")
    torch.manual_seed(42)
    with open(args.data) as source:
        records = [json.loads(line) for line in source if line.strip()]
    if not records or any(set(r) != {"graph", "question", "answer"} or not isinstance(r["answer"], str) or not r["answer"].strip() for r in records):
        parser.error("JSONL records must contain graph, question and nonempty answer")
    encoder = SentenceEncoder(args.encoder) if args.encoder else HashEncoder()
    retrievals = [retrieve(r["graph"], r["question"], encoder) for r in records]
    if args.model:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        from g_retriever.demo import tiny_model
        model, tokenizer = tiny_model()
        print("Offline random-model training demo; not a pretrained QA system.")
    retriever = GraphRetriever(model, tokenizer, encoder.dimension, hidden_dim=args.hidden_dim)
    if args.resume:
        retriever.load_adapter(args.resume)
    optimizer = torch.optim.AdamW([p for p in retriever.parameters() if p.requires_grad], lr=args.learning_rate)
    retriever.train()
    for epoch in range(args.epochs):
        total = 0.0
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            optimizer.zero_grad()
            loss = retriever(retrievals[start:start + args.batch_size],
                [r["question"] for r in batch], [r["answer"] for r in batch])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in retriever.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
        print(f"epoch={epoch + 1} loss={total / len(records):.6f}")
    retriever.save_adapter(args.output)
    print("Saved graph encoder/projector:", args.output)


if __name__ == "__main__":
    main()
