"""Tiny randomly initialized Llama and byte tokenizer for offline plumbing checks."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM


class ByteTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0
    def encode(self, text, add_special_tokens=False):
        return [byte + 3 for byte in text.encode("utf-8")]
    def batch_decode(self, rows, skip_special_tokens=True):
        return [bytes(token - 3 for token in row.tolist() if 3 <= token < 259).decode("utf-8", errors="replace") for row in rows]


def tiny_model():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = LlamaConfig(vocab_size=259, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=2048, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    return LlamaForCausalLM(config), ByteTokenizer()
