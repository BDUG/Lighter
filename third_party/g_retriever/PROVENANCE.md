# Vendored G-Retriever

Source: https://github.com/XiaoxinHe/G-Retriever

Revision: `315b0ff8a206536067602fb97e77c10f4d646d5d`

Retrieved: 2026-10-04. License: [MIT](LICENSE), copyright 2024 Xiaoxin He.

`src/`, `train.py`, `inference.py`, `run.sh` and `README.md` are unchanged copies
from that revision. Datasets and figures are excluded. Dataset paths in the upstream
scripts are relative to their working directory; follow the upstream README to obtain
and preprocess the datasets. Its figures are available in the original repository.

The Lighter integration in `examples/python/g_retriever/` adapts the upstream PCST
retrieval algorithm and graph soft-prompt design. It imports the unchanged GNN
implementations by file path. Adaptations include CPU support, dynamic language-model
embedding dimensions, native PyTorch mean pooling, validated property-graph inputs,
stable tie handling and a small offline demonstration. See the integration guide for
behavior differences and dependencies.
