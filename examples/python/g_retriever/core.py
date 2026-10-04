"""Adapted from XiaoxinHe/G-Retriever (MIT, copyright 2024 Xiaoxin He).

PCST retrieval follows src/dataset/utils/retrieval.py; graph soft prompting follows
src/model/graph_llm.py. Original source and license: third_party/g_retriever/.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

# pcst-fast 1.0.10 wheels return corrupted index arrays with NumPy 2.x.
if int(np.__version__.split(".")[0]) >= 2:
    raise ImportError("G-Retriever requires numpy>=1.26,<2 for pcst-fast 1.0.10")

from pcst_fast import pcst_fast
import torch
from torch import nn
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("lighter_upstream_gnn", ROOT / "third_party/g_retriever/src/model/gnn.py")
_gnn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gnn)


class HashEncoder:
    """Deterministic lexical demo embeddings, not pretrained semantic embeddings."""
    def __init__(self, dimension=64):
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    def encode(self, texts):
        output = torch.zeros(len(texts), self.dimension)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little") % self.dimension
                output[row, index] += 1
        return nn.functional.normalize(output, dim=-1)


class SentenceEncoder:
    """Hugging Face encoder with masked mean pooling, as in upstream preprocessing."""
    def __init__(self, model="sentence-transformers/all-roberta-large-v1", device="cpu"):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(device).eval()
        self.dimension = self.model.config.hidden_size
        self.device = device

    @torch.no_grad()
    def encode(self, texts, batch_size=32):
        if not texts:
            return torch.empty(0, self.dimension)
        batches = []
        for start in range(0, len(texts), batch_size):
            inputs = self.tokenizer(texts[start:start + batch_size], padding=True, truncation=True, return_tensors="pt").to(self.device)
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            batches.append(nn.functional.normalize(pooled, dim=-1).cpu())
        return torch.cat(batches)


@dataclass
class Retrieval:
    graph: Data
    property_graph: dict
    description: str
    node_ids: list
    edge_ids: list
    node_prizes: list
    edge_prizes: list
    omitted_node_ids: list
    omitted_edge_ids: list

    def metadata(self):
        return {key: getattr(self, key) for key in ("node_ids", "edge_ids", "node_prizes", "edge_prizes", "omitted_node_ids", "omitted_edge_ids")}


def _validate(graph):
    if not isinstance(graph, dict) or set(graph) - {"nodes", "edges"}:
        raise ValueError("graph must contain only nodes and edges")
    nodes, edges = graph.get("nodes"), graph.get("edges")
    if not isinstance(nodes, list) or not nodes or not isinstance(edges, list):
        raise ValueError("graph requires a nonempty nodes array and an edges array")
    def text(value):
        return isinstance(value, str) and bool(value.strip())
    node_ids = set()
    for node in nodes:
        if not isinstance(node, dict) or set(node) - {"id", "label", "properties"} or not text(node.get("id")) or not text(node.get("label")) or node["id"] in node_ids:
            raise ValueError("invalid or duplicate node")
        node_ids.add(node["id"])
    edge_ids = set()
    for edge in edges:
        if not isinstance(edge, dict) or set(edge) - {"id", "source", "relation", "target", "properties"} or not text(edge.get("id")) or not text(edge.get("relation")) or edge["id"] in edge_ids:
            raise ValueError("invalid or duplicate edge")
        if not text(edge.get("source")) or not text(edge.get("target")) or edge["source"] not in node_ids or edge["target"] not in node_ids:
            raise ValueError("edge endpoint is absent")
        edge_ids.add(edge["id"])
    for item in nodes + edges:
        properties = item.get("properties", {})
        if not isinstance(properties, dict) or any(not text(k) for k in properties):
            raise ValueError("properties require nonblank keys")
        json.dumps(properties, allow_nan=False)
    return nodes, edges


def retrieve(graph, question, encoder, topk=3, topk_edges=3, edge_cost=0.5):
    """Cosine rank prizes, virtual edge-prize nodes and unrooted GW PCST pruning.

    Solve connectivity as undirected, then restore original directed relations.
    Returned embeddings are CPU tensors, with original IDs recorded separately.
    """
    nodes, edges = _validate(graph)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonblank")
    if any(isinstance(k, bool) or not isinstance(k, int) or k < 0 for k in (topk, topk_edges)) or not np.isfinite(edge_cost) or edge_cost < 0:
        raise ValueError("rank counts must be nonnegative integers and edge cost finite/nonnegative")
    if topk == 0 and (topk_edges == 0 or not edges):
        raise ValueError("at least one prize source must be enabled")
    node_text = [n["label"] + " " + json.dumps(n.get("properties", {}), sort_keys=True, ensure_ascii=False) for n in nodes]
    edge_text = [e["relation"] + " " + json.dumps(e.get("properties", {}), sort_keys=True, ensure_ascii=False) for e in edges]
    x = torch.as_tensor(encoder.encode(node_text)).detach().cpu().float()
    query = torch.as_tensor(encoder.encode([question])).detach().cpu().float()
    attrs = torch.as_tensor(encoder.encode(edge_text)).detach().cpu().float()
    dimension = x.shape[1] if x.ndim == 2 else 0
    if x.shape != (len(nodes), dimension) or dimension == 0 or query.shape != (1, dimension) or attrs.shape != (len(edges), dimension) or not all(torch.isfinite(t).all() for t in (x, query, attrs)):
        raise ValueError("encoder must return finite, equally sized embedding rows")
    indices = {n["id"]: i for i, n in enumerate(nodes)}
    original_edges = np.asarray([(indices[e["source"]], indices[e["target"]]) for e in edges], dtype=np.int64).reshape(-1, 2)
    nprizes = np.zeros(len(nodes), dtype=np.float64)
    nscores = nn.functional.cosine_similarity(query, x).numpy()
    count = min(topk, len(nodes))
    # Stable index tie breaking; descending integer ranks match upstream.
    for rank, index in enumerate(np.argsort(-nscores, kind="stable")[:count]):
        nprizes[index] = count - rank
    eprizes = np.zeros(len(edges), dtype=np.float64)
    if edges and topk_edges:
        scores = nn.functional.cosine_similarity(query, attrs).numpy()
        values = np.unique(scores)[::-1][:topk_edges]
        previous = float(len(values))
        for rank, score in enumerate(values):
            members = scores == score
            prize = min((len(values) - rank) / int(members.sum()), previous)
            eprizes[members] = prize
            previous = prize * 0.99
        edge_cost = min(edge_cost, float(eprizes.max()) * 0.995)
    transformed, costs, virtual_prizes = [], [], []
    real_mapping, virtual_mapping = {}, {}
    # Nonpositive edge prizes become discounted costs; positive surplus is a
    # virtual node prize attached to both original endpoints by zero-cost edges.
    for i, (source, target) in enumerate(original_edges):
        if eprizes[i] <= edge_cost:
            real_mapping[len(transformed)] = i
            transformed.append((source, target))
            costs.append(edge_cost - eprizes[i])
    real_count = len(transformed)
    for i, (source, target) in enumerate(original_edges):
        if eprizes[i] > edge_cost:
            virtual = len(nodes) + len(virtual_prizes)
            virtual_mapping[virtual] = i
            transformed.extend([(source, virtual), (virtual, target)])
            costs.extend([0.0, 0.0])
            virtual_prizes.append(eprizes[i] - edge_cost)
    if not edges:
        # Upstream returns all nodes for edge-free graphs; preserve that behavior.
        selected_nodes, selected_edges = list(range(len(nodes))), []
    else:
        vertices, selected = pcst_fast(
            np.asarray(transformed, dtype=np.int64).reshape(-1, 2),
            np.concatenate([nprizes, virtual_prizes]), np.asarray(costs, dtype=np.float64),
            -1, 1, "gw", 0)
        selected_edges = sorted({real_mapping[int(e)] for e in selected if e < real_count} | {virtual_mapping[int(v)] for v in vertices if v >= len(nodes)})
        selected_nodes = sorted({int(v) for v in vertices if v < len(nodes)} | {int(v) for i in selected_edges for v in original_edges[i]})
    if not selected_nodes:
        raise ValueError("PCST selected no original nodes; adjust retrieval parameters")
    remap = {node: i for i, node in enumerate(selected_nodes)}
    edge_index = torch.tensor([(remap[int(original_edges[i, 0])], remap[int(original_edges[i, 1])]) for i in selected_edges], dtype=torch.long).reshape(-1, 2).T.contiguous()
    data = Data(x=x[selected_nodes], edge_index=edge_index, edge_attr=attrs[selected_edges], num_nodes=len(selected_nodes))
    node_frame = pd.DataFrame([{"node_id": nodes[i]["id"], "node_attr": nodes[i]["label"], "properties": json.dumps(nodes[i].get("properties", {}), sort_keys=True)} for i in selected_nodes])
    edge_frame = pd.DataFrame([{"edge_id": edges[i]["id"], "src": edges[i]["source"], "edge_attr": edges[i]["relation"], "dst": edges[i]["target"], "properties": json.dumps(edges[i].get("properties", {}), sort_keys=True)} for i in selected_edges], columns=["edge_id", "src", "edge_attr", "dst", "properties"])
    return Retrieval(data, {"nodes": [nodes[i] for i in selected_nodes], "edges": [edges[i] for i in selected_edges]}, node_frame.to_csv(index=False) + "\n" + edge_frame.to_csv(index=False),
        [nodes[i]["id"] for i in selected_nodes], [edges[i]["id"] for i in selected_edges],
        nprizes.tolist(), eprizes.tolist(), [n["id"] for i,n in enumerate(nodes) if i not in selected_nodes],
        [e["id"] for i,e in enumerate(edges) if i not in selected_edges])


class GraphRetriever(nn.Module):
    """Upstream GNN -> mean pool -> projector -> single learned LM soft token.

    The language model is frozen; gradients flow into the GNN and projector.
    Accepts Hugging Face causal LMs with inputs_embeds support.
    """
    def __init__(self, model, tokenizer, input_dim, hidden_dim=64, layers=2, heads=4,
                 gnn="gt", max_text_tokens=512, max_new_tokens=64):
        super().__init__()
        if gnn not in _gnn.load_gnn_model or min(input_dim, hidden_dim, heads, max_text_tokens, max_new_tokens) < 1 or layers < 2 or hidden_dim % heads:
            raise ValueError("invalid GNN dimensions, architecture or token limits")
        self.model, self.tokenizer = model, tokenizer
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        embedding = model.get_input_embeddings().weight
        self.graph_encoder = _gnn.load_gnn_model[gnn](input_dim, hidden_dim, hidden_dim, layers, 0.0, heads).to(device=embedding.device, dtype=torch.float32)
        self.projector = nn.Sequential(nn.Linear(hidden_dim, 2048), nn.Sigmoid(), nn.Linear(2048, embedding.shape[1])).to(embedding.device)
        self.max_text_tokens, self.max_new_tokens = max_text_tokens, max_new_tokens
        self.input_dim = input_dim
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer needs a pad or EOS token")
            tokenizer.pad_token = tokenizer.eos_token

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()  # Keep frozen LM dropout disabled during adapter training.
        return self

    def encode_graphs(self, retrievals):
        if not retrievals or any(r.graph.num_nodes < 1 or r.graph.x.shape[1] != self.input_dim for r in retrievals):
            raise ValueError("nonempty graphs with matching input dimensions are required")
        device = self.model.get_input_embeddings().weight.device
        graphs = Batch.from_data_list([r.graph for r in retrievals]).to(device)
        # BatchNorm in the unchanged upstream GNN needs at least two node rows.
        if self.training and graphs.x.shape[0] < 2:
            raise ValueError("training requires at least two total nodes per batch")
        hidden, _ = self.graph_encoder(graphs.x, graphs.edge_index, graphs.edge_attr)
        pooled = hidden.new_zeros((len(retrievals), hidden.shape[1]))
        pooled.index_add_(0, graphs.batch, hidden)
        counts = torch.bincount(graphs.batch, minlength=len(retrievals)).clamp_min(1)
        return self.projector(pooled / counts.unsqueeze(-1))

    def _inputs(self, retrievals, questions, answers=None):
        if len(retrievals) != len(questions) or (answers is not None and len(answers) != len(questions)) or not questions or any(not isinstance(q,str) or not q.strip() for q in questions):
            raise ValueError("aligned nonempty graph/question/answer batches are required")
        embedding = self.model.get_input_embeddings()
        device, dtype = embedding.weight.device, embedding.weight.dtype
        graph_tokens = self.encode_graphs(retrievals).to(dtype=dtype)
        rows, labels = [], []
        for i, (retrieval, question) in enumerate(zip(retrievals, questions)):
            description = self.tokenizer.encode(retrieval.description, add_special_tokens=False)[:self.max_text_tokens]
            question_ids = self.tokenizer.encode("\nQuestion: " + question + "\nAnswer:", add_special_tokens=False)
            bos = [] if self.tokenizer.bos_token_id is None else [self.tokenizer.bos_token_id]
            bos_emb = embedding(torch.tensor(bos, device=device, dtype=torch.long))
            ids = description + question_ids
            target = []
            if answers is not None:
                target = self.tokenizer.encode(answers[i], add_special_tokens=False)[:self.max_new_tokens]
                if self.tokenizer.eos_token_id is not None:
                    target.append(self.tokenizer.eos_token_id)
                if not target:
                    raise ValueError("answer must tokenize to at least one target token")
            row = torch.cat([bos_emb, graph_tokens[i:i+1], embedding(torch.tensor(ids + target, device=device, dtype=torch.long))])
            rows.append(row)
            labels.append([-100] * (len(row) - len(target)) + target)
        length = max(len(row) for row in rows)
        context = getattr(self.model.config, "max_position_embeddings", None)
        if context and length + (self.max_new_tokens if answers is None else 0) > context:
            raise ValueError("graph prompt exceeds language-model context; reduce text/output limits")
        padded, masks, padded_labels = [], [], []
        pad = embedding(torch.tensor(self.tokenizer.pad_token_id, device=device))
        for row, label in zip(rows, labels):
            count = length - len(row)
            padded.append(torch.cat([pad.expand(count, -1), row]))
            masks.append([0] * count + [1] * len(row))
            padded_labels.append([-100] * count + label)
        return torch.stack(padded), torch.tensor(masks, device=device), torch.tensor(padded_labels, device=device)

    def forward(self, retrievals, questions, answers):
        embeddings, mask, labels = self._inputs(retrievals, questions, answers)
        return self.model(inputs_embeds=embeddings, attention_mask=mask, labels=labels).loss

    @torch.no_grad()
    def generate(self, retrievals, questions):
        was_training = self.training
        self.eval()
        try:
            embeddings, mask, _ = self._inputs(retrievals, questions)
            output = self.model.generate(inputs_embeds=embeddings, attention_mask=mask,
                max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.batch_decode(output, skip_special_tokens=True)
        finally:
            self.train(was_training)

    def save_adapter(self, path):
        torch.save({"graph_encoder": self.graph_encoder.state_dict(), "projector": self.projector.state_dict()}, path)

    def load_adapter(self, path):
        state = torch.load(path, map_location=self.model.get_input_embeddings().weight.device, weights_only=True)
        self.graph_encoder.load_state_dict(state["graph_encoder"])
        self.projector.load_state_dict(state["projector"])
