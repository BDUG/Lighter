"""CPU/offline integration tests: python -m unittest discover -s examples/python/g_retriever."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from g_retriever import GraphRetriever, HashEncoder, retrieve
from g_retriever.demo import tiny_model

ROOT = Path(__file__).resolve().parents[3]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.graph = json.loads((ROOT / "examples/data/graph2text.json").read_text())
        self.encoder = HashEncoder(16)

    def test_real_pcst_selects_connected_relations_and_restores_direction(self):
        result = retrieve(self.graph, "Who wrote notes about the Analytical Engine?", self.encoder)
        self.assertEqual(set(result.node_ids), {"ada", "babbage", "engine"})
        self.assertEqual(set(result.edge_ids), {"notes", "design"})
        self.assertEqual(result.graph.edge_index.shape, (2, 2))
        for column, eid in enumerate(result.edge_ids):
            edge = next(e for e in self.graph["edges"] if e["id"] == eid)
            source, target = result.graph.edge_index[:, column].tolist()
            self.assertEqual(result.node_ids[source], edge["source"])
            self.assertEqual(result.node_ids[target], edge["target"])
        self.assertIn("1843", result.description)

    def test_disconnected_graph_retrieves_one_component(self):
        self.graph["nodes"].append({"id":"isolated", "label":"Unrelated island"})
        r = retrieve(self.graph, "Analytical Engine notes", self.encoder, topk=1, topk_edges=1)
        self.assertIn("isolated", r.omitted_node_ids)
        self.assertTrue(r.edge_ids)

    def test_edgeless_graph_preserves_nodes_and_empty_shapes(self):
        self.graph["edges"] = []
        r = retrieve(self.graph, "Ada", self.encoder)
        self.assertEqual(r.graph.edge_index.shape, (2, 0))
        self.assertEqual(r.graph.edge_attr.shape, (0, 16))
        self.assertEqual(len(r.node_ids), 3)

    def test_prizes_ties_parallel_edges_and_self_loops(self):
        self.graph["edges"].append(dict(self.graph["edges"][0], id="parallel"))
        self.graph["edges"].append({"id":"loop", "source":"ada", "target":"ada", "relation":"knows"})
        r = retrieve(self.graph, "notes", self.encoder, topk_edges=2)
        self.assertEqual(r.edge_prizes[0], r.edge_prizes[2])
        self.assertEqual(len(r.node_ids), len(set(r.node_ids)))
        self.assertEqual(len(r.edge_ids), len(set(r.edge_ids)))

    def test_zero_edge_prizes_and_high_cost(self):
        r = retrieve(self.graph, "Ada Lovelace", self.encoder, topk=1, topk_edges=0, edge_cost=100)
        self.assertEqual(r.edge_ids, [])
        self.assertEqual(len(r.node_ids), 1)
        r = retrieve(self.graph, "notes", self.encoder, topk=0, topk_edges=1)
        self.assertTrue(r.edge_ids)

    def test_invalid_inputs_fail_before_solver(self):
        for kwargs in [{"topk":-1},{"topk":0,"topk_edges":0},{"edge_cost":float("nan")},{"topk":True}]:
            with self.assertRaises(ValueError):
                retrieve(self.graph, "question", self.encoder, **kwargs)
        for bad in [{"nodes":[],"edges":[]},dict(self.graph, unknown=True)]:
            with self.assertRaises(ValueError):
                retrieve(bad,"question",self.encoder)
        self.graph["edges"][0]["source"]="absent"
        with self.assertRaises(ValueError):
            retrieve(self.graph,"question",self.encoder)

    def test_deterministic_embeddings_and_repeat_retrieval(self):
        self.assertTrue(torch.equal(self.encoder.encode(["Ada Lovelace"]), HashEncoder(16).encode(["Ada Lovelace"])))
        a = retrieve(self.graph, "question", self.encoder)
        b = retrieve(self.graph, "question", self.encoder)
        self.assertEqual(a.metadata(), b.metadata())


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.encoder = HashEncoder(16)
        graph = json.loads((ROOT / "examples/data/graph2text.json").read_text())
        self.retrieval = retrieve(graph, "Who wrote notes?", self.encoder)
        model, tokenizer = tiny_model()
        self.retriever = GraphRetriever(model, tokenizer,16,hidden_dim=16,max_text_tokens=128,max_new_tokens=4)

    def test_loss_backward_reaches_gnn_and_projector_but_not_frozen_lm(self):
        loss = self.retriever([self.retrieval], ["Who wrote notes?"], ["Ada"])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for module in [self.retriever.graph_encoder,self.retriever.projector]:
            self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in module.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.retriever.model.parameters()))
        self.assertFalse(self.retriever.model.training)
        self.assertEqual(self.retriever.encode_graphs([self.retrieval]).shape,(1,32))

    def test_checkpoint_roundtrip_generation_and_batch_padding(self):
        self.retriever.eval()
        before = self.retriever.encode_graphs([self.retrieval]).detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"adapter.pt"
            self.retriever.save_adapter(path)
            with torch.no_grad():
                next(self.retriever.projector.parameters()).add_(1)
            self.retriever.load_adapter(path)
        torch.testing.assert_close(before,self.retriever.encode_graphs([self.retrieval]))
        questions=["Who?","Who designed the Analytical Engine?"]
        inputs, masks, labels=self.retriever._inputs([self.retrieval]*2, questions,["Ada","Babbage"])
        self.assertEqual(inputs.shape[:2], masks.shape)
        self.assertTrue((masks[0]==0).any())
        self.assertTrue((labels[masks==0]==-100).all())
        self.assertEqual(len(self.retriever.generate([self.retrieval]*2,questions)),2)
        self.assertFalse(self.retriever.training)

    def test_architectures_and_context_validation(self):
        for gnn in ["gcn","gat","gt"]:
            model, tokenizer=tiny_model()
            r=GraphRetriever(model,tokenizer,16,hidden_dim=16,gnn=gnn)
            r.eval()
            self.assertEqual(r.encode_graphs([self.retrieval]).shape,(1,32))
        self.retriever.model.config.max_position_embeddings=10
        with self.assertRaises(ValueError):
            self.retriever.generate([self.retrieval],["Question?"])
        with self.assertRaises(ValueError):
            self.retriever([],[],[])


if __name__ == "__main__":
    unittest.main()
