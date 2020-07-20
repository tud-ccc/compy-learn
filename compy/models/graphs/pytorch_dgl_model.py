import torch
import numpy as np

from torch import nn
from torch.optim import Adam

from compy.models.model import Model


class Net(nn.Module):
    def __init__(self, config):
        from dgl.nn.pytorch import (
            GatedGraphConv,
        )  # Prevents DGL from clashing with TensorFlow backend
        from dgl.nn.pytorch import GlobalAttentionPooling

        super(Net, self).__init__()

        annotation_size = config["hidden_size_orig"]
        self.annotation_size = annotation_size
        hidden_size = config["gnn_h_size"]
        n_steps = config["num_timesteps"]
        n_etypes = config["num_edge_types"]
        num_cls = 2

        self.reduce_layer = nn.Linear(annotation_size, hidden_size)
        self.ggnn = GatedGraphConv(
            in_feats=hidden_size,
            out_feats=hidden_size,
            n_steps=n_steps,
            n_etypes=n_etypes,
        )

        pooling_gate_nn = nn.Linear(hidden_size * 2, 1)
        self.pooling = GlobalAttentionPooling(pooling_gate_nn)
        self.output_layer = nn.Linear(hidden_size * 2, num_cls)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, graph, labels=None):
        etypes = graph.edata.pop("type")
        annotation = graph.ndata.pop("annotation").float()
        assert annotation.size()[-1] == self.annotation_size

        annotation = self.reduce_layer(annotation)

        out = self.ggnn(graph, annotation, etypes)
        out = torch.cat([out, annotation], -1)
        out = self.pooling(graph, out)

        logits = self.output_layer(out)
        preds = torch.argmax(logits, -1)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, preds
        return preds


class GnnPytorchDGLModel(Model):
    def __init__(self, config=None, num_types=None):
        if not config:
            config = {
                "num_timesteps": 4,
                "hidden_size_orig": num_types,
                "gnn_h_size": 32,
                "gnn_m_size": 2,
                "num_edge_types": 4,
                "learning_rate": 0.001,
                "batch_size": 64,
                "num_epochs": 1000,
            }
        super().__init__(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Net(config)
        self.model = self.model.to(self.device)

    def __process_data(self, data):
        return [
            {
                "nodes": data["x"]["code_rep"].get_node_list(),
                "edges": data["x"]["code_rep"].get_edge_list(),
                "aux_in": data["x"]["aux_in"],
                "label": data["y"],
            }
            for data in data
        ]

    def __build_dgl_graph_and_labels(self, batch_graphs):
        import dgl  # Import DGL locally to prevent clashing with TensorFlow import

        dgl_graphs = []
        labels = []
        for batch_graph in batch_graphs:
            # Graph
            g = dgl.DGLGraph()

            # - nodes
            g.add_nodes(len(batch_graph["nodes"]))
            g.ndata["annotation"] = torch.zeros(
                [len(batch_graph["nodes"]), self.config["hidden_size_orig"]],
                dtype=torch.long,
            )

            # -edges
            edge_types = []
            for edge in batch_graph["edges"]:
                g.add_edge(edge[0], edge[2])
                edge_types.append(edge[1])
            g.edata["type"] = torch.tensor(edge_types, dtype=torch.long)

            dgl_graphs.append(g)

            # Label
            labels.append(batch_graph["label"])

        # Put small graphs into a large graph with disconnected subgraphs
        dgl_graph = dgl.batch(dgl_graphs)

        labels = torch.tensor(labels, dtype=torch.long)

        dgl_graph = dgl_graph.to(self.device)
        labels = labels.to(self.device)

        return dgl_graph, labels

    def _train_init(self, data_train, data_valid):
        self.opt = Adam(self.model.parameters(), lr=self.config["learning_rate"])

        return self.__process_data(data_train), self.__process_data(data_valid)

    def _train_with_batch(self, batch):
        g, labels = self.__build_dgl_graph_and_labels(batch)

        self.model.train()
        self.opt.zero_grad()

        loss, pred = self.model(g, labels)
        loss.backward()
        self.opt.step()

        train_accuracy = (
            np.equal(
                labels.cpu().data.numpy().tolist(), pred.cpu().data.numpy().tolist()
            )
            .astype(np.float)
            .tolist()
        )
        train_accuracy = sum(train_accuracy) / len(train_accuracy)
        train_loss = loss / len(batch)

        return train_loss, train_accuracy

    def _test_init(self):
        self.model.eval()

    def _predict_with_batch(self, batch):
        g, labels = self.__build_dgl_graph_and_labels(batch)

        with torch.no_grad():
            loss, pred = self.model(g, labels)

        valid_accuracy = (
            np.equal(
                labels.cpu().data.numpy().tolist(), pred.cpu().data.numpy().tolist()
            )
            .astype(np.float)
            .tolist()
        )
        valid_accuracy = sum(valid_accuracy) / len(valid_accuracy)

        return valid_accuracy, pred
