import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import GlobalAttention
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

from compy.models.model import Model


class Net(torch.nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        annotation_size = config["hidden_size_orig"]
        hidden_size = config["gnn_h_size"]
        n_steps = config["num_timesteps"]
        num_cls = 2

        self.reduce = nn.Linear(annotation_size, hidden_size)
        self.conv = GatedGraphConv(hidden_size, n_steps)
        self.agg = GlobalAttention(nn.Linear(hidden_size, 1), nn.Linear(hidden_size, 2))
        self.lin = nn.Linear(hidden_size, num_cls)

    def forward(
        self, graph,
    ):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch

        x = self.reduce(x)

        x = self.conv(x, edge_index)
        x = self.agg(x, batch)

        x = F.log_softmax(x, dim=1)

        return x


class GnnPytorchGeomModel(Model):
    def __init__(self, config=None, num_types=None):
        if not config:
            config = {
                "num_timesteps": 4,
                "hidden_size_orig": num_types,
                "gnn_h_size": 32,
                "gnn_m_size": 2,
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

    def __build_pg_graphs(self, batch_graphs):
        pg_graphs = []

        for batch_graph in batch_graphs:
            # Graph
            # - nodes
            one_hot = np.zeros(
                (len(batch_graph["nodes"]), self.config["hidden_size_orig"])
            )
            one_hot[np.arange(len(batch_graph["nodes"])), batch_graph["nodes"]] = 1
            x = torch.tensor(one_hot, dtype=torch.float)

            # -edges
            edge_index, edge_features = [], []
            for edge in batch_graph["edges"]:
                edge_index.append([edge[0], edge[2]])
                edge_features.append([edge[1]])
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_features = torch.tensor(edge_features, dtype=torch.long)

            graph = Data(
                x=x,
                edge_index=edge_index.t().contiguous(),
                edge_features=edge_features,
                y=batch_graph["label"],
            )
            pg_graphs.append(graph)

        return pg_graphs

    def _train_init(self, data_train, data_valid):
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )

        return self.__process_data(data_train), self.__process_data(data_valid)

    def _train_with_batch(self, batch):
        loss_sum = 0
        correct_sum = 0

        graphs = self.__build_pg_graphs(batch)
        loader = DataLoader(graphs, batch_size=999999)
        for data in loader:
            data = data.to(self.device)

            self.model.train()
            self.opt.zero_grad()

            pred = self.model(data)
            loss = F.nll_loss(pred, data.y)
            loss.backward()
            self.opt.step()

            loss_sum += loss
            correct_sum += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()

        train_accuracy = correct_sum / len(loader.dataset)
        train_loss = loss_sum / len(loader.dataset)

        return train_loss, train_accuracy

    def _test_init(self):
        self.model.eval()

    def _predict_with_batch(self, batch):
        correct = 0

        graphs = self.__build_pg_graphs(batch)
        loader = DataLoader(graphs, batch_size=999999)
        for data in loader:
            data = data.to(self.device)

            with torch.no_grad():
                pred = self.model(data)

            correct += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
        valid_accuracy = correct / len(loader.dataset)

        return valid_accuracy, pred
