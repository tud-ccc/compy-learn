import networkx as nx

from compy.models.graphs.pytorch_dgl_model import GnnPytorchDGLModel
from compy.representations.common import Graph


def test_model():
    dummy_graph = nx.MultiDiGraph()
    dummy_graph.add_node("n1", attr="a")
    dummy_graph.add_node("n2", attr="b")
    dummy_graph.add_node("n3", attr="c")
    dummy_graph.add_edge("n1", "n2", attr="dummy")
    dummy_graph.add_edge("n2", "n3", attr="dummy")

    config = {
        "num_timesteps": 2,
        "hidden_size_orig": len(dummy_graph),
        "gnn_h_size": 4,
        "gnn_m_size": 2,
        "num_edge_types": 1,
        "learning_rate": 0.001,
        "batch_size": 4,
        "num_epochs": 1,
    }
    model = GnnPytorchDGLModel(config=config)

    data = [
        {
            "x": {
                "code_rep": Graph(dummy_graph, ["a", "b", "c"], ["dummy"]),
                "aux_in": [0, 0],
            },
            "y": 0,
        }
    ]
    model.train(data, data)
