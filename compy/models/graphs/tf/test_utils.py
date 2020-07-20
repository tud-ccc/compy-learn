import numpy as np

from compy.models.graphs.tf import utils


ONE_GRAPH = [(0, 0, 1), (1, 0, 2), (2, 1, 1)]
TWO_GRAPHS = [(0, 0, 1), (1, 0, 2), (2, 1, 1), (3, 0, 4), (4, 1, 5), (5, 0, 4)]


# Helper functions
def get_test_data(
    config: dict, num_graphs: int
) -> (np.ndarray, dict, list, list, int, int):
    h_dim = config["gnn_h_size"]
    v_dim = len(ONE_GRAPH)

    if num_graphs == 1:
        node_features = np.ones((v_dim, h_dim))
        adjacency_lists = utils.graph_to_adjacency_lists(ONE_GRAPH, False)[0]
        embeddings_to_graph_mappings = [[0, 0, 0]]
        embeddings_last_added_node_idxs = [2]
        last_added_node_types = [1]
        labels = np.array([[1, 0]])
        a1_labels = np.array([1])
        a2_labels = np.array([1])
        a3_labels = np.array([[0, 1], [0, 0], [0, 0]])

    elif num_graphs == 2:
        node_features = np.ones((v_dim * 2, h_dim))
        adjacency_lists = utils.graph_to_adjacency_lists(TWO_GRAPHS, False)[0]
        embeddings_to_graph_mappings = [[0, 0, 0, 1, 1, 1]]
        embeddings_last_added_node_idxs = [2, 5]
        last_added_node_types = [1, 2]
        labels = np.array([[1, 0], [1, 0]])
        a1_labels = np.array([1])
        a2_labels = np.array([1])
        a3_labels = np.array([[0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [1, 0]])

    return {
        "node_features": node_features,
        "adjacency_lists": adjacency_lists,
        "embeddings_to_graph_mappings": embeddings_to_graph_mappings,
        "embeddings_last_added_node_idxs": embeddings_last_added_node_idxs,
        "last_added_node_types": last_added_node_types,
        "labels": labels,
        "a1_labels": a1_labels,
        "a2_labels": a2_labels,
        "a3_labels": a3_labels,
        "v_dim": v_dim,
        "h_dim": h_dim,
    }
