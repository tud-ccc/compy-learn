import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from compy.models.graphs.tf.layer.gnn_model_layer import (
    GGNNModelLayer,
    GGNNModelLayerState,
)
from compy.models.graphs.tf import test_utils
from compy.models.graphs.tf import utils


CONFIG = {
    "num_timesteps": 4,
    "graph_rnn_cell": "gru",
    "gnn_h_size": 4,
    "num_edge_types": 2,
    "use_edge_bias": 0,
}


def run_propagation_1_layer(graph_layer):
    # Get data
    test_data = test_utils.get_test_data(CONFIG, 1)

    # Process embeddings
    embeddings_in = tf.compat.v1.placeholder(
        tf.float32, [None, test_data["h_dim"]], name="embeddings_in"
    )
    embeddings_out = graph_layer.compute_embeddings(embeddings_in)

    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())

        fetch_list = [embeddings_out]
        feed_dict = {
            graph_layer.placeholders["adjacency_lists"][0]: test_data[
                "adjacency_lists"
            ][0],
            graph_layer.placeholders["adjacency_lists"][1]: test_data[
                "adjacency_lists"
            ][1],
            embeddings_in: test_data["node_features"],
        }

        result = session.run(fetch_list, feed_dict=feed_dict)

        # Check if shape is [v_dim, h_dim]
        assert len(result[0][0]) == test_data["v_dim"]
        assert len(result[0][0][0]) == test_data["h_dim"]


def run_propagation_model_1_layer_sparse(graph_layer):
    # Get data
    test_data = test_utils.get_test_data(CONFIG, 1)

    # Process embeddings
    embeddings_in = tf.compat.v1.placeholder(
        tf.float32, [None, test_data["h_dim"]], name="embeddings_in"
    )
    embeddings_out = graph_layer.compute_embeddings(embeddings_in)

    test_data["adjacency_lists"] = {0: [[0, 1]], 1: [[0, 1]]}

    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())

        fetch_list = [embeddings_out]
        feed_dict = {
            graph_layer.placeholders["adjacency_lists"][0]: test_data[
                "adjacency_lists"
            ][0],
            graph_layer.placeholders["adjacency_lists"][1]: test_data[
                "adjacency_lists"
            ][1],
            embeddings_in: test_data["node_features"],
        }

        result = session.run(fetch_list, feed_dict=feed_dict)

        assert result


# GNN Tests
def test_ggnn_propagation_model_1_layer():
    state = GGNNModelLayerState(CONFIG)
    graph_layer = GGNNModelLayer(CONFIG, state)

    run_propagation_1_layer(graph_layer)


def test_ggnn_propagation_model_1_layer_sparse():
    state = GGNNModelLayerState(CONFIG)
    graph_layer = GGNNModelLayer(CONFIG, state)

    run_propagation_model_1_layer_sparse(graph_layer)
