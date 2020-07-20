import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from compy.models.graphs.tf.cell.prediction_cell import (
    PredictionCell,
    PredictionCellState,
)
from compy.models.graphs.tf.layer.gnn_model_layer import (
    GGNNModelLayer,
    GGNNModelLayerState,
)
from compy.models.graphs.tf import test_utils


CONFIG = {
    "hidden_size_orig": 4,
    "num_timesteps": 4,
    "graph_rnn_cell": "gru",
    "gnn_h_size": 4,
    "gnn_m_size": 2,
    "num_node_types": 2,
    "num_edge_types": 2,
    "use_edge_bias": 0,
    "use_edge_msg_avg_aggregation": 0,
    "with_aux_in": 0,
    "prediction_cell": {
        "mlp_f_m_dims": [64, 64],
        "mlp_f_m_activation": "relu",
        "mlp_g_m_dims": [64, 64],
        "mlp_g_m_activation": "relu",
        "mlp_reduce_dims": [64, 64],
        "mlp_reduce_activation": "relu",
        "mlp_reduce_after_aux_in_1_dims": [],
        "mlp_reduce_after_aux_in_1_activation": "relu",
        "mlp_reduce_after_aux_in_1_out_dim": 32,
        "mlp_reduce_after_aux_in_2_dims": [],
        "mlp_reduce_after_aux_in_2_activation": "sigmoid",
        "mlp_reduce_after_aux_in_2_out_dim": 2,
        "output_dim": 2,
        "mlp_reduce_out_dim": 8,
    },
}


# Helper functions
def setup_deepgmg_cell_prediction_and_fetch_op(num_graphs: int):
    # Get data
    test_data = test_utils.get_test_data(CONFIG, num_graphs)

    embeddings_in = tf.compat.v1.placeholder(
        tf.float32, [None, test_data["h_dim"]], name="embeddings_in"
    )

    # Create state
    ggnn_layer_state = GGNNModelLayerState(CONFIG)
    prediction_cell_state = PredictionCellState(CONFIG)

    # Create layer and propagate
    ggnn_layer = GGNNModelLayer(CONFIG, ggnn_layer_state)
    embeddings = ggnn_layer.compute_embeddings(embeddings_in)

    # Create cell and predict
    prediction_cell = PredictionCell(
        CONFIG, False, prediction_cell_state, CONFIG["with_aux_in"]
    )
    prediction_cell.initial_embeddings = tf.compat.v1.placeholder(
        tf.float32, [None, CONFIG["hidden_size_orig"]], name="embeddings_in"
    )
    prediction_cell.compute_predictions(embeddings)

    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())

        fetch_list = [prediction_cell.ops["output"]]
        feed_dict = {
            ggnn_layer.placeholders["adjacency_lists"][0]: test_data["adjacency_lists"][
                0
            ],
            ggnn_layer.placeholders["adjacency_lists"][1]: test_data["adjacency_lists"][
                1
            ],
            prediction_cell.initial_embeddings: test_data["node_features"],
            embeddings_in: test_data["node_features"],
            prediction_cell.placeholders["embeddings_to_graph_mappings"]: test_data[
                "embeddings_to_graph_mappings"
            ],
            prediction_cell.placeholders["is_training"]: False,
        }

        result = session.run(fetch_list, feed_dict=feed_dict)

        return result


def setup_deepgmg_cell_training_and_fetch_op(num_graphs: int):
    # Get data
    test_data = test_utils.get_test_data(CONFIG, num_graphs)

    embeddings_in = tf.compat.v1.placeholder(
        tf.float32, [None, test_data["h_dim"]], name="embeddings_in"
    )

    # Create state
    ggnn_layer_state = GGNNModelLayerState(CONFIG)
    prediction_cell_state = PredictionCellState(CONFIG)

    # Create layer and propagate
    ggnn_layer = GGNNModelLayer(CONFIG, ggnn_layer_state)
    embeddings = ggnn_layer.compute_embeddings(embeddings_in)

    # Create cell and predict
    prediction_cell = PredictionCell(
        CONFIG, True, prediction_cell_state, CONFIG["with_aux_in"]
    )
    prediction_cell.initial_embeddings = tf.compat.v1.placeholder(
        tf.float32, [None, CONFIG["hidden_size_orig"]], name="embeddings_in"
    )
    prediction_cell.compute_predictions(embeddings)

    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())

        fetch_list = [prediction_cell.ops["loss"]]
        feed_dict = {
            ggnn_layer.placeholders["adjacency_lists"][0]: test_data["adjacency_lists"][
                0
            ],
            ggnn_layer.placeholders["adjacency_lists"][1]: test_data["adjacency_lists"][
                1
            ],
            prediction_cell.initial_embeddings: test_data["node_features"],
            embeddings_in: test_data["node_features"],
            prediction_cell.placeholders["embeddings_to_graph_mappings"]: test_data[
                "embeddings_to_graph_mappings"
            ],
            prediction_cell.placeholders["labels"]: test_data["labels"],
            prediction_cell.placeholders["is_training"]: True,
        }

        result = session.run(fetch_list, feed_dict=feed_dict)

        return result


# Prediction Tests
def test_prediction_cell_1_graph():
    result = setup_deepgmg_cell_prediction_and_fetch_op(1)

    assert isinstance(result[0], np.ndarray)


def test_prediction_cell_2_graphs():
    result = setup_deepgmg_cell_prediction_and_fetch_op(2)

    assert isinstance(result[0], np.ndarray)
    assert result[0].shape[0] == 2


# Training Tests
# 1 Graph
def test_training_cell_1_graph():
    result = setup_deepgmg_cell_training_and_fetch_op(1)


def test_training_cell_2_graphs():
    result = setup_deepgmg_cell_training_and_fetch_op(2)
