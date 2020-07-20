import os
import networkx as nx
import tensorflow.python.util.deprecation as deprecation

from compy.models.graphs.tf_graph_model import GnnTfModel
from compy.models.graphs.tf_graph_model import GnnTfModelState
from compy.representations.common import Graph

# Disable TensorFlow messages and deprecation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
deprecation._PRINT_DEPRECATION_WARNINGS = False


CONFIG = {
    "graph_rnn_cell": "gru",
    "num_timesteps": 2,
    "gnn_h_size": 4,
    "gnn_m_size": 2,
    "num_edge_types": 1,
    "prediction_cell": {
        "mlp_f_m_dims": [],
        "mlp_f_m_activation": "relu",
        "mlp_g_m_dims": [],
        "mlp_g_m_activation": "relu",
        "mlp_reduce_dims": [],
        "mlp_reduce_activation": "relu",
        "mlp_reduce_after_aux_in_1_dims": [],
        "mlp_reduce_after_aux_in_1_activation": "relu",
        "mlp_reduce_after_aux_in_1_out_dim": 2,
        "mlp_reduce_after_aux_in_2_dims": [],
        "mlp_reduce_after_aux_in_2_activation": "sigmoid",
        "mlp_reduce_after_aux_in_2_out_dim": 2,
        "mlp_reduce_out_dim": 8,
        "output_dim": 2,
    },
    "embedding_layer": {"mapping_dims": []},
    "learning_rate": 0.0005,
    "clamp_gradient_norm": 1.0,
    "L2_loss_factor": 0,
    "batch_size": 4,
    "num_epochs": 1,
    "tie_fwd_bkwd": 1,
    "use_edge_bias": 0,
    "save_best_model_interval": 1,
    "with_aux_in": 0,
    "with_gradient_monitoring": 1,
    "seed": 0,
}


def test_train_model():
    dummy_graph = nx.MultiDiGraph()
    dummy_graph.add_node("n1", attr="a")
    dummy_graph.add_node("n2", attr="b")
    dummy_graph.add_node("n3", attr="c")

    config = CONFIG
    config["hidden_size_orig"] = len(dummy_graph)

    state = GnnTfModelState(config)
    model = GnnTfModel(config, state)

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

    state.backup_best_weights()
    state.restore_best_weights()

    num_params = state.count_number_trainable_params()
    assert num_params
