import numpy as np
import tensorflow as tf

from compy.models.graphs.tf.cell.prediction_cell import PredictionCell
from compy.models.graphs.tf.cell.prediction_cell import PredictionCellState
from compy.models.graphs.tf.layer.embedding_layer import EmbeddingLayer
from compy.models.graphs.tf.layer.embedding_layer import EmbeddingLayerState
from compy.models.graphs.tf.layer.gnn_model_layer import GGNNModelLayer
from compy.models.graphs.tf.layer.gnn_model_layer import GGNNModelLayerState
from compy.models.graphs.tf import utils
from compy.models.model import Model


class GnnTfModelState(object):
    """Holds the state of the prediction model."""

    def __init__(self, config):
        self.graph = tf.Graph()

        self.best_epoch_weights = None

        seed = config["seed"]
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)

        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(graph=self.graph, config=tf_config)

        with self.graph.as_default():
            self.embedding_layer_state = EmbeddingLayerState(config)
            self.ggnn_layer_state = GGNNModelLayerState(config)
            self.prediction_cell_state = PredictionCellState(config)

    def __get_weights(self):
        weights = {}
        for variable in self.sess.graph.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES
        ):
            weights[variable.name] = self.sess.run(variable)

        return weights

    def __set_weights(self, weights):
        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES
            ):
                used_vars.add(variable.name)
                if variable.name in weights:
                    restore_ops.append(variable.assign(weights[variable.name]))
                else:
                    # print('Initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in weights:
                if var_name not in used_vars:
                    print("Saved weights for %s not used by model." % var_name)
            self.sess.run(restore_ops)

    def backup_best_weights(self):
        """Backs up current state of the model."""
        self.best_epoch_weights = self.__get_weights()

    def restore_best_weights(self):
        """Restores best state of the model."""
        if self.best_epoch_weights:
            self.__set_weights(self.best_epoch_weights)

    def save_weights_to_disk(self, path):
        """Saves model weights to given file."""
        data = self.__get_weights()

        with open(path, "wb") as out_file:
            pickle.dump(data, out_file, pickle.HIGHEST_PROTOCOL)

    def restore_weights_from_disk(self, path):
        """Saves model weights to given file."""
        # print("Restoring weights from file %s." % path)
        with open(path, "rb") as in_file:
            data = pickle.load(in_file)

        self.__set_weights(data)

    def count_number_trainable_params(self):
        """Counts the number of trainable variables.
        Taken from https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
        """
        tot_nb_params = 0
        for trainable_variable in self.sess.graph.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES
        ):
            shape = trainable_variable.get_shape()  # e.g [D,F] or [W,H,C]
            current_nb_params = self.get_nb_params_shape(shape)
            tot_nb_params = tot_nb_params + current_nb_params
        return tot_nb_params

    def get_nb_params_shape(self, shape):
        """Computes the total number of params for a given shape.
        Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
        Taken from https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
        """
        nb_params = 1
        for dim in shape:
            nb_params = nb_params * int(dim)
        return nb_params


class GnnTfModel(Model):
    def __init__(self, config=None, state=None, num_types=None):
        if not config:
            config = {
                "graph_rnn_cell": "gru",
                "num_timesteps": 8,
                "hidden_size_orig": num_types,
                "gnn_h_size": 32,
                "gnn_m_size": 2,
                "num_edge_types": 4,
                "prediction_cell": {
                    "mlp_f_m_dims": [32, 32],
                    "mlp_f_m_activation": "relu",
                    "mlp_g_m_dims": [32, 32],
                    "mlp_g_m_activation": "relu",
                    "mlp_reduce_dims": [32, 32],
                    "mlp_reduce_activation": "relu",
                    "mlp_reduce_out_dim": 64,
                    "mlp_reduce_after_aux_in_1_dims": [],
                    "mlp_reduce_after_aux_in_1_activation": "relu",
                    "mlp_reduce_after_aux_in_1_out_dim": 32,
                    "mlp_reduce_after_aux_in_2_dims": [],
                    "mlp_reduce_after_aux_in_2_activation": "sigmoid",
                    "mlp_reduce_after_aux_in_2_out_dim": 2,
                    "output_dim": 2,
                },
                "embedding_layer": {"mapping_dims": [32, 32]},
                "learning_rate": 0.001,
                "clamp_gradient_norm": 1.0,
                "L2_loss_factor": 0,
                "batch_size": 64,
                "num_epochs": 1000,
                "tie_fwd_bkwd": 0,
                "use_edge_bias": 0,
                "save_best_model_interval": 1,
                "with_aux_in": 1,
                "seed": 0,
            }
        super().__init__(config)

        if not state:
            state = GnnTfModelState(config)
        self.state = state

        self.ggnn_layers = []
        self.cells = []

        with self.state.graph.as_default():
            self.ops = {}
            self.placeholders = {}

        self.with_gradient_monitoring = (
            True
            if "gradient_monitoring" in self.config
            and self.config["gradient_monitoring"] == 1
            else False
        )
        self.with_aux_in = (
            True
            if "with_aux_in" in self.config and self.config["with_aux_in"] == 1
            else False
        )

        with self.state.graph.as_default():
            # Create and initialize model
            self.__make_model(True)
            self.__make_train_step()
            self.__initialize_model()

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

    def __initialize_model(self) -> None:
        """
        Init tf model
        """
        init_op = tf.group(
            tf.compat.v1.global_variables_initializer(),
            tf.compat.v1.local_variables_initializer(),
        )
        self.state.sess.run(init_op)

    def __graphs_to_batch_feed_dict(
        self, graphs: list, graph_sizes: list, is_training: bool
    ) -> dict:
        """Creates feed dict that is the format the tf model is trained with."""
        num_edge_types = self.config["num_edge_types"]

        batch_data = {
            # Graph model
            "adjacency_lists": [[] for _ in range(self.config["num_edge_types"])],
            "embeddings_to_graph_mappings": [],
            "labels": [],
            "embeddings_in": [],
            "node_values": [],
        }

        if self.with_aux_in:
            batch_data["aux_in"] = []

        for graph_idx, graph in enumerate(graphs):
            num_nodes = graph_sizes[graph_idx]

            if self.config["use_edge_bias"] == 1:
                batch_data["num_incoming_edges_dicts_per_type"] = []

            # Aux in
            if self.with_aux_in:
                if "aux_in" in graph:
                    batch_data["aux_in"].append(graph["aux_in"])

            # Labels
            if "label" in graph:
                batch_data["labels"].append(graph["label"])

            # Graph model: Adj list
            adj_lists = graph[utils.AE.ADJ_LIST]
            for idx, adj_list in adj_lists.items():
                if idx >= self.config["num_edge_types"]:
                    continue
                batch_data["adjacency_lists"][idx].append(adj_list)

            if self.config["use_edge_bias"] == 1:
                # Graph model: Incoming edge numbers
                num_incoming_edges_dicts_per_type = (
                    action[utils.AE.NUMS_INCOMING_EDGES_BY_TYPE]
                    if action
                    else utils.graph_to_adjacency_lists(
                        [], self.config["tie_fwd_bkwd"], self.config["edge_type_filter"]
                    )[0]
                )
                num_incoming_edges_per_type = np.zeros((num_nodes, num_edge_types))
                for (
                    e_type,
                    num_incoming_edges_per_type_dict,
                ) in num_incoming_edges_dicts_per_type.items():
                    for (
                        node_id,
                        edge_count,
                    ) in num_incoming_edges_per_type_dict.items():
                        num_incoming_edges_per_type[node_id, e_type] = edge_count
                batch_data["num_incoming_edges_dicts_per_type"].append(
                    num_incoming_edges_per_type
                )

            graph_mappings_all = np.full(num_nodes, graph_idx)
            batch_data["embeddings_to_graph_mappings"].append(graph_mappings_all)

            node_types = utils.get_one_hot(
                np.array(graph["nodes"]), self.config["hidden_size_orig"]
            ).astype(float)
            batch_data["embeddings_in"].append(node_types)

        # Build feed dict
        feed_dict = {}

        # Graph model: Adj list
        for idx, adj_list in enumerate(
            self.ggnn_layers[0].placeholders["adjacency_lists"]
        ):
            feed_dict[adj_list] = np.array(batch_data["adjacency_lists"][idx])
            if len(feed_dict[adj_list]) == 0:
                feed_dict[adj_list] = np.zeros((0, 2), dtype=np.int32)
            else:
                feed_dict[adj_list] = feed_dict[adj_list][0]

        if self.config["use_edge_bias"] == 1:
            # Graph model: Incoming edge numbers
            feed_dict[
                self.ggnn_layers[0].placeholders["num_incoming_edges_per_type"]
            ] = np.concatenate(batch_data["num_incoming_edges_dicts_per_type"], axis=0)

        # Is training
        feed_dict[self.cells[0].placeholders["is_training"]] = is_training

        # Aux in
        if self.with_aux_in:
            feed_dict[self.cells[0].placeholders["aux_in"]] = batch_data["aux_in"]

        # Labels
        if "labels" in self.cells[0].placeholders:
            feed_dict[self.cells[0].placeholders["labels"]] = utils.get_one_hot(
                np.array(batch_data["labels"]),
                self.config["prediction_cell"]["output_dim"],
            )

        # Embeddings
        feed_dict[self.placeholders["embeddings_in"]] = np.concatenate(
            batch_data["embeddings_in"], axis=0
        )
        feed_dict[self.cells[0].placeholders["embeddings_to_graph_mappings"]] = [
            np.concatenate(batch_data["embeddings_to_graph_mappings"], axis=0)
        ]

        return feed_dict

    def __make_model(self, enable_training) -> None:
        """Creates tf model."""
        self.placeholders["embeddings_in"] = tf.compat.v1.placeholder(
            tf.float32, [None, self.config["hidden_size_orig"]], name="embeddings_in"
        )

        # Create model: Unroll network and wire embeddings
        embeddings_reduced = self.placeholders["embeddings_in"]

        # Create embedding layer
        embedding_layer = EmbeddingLayer(self.config, self.state.embedding_layer_state)
        embeddings = embedding_layer.compute_embeddings(embeddings_reduced)

        # Create propagation layer
        ggnn_layer = GGNNModelLayer(self.config, self.state.ggnn_layer_state)
        embeddings = ggnn_layer.compute_embeddings(embeddings)

        # Create prediction cell
        prediction_cell = PredictionCell(
            self.config,
            enable_training,
            self.state.prediction_cell_state,
            self.with_aux_in,
        )
        prediction_cell.initial_embeddings = embeddings_reduced
        prediction_cell.compute_predictions(embeddings)

        self.ggnn_layers.append(ggnn_layer)
        self.cells.append(prediction_cell)

        if enable_training:
            # Cell losses
            losses = []
            for cell in self.cells:
                losses.append(cell.ops["loss"])
            self.ops["losses"] = losses

            # Regularization loss
            vars = tf.compat.v1.trainable_variables()
            lossL2 = tf.add_n(
                [tf.nn.l2_loss(v) for v in vars if "bias" not in v.name]
            ) * float(self.config["L2_loss_factor"])

            self.ops["loss"] = tf.reduce_sum(losses) + lossL2

    def __make_train_step(self) -> None:
        """Creates tf train step."""
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            trainable_vars = self.state.sess.graph.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES
            )

            optimizer = tf.compat.v1.train.AdamOptimizer(self.config["learning_rate"])
            grads_and_vars = optimizer.compute_gradients(
                self.ops["loss"], var_list=trainable_vars
            )

            # Clipping
            clipped_grads = []
            for grad, var in grads_and_vars:
                if grad is not None:
                    clipped_grads.append(
                        (tf.clip_by_norm(grad, self.config["clamp_gradient_norm"]), var)
                    )
                else:
                    clipped_grads.append((grad, var))

            # Monitoring
            if self.with_gradient_monitoring:
                self.ops["gradients"] = tf.summary.merge(
                    [
                        tf.summary.histogram("%s-grad" % g[1].name, g[0])
                        for g in grads_and_vars
                    ]
                )
                self.ops["clipped_gradients"] = tf.summary.merge(
                    [
                        tf.summary.histogram("%s-clipped-grad" % g[1].name, g[0])
                        for g in clipped_grads
                    ]
                )

            # Apply
            self.ops["train_step"] = optimizer.apply_gradients(clipped_grads)

        # Initialize newly-introduced variables:
        self.state.sess.run(tf.compat.v1.local_variables_initializer())

    def __run_batch(self, feed_dict):
        """Trains model with one batch and retrieve result."""
        fetch_list = [
            self.ops["loss"],
            self.cells[0].ops["output"],
            self.ops["train_step"],
        ]
        if self.with_gradient_monitoring:
            offset = len(fetch_list)
            fetch_list.extend([self.ops["gradients"], self.ops["clipped_gradients"]])

        result = self.state.sess.run(fetch_list, feed_dict=feed_dict)

        return result

    def _train_init(self, data_train, data_valid):
        # Train
        # - Convert to format
        data_train = self.__process_data(data_train)

        # - Enrich graphs with adj list
        for graph in data_train:
            graph[utils.AE.ADJ_LIST] = utils.graph_to_adjacency_lists(
                graph["edges"], self.config["tie_fwd_bkwd"] == 1
            )[0]

        # Valid
        # - Convert to format
        data_valid = self.__process_data(data_valid)

        # - Enrich graphs with adj list
        for graph in data_valid:
            graph[utils.AE.ADJ_LIST] = utils.graph_to_adjacency_lists(
                graph["edges"], self.config["tie_fwd_bkwd"] == 1
            )[0]

        return data_train, data_valid

    def __build_tf_graphs(self, data, is_training):
        # Extract graph sizes
        graph_sizes = []
        for graph in data:
            graph_sizes.append(len(graph["nodes"]))

        # Extract train labels
        y_train = []
        for graph in data:
            y_train.append(graph["label"])
        y_train = np.array(y_train)

        # Build feed dict
        feed_dict = self.__graphs_to_batch_feed_dict(data, graph_sizes, is_training)

        return feed_dict, y_train

    def _train_with_batch(self, batch):
        feed_dict, labels = self.__build_tf_graphs(batch, True)
        result = self.__run_batch(feed_dict)

        pred = result[1]
        train_accuracy = np.sum(list(np.argmax(pred, axis=1) == labels)) / len(pred)
        train_loss = result[0]

        return train_loss, train_accuracy

    def _predict_with_batch(self, batch):
        feed_dict, labels = self.__build_tf_graphs(batch, False)

        # Run
        fetch_list = [self.cells[0].ops["output"]]
        result = self.state.sess.run(fetch_list, feed_dict=feed_dict)

        pred = result[0]
        valid_accuracy = np.sum(np.argmax(pred, axis=1) == labels) / len(pred)

        return valid_accuracy, pred
