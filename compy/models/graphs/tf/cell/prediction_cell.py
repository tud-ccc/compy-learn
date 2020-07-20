import tensorflow as tf

from compy.models.graphs.tf import utils


class PredictionCellState(object):
    """Holds the state / weights of a Prediction cell."""

    def __init__(self, config):
        self.config = config

        h_size = self.config["gnn_h_size"]
        h_size_orig = self.config["hidden_size_orig"]
        m_size = self.config["gnn_m_size"]

        self.weights = {}

        self.weights["mlp_f_m"] = utils.MLP(
            h_size,
            h_size * m_size,
            self.config["prediction_cell"]["mlp_f_m_dims"],
            self.config["prediction_cell"]["mlp_f_m_activation"],
            "mlp_regression_transform",
        )
        self.weights["mlp_g_m"] = utils.MLP(
            h_size + h_size_orig,
            h_size * m_size,
            self.config["prediction_cell"]["mlp_g_m_dims"],
            self.config["prediction_cell"]["mlp_g_m_activation"],
            "mlp_regression_gate",
        )

        self.weights["mlp_reduce"] = utils.MLP(
            h_size * m_size,
            h_size * m_size,
            self.config["prediction_cell"]["mlp_reduce_dims"],
            self.config["prediction_cell"]["mlp_reduce_activation"],
            "mlp_reduce",
        )

        offset = 0
        if config["with_aux_in"] == 1:
            offset = 2
        self.weights["mlp_reduce_after_aux_in_1"] = utils.MLP(
            h_size * m_size + offset,
            self.config["prediction_cell"]["mlp_reduce_after_aux_in_1_out_dim"],
            self.config["prediction_cell"]["mlp_reduce_after_aux_in_1_dims"],
            self.config["prediction_cell"]["mlp_reduce_after_aux_in_1_activation"],
            "mlp_reduce_after_aux_in_1",
        )

        self.weights["mlp_reduce_after_aux_in_2"] = utils.MLP(
            self.config["prediction_cell"]["mlp_reduce_after_aux_in_1_out_dim"],
            self.config["prediction_cell"]["mlp_reduce_after_aux_in_2_out_dim"],
            self.config["prediction_cell"]["mlp_reduce_after_aux_in_2_dims"],
            self.config["prediction_cell"]["mlp_reduce_after_aux_in_2_activation"],
            "mlp_reduce_after_aux_in_2",
        )

        self.weights["graph_model_out"] = utils.MLP(
            self.config["prediction_cell"]["mlp_reduce_out_dim"],
            self.config["prediction_cell"]["mlp_reduce_after_aux_in_2_out_dim"],
            [],
            "sigmoid",
            "graph_model_out",
        )


class PredictionCell(object):
    """Implementation of the Prediction cell."""

    def __init__(self, config, enable_training, state, with_aux_in):
        self.config = config
        self.enable_training = enable_training
        self.state = state
        self.with_aux_in = with_aux_in

        self.ops = {}
        self.placeholders = {}

    def compute_predictions(self, embeddings: tf.Tensor) -> tf.Tensor:
        """
        Make prediction based on node embeddings.

        Args:
            embeddings: Tensor of shape [b*v, h].

        Returns:
            Tensor of predictions.
        """
        # Placeholders
        # #########################################
        # # Initial embeddings
        # self.placeholders['initial_embeddings'] = tf.compat.v1.placeholder(tf.float32, [None, self.config['gnn_h_size']], name='initial_embeddings')
        # initial_embeddings = self.placeholders['initial_embeddings']

        # Is training (for batch norm)
        self.placeholders["is_training"] = tf.compat.v1.placeholder(
            tf.bool, None, name="is_training"
        )
        is_training = self.placeholders["is_training"]

        # Embeddings to graph mappings
        self.placeholders["embeddings_to_graph_mappings"] = tf.compat.v1.placeholder(
            tf.int32, [1, None], name="embeddings_to_graph_mappings"
        )
        embeddings_to_graph_mappings = self.placeholders["embeddings_to_graph_mappings"]
        num_graphs = tf.reduce_max(embeddings_to_graph_mappings) + 1  # Scalar

        # Input
        if self.with_aux_in:
            self.placeholders["aux_in"] = tf.compat.v1.placeholder(
                tf.float32, [None, 2], name="aux_in"
            )
            aux_in = tf.cast(self.placeholders["aux_in"], dtype=tf.float32)

        # Graph Model
        # #########################################
        gate_input = tf.concat(
            [embeddings, tf.expand_dims(self.initial_embeddings, 0)], axis=-1
        )  # [b*v, 2h + h_init]
        h_v_G = self.state.weights["mlp_f_m"](embeddings)  # [b*v, 2h]
        g_v_G = self.state.weights["mlp_g_m"](gate_input)  # [b*v, 2h]
        g_v_G = tf.nn.sigmoid(g_v_G)  # [b*v, 2h]

        h_G = h_v_G * g_v_G  # [b*v, 2h]

        # Sum up all nodes per graph
        h_G = tf.compat.v1.unsorted_segment_sum(
            data=h_G, segment_ids=embeddings_to_graph_mappings, num_segments=num_graphs
        )  # [b, 2h]
        h_G = self.state.weights["mlp_reduce"](h_G)  # [b, 2]

        graphmodel_logits = self.state.weights["graph_model_out"](h_G)

        # Prediction Model
        # #########################################
        if self.with_aux_in:
            h_G = tf.concat([h_G, aux_in], axis=-1)  # [b, 2h + 2]

        h_G = tf.compat.v1.layers.batch_normalization(h_G, training=is_training)

        output = self.state.weights["mlp_reduce_after_aux_in_1"](h_G)  # [b, 32]
        output = self.state.weights["mlp_reduce_after_aux_in_2"](output)  # [b, 2]

        predictionmodel_logits = tf.nn.softmax(output)

        # Training
        if self.enable_training:
            # Input
            self.placeholders["labels"] = tf.compat.v1.placeholder(
                tf.int32,
                [None, self.config["prediction_cell"]["output_dim"]],
                name="labels",
            )
            labels = tf.cast(self.placeholders["labels"], dtype=tf.float32)

            # Graph model
            graphmodel_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels, logits=graphmodel_logits
            )  # [b, 2]

            # Prediction model
            predictionmodel_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels, logits=predictionmodel_logits
            )  # [b, 2]

            # Loss
            loss = tf.reduce_sum(graphmodel_loss + 0.2 * predictionmodel_loss)  # [b]
            self.ops["loss"] = loss

        self.ops["output"] = predictionmodel_logits
