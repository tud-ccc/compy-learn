import tensorflow as tf

from compy.models.graphs.tf import utils
from compy.models.graphs.tf.layer.propagation_model_layer import PropagationModelLayer


class EmbeddingLayerState(object):
    """Holds the state / weights of a Embedding Layer."""

    def __init__(self, config):
        self.config = config

        hidden_size_orig = self.config["hidden_size_orig"]
        h_size = self.config["gnn_h_size"]

        self.weights = {}

        self.weights["mapping"] = utils.MLP(
            hidden_size_orig,
            h_size,
            self.config["embedding_layer"]["mapping_dims"],
            "relu",
            "mapping",
        )


class EmbeddingLayer(PropagationModelLayer):
    """Implementation of the Embedding Layer."""

    def __init__(self, config, state):
        super().__init__()

        self.config = config
        self.state = state

    def compute_embeddings(self, embeddings: tf.Tensor) -> tf.Tensor:
        """
        Uses the model layer to process embeddings to new embeddings. All embeddings are in one dimension.
        Propagation is made in one pass with many disconnected graphs.

        Args:
            embeddings: Tensor of shape [v, h].

        Returns:
            Tensor of shape [v, h].
        """
        embeddings_new = self.state.weights["mapping"](embeddings)  # [v, h]

        return embeddings_new
