import tensorflow as tf


class PropagationModelLayer(object):
    def __init__(self):
        self.placeholders = {}

    def compute_embeddings(self, embeddings: tf.Tensor) -> tf.Tensor:
        """Uses the model layer to process embeddings to new embeddings. All embeddings are in one dimension.
        Propagation is made in one pass with many disconnected graphs.

        Args:
            embeddings: Tensor of shape [V, D].

        Returns:
            Tensor of shape [V, D].
        """
        raise NotImplementedError
