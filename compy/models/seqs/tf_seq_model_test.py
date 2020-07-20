import os
import tensorflow.python.util.deprecation as deprecation

from compy.models.seqs.tf_seq_model import RnnTfModel
from compy.representations.common import Sequence

# Disable TensorFlow messages and deprecation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
deprecation._PRINT_DEPRECATION_WARNINGS = False


def test_model():
    config = {
        "batch_size": 4,
        "num_epochs": 1,
    }
    model = RnnTfModel(config, num_types=3)

    data = [
        {
            "x": {
                "code_rep": Sequence(
                    ["a", "b", "c", "a", "b", "b", "a"], ["a", "b", "c"]
                ),
                "aux_in": [0, 0],
            },
            "y": 0,
        }
    ]
    model.train(data, data)
