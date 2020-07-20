import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from compy.models.model import Model


class SummaryCallback(tf.keras.callbacks.Callback):
    def __init__(self, summary):
        self.__summary = summary

    def on_epoch_end(self, epoch, logs=None):
        self.__summary["accuracy"] = logs["dense_2_accuracy"]
        self.__summary["loss"] = logs["loss"]


class RnnTfModel(Model):
    def __init__(self, config=None, num_types=None):
        if not config:
            config = {
                "learning_rate": 0.001,
                "batch_size": 64,
                "num_epochs": 1000,
            }
        super().__init__(config)

        self.__num_types = num_types

        np.random.seed(0)

        # Language model. Takes as inputs source code sequences
        code_in = tf.keras.layers.Input(shape=(1024,), dtype="int32", name="code_in")
        x = tf.keras.layers.Embedding(
            input_dim=num_types + 1, input_length=1024, output_dim=64, name="embedding"
        )(code_in)
        x = tf.keras.layers.LSTM(
            64, implementation=1, return_sequences=True, name="lstm_1"
        )(x)
        x = tf.keras.layers.LSTM(64, implementation=1, name="lstm_2")(x)
        langmodel_out = tf.keras.layers.Dense(2, activation="sigmoid")(x)

        # Auxiliary inputs. wgsize and dsize
        auxiliary_inputs = tf.keras.layers.Input(shape=(2,))

        # Heuristic model. Takes as inputs the language model, outputs 1-hot encoded device mapping
        x = tf.keras.layers.Concatenate()([auxiliary_inputs, x])
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        out = tf.keras.layers.Dense(2, activation="sigmoid")(x)

        self.model = tf.keras.models.Model(
            inputs=[auxiliary_inputs, code_in], outputs=[out, langmodel_out]
        )
        self.model.compile(
            optimizer="adam",
            metrics=["accuracy"],
            loss=["categorical_crossentropy", "categorical_crossentropy"],
            loss_weights=[1.0, 0.2],
        )

    def __process_data(self, data):
        processed = {"sequences": [], "aux_in": [], "label": []}
        for item in data:
            processed["sequences"].append(item["x"]["code_rep"].get_token_list())
            processed["aux_in"].append(item["x"]["aux_in"])
            processed["label"].append(item["y"])

        return processed

    def __process(self, data):
        # Pad sequences
        encoded = np.array(
            tf.keras.preprocessing.sequence.pad_sequences(
                data["sequences"], maxlen=1024, value=self.__num_types
            )
        )
        seqs = np.vstack([np.expand_dims(x, axis=0) for x in encoded])

        aux_in = data["aux_in"]

        # Encode labels one-hot
        ys = tf.keras.utils.to_categorical(data["label"], num_classes=2)

        return seqs, aux_in, ys

    def _train_with_batch(self, batch):
        seqs, aux_in, ys = self.__process(self.__process_data(batch))

        summary = {}
        callback = SummaryCallback(summary)

        self.model.fit(
            x=[np.array(aux_in), np.array(seqs)],
            y=[np.array(ys), np.array(ys)],
            epochs=1,
            batch_size=self.config["batch_size"],
            verbose=False,
            shuffle=True,
            callbacks=[callback],
        )

        return summary["loss"], summary["accuracy"]

    def _predict_with_batch(self, batch):
        seqs, aux_in, ys = self.__process(self.__process_data(batch))

        pred = self.model.predict(
            x=[np.array(aux_in), np.array(seqs)], batch_size=999999, verbose=False
        )[0]

        valid_accuracy = np.sum(np.argmax(pred, axis=1) == np.argmax(ys, axis=1)) / len(
            pred
        )

        return valid_accuracy, pred
