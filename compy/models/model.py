import pprint
import time

import numpy as np


class Model(object):
    def __init__(self, config):
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(config)

        self.config = config

    def train(self, data_train, data_valid):
        train_summary = []
        data_train, data_valid = self._train_init(data_train, data_valid)

        print()
        for epoch in range(self.config["num_epochs"]):
            batch_size = self.config["batch_size"]
            np.random.shuffle(data_train)
            batches = [
                data_train[i * batch_size : (i + 1) * batch_size]
                for i in range((len(data_train) + batch_size - 1) // batch_size)
            ]

            # Train
            start_time = time.time()
            for batch in batches:
                train_loss, train_accuracy = self._train_with_batch(batch)
            end_time = time.time()

            # Valid
            self._test_init()

            batch_size = self.config["batch_size"]
            np.random.shuffle(data_valid)
            batches = [
                data_valid[i * batch_size : (i + 1) * batch_size]
                for i in range((len(data_valid) + batch_size - 1) // batch_size)
            ]

            valid_count = 0
            for batch in batches:
                batch_accuracy, _ = self._predict_with_batch(batch)
                valid_count += batch_accuracy * len(batch)
            valid_accuracy = valid_count / len(data_valid)

            # Logging
            instances_per_sec = len(data_train) / (end_time - start_time)
            print(
                "epoch: %i, train_loss: %.8f, train_accuracy: %.4f, valid_accuracy:"
                " %.4f, train instances/sec: %.2f"
                % (epoch, train_loss, train_accuracy, valid_accuracy, instances_per_sec)
            )

            train_summary.append({"train_accuracy": train_accuracy})
            train_summary.append({"valid_accuracy": valid_accuracy})

        return train_summary

    def predict(self, data):
        _, pred = self._predict_with_batch(data)

        return pred

    def _train_init(self, data_train, data_valid):
        return data_train, data_valid

    def _test_init(self):
        pass

    def _train_with_batch(self, batch):
        raise NotImplementedError

    def _predict_with_batch(self, batch):
        raise NotImplementedError
