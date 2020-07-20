import os
import shutil
from collections import defaultdict
from typing import Dict

import numpy as np
import tensorflow as tf


# Constants
LABEL_OFFSET = 20
I_OFFSET = 40

# Enums
#######
class AE:
    (
        GRAPH_IDX,
        STEP_IDX,
        ACTION,
        LAST_ADDED_NODE_ID,
        LAST_ADDED_NODE_TYPE,
        ACTIONS,
        GRAPH,
        NODE_STATES,
        ADJ_LIST,
        ACTION_CURRENT_IDX,
        ACTION_CURRENT,
        SKIP_NEXT,
        SUBGRAPH_START,
        NUM_NODES,
        PROBABILITY,
        NUMS_INCOMING_EDGES_BY_TYPE,
        KERNEL_NAME,
    ) = range(0, 17)


# Labels
class L:
    LABEL_0, LABEL_1 = range(LABEL_OFFSET, LABEL_OFFSET + 2)


# Type
class T:
    NODES, EDGES, NODE_VALUES = range(30, 33)


# Inputs
class I:
    AUX_IN_0 = range(LABEL_OFFSET, I_OFFSET + 1)


# Functions
###########
def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(
        low=-initialization_range, high=initialization_range, size=shape
    ).astype(np.float32)


def graph_to_adjacency_lists(
    graph, tie_fwd_bkwd, edge_type_filter=[]
) -> (Dict[int, np.ndarray], Dict[int, Dict[int, int]]):
    adj_lists = defaultdict(list)
    num_incoming_edges_dicts_per_type = defaultdict(lambda: defaultdict(lambda: 0))
    for src, e, dest in graph:
        fwd_edge_type = e
        if fwd_edge_type not in edge_type_filter and len(edge_type_filter) > 0:
            continue

        adj_lists[fwd_edge_type].append((src, dest))
        num_incoming_edges_dicts_per_type[fwd_edge_type][dest] += 1

        if tie_fwd_bkwd:
            adj_lists[fwd_edge_type].append((dest, src))
            num_incoming_edges_dicts_per_type[fwd_edge_type][src] += 1

    final_adj_lists = {
        e: np.array(sorted(lm), dtype=np.int32) for e, lm in adj_lists.items()
    }

    return final_adj_lists, num_incoming_edges_dicts_per_type


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


# Classes
#########
class MLP(object):
    def __init__(self, in_size, out_size, hid_sizes, activation, func_name):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.activation = activation
        self.func_name = func_name
        self.params = self.make_network_params()

    def make_network_params(self) -> dict:
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [
            tf.Variable(self.init_weights(s), name="%s_W_layer%i" % (self.func_name, i))
            for (i, s) in enumerate(weight_sizes)
        ]
        biases = [
            tf.Variable(
                np.zeros(s[-1]).astype(np.float32),
                name="%s_b_layer%i" % (self.func_name, i),
            )
            for (i, s) in enumerate(weight_sizes)
        ]

        network_params = {
            "weights": weights,
            "biases": biases,
        }

        return network_params

    def init_weights(self, shape: tuple):
        return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (
            2 * np.random.rand(*shape).astype(np.float32) - 1
        )

    def __call__(self, inputs):
        acts = inputs
        for W, b in zip(self.params["weights"], self.params["biases"]):
            hid = tf.matmul(acts, W) + b
            if self.activation == "relu":
                acts = tf.nn.relu(hid)
            elif self.activation == "sigmoid":
                acts = tf.nn.sigmoid(hid)
            elif self.activation == "linear":
                acts = hid
            else:
                raise Exception("Unknown activation function: %s" % self.activation)
        last_hidden = hid
        return last_hidden
