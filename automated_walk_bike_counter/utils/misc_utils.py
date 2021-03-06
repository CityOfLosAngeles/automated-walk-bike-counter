# Copyright (c) Data Science Research Lab at California State University Los
# Angeles (CSULA), and City of Los Angeles ITA
# Distributed under the terms of the Apache 2.0 License
# www.calstatela.edu/research/data-science
# Designed and developed by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

import fsspec
import numpy as np
import tensorflow as tf


def parse_anchors(anchor_path):
    """
    parse anchors.
    returned data: shape [N, 2], dtype float32
    """
    with fsspec.open(anchor_path, "r") as anchor_file:
        anchors = np.reshape(
            np.asarray(anchor_file.read().split(","), np.float32), [-1, 2]
        )
        return anchors


def read_class_names(class_name_path):
    names = {}
    with fsspec.open(class_name_path, "r") as data:
        for i, name in enumerate(data):
            names[i] = name.strip("\n")
    return names


def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    param:
        var_list: list of network variables.
        weights_file: name of the binary file.
    """
    with fsspec.open(weights_file, "rb") as fp:
        np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if "Conv" in var1.name.split("/")[-2]:
            # check type of next layer
            if "BatchNorm" in var2.name.split("/")[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1 : i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr : ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif "Conv" in var2.name.split("/")[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr : ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr : ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1])
            )
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops
