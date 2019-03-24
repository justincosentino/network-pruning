""" Functions for building and pruning/sparsifying models. """

import tensorflow as tf
from ..utils import get_checkpoint_dir, get_checkpoint_name


def build_model(name="dense_model"):
    """
    Returns a sequential keras model of the following form:

    Layer (type)                 Output Shape              Param #
    =================================================================
    hidden_1 (Dense)             (None, 1000)              784000
    _________________________________________________________________
    hidden_2 (Dense)             (None, 1000)              1000000
    _________________________________________________________________
    hidden_3 (Dense)             (None, 500)               500000
    _________________________________________________________________
    hidden_4 (Dense)             (None, 200)               100000
    _________________________________________________________________
    output (Dense)               (None, 10)                2000
    =================================================================
    Total params: 2,386,000
    Trainable params: 2,386,000
    Non-trainable params: 0
    _________________________________________________________________
    """
    with tf.name_scope(name=name):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(784,), name="input"),
            tf.keras.layers.Dense(
                1000, activation=tf.nn.relu, use_bias=False, name="hidden_1"),
            tf.keras.layers.Dense(
                1000, activation=tf.nn.relu, use_bias=False, name="hidden_2"),
            tf.keras.layers.Dense(
                500, activation=tf.nn.relu, use_bias=False, name="hidden_3"),
            tf.keras.layers.Dense(
                200, activation=tf.nn.relu, use_bias=False, name="hidden_4"),
            tf.keras.layers.Dense(
                10, activation=tf.nn.softmax, use_bias=False, name="output")
        ])
        model.summary()
        return model


def load_model_weights(experiment_dir, learning_rate):
    """
    Loads a saved model's weights from the associated experiment checkpoint
    directory.
    """

    # Throw if the checkpoint directory does not exist
    checkpoint_dir = get_checkpoint_dir(experiment_dir)
    if not tf.io.gfile.exists(checkpoint_dir):
        raise Exception(
            "Model checkpoint directory '{}' ".format(checkpoint_dir) +
            "does not exist. Aborting.")

    # Load the weights and compile the model
    model = build_model()
    model.load_weights(get_checkpoint_name(checkpoint_dir))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

    return model


def convert_dense_to_weight_pruned(model, k=0):
    """
    TODO: Implement me.
    """
    raise Exception("convert_dense_to_weight_pruned:: Not implemented")


def convert_dense_to_unit_pruned(model, k=0):
    """
    TODO: Implement me.
    """
    raise Exception("convert_dense_to_unit_pruned:: Not implemented")


def convert_dense_to_sparse_weight(model, k=0):
    """
    TODO: Implement me.
    """
    raise Exception("convert_dense_to_sparse_weight:: Not implemented")
