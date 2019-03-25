""" Functions for building and pruning/sparsifying models. """

import tensorflow as tf
import numpy as np
from ..utils import get_checkpoint_dir, get_checkpoint_name

FLAGS = tf.app.flags.FLAGS


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
                1000,
                activation=tf.nn.relu,
                kernel_regularizer=tf.keras.regularizers.l1_l2(
                    l1=FLAGS.l1_reg,
                    l2=FLAGS.l2_reg),
                use_bias=False,
                name="hidden_1"),
            tf.keras.layers.Dense(
                1000,
                activation=tf.nn.relu,
                kernel_regularizer=tf.keras.regularizers.l1_l2(
                    l1=FLAGS.l1_reg,
                    l2=FLAGS.l2_reg),
                use_bias=False,
                name="hidden_2"),
            tf.keras.layers.Dense(
                500,
                activation=tf.nn.relu,
                kernel_regularizer=tf.keras.regularizers.l1_l2(
                    l1=FLAGS.l1_reg,
                    l2=FLAGS.l2_reg),
                use_bias=False,
                name="hidden_3"),
            tf.keras.layers.Dense(
                200,
                activation=tf.nn.relu,
                kernel_regularizer=tf.keras.regularizers.l1_l2(
                    l1=FLAGS.l1_reg,
                    l2=FLAGS.l2_reg),
                use_bias=False,
                name="hidden_4"),
            tf.keras.layers.Dense(
                10, activation=tf.nn.softmax, use_bias=False, name="output")
        ])
        model.summary()
        return model


def load_model_weights(experiment_dir):
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

    # TODO: it would be nice to not compile each time we copy values or load
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=FLAGS.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    return model


def convert_dense_to_weight_pruned(model, k=0):
    """
    Given a dense model, perform k%-weight pruning and return a sparsified
    model.
    """
    # Clone the existing dense model
    new_model = tf.keras.models.clone_model(model)
    new_model.set_weights(model.get_weights())

    # Skip output layer
    for layer in new_model.layers[:-1]:
        # l1 norm the weights for the given layer
        weights = layer.get_weights()[0]
        normed = np.absolute(weights)

        # Find the k% threshold value (< threshold_val should be zeroed)
        kth_index = int(weights.size * k)
        threshold_val = np.partition(normed.flatten(), kth_index)[kth_index]

        # Zero k% of weights
        weights[normed < threshold_val] = 0
        layer.set_weights([weights])

    # TODO: it would be nice to not compile each time we clone values or load
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=FLAGS.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    return new_model


def convert_dense_to_unit_pruned(model, k=0):
    """
    Given a dense model, perform k%-unit pruning and return a sparsified model.
    """
    # Clone the existing dense model
    new_model = tf.keras.models.clone_model(model)
    new_model.set_weights(model.get_weights())

    # Skip output layer
    for layer in new_model.layers[:-1]:
        # l2 norm the weights for the given layer
        weights = layer.get_weights()[0]
        normed = np.linalg.norm(weights, ord=2, axis=0)

        # Find the k% threshold value (< threshold_val should be zeroed)
        kth_index = int(normed.size * k)
        threshold_val = np.partition(normed.flatten(), kth_index)[kth_index]

        # Zero k% of units
        weights[:, normed < threshold_val] = 0
        layer.set_weights([weights])

    # TODO: it would be nice to not compile each time we clone values or load
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=FLAGS.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    return new_model


def convert_dense_to_sparse_weight(model, k=0):
    """
    TODO: Implement me.
    """
    raise Exception("convert_dense_to_sparse_weight:: Not implemented")
