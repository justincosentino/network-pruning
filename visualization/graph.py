""" Functions used for visualizing data from experiments. """

import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

sns.set(style="darkgrid")
FLAGS = tf.app.flags.FLAGS


def plot_history(history, experiment_dir, output_name='loss_accuracy.png'):
    """
    Plot the training history from model.fit. Displays the test and training
    accuracy and loss over epochs and writes as a png to the specified
    experiment directory.
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.xticks(np.arange(0, FLAGS.epochs, step=1))

    plt.subplot(1, 2, 2)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.xticks(np.arange(0, FLAGS.epochs, step=1))

    plt.savefig(os.path.join(experiment_dir, output_name))


def plot_weights_l1(model, experiment_dir, output_name="weights.png"):
    """
    Plot the distribution of l1 weight norms for the model's current weights.
    Save the plot as a png in the specified directory.
    """
    plt.figure(figsize=(12, 4))
    plt.subplots_adjust(hspace=0.8, wspace=0.4)

    for i, layer in enumerate(model.layers):
        name = layer.get_config()['name']
        weights = np.absolute(layer.get_weights()[0])
        plt.subplot(2, 3, i+1)
        plt.title(name)
        plt.xlabel('L1 Weights Norm')
        plt.ylabel('Frequency')
        plt.hist(weights.reshape(-1), bins=300)

    plt.savefig(os.path.join(experiment_dir, output_name))


def plot_units_l2(model, experiment_dir, output_name="units.png"):
    """
    Plot the distribution of l2 unit norms for the model's current weights.
    Save the plot as a png in the specified directory.
    """
    plt.figure(figsize=(12, 4))
    plt.subplots_adjust(hspace=0.8, wspace=0.4)

    for i, layer in enumerate(model.layers):
        name = layer.get_config()['name']
        weights = layer.get_weights()[0]
        column_l2 = np.linalg.norm(weights, ord=2, axis=0)
        plt.subplot(2, 3, i+1)
        plt.title(name)
        plt.xlabel('L2 Unit Norm')
        plt.ylabel('Frequency')
        plt.hist(column_l2, bins=100)

    plt.savefig(os.path.join(experiment_dir, output_name))
