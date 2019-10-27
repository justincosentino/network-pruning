""" Functions used for visualizing data from experiments. """

import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

sns.set(style="darkgrid")
FLAGS = tf.app.flags.FLAGS


def plot_history(history, experiment_dir, output_name="loss_accuracy.png"):
    """
    Plot the training history from model.fit. Displays the test and training
    accuracy and loss over epochs and writes as a png to the specified
    experiment directory.
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper right")
    plt.xticks(np.arange(0, FLAGS.epochs, step=1))

    plt.subplot(1, 2, 2)
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.xticks(np.arange(0, FLAGS.epochs, step=1))

    plt.savefig(os.path.join(experiment_dir, output_name))


def plot_weights_l1(model, experiment_dir, k, output_name="weights.png"):
    """
    Plot the distribution of l1 weight norms for the model's current weights.
    Save the plot as a png in the specified directory.
    """
    plt.figure(figsize=(12, 4))
    plt.suptitle("L1 Weight Norms: {:4.3f}% Sparsity".format(k), y=0.99)
    plt.subplots_adjust(hspace=0.8, wspace=0.4)

    for i, layer in enumerate(model.layers):
        name = layer.get_config()["name"]
        weights = np.absolute(layer.get_weights()[0])
        plt.subplot(2, 3, i + 1)
        plt.title(name)
        plt.xlabel("L1 Weights Norm")
        plt.ylabel("Frequency")
        plt.hist(weights[np.nonzero(weights)], bins=300)

    plt.savefig(os.path.join(experiment_dir, output_name))
    plt.close("all")


def plot_units_l2(model, experiment_dir, k, output_name="units.png"):
    """
    Plot the distribution of l2 unit norms for the model's current weights.
    Save the plot as a png in the specified directory.
    """
    plt.figure(figsize=(12, 4))
    plt.subplots_adjust(hspace=0.8, wspace=0.4)
    plt.suptitle("L2 Unit Norms: {:4.3f}% Sparsity".format(k), y=0.99)

    for i, layer in enumerate(model.layers):
        name = layer.get_config()["name"]
        weights = layer.get_weights()[0]
        column_l2 = np.linalg.norm(weights, ord=2, axis=0)
        plt.subplot(2, 3, i + 1)
        plt.title(name)
        plt.xlabel("L2 Unit Norm")
        plt.ylabel("Frequency")
        plt.hist(column_l2, bins=100)

    plt.savefig(os.path.join(experiment_dir, output_name))
    plt.close("all")


def plot_prune_history(
    weight_losses,
    weight_accuracies,
    unit_losses,
    unit_accuracies,
    k_vals,
    experiment_dir,
    output_name="pruned_loss_accuracy.png",
):
    """
    Plot the evaluation loss and accuracy for pruning on the test dataset.
    Compares weight and unit pruning and writes as a png to the specified
    experiment directory.
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(k_vals, weight_losses)
    plt.plot(k_vals, unit_losses)
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Test Loss")
    plt.legend(["weight", "unit"], loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(k_vals, weight_accuracies)
    plt.plot(k_vals, unit_accuracies)
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Test Accuracy")
    plt.legend(["weight", "unit"], loc="lower left")

    plt.savefig(os.path.join(experiment_dir, output_name))
    plt.close("all")
