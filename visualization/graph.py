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
    plt.figure(figsize=(16, 4))
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(121)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.xticks(np.arange(0, FLAGS.epochs, step=1))

    plt.subplot(122)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.xticks(np.arange(0, FLAGS.epochs, step=1))

    plt.savefig(os.path.join(experiment_dir, output_name))
