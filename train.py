"""
Train the model.
"""
import os
import tensorflow as tf
from .data.registry import load_data
from .models.model import build_model
from .utils import unique_id, get_checkpoint_dir, get_checkpoint_name
from .visualization.graph import plot_history

FLAGS = tf.app.flags.FLAGS


def init_flags():
    """ Initialize flags related to model training. """
    tf.flags.DEFINE_boolean(
        "debug",
        default=False,
        help="Debug mode")
    tf.flags.DEFINE_string(
        "dataset",
        default="digits",
        help="The dataset. Valid options are: {'digits' | 'fashion'}.")
    tf.flags.DEFINE_integer(
        "batch_size",
        default=128,
        help="The batch size")
    tf.flags.DEFINE_integer(
        "num_valid",
        default=10000,
        help="The size of the validation dataset")
    tf.flags.DEFINE_integer(
        "epochs",
        20,
        "Number of training epochs to perform.")
    tf.flags.DEFINE_float(
        "learning_rate",
        0.001,
        "The optimizer's learning rate.")
    tf.flags.DEFINE_string(
        "output_dir",
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "results/"),
        help="The output directory for checkpoints, figures, etc.")
    tf.flags.DEFINE_boolean(
        "force_train",
        default=False,
        help="If true, overwrite existing model for given hparam config.")
    tf.flags.DEFINE_string(
        "experiment_id",
        default=unique_id(FLAGS),
        help="A unique name to identify the current model and experiment.")

    # Update logging as needed
    if FLAGS.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)


def train(experiment_dir):
    """ Train the model, saving checkpoints to the specified directory. """
    (x_train,
     y_train,
     x_valid,
     y_valid,
     x_test,
     y_test) = load_data(FLAGS.dataset)(num_valid=FLAGS.num_valid)

    # Build and compile model
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=FLAGS.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

    # Setup checkpointing
    checkpoint_dir = get_checkpoint_dir(experiment_dir)
    tf.io.gfile.mkdir(checkpoint_dir)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        get_checkpoint_name(checkpoint_dir),
        save_best_only=True,
        mode="auto",
        verbose=1)

    # Train and plot
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs,
        validation_data=(x_valid, y_valid),
        callbacks=[checkpoint_callback])
    plot_history(history, experiment_dir)
    model.evaluate(x=x_test, y=y_test)


def main(_):
    """ Prepare specified directory and kickoff training. """
    experiment_dir = os.path.join(
        FLAGS.output_dir,
        FLAGS.experiment_id)
    exists = tf.io.gfile.exists(experiment_dir)

    # Abort if the model exists and the user has not specified --force_training
    if not FLAGS.force_train and exists:
        tf.logging.warning(
            "Experiment directory '{}' already ".format(experiment_dir) +
            "exists. Run with '--force_train' to overwrite. " +
            "Using existing weights for pruning.")
        return

    # If the model already exists and the user has specified --force_training,
    # remove the old model before continuing.
    if exists:
        tf.logging.warning(
            "Force training enabled... deleting '{}'.".format(experiment_dir))
        tf.io.gfile.rmtree(experiment_dir)

    tf.io.gfile.mkdir(experiment_dir)
    train(experiment_dir)


if __name__ == "__main__":
    init_flags()
    tf.app.run(main=main)
