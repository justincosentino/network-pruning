"""
Trains the model and runs pruning experiments.
"""
import os
import tensorflow as tf
from .data.registry import load_data
from .models.model import *
from .utils import *
from .visualization import csv, gifs, graph

FLAGS = None


def init_flags():
    """ Initialize flags related to model training. """
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
    tf.flags.DEFINE_float(
        "l1_reg",
        0,
        "l1 regularization lambda")
    tf.flags.DEFINE_float(
        "l2_reg",
        0,
        "l2 regularization lambda")
    tf.flags.DEFINE_list(
        "k_vals",
        [.0, .25, .50, .60, .70, .80, .90, .95, .97, .99],
        "A list of sparsity values to use in pruning experiments.")
    tf.flags.DEFINE_string(
        "output_dir",
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "results/"),
        help="The output directory for checkpoints, figures, etc.")
    tf.flags.DEFINE_boolean(
        "keep_best",
        default=True,
        help="If true, keep the best validation acc model when checkpointing.")
    tf.flags.DEFINE_boolean(
        "force_train",
        default=False,
        help="If true, overwrite existing model for given hparam config.")
    tf.flags.DEFINE_string(
        "experiment_id",
        default=None,
        help="A unique name to identify the current model and experiment.")


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
        save_best_only=FLAGS.keep_best,
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
    graph.plot_history(history, experiment_dir)
    model.evaluate(x=x_test, y=y_test)


def prune_experiments(experiment_dir):
    """
    Runs weight and unit pruning experiments over the provided k_vals list.
    Evaluates the pruned model on the specified MNIST test subset and saves
    results to the specified experiment directory.
    """

    # Load dense model and test data
    dense_model = load_model_weights(experiment_dir)
    (_, _, _, _, x_test, y_test) = load_data(FLAGS.dataset)(
        num_valid=FLAGS.num_valid)
    print("----------- Dense Model without pruning ----------")
    dense_model.evaluate(x=x_test, y=y_test)

    weight_accuracies, unit_accuracies = [], []
    weight_losses, unit_losses = [], []
    for k in FLAGS.k_vals:
        int_k = int(k*100)

        # Run weight pruning
        print("-------------- Weight pruning k={:.2f} -------------".format(k))
        weight_pruned_model = convert_dense_to_weight_pruned(dense_model, k=k)
        graph.plot_weights_l1(
            weight_pruned_model,
            experiment_dir,
            "pruned_weights_l1_k={}.png".format(int_k))
        weight_loss, weight_acc = weight_pruned_model.evaluate(
            x=x_test,
            y=y_test)
        weight_losses.append(weight_loss)
        weight_accuracies.append(weight_acc)

        # Run unit pruning
        print("--------------   Unit pruning k={:.2f} -------------".format(k))
        unit_pruned_model = convert_dense_to_unit_pruned(dense_model, k=k)
        graph.plot_units_l2(
            unit_pruned_model,
            experiment_dir,
            "pruned_units_l2_k={}.png".format(int_k))
        unit_loss, unit_acc = unit_pruned_model.evaluate(x=x_test, y=y_test)
        unit_losses.append(unit_loss)
        unit_accuracies.append(unit_acc)

    # Plot weight vs unit pruning and write results to csv
    graph.plot_prune_history(
        weight_losses,
        weight_accuracies,
        unit_losses,
        unit_accuracies,
        FLAGS.k_vals,
        experiment_dir)
    gifs.convert_imgs_to_gif(experiment_dir, "pruned_weights")
    gifs.convert_imgs_to_gif(experiment_dir, "pruned_units")
    csv.write_to_csv(
        weight_losses,
        weight_accuracies,
        unit_losses,
        unit_accuracies,
        FLAGS.k_vals,
        experiment_dir)


def main(_):
    """ Prepare specified directory and kickoff training. """
    global FLAGS
    FLAGS = tf.app.flags.FLAGS

    # Set unique id based on flags if not specified
    if not FLAGS.experiment_id:
        FLAGS.experiment_id = unique_id(FLAGS)

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
        prune_experiments(experiment_dir)
        return

    # If the model already exists and the user has specified --force_training,
    # remove the old model before continuing.
    if exists:
        tf.logging.warning(
            "Force training enabled... deleting '{}'.".format(experiment_dir))
        tf.io.gfile.rmtree(experiment_dir)

    tf.io.gfile.mkdir(experiment_dir)
    train(experiment_dir)
    prune_experiments(experiment_dir)


if __name__ == "__main__":
    init_flags()
    tf.app.run(main=main)
