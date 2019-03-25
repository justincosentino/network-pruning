""" Pure functions used as utilities in the training and pruning process. """

import os


def unique_id(flags):
    """
    Given the set of user-specified flags, generate a unique experiment id.

    Currently this is of the form '<dataset_name>-<epochs>-<learning_rate>'.
    """
    return "{}-{}-lr={}-l1={}-l2={}".format(
        flags.dataset,
        flags.epochs,
        flags.learning_rate,
        flags.l1_reg,
        flags.l2_reg)


def get_checkpoint_dir(experiment_dir):
    """
    Given the experiment directly, build the checkpoint directory path.
    """
    return os.path.join(experiment_dir, "checkpoints")


def get_checkpoint_name(checkpoint_dir):
    """
    Given the checkpoint directly, build the checkpoint file name.
    """
    return os.path.join(checkpoint_dir, "weights.best.hdf5")
