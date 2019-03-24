"""
Loads preprocessed MNIST digits from the keras dataset.
"""

import tensorflow as tf
from .loader_utils import load_from_keras
from ..registry import register


@register("digits")
def load_digits(num_valid=10000):
    """
    Returns preprocessed train, validation, and test sets for MNIST digits.
    """
    return load_from_keras(tf.keras.datasets.mnist, num_valid)
