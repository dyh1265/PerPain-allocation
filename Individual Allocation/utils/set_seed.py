import os
import random
import numpy as np
import tensorflow as tf


def setSeed(seed):
    """
    Sets the random seed for reproducibility across various libraries and frameworks.

    This function ensures deterministic behavior by setting the seed for Python's 
    `random` module, NumPy, and TensorFlow. Additionally, it configures environment 
    variables to enforce deterministic operations in TensorFlow.

    Args:
        seed (int): The seed value to be used for random number generation.

    Environment Variables Set:
        - PYTHONHASHSEED: Ensures consistent hashing in Python.
        - TF_DETERMINISTIC_OPS: Forces TensorFlow to use deterministic operations.
        - TF_CUDNN_DETERMINISTIC: Ensures deterministic behavior for TensorFlow's 
          cuDNN backend.

    Note:
        Uncomment the `tf.config.threading.set_inter_op_parallelism_threads` and 
        `tf.config.threading.set_intra_op_parallelism_threads` lines if you want 
        to control TensorFlow's threading behavior for further reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = '0'

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)