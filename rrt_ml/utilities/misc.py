import numpy as np
from coral_pytorch.losses import corn_loss


def remove_trailing_zeros(arr):
    """
    Remove trailing zeros from array (flatten if needed)
    :param arr: input array
    :return: array without trailing zeros
    """

    # Flatten
    if isinstance(arr, np.ndarray):
        if len(arr.shape) > 1:
            arr = arr.flatten()

    # Iterate
    for idx, n in enumerate(arr[:-1]):

        # Check if the rest is zero too
        if n == 0:
            if not np.count_nonzero(arr[idx + 1:]) > 0:
                return np.copy(arr[:idx])

    return arr


def remove_nones(arr):
    """
    Remove nones from array.
    """
    
    return arr[arr != np.array(None)]


class EnsembleLoss:

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, x, y):
        return corn_loss(x, y, self.n_classes)
