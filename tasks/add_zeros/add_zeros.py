import numpy as np
import numpy.typing as npt


def add_zeros(x: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """
    Add zeros between values of given array
    :param x: array,
    :return: array with zeros inserted
    """
    # If the array is empty or has only one element, return it as it is
    if x.size == 0:
        return x

    new_size = x.size * 2 - 1
    result = np.zeros(new_size, dtype=x.dtype)

    result[::2] = x

    return result