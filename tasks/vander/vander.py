import numpy as np
import numpy.typing as npt


def vander(array: npt.NDArray[np.float64 | np.int_]) -> npt.NDArray[np.float64]:
    """
    Compute Vandermonde matrix using given vector as an input.
    :param array: input array,
    :return: Vandermonde matrix
    """
    array = array[:, np.newaxis]
    powers = np.arange(array.shape[0])
    return array ** powers
