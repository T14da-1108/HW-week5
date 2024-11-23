import numpy as np
import numpy.typing as npt

def nearest_value(matrix: npt.NDArray[np.float64], value: float) -> float | None:
    """
    Find closest value in matrix.
    If matrix is empty return None
    :param matrix: input matrix
    :param value: value to find
    :return: nearest value in matrix or None
    """
    # Check if the matrix is empty
    if matrix.size == 0:
        return None

    # Compute the absolute differences between each element and the given value
    diff = np.abs(matrix - value)

    # Find the index of the element with the smallest difference
    idx = np.argmin(diff)

    # Return the value at the index
    return matrix.flat[idx]
