import numpy as np
import numpy.typing as npt


def nearest_value(matrix: npt.NDArray[np.float64], value: float) -> float | None:
    """
    Find the closest value in a matrix.
    If the matrix is empty, return None.

    :param matrix: Input matrix (2D array).
    :param value: The target value.
    :return: The closest value in the matrix or None if the matrix is empty.
    """
    # Check if the matrix is empty
    if matrix.size == 0:
        return None

    # Flatten the matrix for easier indexing and compute absolute differences
    flat_matrix = matrix.ravel()
    diff = np.abs(flat_matrix - value)

    # Find the index of the element with the smallest difference
    idx = np.argmin(diff)

    # Return the value at the index, converted to a Python float
    return float(flat_matrix[idx])
