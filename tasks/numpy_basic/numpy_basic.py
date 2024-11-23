import numpy as np
import numpy.typing as npt


def construct_array(
        matrix: npt.NDArray[np.int_],
        row_indices: npt.NDArray[np.int_] | list[int],
        col_indices: npt.NDArray[np.int_] | list[int]
) -> npt.NDArray[np.int_]:
    """
    Construct slice of given matrix by indices row_indices and col_indices:
    [matrix[row_indices[0], col_indices[0]], ... , matrix[row_indices[N-1], col_indices[N-1]]]
    :param matrix: input matrix
    :param row_indices: list of row indices
    :param col_indices: list of column indices
    :return: matrix slice
    """
    if len(row_indices) != len(col_indices):
        raise ValueError("row_indices and col_indices must have the same length.")
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a NumPy array.")
    return matrix[row_indices, col_indices]


def detect_identical(
        lhs_array: npt.ArrayLike,
        rhs_array: npt.ArrayLike
) -> bool:
    """
    Check whether two arrays are equal or not
    :param lhs_array: first array
    :param rhs_array: second array
    :return: True if input arrays are equal, False otherwise
    """
    if np.isscalar(lhs_array):
        lhs_array = np.array([lhs_array])
    if np.isscalar(rhs_array):
        rhs_array = np.array([rhs_array])

    if not isinstance(lhs_array, np.ndarray) or not isinstance(rhs_array, np.ndarray):
        raise TypeError("Both inputs must be NumPy arrays.")
    return np.array_equal(lhs_array, rhs_array)


def mean_channel(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Given color image (3-dimensional vector of size (n, m, 3).
    Compute mean value for all 3 channels
    :param x: color image
    :return: array of size 3 with mean values
    """

    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("Input must be a 3D array with 3 channels (n, m, 3).")

    if x.size == 0:
        return np.array([np.nan, np.nan, np.nan])

    return np.mean(x, axis=(0, 1))


def get_unique_rows(x: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """
    Compute unique rows of a 2D array
    :param x: 2D array of shape (n,m)
    :return: Array of unique rows, sorted as if they were vectors of length m
    """

    if x.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")
    return np.unique(x, axis=0)


def construct_matrix(
        first_array: npt.NDArray[np.int_], second_array: npt.NDArray[np.int_]
) -> npt.NDArray[np.int_]:
    """
    Construct matrix from  a pair of arrays
    :param first_array: first array
    :param second_array: second array
    :return: constructed matrix
    """

    if len(first_array) != len(second_array):
        raise ValueError("Both arrays must have the same length.")
    if not isinstance(first_array, np.ndarray) or not isinstance(second_array, np.ndarray):
        raise TypeError("Both inputs must be NumPy arrays.")
    return np.stack((first_array, second_array), axis=1)
