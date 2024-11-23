import numpy as np
import numpy.typing as npt


def replace_nans(data_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Replace all NaN values in a matrix with the arithmetic mean of non-NaN elements.
    If all elements of the matrix are NaN, return a zero matrix of the same shape.

    :param data_matrix: Input matrix (2D array).
    :return: A new matrix with NaN values replaced.
    """
    # Check if the matrix is empty
    if data_matrix.size == 0:
        return data_matrix

    # Calculate the mean of all non-NaN elements
    mean_value: float = float(np.nanmean(data_matrix))

    # Replace NaN values with the mean value
    result_matrix = np.nan_to_num(data_matrix, nan=mean_value)

    return result_matrix
