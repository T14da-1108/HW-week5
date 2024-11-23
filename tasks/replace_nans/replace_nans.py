import numpy as np
import numpy.typing as npt

def replace_nans(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Replace all NaN values in the matrix with the mean of the other values in the same column.
    If all values are NaN, return a zero matrix of the same size.

    :param matrix: Input matrix with potential NaN values.
    :return: Matrix with NaN values replaced by column means or zero matrix if all values are NaN.
    """
    # Copy the matrix to avoid modifying the input
    result = matrix.copy()

    # If the entire matrix is NaN, return a zero matrix
    if np.isnan(result).all():
        return np.zeros_like(result)

    # Calculate column-wise means excluding NaNs
    col_means = np.nanmean(result, axis=0)

    # If any column is entirely NaN, set its mean to 0
    col_means[np.isnan(col_means)] = 0

    # Find positions where NaN values are present
    nan_mask = np.isnan(result)

    # Replace NaN values with the column mean
    result[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    return result
