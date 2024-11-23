import numpy as np


def replace_nans(matrix: np.ndarray) -> np.ndarray:
    # Ensure the matrix is of type np.float64
    matrix = matrix.astype(np.float64)

    # Create a mask of NaN values
    nan_mask = np.isnan(matrix)

    # Compute the row-wise means for non-NaN values
    row_means = np.nanmean(matrix, axis=1)

    # Iterate over each row and replace NaN values with the row mean
    for i in range(matrix.shape[0]):
        # Replace NaN elements with the mean of the row (excluding NaNs)
        matrix[i, nan_mask[i]] = row_means[i]

    return matrix
