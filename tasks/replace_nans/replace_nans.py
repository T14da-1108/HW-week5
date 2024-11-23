import numpy as np
import numpy.typing as npt


def replace_nans(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Replace all NaN values in a matrix with the average of non-NaN neighbors.
    If no neighbors are available, replace with 0.

    :param matrix: Input matrix (2D array).
    :return: A new matrix with NaN values replaced.
    """
    if matrix.size == 0:
        return matrix  # Return the empty matrix as is

    # Copy the matrix to avoid mutating the input
    result = np.array(matrix, copy=True)

    # Create a mask of NaN values
    nan_mask = np.isnan(result)

    # We will use the neighboring values for replacement
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if nan_mask[i, j]:
                # Get the neighbors using numpy slicing (avoid loops)
                neighbors = []
                if i > 0 and not np.isnan(result[i - 1, j]):  # above
                    neighbors.append(result[i - 1, j])
                if i < result.shape[0] - 1 and not np.isnan(result[i + 1, j]):  # below
                    neighbors.append(result[i + 1, j])
                if j > 0 and not np.isnan(result[i, j - 1]):  # left
                    neighbors.append(result[i, j - 1])
                if j < result.shape[1] - 1 and not np.isnan(result[i, j + 1]):  # right
                    neighbors.append(result[i, j + 1])

                # If there are valid neighbors, replace with their mean; otherwise, 0
                if neighbors:
                    result[i, j] = np.mean(neighbors)
                else:
                    result[i, j] = 0

    return result
