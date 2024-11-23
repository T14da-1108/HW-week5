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

    # For each element in the matrix, replace NaNs with the mean of its valid neighbors
    rows, cols = result.shape
    for i in range(rows):
        for j in range(cols):
            if nan_mask[i, j]:
                neighbors = []
                if i > 0 and not np.isnan(result[i - 1, j]):  # Check above
                    neighbors.append(result[i - 1, j])
                if i < rows - 1 and not np.isnan(result[i + 1, j]):  # Check below
                    neighbors.append(result[i + 1, j])
                if j > 0 and not np.isnan(result[i, j - 1]):  # Check left
                    neighbors.append(result[i, j - 1])
                if j < cols - 1 and not np.isnan(result[i, j + 1]):  # Check right
                    neighbors.append(result[i, j + 1])

                # Replace with the mean of neighbors if available, otherwise 0
                if neighbors:
                    result[i, j] = np.mean(neighbors)
                else:
                    result[i, j] = 0  # No valid neighbors, set to 0

    return result
