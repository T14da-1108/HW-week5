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

    # Identify the valid neighbors for each NaN location
    for i in range(1, result.shape[0] - 1):
        for j in range(1, result.shape[1] - 1):
            if nan_mask[i, j]:
                neighbors = []
                # Check the neighbors (above, below, left, right)
                if not np.isnan(result[i - 1, j]):
                    neighbors.append(result[i - 1, j])  # Above
                if not np.isnan(result[i + 1, j]):
                    neighbors.append(result[i + 1, j])  # Below
                if not np.isnan(result[i, j - 1]):
                    neighbors.append(result[i, j - 1])  # Left
                if not np.isnan(result[i, j + 1]):
                    neighbors.append(result[i, j + 1])  # Right

                # Replace NaN with the mean of its neighbors
                if neighbors:
                    result[i, j] = np.mean(neighbors)
                else:
                    result[i, j] = 0  # If no neighbors, replace with 0

    return result
