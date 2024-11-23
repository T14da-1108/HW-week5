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

    # Get indices where NaN values are located
    nan_indices = np.argwhere(np.isnan(result))

    for i, j in nan_indices:
        # Extract the neighbors of the current NaN cell
        neighbors = []

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < result.shape[0] and 0 <= nj < result.shape[1]:
                neighbors.append(result[ni, nj])

        # Filter out NaN values from the neighbors
        neighbors = [val for val in neighbors if not np.isnan(val)]

        # Replace NaN with the mean of neighbors or 0 if no valid neighbors exist
        result[i, j] = np.mean(neighbors) if neighbors else 0

    return result
