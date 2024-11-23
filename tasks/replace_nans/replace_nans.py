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
        return matrix

    result = np.array(matrix, copy=True)
    nan_mask = np.isnan(result)
    rows, cols = result.shape

    for i in range(rows):
        for j in range(cols):
            if nan_mask[i, j]:
                neighbors = []
                if i > 0 and not np.isnan(result[i - 1, j]):
                    neighbors.append(result[i - 1, j])
                if i < rows - 1 and not np.isnan(result[i + 1, j]):
                    neighbors.append(result[i + 1, j])
                if j > 0 and not np.isnan(result[i, j - 1]):
                    neighbors.append(result[i, j - 1])
                if j < cols - 1 and not np.isnan(result[i, j + 1]):
                    neighbors.append(result[i, j + 1])

                if neighbors:
                    result[i, j] = np.mean(neighbors)
                else:
                    result[i, j] = 0

    return result
