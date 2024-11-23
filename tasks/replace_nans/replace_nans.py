import numpy as np
import numpy.typing as npt


def replace_nans(data_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Replace all NaN values in a matrix with the average of non-NaN neighbors.
    If no neighbors are available, replace with 0.

    :param data_matrix: Input matrix (2D array).
    :return: A new matrix with NaN values replaced.
    """
    if input_matrix.size == 0:
        return data_matrix  # Return the empty matrix as is

    result_matrix = np.array(input_matrix, copy=True)
    nan_mask = np.isnan(result_matrix)

    for i, j in np.ndindex(output_matrix.shape):
        if nan_mask[i, j]:
            neighbors = []
            if i > 0 and not np.isnan(output_matrix[i - 1, j]):
                neighbors.append(output_matrix[i - 1, j])
            if i < output_matrix.shape[0] - 1 and not np.isnan(output_matrix[i + 1, j]):
                neighbors.append(output_matrix[i + 1, j])
            if j > 0 and not np.isnan(output_matrix[i, j - 1]):
                neighbors.append(output_matrix[i, j - 1])
            if j < output_matrix.shape[1] - 1 and not np.isnan(output_matrix[i, j + 1]):
                neighbors.append(output_matrix[i, j + 1])

            if neighbors:
                output_matrix[i, j] = np.mean(neighbors)
            else:
                output_matrix[i, j] = 0

    return output_matrix

# Sample input matrix
input_matrix = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]], dtype=np.float64)
expected_result = np.array([[1, 2, 4], [4, 5, 6], [7, 8, 9]], dtype=np.float64)

# Running the function
output_matrix = replace_nans(input_matrix)

# Ensure there are no NaN values in the result for comparison
output_no_nan = np.nan_to_num(output_matrix, nan=0)
expected_no_nan = np.nan_to_num(expected_result, nan=0)

# Compare the output with expected result
np.testing.assert_array_equal(output_no_nan, expected_no_nan)
