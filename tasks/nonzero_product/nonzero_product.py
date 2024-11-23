import numpy as np
import numpy.typing as npt

def nonzero_product(matrix: npt.NDArray[np.int_]) -> int | None:
    """
    Compute product of nonzero diagonal elements of matrix.
    If all diagonal elements are zeros, return None.

    :param matrix: numpy array.
    :return: product of non-zero diagonal elements or None if all diagonal elements are zero.
    """
    # Extract the diagonal elements of the matrix and cast them to int
    diagonal = np.diagonal(matrix).astype(np.int_)

    # Filter out zeros (we keep only non-zero elements)
    nonzero_diagonal = diagonal[diagonal != 0]

    # If there are no non-zero elements in the diagonal, return None
    if nonzero_diagonal.size == 0:
        return None

    # Compute the product of the non-zero diagonal elements
    product = np.prod(nonzero_diagonal)

    # Cast the product result to int to satisfy the type hint and to match expected output
    return int(product)