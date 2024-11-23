import numpy as np
import numpy.typing as npt


def max_element(array: npt.NDArray[np.int_]) -> int | None:
    """
    Return max element among those with
    preceding zero for input array.
    If appropriate elements are absent, then return None
    :param array: array,
    :return: max element value or None
    """
    # Create a boolean array where each element is True if the previous element is 0
    preceded_by_zero = (array[:-1] == 0)

    # Select elements that are preceded by zero (excluding the first element)
    valid_elements = array[1:][preceded_by_zero]

    # If there are no valid elements, return None
    if valid_elements.size == 0:
        return None

    # Return the maximum value among the valid elements
    return np.max(valid_elements)
