import ast
import dataclasses
import inspect

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_array_equal

from .add_zeros import add_zeros


@dataclasses.dataclass
class AddZerosCase:
    array: npt.NDArray[np.int_]
    result: npt.NDArray[np.int_]


ADD_ZEROS_TEST_CASES = [
    AddZerosCase(
        array=np.array([1, 2, 3]),
        result=np.array([1, 0, 2, 0, 3])),
    AddZerosCase(
        array=np.array([], dtype=np.int_),
        result=np.array([], dtype=np.int_)),
    AddZerosCase(
        array=np.array([1]),
        result=np.array([1])),
    AddZerosCase(
        array=np.array([1, 1]),
        result=np.array([1, 0, 1])),
    AddZerosCase(
        array=np.array([0]),
        result=np.array([0])),
    AddZerosCase(
        array=np.array([1, 0, 0, 1]),
        result=np.array([1, 0, 0, 0, 0, 0, 1])),
    AddZerosCase(
        array=np.zeros(10, dtype=np.int_),
        result=np.zeros(19, dtype=np.int_)),
]


def test_structural() -> None:
    source = inspect.getsource(add_zeros)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        assert not isinstance(node, (ast.For, ast.While, ast.ListComp)), "Function contains loops or list comp"


@pytest.mark.parametrize('t', ADD_ZEROS_TEST_CASES, ids=str)
def test_construct_matrix(t: AddZerosCase) -> None:
    array_copy = t.array.copy()
    assert_array_equal(add_zeros(t.array), t.result)
    assert_array_equal(t.array, array_copy, "Function shouldn't change the input")
