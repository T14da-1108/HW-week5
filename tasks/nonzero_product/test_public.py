import ast
import dataclasses
import inspect
import typing as tp

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_array_equal

from .nonzero_product import nonzero_product


@dataclasses.dataclass
class NonzeroProductCase:
    matrix: npt.NDArray[np.int_]
    result: tp.Any


NONZERO_PRODUCT_TEST_CASES = [
    NonzeroProductCase(
        matrix=np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 4, 4]]),
        result=3
    ),
    NonzeroProductCase(
        matrix=np.array([[0, 0, 1], [2, 0, 2], [3, 0, 0], [4, 4, 4]]),
        result=None
    ),
    NonzeroProductCase(
        matrix=np.array([[], [], [], []]),
        result=None
    ),
    NonzeroProductCase(
        matrix=np.arange(24).reshape((4, 6)),
        result=2058
    ),
    NonzeroProductCase(
        matrix=np.ones(48, dtype=np.int_).reshape((12, 4)),
        result=1
    ),
    NonzeroProductCase(
        matrix=np.zeros(48, dtype=np.int_).reshape((12, 4)),
        result=None
    ),
    NonzeroProductCase(
        matrix=np.array([[1]]),
        result=1
    ),
    NonzeroProductCase(
        matrix=np.array([[0]]),
        result=None
    ),
    NonzeroProductCase(
        matrix=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
        result=1
    ),
]


def test_structural() -> None:
    source = inspect.getsource(nonzero_product)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        assert not isinstance(node, (ast.For, ast.While, ast.ListComp)), "Function contains loops or list comp"


@pytest.mark.parametrize('t', NONZERO_PRODUCT_TEST_CASES, ids=str)
def test_construct_matrix(t: NonzeroProductCase) -> None:
    matrix_copy = t.matrix.copy()
    assert nonzero_product(t.matrix) == t.result
    assert_array_equal(t.matrix, matrix_copy, "Function shoudn't change the input")
