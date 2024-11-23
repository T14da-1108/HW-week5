import ast
import dataclasses
import inspect
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_array_equal

from .numpy_basic import construct_array, detect_identical, mean_channel, get_unique_rows, construct_matrix


@dataclasses.dataclass
class ConstructArrayCase:
    matrix: npt.NDArray[np.int_]
    row_indices: npt.NDArray[np.int_] | list[int]
    col_indices: npt.NDArray[np.int_] | list[int]
    result: npt.NDArray[np.int_]


CONSTRUCT_ARRAY_TEST_CASES = [
    ConstructArrayCase(matrix=np.array(range(25)).reshape(5, 5),
                       row_indices=[0, 1, 2],
                       col_indices=[0, 1, 2],
                       result=np.array([0, 6, 12])),
    ConstructArrayCase(matrix=np.arange(-10, 10).reshape((5, 4)),
                       row_indices=[1, 2, 3, 3],
                       col_indices=[3, 2, 1, 2],
                       result=np.array([-3, 0, 3, 4])),
    ConstructArrayCase(matrix=np.arange(42).reshape((7, 6)),
                       row_indices=[],
                       col_indices=[],
                       result=np.array([])),
    ConstructArrayCase(matrix=np.arange(42).reshape((7, 6)),
                       row_indices=np.arange(4),
                       col_indices=np.arange(4),
                       result=np.array(np.arange(4)) * 7),
    ConstructArrayCase(matrix=np.arange(42).reshape((42, 1)),
                       row_indices=[0, 1, 41],
                       col_indices=[0, 0, 0],
                       result=np.array([0, 1, 41]))
]


@dataclasses.dataclass
class DetectIdenticalCase:
    lhs_array: npt.ArrayLike
    rhs_array: npt.ArrayLike
    result: bool


DETECT_IDENTICal_TEST_CASES = [
    DetectIdenticalCase(
         lhs_array=np.array([1, 2]),
         rhs_array=np.array([1, 2]),
         result=True),
    DetectIdenticalCase(
         lhs_array=np.array([1., 2]),
         rhs_array=np.array([1, 2.]),
         result=True),
    DetectIdenticalCase(
         lhs_array=np.array([1, 2]),
         rhs_array=np.array([1.0001, 2]),
         result=False),
    DetectIdenticalCase(
         lhs_array=np.array([1, 2]),
         rhs_array=np.array([[1, 2]]),
         result=False),
    DetectIdenticalCase(
         lhs_array=np.array([[1, 2, 3]]),
         rhs_array=np.array([[1, 2]]),
         result=False),
    DetectIdenticalCase(
         lhs_array=np.array([]),
         rhs_array=np.array([]),
         result=True),
    DetectIdenticalCase(
         lhs_array=3,
         rhs_array=3,
         result=True),
    DetectIdenticalCase(
         lhs_array=np.array(range(3)),
         rhs_array=np.array(range(3))[np.newaxis, :],
         result=False),
]


@dataclasses.dataclass
class MeanChannelCase:
    X: npt.NDArray[np.float64]
    result: npt.NDArray[np.float64]


MEAN_CHANNEL_TEST_CASES = [
    MeanChannelCase(
        X=np.array(range(5 * 5 * 3)).reshape(5, 5, 3),
        result=np.array([36, 37, 38])),
    MeanChannelCase(
        X=np.dstack((
            (np.arange(320 * 240) % 64).reshape(320, 240),
            (np.arange(320 * 240) % 64).reshape(320, 240) * 2,
            (np.arange(320 * 240) % 64).reshape(320, 240) * 3)),
        result=np.array([31.5, 63., 94.5])),
    MeanChannelCase(
        X=np.array([]).reshape(0, 0, 3),
        result=np.array([np.nan, np.nan, np.nan]))
]


@dataclasses.dataclass
class GetUniqueRowsCase:
    X: npt.NDArray[np.int_]
    result: npt.NDArray[np.int_]


GET_UNIQUE_ROWS_TEST_CASES = [
    GetUniqueRowsCase(
        X=np.array([[1, 2, 3]]),
        result=np.array([[1, 2, 3]])),
    GetUniqueRowsCase(
        X=np.array([[4, 5, 6], [0, 1, 2], [1, 2, 3], [0, 1, 2], [4, 5, 6], [1, 2, 3]]),
        result=np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6]])),
]


@dataclasses.dataclass
class ConstructMatrixCase:
    first_array: npt.NDArray[np.int_]
    second_array: npt.NDArray[np.int_]
    result: npt.NDArray[np.int_]


CONSTRUCT_MATRIX_TEST_CASES = [
    ConstructMatrixCase(
        first_array=np.array([1, 2, 3]),
        second_array=np.array([4, 5, 6]),
        result=np.array([[1, 4], [2, 5], [3, 6]])),
    ConstructMatrixCase(
        first_array=np.array([]),
        second_array=np.array([]),
        result=np.array([]).reshape(0, 2)),
    ConstructMatrixCase(
        first_array=np.array([1]),
        second_array=np.array([2]),
        result=np.array([[1, 2]])),
    ConstructMatrixCase(
        first_array=np.arange(0, 100, 2),
        second_array=np.arange(1, 100, 2),
        result=np.arange(100).reshape(50, 2))
]


def test_structural() -> None:
    for function in [construct_array, detect_identical, mean_channel, get_unique_rows, construct_matrix]:
        source = inspect.getsource(function) # type: ignore[arg-type]
        tree = ast.parse(source)
        for node in ast.walk(tree):
            assert not isinstance(node, (ast.For, ast.While)), (
                    "Function '" + function.__name__ + "' contains loops")


@pytest.mark.parametrize('t', CONSTRUCT_ARRAY_TEST_CASES, ids=str)
def test_construct_array(t: ConstructArrayCase) -> None:
    matrix_copy, row_indices_copy, col_indices_copy = t.matrix.copy(), t.row_indices.copy(), t.col_indices.copy()
    assert_array_equal(construct_array(t.matrix, t.row_indices, t.col_indices), t.result)
    assert_array_equal(t.matrix, matrix_copy, "Function shouldn't change the input")
    assert_array_equal(t.row_indices, row_indices_copy, "Function shouldn't change the input")
    assert_array_equal(t.col_indices, col_indices_copy, "Function shouldn't change the input")


@pytest.mark.parametrize('t', DETECT_IDENTICal_TEST_CASES, ids=str)
def test_detect_identical(t: DetectIdenticalCase) -> None:
    lhs_array_copy: np.ndarray[Any, Any] | None = None
    rhs_array_copy: np.ndarray[Any, Any] | None = None
    if isinstance(t.lhs_array, np.ndarray) and isinstance(t.rhs_array, np.ndarray):
        lhs_array_copy, rhs_array_copy = t.lhs_array.copy(), t.rhs_array.copy()
    assert_array_equal(detect_identical(t.lhs_array, t.rhs_array), t.result)
    if lhs_array_copy is not None and rhs_array_copy is not None:
        assert_array_equal(t.lhs_array, lhs_array_copy, "Function shouldn't change the input")
        assert_array_equal(t.rhs_array, rhs_array_copy, "Function shouldn't change the input")


@pytest.mark.parametrize('t', MEAN_CHANNEL_TEST_CASES, ids=str)
def test_mean_channel(t: MeanChannelCase) -> None:
    x_copy = t.X.copy()
    assert_array_equal(mean_channel(t.X), t.result)
    assert_array_equal(t.X, x_copy, "Function shouldn't change the input")


@pytest.mark.parametrize('t', GET_UNIQUE_ROWS_TEST_CASES, ids=str)
def test_get_unique_rows(t: GetUniqueRowsCase) -> None:
    x_copy = t.X.copy()
    assert_array_equal(get_unique_rows(t.X), t.result)
    assert_array_equal(t.X, x_copy, "Function shouldn't change the input")


@pytest.mark.parametrize('t', CONSTRUCT_MATRIX_TEST_CASES, ids=str)
def test_construct_matrix(t: ConstructMatrixCase) -> None:
    first_array_copy, second_array_copy = t.first_array.copy(), t.second_array.copy()
    assert_array_equal(construct_matrix(t.first_array, t.second_array), t.result)
    assert_array_equal(t.first_array, first_array_copy, "Function shouldn't change the input")
    assert_array_equal(t.second_array, second_array_copy, "Function shouldn't change the input")
