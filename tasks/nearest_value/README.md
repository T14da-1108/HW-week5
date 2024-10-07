## Nearest Value

`numpy`

### Task

Implement a function which accepts matrix `X` and some real number `a` and
returns the element of a matrix which is closest (e.g. minimal
absolute difference) to `a`.

For example, with ```X = np.arange(0,10).reshape((2, 5))``` and ```a = 3.6``` the answer is 4
(`np.arange()` is similar to Python's `range()`; in this case, it returns `np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])` which
is then reshaped to `np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])`).

### Notes

* You cannot use loops (keywords *for* and *while*) as well as list comprehension, *map*, etc.
* As in “Introduction to NumPy”, here *matrix* means a two-dimensional NumPy array.
