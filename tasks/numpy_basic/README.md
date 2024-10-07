## Introduction to Numpy (3 POINTS)

`numpy`

### Tasks
1. Implement a function that accepts a ```matrix``` (from here on we call two-dimensional NumPy array *a matrix*)
   and two index arrays
   ```row_indices``` and ```col_indices``` of the same length 
   and returns a NumPy array (`np.array`) consisting of the sequence of elements 

   ```[matrix[row_indices[0], col_indices[0]], ... , matrix[row_indices[N-1], col_indices[N-1]]]``` 

2. Implement a function that accepts arrays ```lhs_array``` 
   and ```rhs_array``` (arrays can be n-dimensional, of different sizes),
   and returns **True** if they are equal and **False** otherwise. 

3. Implement a function that accepts a bitmap image ```X``` 
   and returns the mean value per each of the three color channels (a vector of length 3).
    
   Image is represented as a three-dimensional (n, m, 3) NumPy array. For example,
   `X[i, j, color]` represents the intensity of `color` in pixel located in `i`th row and
   `j`th column.

   Example:
   
   ```
   [[[0, 1, 2], [3, 4, 5]],     ->       [4.5, 5.5, 6.5]
    [[6, 7, 8], [9, 10, 11]]]
   ```

4. Implement a function that accepts a matrix ```X``` of dimensions (n, m) and returns 
   a matrix which consists only of unique rows of `X` sorted as if they were m-dimensional
   vectors.

   Example:

   ```
   [[1, 2, 3, 4],    ->    [[1, 2, 3, 4],
    [3, 0, 0, 0],           [1, 9, 0, 0],
    [1, 9, 0, 0],           [3, 0, 0, 0]]
    [1, 2, 3, 4]]
   ```

5. Implement a function that accepts two one-dimensional arrays ```x``` and ```y``` of the same length
   and returns a matrix in which the first column is `x`, and the second column is `y`.

   It is **prohibited** to use the transpose operation;
   [reshape](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.reshape.html) is recommended instead.

   Example:

   ```
      [1, 2, 3], [4, 5, 6]    ->    [[1, 4],
                                     [2, 5],
                                     [3, 6]]
   ```

### Notes

Don't use loops. Instead, check out [list comprehension](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions),
[map etc.](https://realpython.com/python-functional-programming/#applying-a-function-to-an-iterable-with-map)

---