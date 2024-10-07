## Replace NaNs (1 POINT)

`numpy`

### Task
Implement a function that accepts real matrix ```X``` and returns a new matrix,
which is ```X``` with all ```nan``` values replaced by the arithmetic mean of all other elements. 
If all elements of the matrix are ```nan```, then return a zero matrix of the same shape.

For example, the matrix ```np.array([[nan, 1, 2, 3], [4, nan, 5, nan]])``` 
will become ```np.array([[3, 1, 2, 3], [4, 3, 5, 3]])``` because mean of 
all non-`nan` elements in the original matrix is `(1 + 2 + 3 + 4 + 5) // 5 = 3`.
 
### Notes

* You cannot use loops (keywords *for* and *while*) as well as list comprehension, *map*, etc.
* As in “Introduction to NumPy”, here *matrix* means a two-dimensional NumPy array.

---