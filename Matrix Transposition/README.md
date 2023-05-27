# CUDA Matrix Transposition Experiment
## Aim
The aim of this experiment is to compare the performance of different matrix transposition implementations using CUDA.

## Procedure
1. The code implements various matrix transposition techniques using shared memory in CUDA.
2. The different implementations include:
   - `setRowReadRow`: Transpose matrix using row-major ordering for both read and write operations.
   - `setColReadCol`: Transpose matrix using column-major ordering for both read and write operations.
   - `setColReadCol2`: Transpose matrix using column-major ordering for write operation and row-major ordering for read operation.
   - `setRowReadCol`: Transpose matrix using row-major ordering for write operation and column-major ordering for read operation.
   - `setRowReadColDyn`: Transpose matrix using dynamic shared memory and row-major ordering for write operation and column-major ordering for read operation.
   - `setRowReadColPad`: Transpose matrix using row-major ordering for write operation and column-major ordering for read operation, with padding.
   - `setRowReadColDynPad`: Transpose matrix using dynamic shared memory, row-major ordering for write operation, column-major ordering for read operation, with padding.
3. The code measures the execution time of each implementation using CUDA events.
4. The results of the matrix transposition are verified by comparing the output with the expected result.
5. The performance of each implementation is compared based on their execution times.

## Output
The output of the program includes:
- The transposed matrix for each implementation (if `iprintf` flag is set).
- Verification result indicating whether the transposed matrices match the expected result.
- Execution time for each implementation.

![image](https://github.com/Marinto-Richee/Parallel-Computing-Architecture/assets/65499285/65657470-fbdb-4c1c-881a-2b1bd6eb1e97)


## Result
The experiment aims to compare the performance of different matrix transposition techniques using shared memory in CUDA. By measuring the execution time of each implementation, we can identify the most efficient technique for matrix transposition.
