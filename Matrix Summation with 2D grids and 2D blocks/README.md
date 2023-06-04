# Matrix Summation using 2D grids and 2D blocks

## Aim:
To perform matrix addition on CPU and GPU and compare their execution time.


## Procedure:
1. Include the necessary header files:
   - `cuda_runtime.h`: CUDA runtime APIs.
   - `stdio.h`: Standard input/output functions.
   - `stdlib.h`: Standard library functions.
   - `time.h`: Time-related functions.
   - `math.h`: Mathematical functions.
   - `windows.h`: Windows-specific functions.
   - `device_launch_parameters.h`: Device launch parameters for CUDA.
   - `windows.h`: Windows-specific functions.

2. Define the `seconds()` function:
   - This function measures the current time using a high-resolution timer and returns the time in seconds.

3. Define the `CHECK()` macro:
   - This macro checks the return value of a CUDA function call and prints an error message if there is an error.

4. Define the `initialData()` function for initializing the input matrices:
   - There are two overloaded versions of this function, one for initializing `float` data and another for initializing `int` data.
   - The function fills the input array with random values.

5. Define the `sumMatrixOnHost()` function for performing matrix addition on the host:
   - There are two overloaded versions of this function, one for `float` matrices and another for `int` matrices.
   - The function adds corresponding elements of matrices `A` and `B` and stores the result in matrix `C`.

6. Define the `checkResult()` function for verifying the GPU computation:
   - There are two overloaded versions of this function, one for `float` matrices and another for `int` matrices.
   - The function compares the elements of the host and GPU result matrices and checks if they match within a specified epsilon (for `float` matrices) or exact match (for `int` matrices).

7. Define the GPU kernel functions `sumMatrixOnGPU2D()`:
   - There are two overloaded versions of this kernel, one for adding `float` matrices and another for adding `int` matrices.
   - The kernel uses 2D thread and block indices to perform the matrix addition in parallel on the GPU.

8. Implement the `main()` function:
   - Set the device for CUDA computation and retrieve the device properties.
   - Set the size of the matrices (`nx` and `ny`) and calculate the total number of elements (`nxy`).
   - Allocate host memory for input and result matrices (`h_A`, `h_B`, `hostRef_float`, `gpuRef_float`, `h_A_int`, `h_B_int`, `hostRef_int`, `gpuRef_int`).
   - Initialize the host input matrices using the `initialData()` function and measure the initialization time.
   - Allocate device memory for the input and result matrices (`d_MatA_float`, `d_MatB_float`, `d_MatC_float`, `d_MatA_int`, `d_MatB_int`, `d_MatC_int`).
   - Transfer the input matrices from the host to the device using `cudaMemcpy()`.
   - Warm up the GPU kernel by launching it once for each data type (`float` and `int`).
   - Launch the GPU kernel for `float` matrices and measure the execution time.
   - Copy the result matrix (`d_MatC_float`) back from the device to the host using `cudaMemcpy()`.
   - Verify the GPU computation using the `checkResult()` function for `float` matrices.
   - Launch the GPU kernel for `int` matrices and measure the execution time.
   - Copy the result matrix (`d_MatC_int`) back from the device to the host using `cudaMemcpy()`.
   - Verify the GPU computation using the `checkResult()` function for `int` matrices.
   - Free the allocated memory (`h_A`, `h_B`, `hostRef_float`, `gpuRef_float`, `h_A_int`, `h_B_int`, `hostRef_int`, `gpuRef_int`, `d_MatA_float`, `d_MatB_float`, `d_MatC_float`, `d_MatA_int`, `d_MatB_int`, `d_MatC_int`).
   - Reset the device and exit the program.

## Output:

![image](https://github.com/Marinto-Richee/Parallel-Computing-Architecture/assets/65499285/dee5a19d-b22f-4b3b-a35b-3d959d22cc1a)


## Result:
The program prints device information, matrix size, execution time for matrix operations, memory transfer, and verifies if the results from the host and device match, providing insights into the performance and suggesting the use of float variables for better performance.
