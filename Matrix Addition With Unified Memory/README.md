# Matrix Addition With Unified Memory

## Aim
The aim of this experiment is to demonstrate matrix addition using CUDA programming with unified memory.

## Procedure
Here's a condensed version of the step-by-step procedure, including the functions used:

1. Include the required header files and define necessary macros.
   - `cuda_runtime.h`, `stdio.h`, `stdlib.h`, `time.h`, `math.h`, `windows.h`, `device_launch_parameters.h`

2. Define helper functions for time measurement and error checking.
   - `seconds()`: Measures the current time in seconds using Windows API functions.
   - `CHECK(call)`: Macro to check CUDA function calls for errors.

3. Implement functions for data initialization, host matrix addition, and result checking.
   - `initialData(float* ip, const int size)`: Initializes the input matrices with random values.
   - `sumMatrixOnHost(float* A, float* B, float* C, const int nx, const int ny)`: Performs matrix addition on the host side.
   - `checkResult(float* hostRef, float* gpuRef, const int N)`: Compares the host and GPU results for verification.

4. Define the GPU kernel function for matrix addition.
   - `sumMatrixGPU(float* MatA, float* MatB, float* MatC, int nx, int ny)`: Performs matrix addition on the GPU using the grid and block dimensions.

5. In the main function:
   - Set up the CUDA device and determine the matrix size.
     - `cudaGetDeviceProperties(&deviceProp, dev)`
     - `cudaSetDevice(dev)`

   - Allocate memory on the host for matrices A, B, hostRef, and gpuRef.
     - `cudaMallocManaged((void**)&A, nBytes)`
     - `cudaMallocManaged((void**)&B, nBytes)`
     - `cudaMallocManaged((void**)&gpuRef, nBytes)`
     - `cudaMallocManaged((void**)&hostRef, nBytes)`

   - Initialize data on the host and perform matrix addition on the host side.
     - `initialData(A, nxy)`
     - `initialData(B, nxy)`
     - `sumMatrixOnHost(A, B, hostRef, nx, ny)`

   - Allocate and initialize GPU result matrices.
     - `memset(hostRef, 0, nBytes)`
     - `memset(gpuRef, 0, nBytes)`

   - Configure the GPU grid and block dimensions.
     - `dim3 block(dimx, dimy)`
     - `dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y)`

   - Perform a warm-up kernel launch.
     - `sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, 1, 1)`

   - Measure the execution time of the GPU matrix addition.
     - `sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, nx, ny)`
     - `cudaDeviceSynchronize()`

   - Synchronize and check for kernel errors.
     - `cudaGetLastError()`

   - Compare host and GPU results for verification.
     - `checkResult(hostRef, gpuRef, nxy)`

   - Free allocated memory on the device and reset the device.
     - `cudaFree(A)`
     - `cudaFree(B)`
     - `cudaFree(hostRef)`
     - `cudaFree(gpuRef)`
     - `cudaDeviceReset()`

6. Return and terminate the program.

## Output

![image](https://github.com/Marinto-Richee/Parallel-Computing-Architecture/assets/65499285/07a8472f-09d2-4601-a9c3-baefbec34321)

![image](https://github.com/Marinto-Richee/Parallel-Computing-Architecture/assets/65499285/f125fd3b-d021-41c3-96e6-007b1b78ea9f)

![image](https://github.com/Marinto-Richee/Parallel-Computing-Architecture/assets/65499285/eb8a8c46-1e57-4b60-be1f-85b0aa6f39c9)

## Result
Removing the `memset` calls does not affect the correctness of the program and has no significant impact on its performance. It is good practice to remove unnecessary code to improve code readability and maintainability.
<br>
The result provides insights on the advantages of Unified memory. And profiling provides detailled information about the resource utilization.

