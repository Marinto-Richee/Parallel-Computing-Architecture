# Simple Warp Divergence

## Aim
The aim of this experiment is to compare the performance of two CUDA kernels, `reduceUnrolling8` and `reduceUnrolling16`, which handle 8 and 16 data blocks per thread, respectively.

## Procedure
The experiment follows the following steps:
1. Initialize an input array of size 1024.
2. Launch the `reduceUnrolling8` kernel, which performs reduction using 8 data blocks per thread.
3. Launch the `reduceUnrolling16` kernel, which performs reduction using 16 data blocks per thread.
4. Compare the results obtained from both kernels.

## Output
The program outputs the results of the reduction performed by each kernel. Specifically, it displays the final reduced value obtained from the `reduceUnrolling8` kernel and the `reduceUnrolling16` kernel.

![image](https://github.com/Marinto-Richee/Parallel-Computing-Architecture/assets/65499285/5bf49845-df2d-4598-92d3-25d46a5e8f3a)

![image](https://github.com/Marinto-Richee/Parallel-Computing-Architecture/assets/65499285/028e621e-fa75-453c-9d07-e2f815c2d238)

![image](https://github.com/Marinto-Richee/Parallel-Computing-Architecture/assets/65499285/2dc932f0-1293-496c-b65c-81b8b78ca22c)

## Results
The performance of the two kernels can be compared based on the reduction results. A higher reduction result indicates a more efficient reduction algorithm.
The comparison between the two results can provide insights into the performance difference between the kernels.