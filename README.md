# Matrix Multiplication on GPU with CUDA

This project aims to optimize the task of matrix multiplication on GPU architectures. To get optimal performance, the tiling algorithm, use of shared memory, thread level, and instruction level parallelism were explored. The [report](Report.pdf) explains all these methods in detail along with performance results.

## Usage

To build the binary:<br>
make -C build_T4 `cat src_todo_T4/OPTIONS.txt`

To build the binary with cublas:<br>
make -C build_T4 cublastest=1 `cat src_todo_T4/OPTIONS.txt`

To clean the output files of the make command:<br>
make -C build_T4 clean

To Run:<br>
./mmpy `cat src_todo_T4/OPTIONS_RUNTIME.txt` -n 256

To run Script in tools folder:<br>
./tools/run_ncu.sh

If you get Permission denied error when executing a file:<br>
chmod +x name_of_file eg: chmod +x tools/*

Find GPU chipset:<br>
lspci | grep -i --color 'vga\|3d\|2d'

## Results

### Comparison with Naive Implementation

![](naive.png)

### Comparison with BLAS and CuBLAS

![](cublas.png)

### Roofline plots

![](roofline220.png)

![](roofline320.png)

## References

<ol>
  <li>Andrew Kerr, Duane Merrill, Julien Demouth and John Tran, CUTLASS: Fast Linear Algebra in Cuda C++, December 2017</li>
  <li>Nvidia Cutlass github - https://github.com/NVIDIA/cutlass/blob/master/media/docs/efficient_gemm.md</li>
  <li>Volkov, Demmel, Benchmarking GPUs to Tune Dense Linear Algebra, SC2008</li>
  <li>Volkov, Better Performance at lower Occupancy, GTC2010</li>
  <li>Cuda C++ programming guide - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html</li>
  <li>Cuda documentation - https://docs.nvidia.com/cuda/index.html</li>
  <li>Jia, Maggioni, Smith, Scarpazza, ”Dissecting the NVidia Turing T4 GPU via Microbenchmarking”: https://arxiv.org/abs/1903.07486</li>
</ol>
