# MonteCarloIsing
Monte Carlo simulation of the 2D Ising model on the square lattice of size LxL 
This program is written in CUDA and tested on Ubuntu 21.04 with CUDA-tool-kit 11.

1) hwo to compile
  $ nvcc -O2 cudaIsing_AB.cu -lculand_static -lculibos -lm

2) how to run
  $ ./a.out 1024 1000 10
  Three command line arguments should be given:
  L (=1024): width and height of the square lattice
  Tmax (=1000): MC time steps
  delt (=10): output will be written in every delt time steps

3) output 
   L beta=1/T t Energy MAGNETIZATION
