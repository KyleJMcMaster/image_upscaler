#include "../headers/CUDAKernals.cuh"
#include <iostream>

__global__ void shift(int size, int max_col, int *output, int shift_amount){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int x;
    for(int i = index; i < size; i += stride){
        x = output[i]+shift_amount;
        output[i] = (x>=max_col)*max_col + (x<max_col)*x;
    }
}