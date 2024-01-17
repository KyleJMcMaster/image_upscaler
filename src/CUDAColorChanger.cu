#include "../headers/CUDAColorChanger.cuh"
#include "../headers/CUDAKernels.cuh"
#include <iostream>

void CUDAColorChanger::shift_color(Image* img, int channel, int shift_amount){
    std::vector<int> dims;
    cudaError_t error;
    int *channel_value, *output;
    int size;
    int max_col;

    max_col = img->get_max_col();
    dims = img->get_size();
    size = dims[0] * dims[1];

    cudaMallocManaged(&output, size*sizeof(int));

    channel_value = &((*(img->get_channels()))[channel][0]);

    for (int i = 0; i < size; i++) {
        output[i] = channel_value[i];
    }

    int blockSize = 256;
    int numBlocks = 10;


    shift<<<numBlocks, blockSize>>>(size, max_col, output, shift_amount);

    cudaDeviceSynchronize();

    error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

    std::vector<int> out_vec(output,output+size);
    img->set_channel(channel, out_vec);
    cudaFree(output);
}


