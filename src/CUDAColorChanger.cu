#include "../headers/CUDAColorChanger.cuh"

void CUDAColorChanger::shift_color(Image* img, int channel, int shift_amount){
    std::vector<int> dims;
    int* channel_value;
    int size;
    int max_col;

    max_col = img->get_max_col();
    dims = img->get_size();
    size = dims[0] * dims[1];
    channel_value = &((*(img->get_channels()))[channel][0]);

    cudaMallocManaged(&channel_value, size*sizeof(int));
    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    CUDAColorChanger::shift<<<numBlocks, blockSize>>>(size, max_col, channel_value, shift_amount);

    cudaDeviceSynchronize();

    cudaFree(channel_value);
}

__global__ void CUDAColorChanger::shift(int size, int max_col, int* channel, int shift_amount){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < size; i += stride){
        int x = channel[i]+shift_amount;
        channel[i] = (x>=max_col)*max_col + (x<max_col)*x;
    }
}