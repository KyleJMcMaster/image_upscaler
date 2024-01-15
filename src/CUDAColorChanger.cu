#include "../CUDAColorChanger.cuh"

void CUDAColorChanger::shift_color(Image* img, int channel, int shift_amount){
    std::vector<int> dims = img->get_size();
    int size = dims[0] * dims[1];
    std::vector<int>

}

__global__ void CUDAColorChanger::shift(int size, std::vector<int> channel, int shift_amount){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < size; i += stride){
        int x = channel[i]+shift_amount;
        channel[i] = (x>=max_col)*max_col + (x<max_col)*x;
    }
}