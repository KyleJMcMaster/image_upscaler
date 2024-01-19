#ifndef KERNEL_H
#define KERNEL_H

#include "../headers/Image.h"
#include <fftw3.h>
#include <vector>

class Kernel
{
private:
    int size_x;
    int size_y;
    int padded_size_x;
    int padded_size_y;
    std::vector<double> data;
    std::vector<double> padded_data;
    bool fft_loaded;//has a plan been created?
    bool transformed;//has a transform been performed?
    bool padded;//has the kernel been resized?
    double *fft_data;
    fftw_complex *cfft_data;
    fftw_plan fft_plan;

public:
    Kernel(int size_x, int size_y, std::vector<double> value, bool load_fft);
    bool is_fft_loaded();
    bool is_transformed();
    std::vector<int> get_size();
    void pad_kernel(int size_x, int size_y);
    std::vector<double> * get_data();
    fftw_complex * get_cfft_result();
    void load_fft();
    void destroy_fft();
    void transform();
    void convolve(int size_x, int size_y, fftw_complex * img_data);

};

#endif