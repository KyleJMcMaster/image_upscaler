#ifndef KERNEL_H
#define KERNEL_H

#include <fftw3.h>
#include <vector>

class Kernel
{
private:
    int size_x;
    int size_y;
    std::vector<double> data;
    bool fft_loaded;//has a plan been created?
    bool transformed;//has a transform been performed?
    double *fft_data;
    fftw_complex *cfft_data;
    fftw_plan fft_plan;

public:
    Kernel(int size_x, int size_y, std::vector<double> value, bool load_fft);
    bool is_fft_loaded();
    bool is_transformed();
    std::vector<int> get_size();
    void pad_size(int size_x, int size_y);
    std::vector<double> * get_data();
    fftw_complex * get_cfft_result();
    void load_fft();
    void destroy_fft();
    void transform();

};

#endif