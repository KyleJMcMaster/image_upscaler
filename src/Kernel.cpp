#include "../headers/Kernel.h"
#include <iostream>


    Kernel::Kernel(int size_x, int size_y, std::vector<double> value, bool do_load_fft){
        data = value;
        this->size_x = size_x;
        this->size_y = size_y;
        if(do_load_fft){
            load_fft();
        }
    }
    bool Kernel::is_fft_loaded(){return fft_loaded;}
    bool Kernel::is_transformed(){return transformed;}
    std::vector<int> Kernel::get_size(){
        std::vector<int> dimensions = {size_x, size_y};
        return dimensions;
    }
    void Kernel::pad_kernel(int target_size_x, int target_size_y){
        std::vector<double> new_data(target_size_x*target_size_y);
        for(int i = 0; i < target_size_x*target_size_y; i++){
            if(i%target_size_x >= size_x || i/target_size_x >= size_y){
                new_data[i] = 0;
            }else{
                new_data[i] = data[i%target_size_x + i/target_size_x*size_x];
            }
        }
        padded_data = new_data;
        padded_size_x = target_size_x;
        padded_size_y = target_size_y;
        padded = true;
        if(fft_loaded){
            destroy_fft();
        }
    }
    std::vector<double> * Kernel::get_data(){
        return &data;
    }
    fftw_complex * Kernel::get_cfft_result(){
        return cfft_data;
    }
    void Kernel::load_fft(){
        if(!padded){
            std::cout<<"kernel not padded... aborting load_fft()";
            return;
        }
        if(fft_loaded){
            destroy_fft();
        }
        fft_data = fftw_alloc_real(padded_size_y * 2 * (padded_size_x/2 + 1));
        cfft_data = (fftw_complex*) &fft_data[0];

        fft_plan = fftw_plan_dft_r2c_2d(padded_size_y, padded_size_x, fft_data, cfft_data, FFTW_ESTIMATE);

        for(int i = 0; i < padded_size_y*padded_size_x; i++){
            fft_data[i] = padded_data[i];
        }
        fft_loaded = true;
    }
    void Kernel::destroy_fft(){
        if(!fft_loaded){
            return;
        }
        fftw_destroy_plan(fft_plan);
        fftw_free(fft_data);
        fft_loaded = false;
        transformed = false;
    }
    void Kernel::transform(){
        if(!fft_loaded){
            return;
        }
        if(transformed){
            return;
        }
        fftw_execute(fft_plan);
        transformed = true;
    }
    void Kernel::convolve(int size_x, int size_y, fftw_complex * img_data){
        if(!padded){
            pad_kernel(size_x, size_y);
        }
        if(size_x != padded_size_x || size_y != padded_size_y){
            pad_kernel(size_x, size_y);
        }
        if(!fft_loaded){
            load_fft();
        }
        if(!transformed){
            transform();
        }
        double tmp;
        for(int i = 0; i < 100; i++){
            std::cout<<img_data[i][0]<<", "<<img_data[i][1]<<"\n";
        }
        std::cout<<"input---------\n";
        for(int i = 0; i < size_y*(size_x/2+1); i++){
            
            tmp = img_data[i][0] * cfft_data[i][0] - img_data[i][1] * cfft_data[i][1];
            img_data[i][1] = img_data[i][0] * cfft_data[i][1] + img_data[i][1] * cfft_data[i][0];
            img_data[i][0] = tmp;
        }
        for(int i = 0; i < 100; i++){
            std::cout<<img_data[i][0]<<", "<<img_data[i][1]<<"\n";
        }
        std::cout<<"output---------\n";

    }