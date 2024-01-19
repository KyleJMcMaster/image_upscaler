#include "../headers/Image.h"
#include "../headers/ColorChanger.h"
#include "../headers/CUDAColorChanger.cuh"
#include "../headers/Kernel.h"
#include <iostream>
#include <string>
#include <vector>

int main()
{
    int size = 100;
    std::vector<double> k1_data(size*size);
    for(int i = 0; i<size*size; i++){
        k1_data[i] = size;
    }
    Kernel k1 = Kernel(size,size,k1_data,false);

    Image im1("resources/dog_water.ppm", true);
    im1.set_save_filepath("resources/dog_water_blur.ppm");
    std::vector<int> dims = im1.get_size();
    //CUDAColorChanger::shift_color(&im1, 1, 125);
    //std::cout<<"shifted\n";
    im1.load_fft(false);
    std::cout<<"loaded\n";
    im1.transform();
    std::cout<<"transformed\n";

    k1.convolve(dims[0], dims[1], im1.get_cfft_result(0));
    k1.convolve(dims[0], dims[1], im1.get_cfft_result(1));
    k1.convolve(dims[0], dims[1], im1.get_cfft_result(2));


    fftw_complex * img_data = im1.get_cfft_result(0);
    for(int i = 0; i < 100; i++){
            std::cout<<img_data[i][0]<<", "<<img_data[i][1]<<"\n";
        }

    im1.inv_transform();
    std::cout<<"inv_transformed\n";
    im1.save_fft();
    std::cout<<"saved\n";
    im1.destroy_fft();
    std::cout<<"plan destroyed\n";
    im1.save_image();
    return 1;
}
