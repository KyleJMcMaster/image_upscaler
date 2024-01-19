#include "../headers/Image.h"
#include "../headers/ColorChanger.h"
#include "../headers/CUDAColorChanger.cuh"
#include "../headers/Kernel.h"
#include <iostream>
#include <string>
#include <vector>

int main()
{
    
    std::vector<double> k1_data = {0.1, 0.2, 0.3, 0.4};
    Kernel k1 = Kernel(2,2,k1_data,false);
    k1.pad_kernel(4,3);
    std::vector<double> k1_padded_data = *(k1.get_data());
    for(int i =0; i<12;i++){
        std::cout<<k1_padded_data[i]<<" ";
        if(i%4==3){
            std::cout<<"\n";
        }
    }
    k1.transform();

    Image im1("resources/dog_water.ppm", true);
    im1.set_save_filepath("resources/dog_water_red.ppm");
    CUDAColorChanger::shift_color(&im1, 1, 125);
    std::cout<<"shifted\n";
    im1.load_fft(false);
    std::cout<<"loaded\n";


    im1.transform();
    std::cout<<"transformed\n";


    im1.inv_transform();
    std::cout<<"inv_transformed\n";
    im1.save_fft();
    std::cout<<"saved\n";
    im1.destroy_fft();
    std::cout<<"plan destroyed\n";
    im1.save_image();
    return 1;
}
