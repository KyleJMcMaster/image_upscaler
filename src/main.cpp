#include "../headers/Image.h"
#include "../headers/ColorChanger.h"
#include "../headers/CUDAColorChanger.cuh"
#include <iostream>
#include <string>
#include <vector>

int main()
{
    Image im1("resources/dog_water.ppm", true);
    im1.set_save_filepath("resources/dog_water_red.ppm");
    CUDAColorChanger::shift_color(&im1, 0, 125);
    im1.load_fft(false);
    im1.transform();
    im1.inv_transform();
    im1.destroy_fft();
    im1.save_image();
    return 1;
}
