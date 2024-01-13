#include "../headers/Image.h"
#include "../headers/ColorChanger.h"
#include <iostream>
#include <string>
#include <vector>

int main()
{
    std::cout << "here-1";
    Image im1("resources/dog_water.ppm", true);
    im1.set_save_filepath("resources/dog_water_red.ppm");
    std::cout << "here";
    ColorChanger::shift_color(&im1, 0, 50);
    std::cout << "here1";
    im1.save_image();
    return 0;
}
