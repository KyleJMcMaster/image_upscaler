#include "../headers/Image.h"
#include "../headers/ColorChanger.h"
#include <iostream>
#include <string>
#include <vector>

int main()
{
    Image im1("resources/dog_water.ppm", true);
    im1.set_save_filepath("resources/dog_water_red.ppm");
    ColorChanger::shift_color(&im1, 2, 78);
    im1.save_image();
    return 0;
}
