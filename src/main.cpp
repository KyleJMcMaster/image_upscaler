#include "../headers/Image.h"
#include <iostream>
#include <string>
#include <vector>

int main()
{

    Image im1("resources/dog_water.ppm", true);
    im1.set_save_filepath("resources/dog_water_red.ppm");

    std::vector<int> dim = im1.get_size();
    std::vector<int> channel;
    int length = dim[0]*dim[1];
    for(int i = 0; i < length; i++){
        channel.push_back(0);
    }
    im1.set_red_channel(channel);
    im1.save_image();
    return 0;
}
