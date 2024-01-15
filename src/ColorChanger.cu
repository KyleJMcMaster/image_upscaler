#include "../headers/ColorChanger.h"
#include <iostream>

void ColorChanger::set_color(Image* img, int channel, std::vector<int> &new_value){
    std::vector<int> dims = img->get_size();
    int size = dims[0] * dims[1];
    new_value.resize(size, 0); //make sure sizes match
    img->set_channel(channel, new_value);
}

void ColorChanger::shift_color(Image* img, int channel, int shift_amount){
    std::vector<int> dims;
    std::vector<int> *channel_value; //= new std::vector<int>();
    int max_col;
    int size;

    max_col = img->get_max_col();
    dims = img->get_size();
    size = dims[0] * dims[1];
    channel_value = &((*(img->get_channels()))[channel]);
    std::cout << channel_value << "\n";
    std::cout <<(*channel_value)[0] << "\n";

    for (int i = 0; i < size; i++) {
        int x = (*channel_value)[i]+shift_amount;
        (*channel_value)[i] = (x>=max_col)*max_col + (x<max_col)*x; //limit to max_col without conditional
    }
}
