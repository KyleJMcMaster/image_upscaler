#include "../headers/ColorChanger.h"


void ColorChanger::set_color(Image* img, int channel, std::vector<int> new_value){
    std::vector<int> dims = img->get_size();
    int size = dims[0] * dims[1];
    new_value.resize(size, 0); //make sure sizes match
    img->set_channel(channel, new_value);
}

void ColorChanger::shift_color(Image* img, int channel, int shift_amount){
    int max_col = img->get_max_col();
    std::vector<int> dims = img->get_size();
    int size = dims[0] * dims[1];
    std::vector<int> value = img->get_channels()[channel];
    for (int i = 0; i < size; i++) {
        int x = value[i]+shift_amount;
        value[i] = (x>=max_col)*max_col + (x<max_col)*x; //limit to max_col without conditional
    }
    img->set_channel(channel, value);
}
