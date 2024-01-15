#ifndef COLORCHANGER_H
#define COLORCHANGER_H
#include "../headers/Image.h"
#include <vector>

class ColorChanger {

public:
    static void set_color(Image* img, int channel, std::vector<int> &new_value);
    void shift_color(Image* img, int channel, int shift_amount);
};

#endif