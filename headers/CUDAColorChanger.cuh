#ifndef CUDACOLORCHANGER_H
#define CUDACOLORCHANGER_H
#include "../headers/Image.h"
#include "../headers/ColorChanger.h"
#include <vector>

class CUDAColorChanger: public ColorChanger {

public:
    static void set_color(Image* img, int channel, std::vector<int> new_value);
    static void shift_color(Image* img, int channel, int shift_amount);

};

#endif
