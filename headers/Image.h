#ifndef IMAGE_H
#define IMAGE_H
#include <string>
#include <vector>
#include <fftw3.h>
class Image
{
private:
    std::string load_filepath;
    std::string save_filepath;
    int size_x;
    int size_y;
    int max_col;
    bool saved; // does the saved colour data match the data at save_filepath
    bool loaded; //is there colour data to read
    std::vector<std::vector<int>> data;
    bool fft_loaded;//has a plan been created?
    bool transformed;//has a transform been performed?
    double *fft_data_red;
    double *fft_data_green;
    double *fft_data_blue;
    fftw_complex *cfft_data_red;
    fftw_complex *cfft_data_green;
    fftw_complex *cfft_data_blue;
    fftw_plan fft_plan_red;
    fftw_plan fft_plan_green;
    fftw_plan fft_plan_blue;
    fftw_plan ifft_plan_red;
    fftw_plan ifft_plan_green;
    fftw_plan ifft_plan_blue;
    

public:
    Image(std::string filepath, bool load);
    Image (std::string load_fp, std::string save_fp, bool load);
    void load_image_from_file();
    void load_image_from_channels(int size_x, int size_y, int max_col, std::vector<int> &red, std::vector<int> &green, std::vector<int> &blue);
    void save_image();
    bool is_loaded();
    bool is_saved();

    int get_max_col();
    std::vector<std::vector<int>> * get_channels();
    std::vector<int> get_size();

    void set_save_filepath(std::string filepath);
    void set_channels(std::vector<std::vector<int>> &channels);
    void set_channel(int channel, std::vector<int> &value);

    void load_fft(bool quick_load);
    void save_fft();
    void transform();
    void inv_transform();
    void destroy_fft();
    
    //add fcn to reload image from transformed data
};

#endif