#include "../headers/Image.h"
#include <fstream>
#include <sstream>
#include <iostream>



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
    double *fft_data;
    fftw_plan fft_plan;
    fftw_plan ifft_plan;





    Image::Image(std::string filepath, bool load = false){
        load_filepath = filepath;
        save_filepath = filepath;

        if (load)
        {
            data = {{},{},{}};
            load_image_from_file();
            loaded = true;
        }else
        {
            size_x = 0;
            size_y = 0;
            max_col = 0;
            loaded = false;
            data = {{},{},{}};
        }
        saved = true;
        fft_loaded=false;
        transformed=false;
    }

    Image::Image (std::string load_fp, std::string save_fp, bool load = false){
        load_filepath = load_fp;
        save_filepath = save_fp;
        if (load)
        {
            data = {{},{},{}};
            load_image_from_file();
            loaded = true;
        }else
        {
            size_x = 0;
            size_y = 0;
            max_col = 0;
            loaded = false;
            data = {{},{},{}};
        }
        saved = true;
        fft_loaded=false;
        transformed=false;
    }



    void Image::load_image_from_file(){

        std::ifstream image;
        std::string type = "none";
        std::string red_str, green_str, blue_str;
        
        int r,g,b;
        int index = 0;
        image.open(load_filepath);
        
        image >> type;
        image >> size_x;
        image >> size_y;
        image >> max_col;
        if(type == "none"){
            std::cout << "no file found at " << load_filepath<<"\n";
            return;
        }

        while(!image.eof()){
            std::cout << "loading pixels: " << index << ":" <<size_x*size_y<<"           \r";
            index++;
            image >> red_str;
            image >> green_str;
            image >> blue_str;

            std::stringstream redstream(red_str);
            std::stringstream greenstream(green_str);
            std::stringstream bluestream(blue_str);

            redstream >> r;
            greenstream >> g;
            bluestream >> b;

            data[0].push_back(r);
            data[1].push_back(g);
            data[2].push_back(b);
        }
        std::cout << size_x*size_y << " pixels loaded                  \n";
        image.close();
        saved = true;
        loaded = true;
        fft_loaded=false;
        transformed=false;
    }

    void Image::load_image_from_channels(int size_x, int size_y, int max_col, std::vector<int> &red, std::vector<int> &green, std::vector<int> &blue){
        size_x = size_x;
        size_y = size_y;
        max_col = max_col;
        data[0] = red;
        data[1] = green;
        data[2] = blue;

        saved = false;
        loaded = true;
        fft_loaded=false;
        transformed=false;
    }

    void Image::save_image(){
        std::ofstream image;
        image.open(save_filepath);

        image << "P3" << "\n";
        image << size_x << " " << size_y << "\n";
        image << max_col << "\n";

        for(int i = 0; i < size_x * size_y; i ++){
            image << data[0][i] << " ";
            image << data[1][i] << " ";
            image << data[2][i] << " ";
        }
        image.close();
        saved = true;
    }

    bool Image::is_loaded(){return loaded;}
    bool Image::is_saved(){return saved;}
    int Image::get_max_col(){return max_col;}
    std::vector<std::vector<int>> * Image::get_channels()
    {
        return &data;
    }
    std::vector<int> Image::get_size()
    {
        std::vector<int> dimensions = {size_x, size_y};
        return dimensions;
    }

    void Image::set_save_filepath(std::string filepath)
     {
        save_filepath = filepath;
        saved = false;
    }
    void Image::set_channels(std::vector<std::vector<int>> &channels)
    {
        data[0] = channels[0];
        data[1] = channels[1];
        data[2] = channels[2];
        saved = false;
    }
    void Image::set_channel(int channel, std::vector<int> &value)
    {
        data[channel] = value;
    }

    void Image::load_fft(bool quick_load = false)
    {
        fftw_complex *cfft_data;
        fft_data = fftw_alloc_real(size_y * 2 * (size_x/2 + 1));
        cfft_data = (fftw_complex*) &fft_data;
        if (quick_load){
            fft_plan = fftw_plan_dft_r2c_2d(size_y, size_x, fft_data, cfft_data, FFTW_ESTIMATE)
            ifft_plan = fftw_plan_dft_c2r_2d(size_y, size_x, fft_data, cfft_data, FFTW_ESTIMATE)

        }
    }
    void Image::transform();
    void Image::inv_transform();
    void Image::destroy_fft();
