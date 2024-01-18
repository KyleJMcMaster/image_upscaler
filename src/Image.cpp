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
        if(fft_loaded){
            destroy_fft();
        }

        //allocate for in-place transform
        fft_data_red = fftw_alloc_real(size_y * 2 * (size_x/2 + 1));
        cfft_data_red = (fftw_complex*) &fft_data_red;

        fft_data_green = fftw_alloc_real(size_y * 2 * (size_x/2 + 1));
        cfft_data_green  = (fftw_complex*) &fft_data_green ;

        fft_data_blue = fftw_alloc_real(size_y * 2 * (size_x/2 + 1));
        cfft_data_blue = (fftw_complex*) &fft_data_blue;

        
        //make plans
        if (quick_load){
            fft_plan_red = fftw_plan_dft_r2c_2d(size_y, size_x, fft_data_red, cfft_data_red, FFTW_ESTIMATE);
            ifft_plan_red = fftw_plan_dft_c2r_2d(size_y, size_x, cfft_data_red, fft_data_red, FFTW_ESTIMATE);

            fft_plan_green = fftw_plan_dft_r2c_2d(size_y, size_x, fft_data_green, cfft_data_green, FFTW_ESTIMATE);
            ifft_plan_green = fftw_plan_dft_c2r_2d(size_y, size_x, cfft_data_green, fft_data_green,  FFTW_ESTIMATE);

            fft_plan_blue = fftw_plan_dft_r2c_2d(size_y, size_x, fft_data_blue, cfft_data_blue, FFTW_ESTIMATE);
            ifft_plan_blue = fftw_plan_dft_c2r_2d(size_y, size_x, cfft_data_blue, fft_data_blue, FFTW_ESTIMATE);
        } else{
            fft_plan_red = fftw_plan_dft_r2c_2d(size_y, size_x, fft_data_red, cfft_data_red, FFTW_MEASURE);
            ifft_plan_red = fftw_plan_dft_c2r_2d(size_y, size_x, cfft_data_red, fft_data_red, FFTW_MEASURE);

            fft_plan_green = fftw_plan_dft_r2c_2d(size_y, size_x, fft_data_green, cfft_data_green, FFTW_MEASURE);
            ifft_plan_green = fftw_plan_dft_c2r_2d(size_y, size_x, cfft_data_green, fft_data_green,  FFTW_MEASURE);

            fft_plan_blue = fftw_plan_dft_r2c_2d(size_y, size_x, fft_data_blue, cfft_data_blue, FFTW_MEASURE);
            ifft_plan_blue = fftw_plan_dft_c2r_2d(size_y, size_x, cfft_data_blue, fft_data_blue, FFTW_MEASURE);
        }

        //load data to 2d array
        for(int i = 0; i < size_y; i++){
            for(int j = 0; j < size_x; j++){
                fft_data_red[i][j] = data[0][i*size_y+j];
                fft_data_green[i][j] = data[1][i*size_y+j];
                fft_data_blue[i][j] = data[2][i*size_y+j];
            }
        }
        fft_loaded = true;
    }
    void Image::transform(){
        if(!transformed)
        fftw_execute(fft_plan_red);
        fftw_execute(fft_plan_green);
        fftw_execute(fft_plan_blue);
        transformed = true;
    }
    void Image::inv_transform(){
        if(transformed)
        fftw_execute(ifft_plan_red);
        fftw_execute(ifft_plan_green);
        fftw_execute(ifft_plan_blue);
        transformed = false;
    }
    void Image::destroy_fft(){
        if(!fft_loaded){
            return;
        }

        fftw_destroy_plan(fft_plan_red);
        fftw_destroy_plan(fft_plan_green);
        fftw_destroy_plan(fft_plan_blue);
        fftw_destroy_plan(ifft_plan_red);
        fftw_destroy_plan(ifft_plan_green);
        fftw_destroy_plan(ifft_plan_blue);

        fftw_free(fft_data_red);
        fftw_free(fft_data_green);
        fftw_free(fft_data_blue);

        fft_loaded = false;
        transformed = false;
    }
