#include "../headers/Image.h"
#include <string>
#include <vector>
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
    std::vector<int> red;
    std::vector<int> green;
    std::vector<int> blue;



    Image::Image(std::string filepath, bool load = false){
        load_filepath = filepath;
        save_filepath = filepath;
        if (load)
        {
            load_image_from_file();
            loaded = true;
            saved = true;
        }else
        {
            size_x = 0;
            size_y = 0;
            max_col = 0;
            saved = true;
            loaded = false;
            red = {};
            green = {};
            blue = {};
        }
    }

    Image::Image (std::string load_fp, std::string save_fp, bool load = false){
        load_filepath = load_fp;
        save_filepath = save_fp;
        if (load)
        {
            load_image_from_file();
            loaded = true;
            saved = true;
        }else
        {
            size_x = 0;
            size_y = 0;
            max_col = 0;
            saved = true;
            loaded = false;
            red = {};
            green = {};
            blue = {};
        }
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

            red.push_back(r);
            green.push_back(g);
            blue.push_back(b);
        }
        image.close();
        saved = true;
        loaded = true;
    }

    void Image::load_image_from_channels(int size_x, int size_y, int max_col, std::vector<int>* red, std::vector<int>* green, std::vector<int>* blue){
        size_x = size_x;
        size_y = size_y;
        max_col = max_col;
        red = red;
        green = green;
        blue = blue;

        saved = false;
        loaded = true;
    }

    void Image::save_image(){
        std::ofstream image;
        image.open(save_filepath);

        image << "P3" << "\n";
        image << size_x << " " << size_y << "\n";
        image << max_col << "\n";

        for(int i = 0; i < size_x * size_y; i ++){
            image << red[i] << " ";
            image << green[i] << " ";
            image << blue[i] << " ";
        }
        image.close();
        saved = true;
    }

    bool Image::is_loaded(){return loaded;}
    bool Image::is_saved(){return saved;}
    int Image::get_max_col(){return max_col;}
    std::vector<std::vector<int>> Image::get_channels()
    {
        std::vector<std::vector<int>> channels = {red, green, blue};
        return channels;
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
    void Image::set_channels(std::vector<std::vector<int>> channels)
    {
        red = channels[0];
        green = channels[1];
        blue = channels[2];
        saved = false;
    }
    void Image::set_red_channel(std::vector<int> channel)
    {
        red = channel;
    }
    void Image::set_green_channel(std::vector<int> channel)
    {
        green = channel;
    }
    void Image::set_blue_channel(std::vector<int> channel)
    {
        blue = channel;
    }
