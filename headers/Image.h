#ifndef IMAGE_H
#define IMAGE_H
#include <string>
#include <vector>
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
    std::vector<int> red;
    std::vector<int> green;
    std::vector<int> blue;

public:
    Image(std::string filepath, bool load);
    Image (std::string load_fp, std::string save_fp, bool load);
    void load_image_from_file();
    void load_image_from_channels(int size_x, int size_y, int max_col, std::vector<int>* red, std::vector<int>* green, std::vector<int>* blue);
    void save_image();
    bool is_loaded();
    bool is_saved();

    int get_max_col();
    std::vector<std::vector<int>> get_channels();
    std::vector<int> get_size();

    void set_save_filepath(std::string filepath);
    void set_channels(std::vector<std::vector<int>> channels);
    void set_red_channel(std::vector<int> channel);
    void set_green_channel(std::vector<int> channel);
    void set_blue_channel(std::vector<int> channel);
};
#endif