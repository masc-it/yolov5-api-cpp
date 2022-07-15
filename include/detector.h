#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

struct PadInfo
{
    float scale;
    int top;
    int left;
};

struct Detection
{
    PadInfo info;
    std::vector<cv::Mat> detection;
};

class Colors
{
public:
    std::vector<std::string> hex_str;
    std::vector<cv::Scalar> palette;
    int n = 20;
    Colors()
    {
        palette.push_back(hex2rgb("FFFFFF"));
        /*this->hex_str = {
            "FFFFFF", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
            "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7"};
        for (auto &ele : this->hex_str)
        {
            palette.push_back(hex2rgb(ele));
        }*/
    }
    cv::Scalar hex2rgb(const std::string& hex_color)
    {
        int b, g, r;
        sscanf(hex_color.substr(0, 2).c_str(), "%x", &r);
        sscanf(hex_color.substr(2, 2).c_str(), "%x", &g);
        sscanf(hex_color.substr(4, 2).c_str(), "%x", &b);
        return cv::Scalar(b, g, r);
    }
};

struct Config
{
    float confThreshold;
    float nmsThreshold;
    std::string weightPath;
    std::string classNamePath;
    cv::Size size;
    bool _auto;
};

class Detector
{
public:
    Detector(Config &config);
    Detection detect(cv::Mat &img);
    void postProcess(cv::Mat &img, Detection &detection,Colors&cl);
    PadInfo letterbox(cv::Mat &img, cv::Size new_shape, cv::Scalar color, bool _auto, bool scaleFill, bool scaleup, int stride);

private:
    float nmsThreshold;
    float confThreshold;
    cv::Size inSize;
    bool _auto; // not scaled to inSize but   minimum rectangle ,https://github.com/ultralytics/yolov5/blob/master/utils/augmentations.py line 106
    std::vector<std::string> classNames;
    cv::dnn::Net model;
    void drawPredection(cv::Mat &img, std::vector<cv::Rect> &boxes, std::vector<float> &sc, std::vector<int> &clsIndexs, std::vector<int> &ind,Colors&cl);
};
