//
// Created by mascIT on 15/07/2022.
//
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#ifndef APP_MODEL_H
#define APP_MODEL_H


struct PadInfo
{
    float scale;
    int top;
    int left;
};

struct Detection
{
    PadInfo info;
    std::vector<cv::Mat> detections;
};

struct DetectionOutput
{
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> indices;
    std::vector<int> classIndexList;
    cv::Mat img_with_bboxes;
};

class Model {

public:

    Model(std::string &model_path, std::vector<std::string> &class_names, int input_size);
    cv::dnn::Net get_model();
    std::vector<std::string> get_class_names();

    cv::Size get_input_size() const;

    DetectionOutput detect(cv::Mat &img, float conf_thresh, float nms_thresh);

private:
    cv::dnn::Net model;

    std::vector<std::string> class_names;

    cv::Size input_size;

    void drawPredection(cv::Mat &img, DetectionOutput &detectionOutput);
    PadInfo letterbox(cv::Mat &img, cv::Size new_shape, cv::Scalar color, bool _auto, bool scaleFill, bool scaleup, int stride);
    DetectionOutput postProcess(cv::Mat &img, Detection &detection, float conf_thresh, float nms_thresh);

};





#endif //APP_MODEL_H
