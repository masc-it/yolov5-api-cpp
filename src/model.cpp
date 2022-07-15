//
// Created by masc9 on 15/07/2022.
//
#include "model.h"

Model::Model(std::string &model_path, std::vector<std::string> &class_names, int input_size) {

    this->model = cv::dnn::readNetFromONNX(model_path);
    this->model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    this->model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    this->class_names = class_names;

    this->input_size = cv::Size(input_size, input_size);
}

cv::dnn::Net Model::get_model() {
    return this->model;
}

std::vector<std::string> Model::get_class_names() {
    return this->class_names;
}

cv::Size Model::get_input_size() const {
    return this->input_size;
}

DetectionOutput Model::detect(cv::Mat &img, float conf_thresh, float nms_thresh) {

    cv::Mat im;
    img.copyTo(im);
    PadInfo padInfo = letterbox(im, this->input_size, cv::Scalar(114, 114, 114), false, false, true, 32);

    //std::cout << "letterbox ok" << std::endl;
    cv::Mat blob;
    cv::dnn::blobFromImage(im, blob, 1 / 255.0f, cv::Size(im.cols, im.rows), cv::Scalar(0, 0, 0), true, false);
    std::vector<std::string> outLayerNames = this->model.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outs;
    this->model.setInput(blob);
    this->model.forward(outs, outLayerNames);

    //std::cout << "forward ok" << std::endl;
    Detection detection = {padInfo, outs};

    DetectionOutput detectionOutput = this->postProcess(im, detection, conf_thresh, nms_thresh);

    this->drawPredection(im, detectionOutput);

    detectionOutput.img_with_bboxes = im;
    //std::cout << "draw ok" << std::endl;
    //imwrite("D:\\Download\\ONNX-yolov5\\assets\\output.jpg", im);
    return detectionOutput;

}

void Model::drawPredection(cv::Mat &img, DetectionOutput &detectionOutput) {

    auto boxes = detectionOutput.boxes;
    auto scores = detectionOutput.scores;
    auto indices = detectionOutput.indices;
    auto classIndexList = detectionOutput.classIndexList;

    for (int i = 0; i < indices.size(); i++)
    {
        cv::Rect rect = boxes[indices[i]];

        std::string label = this->class_names[classIndexList[i]];

        cv::Scalar color = cv::Scalar::all(255);
        cv::rectangle(img, rect, color);

        int baseLine = 0;

        cv::Size textSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseLine);

        rectangle(img, cv::Rect(rect.x, rect.y - textSize.height - 1, rect.width, textSize.height + 1), color, -1);
        putText(img, label, cv::Point(rect.x, rect.y - 1), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar::all(0), 1, 16);
    }

}

PadInfo Model::letterbox(cv::Mat &img, cv::Size new_shape, cv::Scalar color, bool _auto, bool scaleFill, bool scaleup,
                         int stride) {
    float width = img.cols;
    float height = img.rows;

    float r = std::min(new_shape.width / width, new_shape.height / height);
    if (!scaleup)
        r = std::min(r, 1.0f);
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    int dw = new_shape.width - new_unpadW;
    int dh = new_shape.height - new_unpadH;
    if (_auto)
    {
        dw %= stride;
        dh %= stride;
    }
    dw /= 2, dh /= 2;
    cv::Mat dst;
    resize(img, img, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_LINEAR);
    int top = int(round(dh - 0.1));
    int bottom = int(round(dh + 0.1));
    int left = int(round(dw - 0.1));
    int right = int(round(dw + 0.1));
    copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    return PadInfo{r, top, left};
}

DetectionOutput Model::postProcess(cv::Mat &img, Detection &detection, float conf_thresh, float nms_thresh) {

    //letterbox(img, this->input_size, cv::Scalar(114, 114, 114), false, false, true, 32);
    std::vector<cv::Mat> outs = detection.detections;

    cv::Mat out(outs[0].size[1], outs[0].size[2], CV_32F, outs[0].ptr<float>());

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> indices;
    std::vector<int> classIndexList;
    for (int r = 0; r < out.rows; r++)
    {
        float cx = out.at<float>(r, 0);
        float cy = out.at<float>(r, 1);
        float w = out.at<float>(r, 2);
        float h = out.at<float>(r, 3);
        float sc = out.at<float>(r, 4);
        cv::Mat confs = out.row(r).colRange(5, out.row(r).cols);
        confs *= sc;
        double minV, maxV;
        cv::Point minI, maxI;
        minMaxLoc(confs, &minV, &maxV, &minI, &maxI);
        scores.push_back(maxV);
        boxes.emplace_back(cx - w / 2, cy - h / 2, w, h);
        indices.push_back(r);
        classIndexList.push_back(maxI.x);
    }

    cv::dnn::NMSBoxes(boxes, scores, conf_thresh, nms_thresh, indices);

    std::vector<int> clsIndexs;
    for (int index : indices)
    {
        clsIndexs.push_back(classIndexList[index]);
    }

    return DetectionOutput{
            boxes, scores, indices, clsIndexs, cv::Mat()
    };


}





