#include "detector.h"

Detector::Detector(Config &config)
{
    this->nmsThreshold = config.nmsThreshold;
    this->confThreshold = config.confThreshold;

    std::ifstream ifs(config.classNamePath);
    std::string line;
    while (getline(ifs, line))
        this->classNames.push_back(line);
    ifs.close();

    this->model = cv::dnn::readNetFromONNX(config.weightPath);
    this->model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    this->model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    this->inSize = config.size;
    this->_auto = config._auto;
}

PadInfo Detector::letterbox(cv::Mat &img, cv::Size new_shape, cv::Scalar color, bool _auto, bool scaleFill, bool scaleup, int stride)
{
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
    return {r, top, left};
}

Detection Detector::detect(cv::Mat &img)
{
    // 预处理 添加border
    cv::Mat im;
    img.copyTo(im);
    PadInfo padInfo = letterbox(im, this->inSize, cv::Scalar(114, 114, 114), this->_auto, false, true, 32);
    cv::Mat blob;
    cv::dnn::blobFromImage(im, blob, 1 / 255.0f, Size(im.cols, im.rows), cv::Scalar(0, 0, 0), true, false);
    std::vector<std::string> outLayerNames = this->model.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outs;
    this->model.setInput(blob);
    this->model.forward(outs, outLayerNames);
    return {padInfo, outs};
}
void Detector::postProcess(cv::Mat &img, Detection &detection, Colors &cl)
{

    letterbox(img, this->inSize, cv::Scalar(114, 114, 114), this->_auto, false, true, 32);
    std::vector<cv::Mat> outs = detection.detection;

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

    cv::dnn::NMSBoxes(boxes, scores, this->confThreshold, this->nmsThreshold, indices);

    std::vector<int> clsIndexs;
    for (int index : indices)
    {
        clsIndexs.push_back(classIndexList[index]);
    }

    drawPredection(img, boxes, scores, clsIndexs, indices, cl);
}

void Detector::drawPredection(cv::Mat &img, std::vector<cv::Rect> &boxes, std::vector<float> &scores, std::vector<int> &clsIndexs, std::vector<int> &ind, Colors &cl)
{

    for (int i = 0; i < ind.size(); i++)
    {
        cv::Rect rect = boxes[ind[i]];
        float score = scores[ind[i]];
        std::string name = this->classNames[clsIndexs[i]];
        //int color_ind = clsIndexs[i] % 20;
        cv::Scalar color = cl.palette[0];
        rectangle(img, rect, color);
        char s_text[80];
        sprintf(s_text, "%.2f", round(score * 1e3) / 1e3);
        std::string label = name;  // + " " + s_text;

        int baseLine = 0;

        cv::Size textSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseLine);

        rectangle(img, cv::Rect(rect.x, rect.y - textSize.height - 1, rect.width, textSize.height + 1), color, -1);
        putText(img, label, cv::Point(rect.x, rect.y - 1), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 1, 16);
    }
    //imshow("rst", img);
    //waitKey(0);
}
