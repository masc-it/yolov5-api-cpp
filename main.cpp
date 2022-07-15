#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <vector>

#include "model.h"
#include "json.hpp"
#include "crow_all.h"

using json = nlohmann::json;


Model load_from_config(){

    std::ifstream inFile;
    inFile.open("data/config.json");

    std::stringstream strStream;
    strStream << inFile.rdbuf(); //read the file
    std::string config_string = strStream.str();
    json config_json = json::parse(config_string);

    std::string model_path = config_json["model_path"].get<std::string>();
    std::vector<std::string> class_names = config_json["class_names"].get<std::vector<std::string>>();

    return Model(model_path, class_names, config_json["input_size"].get<int>());

}


int main(int argc, char *argv[])
{

    Model model = load_from_config();

    crow::SimpleApp app;
    CROW_ROUTE(app, "/health-check").methods("GET"_method)
            ([&model](const crow::request& request, crow::response& res) {

            res.write("YoloV5 API");
            res.end();
            });

    CROW_ROUTE(app, "/predict").methods("POST"_method)
        ([&model](const crow::request& request, crow::response& res) {

            try {
                std::string img_str = request.body;

                // Check headers and sanitize
                auto conf_thresh_str = request.get_header_value("X-Confidence-Thresh");

                if (conf_thresh_str.empty()){
                    throw std::runtime_error("X-Confidence-Thresh undefined");
                }

                auto nms_thresh_str = request.get_header_value("X-NMS-Thresh");

                if (nms_thresh_str.empty()){
                    throw std::runtime_error("X-NMS-Thresh undefined");
                }

                auto conf_thresh = std::stof(conf_thresh_str);
                auto nms_thresh = std::stof(nms_thresh_str);

                if (conf_thresh <= 0 || conf_thresh > 1){
                    conf_thresh = 0.5;
                }

                if (nms_thresh <= 0 || nms_thresh > 1){
                    nms_thresh = 0.5;
                }

                std::vector<uchar> data(img_str.begin(), img_str.end());
                cv::Mat img = cv::imdecode(data, cv::IMREAD_UNCHANGED);

                auto detection = model.detect(img, conf_thresh, nms_thresh);

                // TODO Return image or JSON
                std::vector<uchar> buf;
                cv::imencode(".jpg",detection.img_with_bboxes,buf);
                std::string img_out(buf.begin(), buf.end());

                res.write(img_out);

                res.add_header("Content-Type", "image/jpeg");
                res.end();
            } catch(std::exception &e){
                std::ostringstream stringStream;
                stringStream << R"({"status": "ERR", "msg": ")";
                stringStream << e.what();
                stringStream << "\"}";
                std::string msg = stringStream.str();

                res.write(msg); // R"({"status": "ERR", "msg": e.what()})"
                res.add_header("Content-Type", "application/json");
                res.end();
            }


        });

    app
    .port(5000)
    .multithreaded()
    .run();

    return 0;
}
