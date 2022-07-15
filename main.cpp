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
    inFile.open("config.json");

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
    CROW_ROUTE(app, "/test").methods("GET"_method)
            ([&model](const crow::request& request, crow::response& res) {

            res.write("Hello!");
            res.end();
            });
    CROW_ROUTE(app, "/").methods("POST"_method)
        ([&model](const crow::request& request, crow::response& res) {

            try {
                std::string img_str = request.body;
                std::vector<uchar> data(img_str.begin(), img_str.end());
                cv::Mat img = cv::imdecode(data, cv::IMREAD_UNCHANGED);

                auto detection = model.detect(img, 0.5, 0.45);

                /*cv::imshow("Image", detection.img_with_bboxes);
                cv::waitKey();*/

                std::vector<uchar> buf;
                cv::imencode(".jpg",detection.img_with_bboxes,buf);
                std::string img_out(buf.begin(), buf.end());

                res.write(img_out);
                /*res.add_header("Access-Control-Allow-Origin", "*");
                res.add_header("Access-Control-Allow-Headers", "Content-Type");*/
                res.add_header("Content-Type", "image/jpeg");
                res.end();
            } catch(std::exception &e){
                res.write(R"({"status": "ERR"})");
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
