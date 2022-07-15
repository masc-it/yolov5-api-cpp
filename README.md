# YoloV5-API

API to run inferences with YoloV5 models. Written in C++, based on OpenCV 4.5.5

## Setup

**Data** directory must contain your config.json

**config.json** defines:
- ONNX absolute model path
- input size (640 default)
- array of class names

A dummy example is available in the _data/_ folder


## Docker

    docker pull mascit/yolov5-api

To run the container, you first need to mount your data folder containing config.json and your onnx model.

    docker run --name yolov5-api -v <path to data on host>:/app/data -p <port>:5000 mascit/yolov5-api

Remember to use a container-relative path for your _model_path_ field in **config.json**

## Build

Or, just build it from source.

    cmake --configure .
    cmake --build . --target main -j <num jobs>

# Endpoints

## /predict [POST]

### Body
- Image bytes (binary in Postman)

### Headers
- X-Confidence-Thresh
  - default 0.5
- X-NMS-Thresh
  - default 0.45
- X-Return
  - image_with_boxes
    - A JPEG image with drawn predictions
  - json (default)
    - A json array containing predictions. Each object defines: xmin, ymin, xmax, ymax, conf, class_name
