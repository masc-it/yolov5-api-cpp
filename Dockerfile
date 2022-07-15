FROM ubuntu:20.04

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update -y && apt-get install -y build-essential \
  gcc \
  cmake \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libv4l-dev \
  libxvidcore-dev \
  libx264-dev \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  libatlas-base-dev \
  libtbb2 \
  libtbb-dev \
  libdc1394-22-dev \
  libboost-all-dev \
  wget \
  unzip

RUN wget -O opencv.zip "https://github.com/opencv/opencv/archive/4.5.5.zip" \
    && unzip "opencv.zip" \
    && mkdir -p build && cd build

RUN cmake -DBUILD_opencv_java=OFF \
    -DWITH_QT=OFF -DWITH_GTK=OFF \
    -DBUILD_opencv_python=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_OPENGL=OFF \
    -DWITH_OPENCL=ON \
    -DWITH_IPP=ON \
    -DWITH_TBB=ON \
    -DWITH_EIGEN=ON \
    -DWITH_V4L=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
     "../opencv-4.5.5"

RUN cmake --build . -j 10

WORKDIR /app
COPY . .

RUN cmake --configure .
RUN cmake --build . --target main -j 6

EXPOSE 5000
CMD ["./main"]
