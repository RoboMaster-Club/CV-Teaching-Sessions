#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <inRange_gpu.h>

using namespace cv;

int main() {
    Mat msrc = imread("./image_1.jpg");
    Size size = msrc.size();
    int type = msrc.type();
    Mat mrst;
    cuda::GpuMat gsrc(size, type), grst(size, CV_8UC1);
    gsrc.upload(msrc);
    clock_t startTime, endTime;
    startTime = clock();
    for (int i = 0; i < 100; i++) {
        inRange_gpu(gsrc, Scalar(0, 0, 200), Scalar(179, 200, 255), grst);
    }
    endTime = clock();
    grst.download(mrst);
#ifndef NDEBUG
    imshow("GPU Result", mrst);
#endif
    std::cout << "GPU calculation time: " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;

    startTime = clock();
    for(int i = 0; i < 100; i++) {
        inRange(msrc, Scalar(0, 0, 200), Scalar(179, 200, 255), mrst);
    }
    endTime = clock();
    std::cout << "CPU calculation time: " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;

#ifndef NDEBUG
    imshow("CPU Result", mrst);
#endif
    waitKey(0);
    return 0;
}