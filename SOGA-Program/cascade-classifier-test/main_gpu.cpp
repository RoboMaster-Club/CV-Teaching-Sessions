#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/cudaarithm.hpp>
#include "opencv2/cudaobjdetect.hpp"

#include <iostream>
#include <opencv2/cudaimgproc.hpp>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
Ptr<cuda::CascadeClassifier> cascade = cuda::CascadeClassifier::create(
        "./cascade.xml");

/** @function main */
int main(int argc, const char **argv) {
    VideoCapture capture;
    //-- 2. Read the video stream
    capture.open("./robot_blue_3m_480p.mp4");
    if (!capture.isOpened()) {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }

    Mat frame;
    time_t startTime, endTime;
    startTime = clock();
    unsigned long frameCount = 0;
    while (capture.read(frame)) {
        if (frame.empty()) {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }

        //-- 3. Apply the classifier to the frame
        detectAndDisplay(frame);
        frameCount++;

        if (waitKey(10) == 27) {
            break; // escape
        }
    }
    endTime = clock();
    std::cout << "CPU Performance: " << frameCount / ((double) (endTime - startTime) / CLOCKS_PER_SEC) << "FPS"
              << std::endl;
    return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame) {
    Size size = frame.size();
    int type = frame.type();

    cuda::GpuMat gpu_frame(size, type), frame_gray(size, CV_8UC1), results_gpu;;
    gpu_frame.upload(frame);

    cuda::cvtColor(gpu_frame, frame_gray, COLOR_BGR2GRAY);
    cuda::equalizeHist(frame_gray, frame_gray);

    //-- Detect results
    std::vector<Rect> results;
    cascade->detectMultiScale(frame_gray, results_gpu);
    cascade->convert(results_gpu, results);

#ifndef NDEBUG

    for (size_t i = 0; i < results.size(); i++) {
        Point center(results[i].x + results[i].width / 2, results[i].y + results[i].height / 2);
        ellipse(frame, center, Size(results[i].width / 2, results[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
    }

    //-- Show what you got
    imshow("Capture - Face detection", frame);

#endif
}