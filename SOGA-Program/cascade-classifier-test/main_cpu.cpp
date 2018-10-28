#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
CascadeClassifier cascade;
//CascadeClassifier eyes_cascade;

/** @function main */
int main(int argc, const char **argv) {
    //-- 1. Load the cascades
    if (!cascade.load("./cascade.xml")) {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };

    //-- 2. Read the video stream
    VideoCapture capture("./robot_blue_3m.mp4");
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
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    //-- Detect faces
    std::vector<Rect> faces;
    cascade.detectMultiScale(frame_gray, faces);

#ifndef NDEBUG

    for (size_t i = 0; i < faces.size(); i++) {
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);

        Mat faceROI = frame_gray(faces[i]);

        //-- Show what you got
        imshow("Capture - Face detection", frame);
    }

#endif
}
