#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    VideoCapture cap(0);

    Mat input;
    bool pause = false;
    namedWindow("Camera");
    while (cap.isOpened()) {
        if (!pause) {
            cap >> input;
            imshow("Camera", input);
        }
        char c = (char) waitKey(1);
        if (c == 27) break;
        else if (c == ' ') pause = !pause;
    }
    cap.release();
    destroyAllWindows();
    return 0;
}