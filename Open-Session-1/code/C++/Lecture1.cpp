#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
    //read an image
    Mat img = imread("../../image_1.png");
    //get the size and type
    Size size = img.size();
    int type = img.type();

    imshow("image", img);
    //use waitKey to pause the program
    waitKey(0);
//    destroyAllWindows();

    //convert to gray scale
    Mat gray_image(size, CV_8UC1);
    cvtColor(img, gray_image, COLOR_BGR2GRAY);
    imshow("gray scale", gray_image);
    waitKey(0);
    destroyAllWindows();

    Mat car_img = imread("../../car.jpg");
    size = car_img.size();
    type = car_img.type();

    //conver to HSV color space
    Mat hsv_image(size, type);
    cvtColor(car_img, hsv_image, COLOR_BGR2HSV);
    imshow("hsv", hsv_image);
    waitKey(0);
//    destroyAllWindows();

    //get the upper and lower part of red
    Mat upper(size, CV_8UC1), lower(size, CV_8UC1);
    inRange(hsv_image, Scalar(0, 30, 0), Scalar(10, 255, 255), upper);
    inRange(hsv_image, Scalar(160, 30, 0), Scalar(179, 255, 255), lower);

    //create a mask
    Mat mask(size, CV_8UC1);
    bitwise_or(lower, upper, mask);
    imshow("mask", mask);
    waitKey(0);
//    destroyAllWindows();

    //show what the mask represent in the original image
    Mat bgr_headlights(size, type), blurred_headlights(size, type);
    bitwise_or(car_img, car_img, bgr_headlights, mask);
    imshow("headlights", bgr_headlights);
    waitKey(0);
//    destroyAllWindows();

    //use Canny Edge Detection to get the contours
    blur(mask, blurred_headlights, Size(3, 3));
    Canny(mask, mask, 100, 200);
    vector<Vec4i> hierarchy;
    std::vector<std::vector<Point>> contours;
    findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    //draw contours
    Mat mContours(size, type);
    int len = (int)contours.size();
    for (int i = 0; i < len; i++) {
        drawContours(mContours, contours, i, Scalar(255, 255, 255));
    }
    imshow("contours", mContours);
    waitKey(0);
    destroyAllWindows();

    return 0;
}