#include <opencv2/opencv.hpp>
using namespace cv;

int main() {

    Mat pSrcImage = imread("/home/why/CV-Teaching-Sessions/Team-Session-1/example.jpg");
    imshow("original image", pSrcImage);
    Size size = pSrcImage.size();
    int type = pSrcImage.type();

    ///create a lookup table
    Mat lookUpTable(1, 256, CV_8U);
    uchar *p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, 10) * 255.0);

    Mat pDarkImage(size, type);
    ///gamma correction
    LUT(pSrcImage, lookUpTable, pDarkImage);
    imshow("gamma correction", pDarkImage);
    waitKey(0);
}
