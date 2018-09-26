import org.opencv.core.*;

import java.util.ArrayList;

import static org.opencv.core.Core.bitwise_or;
import static org.opencv.core.Core.inRange;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.highgui.HighGui.*;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgproc.Imgproc.*;

public class Lecture1 {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat img = imread("../../image_1.png");
        //get the size and type
        Size size = img.size();
        int type = img.type();

        imshow("image", img);
        //use waitKey to pause the program
        waitKey(0);
//    destroyAllWindows();

        //convert to gray scale
        Mat gray_image = new Mat(size, CV_8UC1);
        cvtColor(img, gray_image, COLOR_BGR2GRAY);
        imshow("gray scale", gray_image);
        waitKey(0);
        destroyAllWindows();

        Mat car_img = imread("../../car.jpg");
        size = car_img.size();
        type = car_img.type();

        //conver to HSV color space
        Mat hsv_image = new Mat(size, type);
        cvtColor(car_img, hsv_image, COLOR_BGR2HSV);
        imshow("hsv", hsv_image);
        waitKey(0);
//    destroyAllWindows();

        //get the upper and lower part of red
        Mat upper = new Mat(size, CV_8UC1), lower = new Mat(size, CV_8UC1);
        inRange(hsv_image, new Scalar(0, 30, 0), new Scalar(10, 255, 255), upper);
        inRange(hsv_image, new Scalar(160, 30, 0), new Scalar(179, 255, 255), lower);

        //create a mask
        Mat mask = new Mat(size, CV_8UC1);
        bitwise_or(lower, upper, mask);
        imshow("mask", mask);
        waitKey(0);
//    destroyAllWindows();

        //show what the mask represent in the original image
        Mat bgr_headlights = new Mat(size, type), blurred_headlights = new Mat(size, type);
        bitwise_or(car_img, car_img, bgr_headlights, mask);
        imshow("headlights", bgr_headlights);
        waitKey(0);
//    destroyAllWindows();

        //use Canny Edge Detection to get the contours
        blur(mask, blurred_headlights, new Size(3, 3));
        Canny(mask, mask, 100, 200);
        ArrayList<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        findContours(mask, contours, hierarchy, 0, 1);
        //draw contours
        Mat mContours = new Mat(size, type);
        int len = contours.size();
        for (int i = 0; i < len; i++) {
            drawContours(mContours, contours, i, new Scalar(255, 255, 255));
        }
        imshow("contours", mContours);
        waitKey(0);
        destroyAllWindows();
    }
}
