#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <iostream>
using namespace std;
using namespace cv;
void detectAndDisplay( Mat frame, rs2::depth_frame & depth );
CascadeClassifier face_cascade("./haarcascade_frontalface_alt.xml");
CascadeClassifier eyes_cascade("./haarcascade_eye_tree_eyeglasses.xml");
int main( int argc, const char** argv )
{
    //-- 1. Read the video stream
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 60);
    cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 60);
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    pipe.start(cfg);
    rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
    rs2::video_frame color = data.get_color_frame();
    rs2::depth_frame depth = data.get_depth_frame();
    const int width = color.get_width();
    const int height = color.get_height();

    Mat frame(Size(width, height), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);
    time_t startTime, endTime;
    unsigned long frameCount = 0;
    startTime = clock();
    while ( frame.data )
    {
        //-- 2. Apply the classifier to the frame
        detectAndDisplay( frame, depth );
        if( waitKey(1) == 27 )
        {
            break; // escape
        }
	data = pipe.wait_for_frames();
	color = data.get_color_frame();
	depth = data.get_depth_frame();
	frame = Mat(Size(width, height), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);
	frameCount++;
    }
    endTime = clock();
    std::cout << "CPU Performance: " << frameCount / ((double) (endTime - startTime) / CLOCKS_PER_SEC) << "FPS" << std::endl;
    pipe.stop();
    cfg.disable_stream(RS2_STREAM_DEPTH);
    cfg.disable_stream(RS2_STREAM_COLOR);
    return 0;
}
void detectAndDisplay( Mat frame, rs2::depth_frame & depth )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces );
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4 );
	float distance = depth.get_distance(center.x, center.y);
	putText(frame, to_string(distance) + " m", Point(center.x, center.y),
                        FONT_HERSHEY_SIMPLEX,
                        1, Scalar(255, 0, 255), 2);
        Mat faceROI = frame_gray( faces[i] );
        //-- In each face, detect eyes
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale( faceROI, eyes );
        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4 );
        }
    }
    //-- Show what you got
    imshow( "Capture - Face detection", frame );
}
