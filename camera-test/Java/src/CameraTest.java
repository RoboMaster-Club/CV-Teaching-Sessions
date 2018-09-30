import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;

import static org.opencv.highgui.HighGui.*;

public class CameraTest {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        VideoCapture cap = new VideoCapture(0);

        Mat input = new Mat();
        boolean pause = false;
        namedWindow("Camera");
        while (cap.isOpened()) {
            if (!pause) {
                cap.read(input);
                imshow("Camera", input);
            }
            char c = (char) waitKey(1);
            if (c == 27) break;
            else if (c == ' ') pause = !pause;
        }
        cap.release();
        destroyAllWindows();
    }
}