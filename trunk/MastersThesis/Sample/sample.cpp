// Video Image PSNR and SSIM
#include <iostream> // for standard I/O
#include <string> // for strings
#include <iomanip> // for controlling float print precision
#include <sstream> // string to number conversion
#include <opencv2/imgproc/imgproc.hpp> // Gaussian Blur
#include <opencv2/core/core.hpp> // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp> // OpenCV window I/O
using namespace std;
using namespace cv;

int main(int argc, char *argv[], char *window_name)
{
	Mat img = imread("../data/images/Koala.jpg");

	namedWindow("OpenCV Image", CV_GUI_EXPANDED );
	imshow("OpenCV Image", img );
	waitKey();
	return 0;
}
