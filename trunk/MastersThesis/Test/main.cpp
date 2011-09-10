#include <iostream>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat img = imread("../Data/images/lena.jpg", 1 );
	imshow( "OpenCV window", img );
	waitKey();
	return 0;
};