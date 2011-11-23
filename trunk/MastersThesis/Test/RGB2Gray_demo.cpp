#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace std;
using namespace cv;

int main ()
{
	const string imfname = "../data/facedb/imm/23-5m.jpg";
	const string wndNameRGB2Gray = "RGB2Gray",
		wndNameBGR2Gray = "BGR2Gray",
		wndNameSrc = "Source";

	Mat imsrc, imrgbgray, imbgrgray;

	namedWindow( wndNameSrc );
	namedWindow( wndNameRGB2Gray );
	namedWindow( wndNameBGR2Gray );

	imsrc = imread( imfname );
	imshow( wndNameSrc, imsrc );

	cvtColor( imsrc, imrgbgray, CV_RGB2GRAY );
	imshow( wndNameRGB2Gray, imrgbgray );

	cvtColor( imsrc, imbgrgray, CV_BGR2GRAY );
	imshow( wndNameBGR2Gray, imbgrgray );

	waitKey();

	return 0;
}