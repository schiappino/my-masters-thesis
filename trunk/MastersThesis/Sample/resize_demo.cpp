#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

int main ()
{
	const string im_fname = "../data/images/exp-op subject IMM 38-01m.png";
	const string wnd_name = "Image";
	Size dst_size = Size( 320, 240 );

	Mat imsrc, imres;

	namedWindow( wnd_name, CV_GUI_EXPANDED );

	imsrc = imread( im_fname, CV_LOAD_IMAGE_UNCHANGED );
	
	float dx = (float)dst_size.width / (float)imsrc.cols;
	float dy = (float)dst_size.height / (float)imsrc.rows;
	float scale;
	( dx >= dy ) ? scale = dx : scale = dy;
	
	resize( imsrc, imres, Size(), scale, scale, CV_INTER_AREA );
	imshow( wnd_name, imres );
	waitKey();
	return 0;
}