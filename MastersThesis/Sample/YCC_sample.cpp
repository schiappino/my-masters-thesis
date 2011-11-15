#include <iostream>
#include <vector>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
	const string input_file_name = "../data/images/mountain-view.jpg";
	Mat imsrc, imycc;
	vector <Mat> channels;

	imsrc = imread( input_file_name );
	cvtColor( imsrc, imycc, CV_RGB2YCrCb );
	split( imycc, channels );

	imshow( "Original", imsrc );
	imshow( "YCC", imycc );
	imshow( "Y", channels[0] );
	imshow( "Cr", channels[1] );
	imshow( "Cb", channels[2] );

	waitKey();
	return 0;
}