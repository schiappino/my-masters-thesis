#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main()
{
	const char		* fileName  = "../data/video sequences/VIDEO0020.3gp",
					* wndName	= "Video";
	double			exec_time;
	Mat				frame;
	VideoCapture	cap;


	namedWindow( wndName, CV_WINDOW_KEEPRATIO );
	cap.open( fileName );
	if( !cap.isOpened() ) { cout << "Error opening file" << endl; return -1; }

	for(;;)
	{
		exec_time = (double) getTickCount();

		cap >> frame;
		if( frame.empty() ){ cout << "Could not load frame" << endl; return -1; }
		imshow( wndName, frame );

		exec_time = 1000 * ((double)getTickCount() - exec_time)/ getTickFrequency(); 
		waitKey(1);
	}
}