#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

int main()
{
	const char	* wndName = "Source image",
				* wndNameGray = "Gray img", 
				* wndNameOut = "Out",
				* filename = "../data/images/eyes-exp-transform.png";

	Mat src, gray, thresh, binary;
	Mat out;
	vector<KeyPoint> keyPoints;

	SimpleBlobDetector::Params params;
	params.minThreshold = 10;
	params.blobColor = 255;

	namedWindow( wndNameOut, CV_GUI_NORMAL );
	//namedWindow( wndName, CV_GUI_NORMAL );

	src = imread( filename, CV_LOAD_IMAGE_GRAYSCALE );

	SimpleBlobDetector blobDetector;
	blobDetector.create("SimpleBlob");

	for(;;)
	{

		blobDetector.detect( src, keyPoints );
		drawKeypoints( src, keyPoints, out, CV_RGB(0,255,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
		cout << "Keypoints " << keyPoints.size() << endl;
	
		imshow( wndNameOut, out );
		waitKey(0);
	}
}