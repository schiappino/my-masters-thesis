#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace std;
using namespace cv;


const double K_EXP_OPERATOR = 0.0217304452751310829264530948549876073716129212732431841605;

void exponentialOperator( Mat src, Mat dst )
{
};

int main()
{
	const string wndNameSrc = "Src Image",
		wndNameRed = "Red Channel",
		wndNameGreen = "Green channel",
		wndNameBlue = "Blue channel",
		wndNameNot = "Complementary Image",
		wndNameExp = "Exponential transf";

	const string imFileName = "../data/facedb/imm/28-2m.jpg";

	vector <Mat> imRgbPlanes;
	Mat imSrc, imNot, imRed, imGreen, imBlue, imExp;

	int GUI_FLAGS = CV_WND_PROP_AUTOSIZE;

	namedWindow( wndNameSrc, GUI_FLAGS );
	namedWindow( wndNameNot, GUI_FLAGS );
	namedWindow( wndNameRed, GUI_FLAGS );
	namedWindow( wndNameGreen, GUI_FLAGS );
	namedWindow( wndNameBlue, GUI_FLAGS );
	namedWindow( wndNameExp, GUI_FLAGS );


	imSrc = imread( imFileName, CV_LOAD_IMAGE_UNCHANGED );
	imshow( wndNameSrc, imSrc );

	split( imSrc, imRgbPlanes );
	imshow( wndNameRed, imRgbPlanes[2] );
	imshow( wndNameGreen, imRgbPlanes[1] );
	imshow( wndNameBlue, imRgbPlanes[0] );

	imRgbPlanes[2].copyTo( imRed );
	//equalizeHist( imRed, imRed );
	bitwise_not( imRed, imNot );
	imshow( wndNameNot, imNot );

	// Initialize Exponential Operator Look-up Table
	Mat lookUpTable( 1, 256, CV_8U );
	uchar* p = lookUpTable.data;
	for( int i = 0; i < 256; ++i ){ p[i] = (uchar)exp( i * K_EXP_OPERATOR ); }
	LUT( imNot, lookUpTable, imExp );

	imshow( wndNameExp, imExp );


	waitKey();
	return 0;
}