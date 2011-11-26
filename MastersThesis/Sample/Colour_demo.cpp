#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

bool loadFileListFromFile( const string fileName, vector <string>& list )
{
	ifstream in;
	string line;
	list.clear();

	in.open( fileName );
	if( !in )
	{
		cout << "--(!) Cannot read input file list" << endl;
		return false;
	}

	while( !in.eof() )
	{
		getline(in, line );
		list.push_back( line );
	}

	if( list.size() > 0 )
		return true;
	else
	{
		cout << "--(!) Cannot read input file list" << endl;
		return false;
	}
}

int main()
{
	const string imfname = "../data/images/IMM face dataset example.png",
		imListFileName = "../data/facedb/color feret/faces list - fa pose.txt",
		wndNameSrc = "Source image",
		wndNameHue = "Hue image",
		wndNameSat_HSV = "HSV Saturation image",
		wndNameSat_HLS = "HLS Saturation image",
		wndNameVal = "Luminance image",
		wndNameGray = "Grayscale image";
	
	Mat imsrc,
		imgray, 
		imhsv,
		imhls;

	vector <Mat> HSV_planes,
				 HLS_planes,
				 RGB_planes;

	// Load list of images from text file
	vector <string> listOfFiles;
	bool isImgListLoaded = loadFileListFromFile( imListFileName, listOfFiles );
	if (!isImgListLoaded){ cerr << "There has been an error processing list of files" << endl; return -1; }
	
	// Initilize GUI
	int gui_flags = CV_WINDOW_KEEPRATIO,
		currImg = 0,
		imgCnt = listOfFiles.size();
	char c;

	namedWindow( wndNameSrc, gui_flags );
	namedWindow( wndNameHue, gui_flags );
	namedWindow( wndNameSat_HSV, gui_flags );
	namedWindow( wndNameSat_HLS, gui_flags );
	namedWindow( wndNameVal, gui_flags );
	namedWindow( wndNameGray, gui_flags );

	bool quit = false;
	while( !quit )
	{
		imsrc = imread( listOfFiles[currImg], CV_LOAD_IMAGE_UNCHANGED );

		cvtColor( imsrc, imhsv, CV_BGR2HSV_FULL );
		cvtColor( imsrc, imhls, CV_BGR2HLS_FULL );
		cvtColor( imsrc, imgray, CV_RGB2GRAY );

		split( imhsv, HSV_planes );
		split( imhsv, HLS_planes );
		split( imsrc, RGB_planes );

		imshow( wndNameSrc, imsrc );
		imshow( wndNameGray, imgray );

		imshow( wndNameHue, HSV_planes[0] );
		imshow( wndNameSat_HSV, HSV_planes[1] );
		imshow( wndNameVal, HSV_planes[2] );

		imshow( wndNameSat_HLS, HLS_planes[2] );

		c = waitKey();

		if( c == 27 ) quit = true;				// esc key pressed
		else if( c == 110 ) ++currImg;			// n key pressed
		else if( c == 98 ) --currImg;			// b key pressed
		else if( c == 109 ) currImg += 10;		// m key pressed
		else if( c == 118 ) currImg -= 10;		// v key pressed

		if( currImg < 0 )
		{	
			currImg = imgCnt -1;
			imsrc = imread( listOfFiles.at( currImg ));
		}
		else if( currImg >= imgCnt )
		{
			currImg = 0;
			imsrc = imread( listOfFiles.at( currImg ));
		}
		else if( c == 110 || c == 98 || c == 109 || c == 118 )
		{
			imsrc = imread( listOfFiles.at( currImg ));
		}
	}


	return 0;
}