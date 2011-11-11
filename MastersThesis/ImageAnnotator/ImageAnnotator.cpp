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

vector <string> imgFileList;
Mat img;
int imIt = 0;
Point origin;
const string colorFeretDB_fa = "../data/facedb/color feret/filelist_fa.txt",
			 groundTruthXml = "00279_940422_fa.xml",
			 wndName = "Image Annotator";

bool loadFileList( const string fileName )
{
	ifstream in;
	string line;

	in.open( fileName );
	if( !in )
	{
		cout << "--(!) Cannot read input file list" << endl;
		return false;
	}

	while( !in.eof() )
	{
		getline(in, line );
		imgFileList.push_back( line );
	}

	if( imgFileList.size() > 0 )
		return true;
	else
	{
		cout << "--(!) Cannot read input file list" << endl;
		return false;
	}
}
void drawPoint( Mat img, Point pt )
{
	Scalar color = CV_RGB(0,255,0);
	int r = 10;
	circle( img, pt, r, color );
	
	Point vertic1 = Point( pt.x, pt.y - r ),
		  vertic2 = Point( pt.x, pt.y + r ),
		  horizo1 = Point( pt.x - r, pt.y ),
		  horizo2 = Point( pt.x + r, pt.y );

	line( img, vertic1, vertic2, color );
	line( img, horizo1, horizo2, color );
}
void onMouse( int event, int x, int y, int, void* )
{
    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN:
        origin = Point(x,y);
		cout << origin << endl;
		drawPoint( img, origin );
        break;
    case CV_EVENT_LBUTTONUP:
        break;
    }
}
int main( int argc, char** argv )
{
	bool isFileLoaded = loadFileList( colorFeretDB_fa );
	if ( !isFileLoaded )
		return -1;

	namedWindow( wndName, CV_GUI_EXPANDED );
	setMouseCallback( wndName, onMouse, 0 );

	img = imread( imgFileList.at( imgFileList.size()-100 ), CV_LOAD_IMAGE_COLOR );
	
	bool finish = true;
	char c;
	while (!finish)
	{
		imshow( wndName, img );
		c = waitKey(1);
		if ( c == 27 ) finish = true;
	}

	FileStorage fs;
	fs.open( groundTruthXml, FileStorage::READ );

	return 0;
}