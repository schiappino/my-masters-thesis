#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

vector <string> im_files,
				gt_files;
vector <Point> fp;

Point getPointFromPtsFile( const string file, int point_no )
{
	ifstream gt;
	string line;
	float x, y;
	int line_no = 2 + point_no;

	gt.open( file, ifstream::in );

	for( int i = 0; (i < line_no) && gt.good(); ++i )
		getline( gt, line );

	gt >> x;
	gt >> y;

	gt.close();

	return Point( cvRound(x), cvRound(y) );
}
bool parseBioIDGroundTruthData( const string file )
{	
	for( int i = 1; i <= 20; ++i )
	{

		Point f = getPointFromPtsFile(file, i);
		fp.push_back( f );
	}
	return true;
}

int main( int argc, char** argv )
{
	const string im_filename = "../data/facedb/bioid/bioid_1520.pgm",
				 gt_filename = "../data/facedb/bioid/bioid_1520.pts",
				 wnd_name = "IMM feature points";
	int r = 2;

	Mat imsrc = imread( im_filename );	
	namedWindow( wnd_name, CV_WND_PROP_AUTOSIZE );

	parseBioIDGroundTruthData( gt_filename );

	for( int i = 0; i < fp.size(); ++i )
	{
		circle( imsrc, fp[i], r, CV_RGB(0,255,0), -1 );
		imshow( wnd_name, imsrc );
		cout << "\r>> Point " << i+1;
		waitKey();
	}

	waitKey();
	return 0;
}