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

Point getPointFromASFFile( const string file, int point_no )
{
	ifstream gt;
	string line;
	string sx, sy;
	int line_no = 17 + point_no;
	const int IMG_WIDTH = 640,
			 IMG_HEIGHT = 480;

	gt.open( file, ifstream::in );

	for( int i = 0; (i < line_no) && gt.good(); ++i )
		getline( gt, line );

	gt.close();
	sx = line.substr(6,10); 
	sy = line.substr(18,10);

	return Point( cvRound(atof(sx.c_str())*IMG_WIDTH), cvRound(atof(sy.c_str())*IMG_HEIGHT) );
}
bool parseIMMGroundTruthData( const string file )
{	
	for( int i = 0; i < 58; ++i )
	{

		Point f = getPointFromASFFile(file, i);
		fp.push_back( f );
	}
	return true;
}

int main( int argc, char** argv )
{
	const string im_filename = "../data/facedb/imm/23-5m.jpg",
				 gt_filename = "../data/facedb/imm/23-5m.asf",
				 wnd_name = "IMM feature points";
	int r = 2;

	Mat imsrc = imread( im_filename );	
	namedWindow( wnd_name, CV_WND_PROP_AUTOSIZE );

	parseIMMGroundTruthData( gt_filename );

	for( int i = 0; i < fp.size(); ++i )
	{
		circle( imsrc, fp[i], r, CV_RGB(0,255,0), -1 );
		imshow( wnd_name, imsrc );
		cout << "\r>> Point " << i;
	}

	waitKey();
	return 0;
}