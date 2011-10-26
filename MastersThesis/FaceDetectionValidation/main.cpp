#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

const string cascadeFNameFace		= "../data/cascades/haarcascade_frontalface_alt.xml";
const string IMMFaceDBFile			= "../data/facedb/imm/filelist.txt";
const string ColorFeretDBFile		= "../data/facedb/color feret/filelist.txt";
const string ColorFeretDBFile_fa	= "../data/facedb/color feret/filelist_fa.txt";
const string BioIDDBDile			= "../data/facedb/BioID/filelist.txt";

vector <string> imgFileList;

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
double startTime(void)
{
	return (double) getTickCount();
}
double calcExecTime( double* time )
{
	*time = 1000 * ((double)getTickCount() - *time)/ getTickFrequency(); 
	return *time;
}


int main( int argc, char** argv )
{
	bool finish = false;
	int found_cnt = 0;
	double exec_time;
	vector <double> execTimes;
	vector <Rect> faces;
	CascadeClassifier cascade;
	Mat imsrc,
		imgray,
		imout;

	if( !cascade.load( cascadeFNameFace) ){ printf("--(!)Error loading\n"); return -1; };
	loadFileList( ColorFeretDBFile_fa );

	for( size_t i = 0; i < imgFileList.size(); ++i )
	{
		cout << "Image " << i;

		imsrc = imread( imgFileList[i], CV_LOAD_IMAGE_UNCHANGED );
		if( imsrc.channels() > 1 ) cvtColor( imsrc, imgray, CV_RGB2GRAY );
		else imgray = imsrc;
		equalizeHist( imgray, imgray );

		//resize( imgray, imout, Size(), 0.3, 0.3, CV_INTER_AREA );
		faces.clear();
		
		exec_time = startTime();
		
		cascade.detectMultiScale( imgray, faces, 1.3, 3, CASCADE_FIND_BIGGEST_OBJECT );
		
		exec_time = calcExecTime( &exec_time );
		execTimes.push_back( exec_time );

		if( faces.size() )
		{
			cout << " faces found " << faces.size();
			found_cnt++;
		}
		cout << endl;
	}

	// Calculate average Haar time
	double total_time = 0;
	for( size_t i = 0; i < execTimes.size(); ++i )
		total_time += execTimes[i];

	double avg_time = total_time / execTimes.size();

	cout << "Images processed: " << imgFileList.size()
		 << " Faces found: " << found_cnt
		 << " that's " << ((float)found_cnt / (float)imgFileList.size()) * 100 << "%" << endl
		 << " with average time: " << avg_time
		 << " that's "
		 << setiosflags(ios::basefield) << setiosflags(ios::fixed) << setprecision(1)
		 << (1000/exec_time) << " FPS" << endl << endl;

	return 0;
}