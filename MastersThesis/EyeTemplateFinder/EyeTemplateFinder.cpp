#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#define FILE_NAME_PREFIX IMM

using namespace std;
using namespace cv;

const string classifierFileNameFace = "../data/cascades/haarcascade_frontalface_alt.xml",
	classifierFileNameEye = "../data/cascades/haarcascade_eye.xml";

const double K_EXP_OPERATOR = 0.0217304452751310829264530948549876073716129212732431841605;

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
Rect detectFace( const string cascadeFileName, const Mat& img )
{
	CascadeClassifier cascade;
	cascade.load( cascadeFileName );

	vector <Rect> faces;
	cascade.detectMultiScale( img, faces, 1.15, 2, CV_HAAR_FIND_BIGGEST_OBJECT );

	if( faces.size() ) 
	{ return faces[0]; }
	else { return Rect(); }
}
Rect detectEye( const string cascadeFileName, const Mat& img )
{
	CascadeClassifier cascade;
	cascade.load( cascadeFileName );

	vector <Rect> eyes;
	cascade.detectMultiScale( img, eyes, 1.15, 2, CV_HAAR_FIND_BIGGEST_OBJECT );

	if( eyes.size() ) 
	{ return eyes[0]; }
	else { return Rect(); }
}
void ExponentialOperator( Mat src, Mat dst, const Mat& lut)
{

}
void extractEyesFromImages( const string list )
{
	vector <string> listOfImages;
	char outFileName[255];
	bool fileListLoaded = loadFileListFromFile( list, listOfImages );
	if( !fileListLoaded ) { cerr << "There has been an error while loding file list" << endl; return; }

	// Initialize Exponential Operator Look-up Table
	Mat lookUpTable( 1, 256, CV_8U );
	uchar* p = lookUpTable.data;
	for( int i = 0; i < 256; ++i ){ p[i] = (uchar)exp( i * K_EXP_OPERATOR ); }

	Mat imsrc, imgray, imred,imexp;
	vector <Mat> bgr;
	Rect roiEyeLeft, roiEyeRight,
		foundRoiEyeLeft, foundRoiEyeRight;
	int imgCnt = listOfImages.size();

	for( int currImg = 0; currImg < imgCnt; ++currImg )
	{
		imsrc = imread( listOfImages[ currImg ], CV_LOAD_IMAGE_UNCHANGED );
		cvtColor( imsrc, imgray, CV_BGR2GRAY );
		equalizeHist( imgray, imgray );
		split( imsrc, bgr );
		bgr[2].copyTo( imred );
		bitwise_not( imred, imred );
		LUT( imred, lookUpTable, imexp );

		Rect face = detectFace( classifierFileNameFace, imgray );

		roiEyeLeft	= Rect( face.x, face.y,					face.width/2., face.height/2. );
		roiEyeRight = Rect( face.x + face.width/2., face.y, face.width/2., face.height/2. );


		Mat imEyeLeft( imgray, roiEyeLeft );
		Mat imEyeRight( imgray, roiEyeRight );
		Rect foundRoiEyeLeft = detectEye( classifierFileNameEye, imEyeLeft );
		Rect foundRoiEyeRight = detectEye( classifierFileNameEye, imEyeRight );

		if( foundRoiEyeLeft.width && foundRoiEyeLeft.height )
		{
			Mat left( imexp, Rect( roiEyeLeft.x + foundRoiEyeLeft.x, roiEyeLeft.y + foundRoiEyeLeft.y,
									foundRoiEyeLeft.width, foundRoiEyeLeft.height ));
			
			sprintf_s( outFileName, "results\\lefteye-%d.png", currImg+1 );
			imwrite( outFileName, left );
		}

		if( foundRoiEyeRight.width && foundRoiEyeRight.height )
		{
			Mat right( imexp, Rect( roiEyeRight.x + foundRoiEyeRight.x, roiEyeRight.y + foundRoiEyeRight.y,
									foundRoiEyeRight.width, foundRoiEyeRight.height ));
		
			sprintf_s( outFileName, "results\\righteye-%d.png", currImg+1 );
			imwrite( outFileName, right );
		}

		// Display progress
		cout << "\r>>> " << setiosflags(ios::basefield) << setiosflags(ios::fixed) << setprecision(1)
			 << ((double)(currImg + 1)/(double)imgCnt) * 100 << " %";
	}
}
int main()
{
	const string imFileList = "../data/facedb/imm/im_filelist(frontal).txt";

	extractEyesFromImages( imFileList );

	waitKey();
	return 0;
}