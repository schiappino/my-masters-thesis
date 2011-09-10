#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define COLOR_FERET_DB_SIZE 4000
#define IMM_DB_SIZE	250

using namespace cv;
using namespace std;

// ********************************** CASCADE FILES ******************************************
const char* cascadeFNameEye   = "../data/cascades/haarcascade_eye_alt.xml";
const char* cascadeFNameFace  = "../data/cascades/haarcascade_frontalface_alt.xml";
const char* cascadeFNameMouth = "../data/cascades/haarcascade_mcs_mouth.xml";

const char* IMMFaceDBFile = "../data/facedb/imm/filelist.txt";
const char* ColorFeretDBFile = "../data/facedb/color feret/filelist.txt";

// *********************************** VIDEO FILES ******************************************


// ****************************** GLOBALS ***************************************************
const int PROGRAM_MODE = 1;

CvCapture* capture = NULL;


CvMemStorage 
	* storage			= NULL;

CascadeClassifier
	cascadeFace,
	cascadeMouth,
	cascadeEye;


CvPoint	point1, point2;
CvFont font;

CvRect           
	* FaceRectROI		= NULL,
	* MouthRectROI		= NULL,
	* EyeRectROI		= NULL;

CvSeq           
	* facesSeq          = NULL,
	* eyesSeq           = NULL,
	* mouthsSeq         = NULL;

Mat        
	img,
	imgRaw,
	imgTemp,
	imgSrc,
	imgGray,
	imgThresh,
	imgEdge,
	imgEyes,
	imgEyebrows,
	imgMouth,

	imgRGB,
	imgRGBRed,
	imgRGBGreen,
	imgRGBBlue,

	imgHSV,
	imgHSVHue,
	imgHSVSat,
	imgHSVVal;


int                     
	counter         = 0,
	posRes          = 0,
	object_size		= 0,
	scale           = 1,
	imIt			= 0,	// image list literator

	TrackbarLoVal   = 2,
	TrackbarHiVal	= 100,
	TrackbarMaxVal	= 255;

bool finish = false;

double          
	fps = 0, 
	sec = 0;

char text[255];

const char      
	* wndNameSrc = "Source",
	* wndNameFace = "Face";

vector<string> 	imgFileList;

void PutTextWithShadow( const char* text, IplImage* img )
{};

void ExponentialOperator( IplImage* src, IplImage* dst )
{};

void handleKeyboard( char c )
{
	if( c == 27 ) finish = true;		// esc key pressed
	//else if( c == 112 ) pause^= true;	// p key pressed
	else if( c == 110 ) ++imIt;			// n key pressed
	else if( c == 98 ) --imIt;			// b key pressed
	else if( c == 109 ) imIt += 10;		// m key pressed
	else if( c == 118 ) imIt -= 10;		// v key pressed

	if( imIt < 0 )
	{	
		imIt = imgFileList.size() -1;
		imgSrc = imread( imgFileList.at( imIt ));
	}
	else if( imIt >= imgFileList.size() )
	{
		imIt = 0;
		imgSrc = imread( imgFileList.at( imIt ));
	}
	else if( c == 110 || c == 98 || c == 109 || c == 118 )
	{
		imgSrc = imread( imgFileList.at( imIt ));
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

bool loadFileList( const char* fileName )
{
	ifstream in;
	string line;

	in.open( fileName );
	if( !in )
		return false;

	while( !in.eof() )
	{
		getline(in, line );
		imgFileList.push_back( line );
	}

	if( imgFileList.size() > 0 )
		return true;
	else
		return false;
}

void InitGUI()
{
	int flags = CV_WINDOW_KEEPRATIO;

	namedWindow( wndNameSrc, flags );
	namedWindow( wndNameFace, flags );
};

int Init()
{
	// Initialize file list containers
	imgFileList.reserve( COLOR_FERET_DB_SIZE );

	loadFileList( ColorFeretDBFile );

	// Load cascades
	if( !cascadeFace.load( cascadeFNameFace) ){ printf("--(!)Error loading\n"); return -1; };
	if( !cascadeEye.load( cascadeFNameEye ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !cascadeMouth.load( cascadeFNameMouth ) ){ printf("--(!)Error loading\n"); return -1; };

	// Initialize font
	cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.0, 1.0, 0, 1, CV_AA);

	// *** MAIN APPLICATION INIT ***
	// Load image or frame form camera/avi capture;
	// PROGRAM_MODE = 1 work on images
	// PROGRAM_MODE = 2 work on frames from avi capture
	// PROGRAM_MODE = 3 work on frames from webcam capture
	//
	if( PROGRAM_MODE == 1 )
	{
		img = imread( imgFileList.at(0) );
	}
	else if( PROGRAM_MODE == 2 )
	{
		cout << "Not implemented" << endl;
		return -1;
	}
	else if( PROGRAM_MODE == 3 )
	{
		cout << "Not implemented" << endl;
		return -1;	
	}
	

	return 0;
};

int ExitNicely(int code)
{
	return code;
};

void DetectFaces()
{};

void DetectEyes()
{};

void DetectEyebrows()
{};

void DetectMouth()
{};

void ProcessAlgorithm()
{
	imshow( wndNameSrc, imgSrc );
	return;
};

int main(int argc, char** argv )
{
	Init();
	InitGUI();

	double exec_time;
	while( !finish )
	{
		if( PROGRAM_MODE == 1 )
		{
			imgSrc = imread( imgFileList.at( imIt ));
		}
	
		exec_time = startTime();
		ProcessAlgorithm();
		calcExecTime( &exec_time );
		cout << "Exec time was "<< exec_time << endl;
		handleKeyboard( waitKey(1) );
	}
	
	
	
	
	
	Mat img = imread("../data/images/lena.jpg");
	namedWindow("OpenCV Image", CV_WINDOW_KEEPRATIO );
	imshow("OpenCV Image", img );
	waitKey();

	ExitNicely(0);
}

