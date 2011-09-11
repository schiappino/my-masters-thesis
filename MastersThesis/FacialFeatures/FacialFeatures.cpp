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
const char* cascadeFNameEye   = "../data/cascades/haarcascade_eye.xml";
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
	imgProcessed,

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
double exec_time;


char text[255];

const char      
	* wndNameSrc = "Source",
	* wndNameFace = "Face";

vector<string> 	imgFileList;
vector<Rect>	faces,
				eyes,
				mouths;


void putTextWithShadow(Mat& img, const char *str, Point org, CvScalar color = CV_RGB(0, 255, 100))
{
	putText( img, str, Point(org.x -1, org.y-1), FONT_HERSHEY_PLAIN, 1, CV_RGB(50, 50, 50), 2 );
	putText( img, str, org, FONT_HERSHEY_PLAIN, 1, color );
};
void displayStats()
{
	sprintf( text, "%dx%d", imgSrc.size().width, imgSrc.size().height );
	putTextWithShadow( imgProcessed, text, Point(5, 35) );

	sprintf( text, "FPS %2.0f", 1000/exec_time);
	putTextWithShadow( imgProcessed, text, Point(5, 55) );

}

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
	int flags = CV_WINDOW_AUTOSIZE;

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
		imgSrc = imread( imgFileList.at(0) );
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

bool DetectFaces()
{
	cascadeFace.detectMultiScale( imgGray, 
		faces, 
		1.1, 
		2, 
		CV_HAAR_FIND_BIGGEST_OBJECT);

	if( faces.size() > 0 )
	{
		rectangle( imgProcessed,
			Point( faces[0].x, faces[0].y),
			Point( faces[0].x + faces[0].width, faces[0].y + faces[0].height),
			CV_RGB( 0, 0, 0));

		return true;
	}
};

void DetectEyes()
{
	Rect eysROI = Rect( faces[0].x, faces[0].y, faces[0].width, 0.5*faces[0].height);
	Mat imgEyesROI (imgGray, eysROI );
	cascadeEye.detectMultiScale(
		imgEyesROI,
		eyes,
		1.2,
		4,
		CV_HAAR_DO_CANNY_PRUNING );
	Mat imgProcessedROI (imgProcessed, eysROI );
	for( int i = 0; i < eyes.size(); ++i )
	{
		rectangle( imgProcessedROI,
			Point( eyes[i].x, eyes[i].y),
			Point( eyes[i].x + eyes[i].width, eyes[i].y + eyes[i].height),
			CV_RGB( 0, 0, 0));
	}
};

void DetectEyebrows()
{};

void DetectMouth()
{};

void ProcessAlgorithm()
{
	imshow( wndNameSrc, imgSrc );

	imgProcessed = imgSrc.clone();
	cvtColor( imgSrc, imgGray, CV_RGB2GRAY );
	equalizeHist( imgGray, imgGray );

	if( DetectFaces() )
	{
		DetectEyes();
	}
	return;
};

int main(int argc, char** argv )
{
	Init();
	InitGUI();

	while( !finish )
	{
		// Start time
		exec_time = startTime();

		ProcessAlgorithm();

		// End time
		calcExecTime( &exec_time );
		cout << "Exec time was "<< exec_time << "\t" << (int) (1000/exec_time) << " FPS" << endl;
		
		displayStats();
		imshow( wndNameFace, imgProcessed );

		handleKeyboard( waitKey(1) );
	}
	
	ExitNicely(0);
}

