#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define COLOR_FERET_DB_SIZE 4000
#define IMM_DB_SIZE	250
#define _DEBUG
//#define _MOUTH_ROI_DEBUG

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

CascadeClassifier
	cascadeFace,
	cascadeMouth,
	cascadeEye;


CvPoint	point1, point2;
CvFont font;

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
	imgHSV,
	imgHSVFull,
	imgHLS,
	imgHLSFull;

vector<Mat> rgb_planes,
			hsv_planes,
			hls_planes,
			hsvfull_planes,
			hlsfull_planes;


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


inline void putTextWithShadow(Mat& img, const char *str, Point org, CvScalar color = CV_RGB(0, 255, 100))
{
	putText( img, str, Point(org.x -1, org.y-1), FONT_HERSHEY_PLAIN, 1, CV_RGB(50, 50, 50), 2 );
	putText( img, str, org, FONT_HERSHEY_PLAIN, 1, color );
};

inline string getCurentFileName( string filePath )
{
	size_t found = filePath.find_last_of("/");
	return filePath.substr( found + 1 );
}

void displayStats()
{
	// Show image size
	sprintf( text, "%dx%d", imgSrc.size().width, imgSrc.size().height );
	putTextWithShadow( imgProcessed, text, Point(5, 35) );

	// Show FPS
	sprintf( text, "FPS %2.0f", 1000/exec_time);
	putTextWithShadow( imgProcessed, text, Point(5, 55) );

	// When working on files 
	if( PROGRAM_MODE == 1 ) 
	{
		// Show current file name 
		putTextWithShadow( imgProcessed, getCurentFileName( imgFileList.at(imIt) ).c_str(), Point(5, 75));
	}

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
	else if( imIt >= (int)imgFileList.size() )
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

	// Load list of images to container
	loadFileList( ColorFeretDBFile );

	// Initialize file list iterator 
	imIt = imgFileList.size() - 250;

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
		// Load first image from image list
		imgSrc = imread( imgFileList.at(imIt) );
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
	else
		return false;
};

void DetectEyes() // DO POPRAWKI
{
	// Start detecting only if face is found
	if( faces.size() )
	{
		Rect eyesROI = Rect( faces[0].x, faces[0].y, faces[0].width, (int)(0.4*faces[0].height) );
		Mat imgEyesROI (imgGray, eyesROI );

		cascadeEye.detectMultiScale(
			imgGray,
			eyes,
			1.2,
			3,
			CV_HAAR_DO_CANNY_PRUNING );
		Mat imgProcessedROI (imgProcessed, eyesROI );
		for( int i = 0; i < (int)eyes.size(); ++i )
		{
			rectangle( imgProcessed,
				Point( eyes[i].x, eyes[i].y),
				Point( eyes[i].x + eyes[i].width, eyes[i].y + eyes[i].height),
				CV_RGB( 0, 0, 0));
		}
	}
};

void DetectEyebrows()
{};

void DetectMouth()
{
	// Create ROI for mouth detection
	Rect mouthROI = Rect(
		(int) (faces[0].x + 0.2*faces[0].width), 
		(int) (faces[0].y + 0.65*faces[0].height),
		(int) (0.6*faces[0].width), 
		(int) (0.45*faces[0].height));

#ifdef _MOUTH_ROI_DEBUG
	// Show up ROI on source image
	rectangle( 
		imgProcessed, 
		Point( mouthROI.x, mouthROI.y ),
		Point( mouthROI.x + mouthROI.width, mouthROI.y + mouthROI.height ),
		CV_RGB(0,255,0)
	);
	putTextWithShadow(
		imgProcessed,
		"Mouth detection ROI",
		Point( mouthROI.x, mouthROI.y )
	);
#endif
	
	// Setup ROI on image where detection will be done
	Mat imgMouthROI( imgGray, mouthROI );

	cascadeMouth.detectMultiScale(
		imgMouthROI,					// image to search
		mouths,							// found objects container
		1.2,							// window increase param
		3,								// min neighbors to accept object
		CV_HAAR_FIND_BIGGEST_OBJECT		// search method
	);

	// Check if detector found anything; if yes draw it
	if( mouths.size() )
	{
		// Setup ROI on output image so that object 
		// coordinates compliy with those on search image
		Mat imgProcessedROI( imgProcessed, mouthROI );

		// ..and draw it
		rectangle( imgProcessedROI, 
			Point( (int) (mouths[0].x - 0.1*mouths[0].width), (int) (mouths[0].y - 0.1*mouths[0].height) ),
			Point( (int) (mouths[0].x + 1.1*mouths[0].width), mouths[0].y + mouths[0].height ),
			CV_RGB(0,0,0) );
	}
};

void ProcessAlgorithm()
{
	//imgProcessed = imgSrc.clone();
	imgSrc.copyTo( imgProcessed );

	cvtColor( imgSrc, imgGray, CV_RGB2GRAY );
	cvtColor( imgSrc, imgHSV, CV_RGB2HSV );
	cvtColor( imgSrc, imgHLS, CV_RGB2HLS );

#ifdef _DEBUG 

	cvtColor( imgSrc, imgHSVFull, CV_RGB2HSV_FULL );
	cvtColor( imgSrc, imgHSVFull, CV_RGB2HLS_FULL );

	equalizeHist( imgGray, imgGray );

	split( imgRGB, rgb_planes );
	split( imgHSV, hsv_planes );
	split( imgHSVFull, hsvfull_planes );
	split( imgHLS, hls_planes );
	split( imgHLSFull, hlsfull_planes );

#endif

	if( DetectFaces() )
	{
		//DetectEyes();
		DetectMouth();
	}
	imshow( wndNameFace, imgProcessed );
	return;
};

int main(int argc, char** argv )
{
	Init();
	InitGUI();

	while( !finish )
	{
		// Show current image or frame
		imshow( wndNameSrc, imgSrc );

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

