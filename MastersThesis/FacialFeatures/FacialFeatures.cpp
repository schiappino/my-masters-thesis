#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define HUE_PLANE 0
#define COLOR_FERET_DB_SIZE 4000
#define IMM_DB_SIZE	250
#define _DEBUG
#define FACE_DETECT_DEBUG
//#define _MOUTH_ROI_DEBUG
#define _CRT_SECURE_NO_WARNINGS 1
//#define EYES_DETECT_SINGLE_CASCADE
//#define EYES_DETECT_MULTI_CASCADE

using namespace cv;
using namespace std;

// ********************************** CASCADE FILES ******************************************
const char* cascadeFNameEye				= "../data/cascades/haarcascade_eye.xml";
const char* cascadeFNameEyeRightSplit	= "../data/cascades/haarcascade_righteye_2splits.xml";
const char* cascadeFNameEyeLeftSplit	= "../data/cascades/haarcascade_lefteye_2splits.xml";
const char* cascadeFNameFace			= "../data/cascades/haarcascade_frontalface_alt.xml";
const char* cascadeFNameMouth			= "../data/cascades/haarcascade_mcs_mouth.xml";

// ********************************** IMAGE FILES *******************************************
const char* IMMFaceDBFile				= "../data/facedb/imm/filelist.txt";
const char* ColorFeretDBFile			= "../data/facedb/color feret/filelist.txt";

// *********************************** VIDEO FILES ******************************************


// ****************************** GLOBALS ***************************************************
const int PROGRAM_MODE = 1;

const double K_EXP_OPERATOR = 0.0217304452751310829264530948549876073716129212732431841605;

CvCapture* capture = NULL;

CascadeClassifier
	cascadeFace,
	cascadeMouth,
	cascadeEye,
	cascadeEyeRight,
	cascadeEyeLeft;


CvPoint	point1, point2;
CvFont font;

CvRect
	foundFaceROI,
	foundMouthROI;

Scalar mouthHueAvg;

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

Mat lookUpTable( 1, 256, CV_8U );

vector<Mat> rgb_planes,
			hls_planes;


int                     
	counter         = 0,
	posRes          = 0,
	object_size		= 0,
	scale           = 1,
	imIt			= 0,	// image list literator

	mouthThreshold	= 0,
	bilateralBlur	= 0,

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
	* wndNameFace = "Face",
	* wndNameMouth = "Mouth",
	* wndNameBlur = "Blur",

	* trckbarMouthThresh = "Mouth THR",
	* trckbarBilateralBlur = "Bilatera blur";

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

	sprintf( text, "avg hue %d", mouthHueAvg[0]);
	putTextWithShadow( imgProcessed, text, Point(5, 95) );
}

inline void exponentialOperator( Mat src, Mat dst )
{
	LUT( src, lookUpTable, dst );
};

void onThresholdTrackbar( int val, void* )
{
	mouthThreshold = val;
};
void onBilateralBlur( int val, void* )
{
	bilateralBlur = val;
};
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
	//namedWindow( wndNameMouth, flags );
	//namedWindow( wndNameBlur, flags );

	createTrackbar( trckbarMouthThresh, wndNameMouth, &mouthThreshold, 255, onThresholdTrackbar );
	createTrackbar( trckbarBilateralBlur, wndNameBlur, &bilateralBlur, 31, onBilateralBlur );
};

int Init()
{
	// Initialize file list containers
	imgFileList.reserve( COLOR_FERET_DB_SIZE );

	// Load list of images to container
	loadFileList( ColorFeretDBFile );

	// Initialize file list iterator 
	imIt = imgFileList.size() - 20;

	// Load cascades
	if( !cascadeFace.load( cascadeFNameFace) )				{ printf("--(!)Error loading\n"); return -1; };
	if( !cascadeEyeRight.load( cascadeFNameEyeRightSplit ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !cascadeEyeLeft.load( cascadeFNameEyeLeftSplit ) )	{ printf("--(!)Error loading\n"); return -1; };
	if( !cascadeMouth.load( cascadeFNameMouth ) )			{ printf("--(!)Error loading\n"); return -1; };
	if( !cascadeMouth.load( cascadeFNameMouth ) )			{ printf("--(!)Error loading\n"); return -1; };

	// Initialize font
	cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.0, 1.0, 0, 1, CV_AA);

	// Initialize Exponential Operator Look-up Table
	uchar* p = lookUpTable.data;
	for( int i = 0; i < 256; ++i )
	{
		p[i] = (uchar)exp( i * K_EXP_OPERATOR );
	}

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
		CV_HAAR_FIND_BIGGEST_OBJECT
	);

	if( faces.size() > 0 )
	{
		#ifdef FACE_DETECT_DEBUG
		// Draw found face
		rectangle( imgProcessed,
			Point( faces[0].x, faces[0].y),
			Point( faces[0].x + faces[0].width, faces[0].y + faces[0].height),
			CV_RGB( 100, 255, 0));
		putTextWithShadow(
			imgProcessed,
			"Found face",
			Point( faces[0].x, faces[0].y )
		);
		#endif

		return true;
	}
	else
		return false;
};

void DetectEyes()
{
	// Start detecting only if face is found
	if( faces.size() )
	{
		Rect eyesROI	 = Rect( faces[0].x,							(int)(faces[0].y + 0.2*faces[0].height), 
								 faces[0].width,						(int)(0.4*faces[0].height) );

		Rect eyeLeftROI	 = Rect( (int)(faces[0].x + 0.1*faces[0].width),(int)(faces[0].y + 0.2*faces[0].height), 
								 (int)(0.4*faces[0].width),				(int)(0.4*faces[0].height) );

		Rect eyeRightROI = Rect( (int)(faces[0].x + 0.5*faces[0].width),(int)(faces[0].y + 0.2*faces[0].height), 
								 (int)(0.4*faces[0].width),				(int)(0.4*faces[0].height) );
		
		// Normalize histogram to improve all shit
		Mat imgEyes ( imgGray, eyesROI );
		equalizeHist( imgEyes, imgEyes );

		#ifdef EYES_DETECT_SINGLE_CASCADE		
		// Here both eyes are found at the same time by single pass
		Mat imgEyesROI (imgGray, eyesROI );
		cascadeEye.detectMultiScale(
			imgEyesROI,
			eyes,
			1.1,
			5,
			CV_HAAR_DO_CANNY_PRUNING );
		
		// Setup roi on image
		Mat imgProcessedROI (imgProcessed, eyesROI );
		
		// draw all found eyes
		for( int i = 0; i < (int)eyes.size(); ++i )
		{
			rectangle( imgProcessedROI,
				Point( eyes[i].x, eyes[i].y),
				Point( eyes[i].x + eyes[i].width, eyes[i].y + eyes[i].height),
				CV_RGB(0, 0, 0)
			);
		}
		imshow( "Foo", imgProcessedROI );
		#endif
		#ifdef EYES_DETECT_MULTI_CASCADE
		// TEMPORARY APPROACH: detecting eye pos in two passes		
		vector<Rect> eyesLeft,
					 eyesRight;
		Mat imgEyeLeft	( imgGray, eyeLeftROI ),
			imgEyeRight ( imgGray, eyeRightROI );
		
		cascadeEyeLeft.detectMultiScale( imgEyeLeft,	eyesLeft, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT );
		cascadeEyeRight.detectMultiScale( imgEyeRight, eyesRight, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT );
		
		Mat imgProcessedWithRightEye ( imgGray, eyeRightROI ),
			imgProcessedWithLeftEye	 ( imgGray, eyeLeftROI );
		
		for( int i = 0; i < (int)eyesRight.size(); ++i )
		{
			rectangle( imgProcessedWithRightEye,
				Point( eyesRight[i].x, eyesRight[i].y),
				Point( eyesRight[i].x + eyesRight[i].width, eyesRight[i].y + eyesRight[i].height),
				CV_RGB(0, 0, 0)
			);
		}
		for( int i = 0; i < (int)eyesLeft.size(); ++i )
		{
			rectangle( imgProcessedWithLeftEye,
				Point( eyesLeft[i].x, eyesLeft[i].y),
				Point( eyesLeft[i].x + eyesLeft[i].width, eyesLeft[i].y + eyesLeft[i].height),
				CV_RGB(0, 0, 0)
			);
		}

		imshow( "Left", imgProcessedWithLeftEye );
		imshow( "Right", imgProcessedWithRightEye );
		#endif
		
		Mat imgEyesRedChannel (rgb_planes[0], eyesROI );
		imshow( "Eyes Red Channel", imgEyesRedChannel );

		equalizeHist( imgEyesRedChannel, imgEyesRedChannel );
		bitwise_not( imgEyesRedChannel, imgEyesRedChannel );
		imshow( "Eyes Inverted Red Channel", imgEyesRedChannel );

		exponentialOperator( imgEyesRedChannel, imgEyesRedChannel );
		imshow( "Eyes Exponential Transform", imgEyesRedChannel );

		Mat imgEyeLeft,
			imgEyeRight;
		imgEyesRedChannel.copyTo( imgEyeLeft );
		imgEyesRedChannel.copyTo( imgEyeRight );
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
		// Adjust found mouth region
		foundMouthROI = Rect(
			(int) (mouthROI.x + mouths[0].x - 0.1*mouths[0].width), (int) (mouthROI.y + mouths[0].y - 0.1*mouths[0].height),
			(int) (1.2*mouths[0].width), mouths[0].height 
		);
		
		#ifdef _MOUTH_ROI_DEBUG
		// Setup ROI on output image so that object 
		// coordinates compliy with those on search image
		Mat imgProcessedROI( imgProcessed, mouthROI );

		// ..and draw it
		rectangle( imgProcessedROI, 
			Point( foundMouthROI.x, foundMouthROI.y ),
			Point( foundMouthROI.x + foundMouthROI.width, foundMouthROI.y + foundMouthROI.height ),
			CV_RGB(0,0,0) 
		);

		putTextWithShadow(
			imgProcessed,
			"Found mouth",
			Point( mouths[0].x, mouths[0].y )
		);
		#endif

		Mat imgMouthHue( hls_planes[HUE_PLANE], foundMouthROI );
		Mat imgMouthThresh ( imgMouthHue.size(), imgMouthHue.type() );
		Mat imgBlurredMouth;
		bilateralFilter( imgMouthHue, imgBlurredMouth, bilateralBlur, bilateralBlur*2, bilateralBlur/2 );
		imshow( wndNameBlur, imgBlurredMouth );
		mouthHueAvg = mean( imgMouthHue );

		mouthThreshold = (int)mouthHueAvg.val[0];
		threshold( imgBlurredMouth, imgMouthThresh, (double) mouthThreshold, 255, THRESH_BINARY_INV );
		imshow( wndNameMouth, imgMouthThresh );
	}
};

void ProcessAlgorithm()
{
	// Make a copy of source image
	imgSrc.copyTo( imgProcessed );

	// Convert image to grayscale and HLS colour space
	cvtColor( imgSrc, imgGray, CV_RGB2GRAY );
	cvtColor( imgSrc, imgHLS, CV_RGB2HLS_FULL );

	// Split multichannel images into separate planes
	split( imgSrc, rgb_planes );
	split( imgHLS, hls_planes );

	if( DetectFaces() )
	{
		DetectEyes();
		//DetectMouth();
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

