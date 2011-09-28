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
//#define EYES_DETECT_HOUGH_TRANSFORM
//#define EYES_DETECT_CONNECTED_COMP
//#define EYES_TEMPLATE_MATCH_DEBUG

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
const char* eyeTemplateFile				= "../data/images/eye_template3.bmp";

// *********************************** VIDEO FILES ******************************************
const char* VideoSequences				= "../data/video sequences/filelist.txt";
const char* VideoSequence1				= "../data/video sequences/VIDEO0020.3gp";

// ****************************** GLOBALS ***************************************************
const int PROGRAM_MODE = 2;

const double K_EXP_OPERATOR = 0.0217304452751310829264530948549876073716129212732431841605;

VideoCapture videoCapture;

CascadeClassifier
	cascadeFace,
	cascadeMouth,
	cascadeEye,
	cascadeEye2,
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
	imgTempl,
	imgSrc,
	imgGray,
	imgThresh,
	imgEdge,
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
			hls_planes,
			hsv_planes;


int                     
	counter         = 0,
	posRes          = 0,
	object_size		= 0,
	face_size		= 0,
	scale           = 1,
	imIt			= 0,	// image list literator
	z				= 10,
	Hough_dp		= 2,
	HoughMinDist	= 50,
	TemplMatchMet	= 4,


	mouthThreshold	= 0,
	bilatBlurVal	= 10,
	eyeThreshold	= 0,
	eyebrowThreshold = 0,

	TrackbarLoVal   = 2,
	TrackbarHiVal	= 100,
	TrackbarMaxVal	= 255;

bool finish = false,
	 pause	= false;

double          
	fps = 0, 
	sec = 0,
	exec_time;


char text[255];

const char      
	* wndNameSrc = "Source",
	* wndNameFace = "Face",
	* wndNameMouth = "Mouth",
	* wndNameBlur = "Blur",
	* wndNameLeftEye = "Left eye",
	* wndNameRightEye = "Right eye",
	* wndNameBilateral = "Bilateral Blur",
	* wndNameEyesExpTrans = "Eyes Exponential Transform",
	* wndNameEyesThresh	 = "Eyes threshold",
	* wndNameTemplRes = "Template Match Res",

	* trckbarMouthThresh = "Mouth THR",
	* trckbarbilateralBlur = "Bilatera blur",
	* trckbarZ = "Z",
	* trckbarEyeThreshold = "Eyes THR";

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
	sprintf_s( text, 256, "%dx%d", imgSrc.size().width, imgSrc.size().height );
	putTextWithShadow( imgProcessed, text, Point(5, 35) );

	// Show FPS
	sprintf_s( text, 256, "FPS %2.0f", 1000/exec_time);
	putTextWithShadow( imgProcessed, text, Point(5, 55) );

	// When working on files 
	if( PROGRAM_MODE == 1 ) 
	{
		// Show current file name 
		putTextWithShadow( imgProcessed, getCurentFileName( imgFileList.at(imIt) ).c_str(), Point(5, 75));

		sprintf_s( text, 256, "Current Image %d", imIt);
		putTextWithShadow( imgProcessed, text, Point(5, 115) );
	}
	else if( PROGRAM_MODE == 2 )
	{
		// Show current frame no.
		sprintf_s( text, 256, "Video pos %d%%", cvRound(videoCapture.get( CV_CAP_PROP_POS_AVI_RATIO)*100));
		putTextWithShadow( imgProcessed, text, Point(5, 75));
	}
}

inline void exponentialOperator( Mat src, Mat dst )
{
	LUT( src, lookUpTable, dst );
};

void onThresholdTrackbar( int val, void* )
{
	mouthThreshold = val;
};
void onEyeThresholdTrackbar( int val, void* )
{
	eyeThreshold = val;
};
void onZTrackbar( int val, void* )
{
	z = val;
};
void onBilateralBlur( int val, void* )
{
	bilatBlurVal = val;
};
void onHough_dp( int val, void* )
{
	Hough_dp = val;
};
void onTemplateMatchingMet( int val, void* )
{
	TemplMatchMet = val;
};
void onEyebrowThresh( int val, void* )
{
	eyebrowThreshold = val;
};

void InitGUI()
{
	int flags = CV_WINDOW_KEEPRATIO | CV_GUI_NORMAL;

	namedWindow( wndNameSrc, flags );
	namedWindow( wndNameFace, flags );
	//namedWindow( wndNameLeftEye, flags );
	//namedWindow( wndNameRightEye, flags );
	//namedWindow( wndNameEyesThresh, flags );
	//namedWindow( wndNameEyesExpTrans, flags );
	//namedWindow( wndNameBilateral, flags );
	//namedWindow( wndNameTemplRes, flags );

	createTrackbar( trckbarMouthThresh, wndNameMouth, &mouthThreshold, 255, onThresholdTrackbar );
	createTrackbar( trckbarbilateralBlur, "", &bilatBlurVal, 20, onBilateralBlur );
	createTrackbar( "Hough dp", "", &Hough_dp, 20, onHough_dp );
	createTrackbar( trckbarEyeThreshold, "", &eyeThreshold, 255, onEyeThresholdTrackbar );
	createTrackbar( trckbarZ, "", &z, 50, onZTrackbar );
	createTrackbar( "Templ Match Met", "", &TemplMatchMet, 5, onTemplateMatchingMet );
	createTrackbar( "Eyebrow THR", "", &eyebrowThreshold, 255, onEyebrowThresh );
};

void handleKeyboard( char c )
{
	if( c == 27 ) finish = true;		// esc key pressed
	else if( c == 112 ) pause^= true;	// p key pressed
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
	if( !cascadeEye.load( cascadeFNameEye ) )				{ printf("--(!)Error loading\n"); return -1; };
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
		videoCapture.open( VideoSequence1 );
		if( !videoCapture.isOpened() )
		{
			cout << "Could not load video file" << endl;
			return -1;
		}
		videoCapture >> imgSrc;
		if( imgSrc.empty() )
		{
			cout << "Could not get first frame from capture" << endl;
			return -1;
		}
	}
	else if( PROGRAM_MODE == 3 )
	{
		cout << "Not implemented" << endl;
		return -1;	
	}
	
	// Load eye template
	imgTempl = imread( eyeTemplateFile, CV_LOAD_IMAGE_GRAYSCALE );
	if( !imgTempl.data )
	{
		cout << "Template file not loaded" << endl;
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

		face_size = faces[0].width;
		return true;
	}
	else
		return false;
};

void EyeTemplateMatching( Mat src, Mat disp, Mat templ, int irisRadius)
{
	Mat result;
	/// Create the result matrix
	int result_cols =  src.cols - templ.cols + 1;
	int result_rows = src.rows - templ.rows + 1;   

	result.create( result_cols, result_rows, CV_32FC1 );

	/// Do the Matching and Normalize
	matchTemplate( src, templ, result, TemplMatchMet );
	normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

	/// Localizing the best match with minMaxLoc
	double minVal, maxVal; 
	Point minLoc, maxLoc, matchLoc;

	minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. 
	/// For all the other methods, the higher the better
	if( TemplMatchMet  == CV_TM_SQDIFF || TemplMatchMet == CV_TM_SQDIFF_NORMED )
	{ matchLoc = minLoc; }
	else  
	{ matchLoc = maxLoc; }
	
	Point center = Point( matchLoc.x + cvRound(templ.cols/2.0), matchLoc.y + cvRound(templ.rows/2.0));

	/// Show me what you got
	circle( disp, center, irisRadius, CV_RGB(0,100,255), 2 );
	circle( result, center, irisRadius, CV_RGB(0,100,255), 2 );

	#ifdef EYES_TEMPLATE_MATCH_DEBUG
	imshow( wndNameTemplRes, result );
	#endif
};

void ColorSegment( vector<Mat> color_planes, Rect roi )
{
	Mat imgHue ( color_planes[0], roi ),
		imgSat ( color_planes[2], roi ),
		imgHueRes, imgSatRes, ImgFinalRes;

	inRange( imgHue, 0, 50, imgHueRes );
	imshow( "Skin Hue" , imgHueRes );
	inRange( imgSat, 0, 30, imgSatRes );
	imshow( "Skin Sat" , imgSatRes );
	bitwise_or( imgSatRes, imgHueRes, ImgFinalRes );

		
	Mat kernel = getStructuringElement( MORPH_ELLIPSE, Size(3, 3) );
	morphologyEx( ImgFinalRes, ImgFinalRes, MORPH_CLOSE, kernel, Point(1,1), 1 );
	morphologyEx( ImgFinalRes, ImgFinalRes, MORPH_OPEN, kernel, Point(1,1), 2 );

	imshow( "Hue and Sat overlapped", ImgFinalRes );
};

void DetectEyes()
{
	// Start detecting only if face is found
	if( faces.size() )
	{
		// Iris is typically 7% of face size
		int irisRadiusMax = cvRound(face_size*0.03);

		Rect eyesROI	 = Rect( faces[0].x,							(int)(faces[0].y + 0.2*faces[0].height), 
								 faces[0].width,						(int)(0.4*faces[0].height) );

		Rect eyeLeftROI	 = Rect( (int)(faces[0].x + 0.1*faces[0].width),(int)(faces[0].y + 0.2*faces[0].height), 
								 (int)(0.4*faces[0].width),				(int)(0.4*faces[0].height) );

		Rect eyeRightROI = Rect( (int)(faces[0].x + 0.5*faces[0].width),(int)(faces[0].y + 0.2*faces[0].height), 
								 (int)(0.4*faces[0].width),				(int)(0.4*faces[0].height) );
		
		// Normalize histogram to improve all shit
		Mat imgGrayEyes ( imgGray, eyesROI );
		equalizeHist( imgGrayEyes, imgGrayEyes );

		#ifdef EYES_DETECT_SINGLE_CASCADE		
		// Here both eyes are found at the same time by single pass
		cascadeEye.detectMultiScale(
			imgGrayEyes,
			eyes,
			1.3,
			3,
			CV_HAAR_DO_CANNY_PRUNING
		);
		
		// Setup roi on image
		Mat imgProcessedROI (imgProcessed, eyesROI );
		
		// draw all found eyes
		for( int i = 0; i < (int)eyes.size(); ++i )
		{
			rectangle( imgProcessedROI,
				Point( eyes[i].x, eyes[i].y),
				Point( eyes[i].x + eyes[i].width, eyes[i].y + eyes[i].height),
				CV_RGB(100, 100, 255)
			);
		}
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
		
		Mat imgEyes		( rgb_planes[0], eyesROI );
		Mat imgEyeLeft	( rgb_planes[0], eyeLeftROI ),
			imgEyeRight ( rgb_planes[0], eyeRightROI );

		equalizeHist( imgEyeRight, imgEyeRight );
		equalizeHist( imgEyeLeft, imgEyeLeft );
		bitwise_not( imgEyes, imgEyes );
		exponentialOperator( imgEyes, imgEyes );
		imshow( wndNameEyesExpTrans, imgEyes );
		
		//Mat kernel = getStructuringElement( MORPH_ELLIPSE, Size(3, 3) );
		//morphologyEx( imgEyes, imgEyes, MORPH_CLOSE, kernel, Point(1,1), 1 );
		//morphologyEx( imgEyes, imgEyes, MORPH_OPEN, kernel, Point(1,1), 1 );
		//threshold( imgEyes, imgEyes, eyeThreshold, 255, THRESH_BINARY );
		//imshow( wndNameEyesThresh, imgEyes );

		Mat imgProcessedLeftEye ( imgProcessed, eyeLeftROI ),
			imgProcessedRightEye ( imgProcessed, eyeRightROI );
		EyeTemplateMatching( imgEyeLeft, imgProcessedLeftEye, imgTempl, irisRadiusMax );
		EyeTemplateMatching( imgEyeRight, imgProcessedRightEye, imgTempl, irisRadiusMax );

		//Mat imgEyesHue ( hls_planes[0], eyesROI );
		//imshow( "Hue: eyes", imgEyesHue );

		//Mat imgEyesSat ( hls_planes[2], eyesROI );
		//imshow( "Sat: eyes", imgEyesSat );

		//Mat imgEyesSat2 ( hsv_planes[1], eyesROI );
		//imshow ( "Sat2: eyes", imgEyesSat2 );

		//ColorSegment( hls_planes, eyesROI );

		#ifdef EYES_DETECT_HOUGH_TRANSFORM
		// --> Hough Circle transform for iris detection
		HoughMinDist = cvRound(face_size/3.0);
		vector<Vec3f> iris;
		Mat imgEyesFiltered;
		bilateralFilter( imgEyes, imgEyesFiltered, bilatBlurVal, bilatBlurVal*2, bilatBlurVal/2 );
		imshow( wndNameBilateral, imgEyesFiltered );

		HoughCircles( imgEyesFiltered, iris, CV_HOUGH_GRADIENT,
			Hough_dp, HoughMinDist, 100, 200, 3, irisRadiusMax );
		for( int i = 0; i < iris.size(); ++i )
		{
			Point center( cvRound(iris[i][0]), cvRound(iris[i][1]) );
			int radius = cvRound(iris[i][2]);

			Mat imgEyesIris ( imgProcessed, eyesROI );
			circle( imgEyesIris, center, radius, CV_RGB(250,0,0) );
		}
		// <-- Hough Circle transform for iris detection
		#endif

		#ifdef EYES_DETECT_CONNECTED_COMP
		Scalar avgIntensityLeftEye = mean( imgEyeLeft );
		Scalar avgIntensityRightEye = mean( imgEyeRight );
		Scalar stdDev, avgIntensity;
		meanStdDev( imgEyeLeft, avgIntensity, stdDev );
		eyeThreshold = (int)(avgIntensity.val[0] + stdDev.val[0]*z/10);

		threshold( imgEyes, imgEyes, eyeThreshold, 255, THRESH_BINARY );
		Mat kernel = getStructuringElement( MORPH_ELLIPSE, Size(3, 3) );
		morphologyEx( imgEyes, imgEyes, MORPH_CLOSE, kernel, Point(1,1), 1 );
		morphologyEx( imgEyes, imgEyes, MORPH_OPEN, kernel, Point(1,1), 1 );

		Mat imgEyeBinaryCopy;
		imgEyeLeft.copyTo( imgEyeBinaryCopy );
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours( imgEyeBinaryCopy, contours, hierarchy,
					  CV_RETR_LIST, CV_CHAIN_APPROX_TC89_KCOS );
		Mat imgProcessedLeftEye ( imgProcessed, eyeLeftROI );
		for( int i = 0; i < contours.size(); ++i )
			drawContours( imgProcessedLeftEye, contours, i, CV_RGB(0,100,255) );
		
		imshow( wndNameLeftEye, imgEyeLeft );
		imshow( wndNameRightEye, imgEyeRight );
		#endif
	}
};

void DetectEyebrows()
{
	if( faces.size() )
	{
		Rect eyesbrowsROI	 = Rect( faces[0].x,							(int)(faces[0].y + 0.2*faces[0].height), 
									 faces[0].width,						(int)(0.2*faces[0].height) );

		Rect eyebrowLeftROI	 = Rect( (int)(faces[0].x + 0.1*faces[0].width),(int)(faces[0].y + 0.2*faces[0].height), 
									 (int)(0.4*faces[0].width),				(int)(0.2*faces[0].height) );

		Rect eyebrowRightROI = Rect( (int)(faces[0].x + 0.5*faces[0].width),(int)(faces[0].y + 0.2*faces[0].height), 
									 (int)(0.4*faces[0].width),				(int)(0.2*faces[0].height) );

		Mat imgEybrows		( rgb_planes[0], eyesbrowsROI ),
			imgEyebrowLeft	( rgb_planes[0], eyebrowLeftROI ),
			imgEyebrowRight	( rgb_planes[0], eyebrowRightROI ),
			imgGrayEyebrows ( imgGray,		 eyesbrowsROI );

		equalizeHist( imgEyebrowLeft, imgEyebrowLeft );
		equalizeHist( imgEyebrowRight, imgEyebrowRight );
		equalizeHist( imgGrayEyebrows, imgGrayEyebrows );

		bitwise_not( imgEyebrowLeft, imgEyebrowLeft );
		bitwise_not( imgEyebrowRight, imgEyebrowRight );

		exponentialOperator( imgEyebrowLeft, imgEyebrowLeft );
		exponentialOperator( imgEyebrowRight, imgEyebrowRight );

		Mat leftSmoothed, rightSmoothed, leftThresh, rightThresh, grayThresh;
		int b = bilatBlurVal;
		bilateralFilter( imgEyebrowLeft, leftSmoothed, b, 2*b, b/2. );
		bilateralFilter( imgEyebrowRight, rightSmoothed, b, 2*b, b/2. );

		Mat kernel = getStructuringElement( CV_SHAPE_ELLIPSE, Size(3,3) );
		morphologyEx( leftSmoothed, leftSmoothed,	CV_MOP_CLOSE,	kernel, Point(-1,-1), 1 );
		morphologyEx( leftSmoothed, leftSmoothed,	CV_MOP_OPEN,	kernel, Point(-1,-1), 1 );
		morphologyEx( rightSmoothed, rightSmoothed, CV_MOP_CLOSE,	kernel, Point(-1,-1), 1 );
		morphologyEx( rightSmoothed, rightSmoothed, CV_MOP_OPEN,	kernel, Point(-1,-1), 1 );

		threshold( leftSmoothed, leftThresh, eyebrowThreshold, 255, CV_THRESH_BINARY );
		threshold( rightSmoothed, rightThresh, eyebrowThreshold, 255, CV_THRESH_BINARY );
		threshold( imgGrayEyebrows, grayThresh, eyebrowThreshold, 255, CV_THRESH_BINARY );

		imshow( "Eyebrow left", imgEyebrowLeft );
		imshow( "Eyebrow right", imgEyebrowRight );

		imshow( "Left smoothed", leftSmoothed );
		imshow( "Right smoothed", rightSmoothed );

		imshow( "Left thr", leftThresh );
		imshow( "Right thr", rightThresh );

		imshow( "Eyebrows gray", imgGrayEyebrows );
		imshow( "Eyebrows gray thr", grayThresh );
	}
};

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
		bilateralFilter( imgMouthHue, imgBlurredMouth, bilatBlurVal, bilatBlurVal*2, bilatBlurVal/2 );
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
	split( imgSrc, hls_planes );

	if( DetectFaces() )
	{
		//DetectEyes();
		//DetectMouth();
		DetectEyebrows();
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
		if( PROGRAM_MODE == 2 && !pause )
		{
			videoCapture >> imgSrc;
			if( imgSrc.empty() )
			{
				videoCapture.set( CV_CAP_PROP_POS_AVI_RATIO, 0 );
				videoCapture >> imgSrc;
			}
		}
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