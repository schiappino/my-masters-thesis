#include "Globals.h"
#include "Eyes.h"
#include "Eyebrows.h"
#include "Mouth.h"
#include "FacialFeatures.h"

using namespace cv;
using namespace std;

// ********************************** CASCADE FILES ******************************************
const char* cascadeFNameEye                             = "../data/cascades/haarcascade_eye.xml";
const char* cascadeFNameEyeRightSplit   = "../data/cascades/haarcascade_righteye_2splits.xml";
const char* cascadeFNameEyeLeftSplit    = "../data/cascades/haarcascade_lefteye_2splits.xml";
const char* cascadeFNameFace                    = "../data/cascades/haarcascade_frontalface_alt.xml";
const char* cascadeFNameMouth                   = "../data/cascades/haarcascade_mcs_mouth.xml";

// ********************************** IMAGE FILES *******************************************
const char* IMMFaceDBFile                               = "../data/facedb/imm/filelist.txt";
const char* ColorFeretDBFile                    = "../data/facedb/color feret/filelist.txt";
const char* eyeTemplateFile                             = "../data/images/eye_template4.bmp";

// *********************************** VIDEO FILES ******************************************
const char* VideoSequences                              = "../data/video sequences/filelist.txt";
const char* VideoSequence1                              = "../data/video sequences/VIDEO0020.3gp";

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


Point point1, point2;
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
        object_size             = 0,
        face_size               = 0,
        scale           = 1,
        imIt                    = 0,    // image list literator
        z                               = 10,
        Hough_dp                = 2,
        HoughMinDist    = 50,
        TemplMatchMet   = 4,
        eyebrowMorph    = 1,


        mouthThreshold  = 0,
        bilatBlurVal    = 12,
        eyeThreshold    = 0,
        eyebrowThreshold = 0,

        TrackbarLoVal   = 2,
        TrackbarHiVal   = 100,
        TrackbarMaxVal  = 255;

bool finish = false,
         pause  = false;

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
        * wndNameEyesThresh      = "Eyes threshold",
        * wndNameTemplRes = "Template Match Res",
        * wndNameBlobs = "Blobs",

        * trckbarMouthThresh = "Mouth THR",
        * trckbarbilateralBlur = "Bilatera blur",
        * trckbarZ = "Z",
        * trckbarEyeThreshold = "Eyes THR";

vector<string>  imgFileList;
vector<Rect>    faces,
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
	sprintf_s( text, 255, "%dx%d", imgSrc.size().width, imgSrc.size().height );
	putTextWithShadow( imgProcessed, text, Point(5, 35) );

	// Show FPS
	sprintf_s( text, 255, "FPS %2.0f", 1000/exec_time);
	putTextWithShadow( imgProcessed, text, Point(5, 55) );

	// When working on files 
	if( PROGRAM_MODE == 1 ) 
	{
		// Show current file name 
		putTextWithShadow( imgProcessed, getCurentFileName( imgFileList.at(imIt) ).c_str(), Point(5, 75));

		sprintf_s( text, 255, "Current Image %d", imIt);
		putTextWithShadow( imgProcessed, text, Point(5, 115) );
	}
	else if( PROGRAM_MODE == 2 )
	{
		// Show current frame no.
		sprintf_s( text, 255, "Video pos %d%%", cvRound(videoCapture.get( CV_CAP_PROP_POS_AVI_RATIO)*100));
		putTextWithShadow( imgProcessed, text, Point(5, 75));
	}
}

void exponentialOperator( Mat src, Mat dst )
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
void onEyebrowMorph( int val, void* )
{
	eyebrowMorph = val;
};
void InitGUI()
{
	int flags = CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED;

	namedWindow( wndNameSrc, flags );
	namedWindow( wndNameFace, flags );
	//namedWindow( wndNameLeftEye, flags );
	//namedWindow( wndNameRightEye, flags );
	//namedWindow( wndNameEyesThresh, flags );
	//namedWindow( wndNameEyesExpTrans, flags );
	//namedWindow( wndNameBilateral, flags );
	//namedWindow( wndNameTemplRes, flags );
	namedWindow( wndNameBlobs, flags );

	createTrackbar( trckbarMouthThresh, wndNameMouth, &mouthThreshold, 255, onThresholdTrackbar );
	createTrackbar( trckbarbilateralBlur, "", &bilatBlurVal, 20, onBilateralBlur );
	createTrackbar( "Hough dp", "", &Hough_dp, 20, onHough_dp );
	createTrackbar( trckbarEyeThreshold, "", &eyeThreshold, 255, onEyeThresholdTrackbar );
	createTrackbar( trckbarZ, "", &z, 50, onZTrackbar );
	createTrackbar( "Templ Match Met", "", &TemplMatchMet, 5, onTemplateMatchingMet );
	createTrackbar( "Eyebrow THR", "", &eyebrowThreshold, 255, onEyebrowThresh );
	createTrackbar( "Eyebrow Morphology", "", &eyebrowMorph, 10, onEyebrowMorph );
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

void ProcessAlgorithm()
{
	double	eyes_exec_time,
			mouth_exec_time,
			eyebrows_exec_time,
			face_exec_time;

	// Make a copy of source image
	imgSrc.copyTo( imgProcessed );

	// Convert image to grayscale and HLS colour space
	cvtColor( imgSrc, imgGray, CV_RGB2GRAY );
	cvtColor( imgSrc, imgHLS, CV_RGB2HLS_FULL );

	// Split multichannel images into separate planes
	split( imgSrc, rgb_planes );
	split( imgSrc, hls_planes );

	face_exec_time = startTime();
	bool isFace = DetectFaces();
	calcExecTime( &face_exec_time );
	cout << "face detect\t\t" << (int)face_exec_time << " ms" << endl;

	if( isFace )
	{
		eyes_exec_time = startTime();
		DetectEyes();
		calcExecTime( &eyes_exec_time );
		cout << "eyes detect\t\t" << (int)eyes_exec_time << " ms" << endl;

		mouth_exec_time = startTime();
		DetectMouth();
		calcExecTime( &mouth_exec_time );
		cout << "mouth detect\t\t" << (int)mouth_exec_time << " ms" << endl;
		
		eyebrows_exec_time = startTime();
		DetectEyebrows();
		calcExecTime( &eyebrows_exec_time );
		cout << "eyesbrows detect\t" << (int)eyebrows_exec_time << " ms" << endl;
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
		cout << "main: exec time\t\t" << (int)exec_time << " ms\t" 
			 << setiosflags(ios::basefield) << setiosflags(ios::fixed) << setprecision(1)
			 << (1000/exec_time) << " FPS" << endl << endl;
		
		displayStats();
		imshow( wndNameFace, imgProcessed );

		handleKeyboard( waitKey(1) );
	}
	ExitNicely(0);
}