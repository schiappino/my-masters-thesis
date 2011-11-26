#include "Globals.h"
#include "Eyes.h"
#include "Eyebrows.h"
#include "Mouth.h"
#include "GUI.h"
#include "GroundTruths.h"
#include "FacialFeatures.h"

using namespace cv;
using namespace std;

// ********************************** CASCADE FILES ******************************************
const char* cascadeFNameEye             = "../data/cascades/haarcascade_eye.xml";
const char* cascadeFNameEyeRightSplit   = "../data/cascades/haarcascade_righteye_2splits.xml";
const char* cascadeFNameEyeLeftSplit    = "../data/cascades/haarcascade_lefteye_2splits.xml";
const char* cascadeFNameFace            = "../data/cascades/haarcascade_frontalface_alt.xml";
const char* cascadeFNameMouth           = "../data/cascades/haarcascade_mcs_mouth.xml";

// ********************************** IMAGE FILES *******************************************
const char* IMMFaceDBFile				= "../data/facedb/imm/im_list(1-2-5pose).txt";
//const char* IMMFaceDBFile				= "../data/facedb/imm/im_filelist(frontal).txt";
const char* ColorFeretDBFile			= "../data/facedb/color feret/filelist.txt";
const char* ColorFeretDBFile_fa			= "../data/facedb/color feret/faces list - fa pose.txt";
const char* eyeTemplateFile             = "../data/images/Eye mean template.png";
//const char* eyeTemplateFile             = "../data/images/eye_template4.bmp";
const char* AverageFaces				= "../data/facedb/average/list.txt";

// *********************************** VIDEO FILES ******************************************
const char* VideoSequences              = "../data/video sequences/filelist.txt";
const char* VideoSequence1              = "../data/video sequences/VIDEO0020.3gp";

// ******************************** GROUND TRUTH FILES **************************************
const string groundTruthsFeret			= "../data/facedb/ground truths/name_value/gt list - fa pose.txt";
const string groundTruthsBioID			= "";
const string groundTruthsIMM			= "../data/facedb/imm/gt_list(1-2-5pose).txt";
//const string groundTruthsIMM			= "../data/facedb/imm/gt_filelist(frontal).txt";

// ****************************** GLOBALS ***************************************************
const int PROGRAM_MODE = 1; 	// PROGRAM_MODE = 1 work on images
								// PROGRAM_MODE = 2 work on frames from avi capture
								// PROGRAM_MODE = 3 work on frames from webcam capture
								// PROGRAM_MODE = 4 validation mode

const double K_EXP_OPERATOR = 0.0217304452751310829264530948549876073716129212732431841605;

FacialFeaturesValidation 
	featuresBioID,
	featuresIMM,
	featuresFeret;

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
        object_size     = 0,
        face_size       = 0,
        scale           = 1,
        imIt			= 0,    // image list literator
        z				= 10,
        Hough_dp		= 2,
        HoughMinDist    = 50,
        TemplMatchMet   = 4,
        eyebrowMorph    = 1,
		maxCorners		= 20,


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
		* wndNameCorners = "Corners",

        * trckbarMouthThresh = "Mouth THR",
        * trckbarbilateralBlur = "Bilatera blur",
        * trckbarZ = "Z",
        * trckbarEyeThreshold = "Eyes THR";

vector<string>  imgFileList;
vector<Rect>    
	faces,
    eyes,
    mouths;





void exponentialOperator( Mat src, Mat dst )
{
	LUT( src, lookUpTable, dst );
};
double startTime(void)
{
	return (double) getTickCount();
}
double calcExecTime( double* time )
{
	*time = 1000 * ((double)getTickCount() - *time)/ getTickFrequency(); 
	return *time;
}
string convertInt(int number)
{
   stringstream ss;
   ss << number;
   return ss.str();
}
double square_distance( double a, double b )
{
	return sqrt( a*a + b*b );
}


bool loadFileList( const char* fileName, vector <string>& list )
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

int Init()
{
	// Initialize file list containers
	imgFileList.reserve( COLOR_FERET_DB_SIZE );

	// Load list of images to container
	loadFileList( IMMFaceDBFile, imgFileList );

	// Initialize file list iterator 
	imIt = 0;

	// Load cascades
	if( !cascadeFace.load( cascadeFNameFace) )				{ cerr << "--(!)Error loading" << endl; return -1; };
	if( !cascadeEye.load( cascadeFNameEye ) )				{ cerr << "--(!)Error loading" << endl; return -1; };
	if( !cascadeEyeRight.load( cascadeFNameEyeRightSplit ) ){ cerr << "--(!)Error loading" << endl; return -1; };
	if( !cascadeEyeLeft.load( cascadeFNameEyeLeftSplit ) )	{ cerr << "--(!)Error loading" << endl; return -1; };
	if( !cascadeMouth.load( cascadeFNameMouth ) )			{ cerr << "--(!)Error loading" << endl; return -1; };
	if( !cascadeMouth.load( cascadeFNameMouth ) )			{ cerr << "--(!)Error loading" << endl; return -1; };

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
	else if( PROGRAM_MODE == 4 )
	{
		cout << "Validation mode turned on" << endl;
	}
	// Load eye template
	imgTempl = imread( eyeTemplateFile, CV_LOAD_IMAGE_GRAYSCALE );
	if( !imgTempl.data )
	{
		cout << "Template file not loaded" << endl;
		return -1;
	}

	// Load ground truth data
	#ifdef VALIDATION
	// Color FERET database
	//getGroundTruthsData( featuresFeret, groundTruthsFeret, FaceDbFlags::COLOR_FERET );
	//
	//if( !(featuresFeret.eyes.left.size() == featuresFeret.eyes.right.size() 
	//	&& featuresFeret.eyes.left.size() == imgFileList.size()) ) 
	//{ cerr << "--(!) Number of ground truth data in not equal" << endl; }

	// IMM database
	getGroundTruthsData( featuresIMM, groundTruthsIMM, FaceDbFlags::IMM );
	#endif

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
	cvtColor( imgSrc, imgGray, CV_BGR2GRAY );
	cvtColor( imgSrc, imgHLS, CV_BGR2HLS_FULL );

	// Split multichannel images into separate planes
	split( imgSrc, rgb_planes );
	split( imgHLS, hls_planes );
	//imshow( "hue channel", hls_planes[0] );

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
		//DetectEyebrows();
		calcExecTime( &eyebrows_exec_time );
		cout << "eyesbrows detect\t" << (int)eyebrows_exec_time << " ms" << endl;
	}
	return;
};

int main(int argc, char** argv )
{
	Init();
	#ifdef GUI
	InitGUI();
	#endif

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
		else if( PROGRAM_MODE == 4 )
		{
			imgSrc = imread( imgFileList.at( imIt ));
			++imIt;
			if( imIt >= imgFileList.size()) { finish = true; break; }
		}
		// Show current image or frame
		#ifdef GUI
		imshow( wndNameSrc, imgSrc );
		#endif

		// Start time
		exec_time = startTime();

		ProcessAlgorithm();

		// End time
		calcExecTime( &exec_time );
		cout << "main: exec time\t\t" << (int)exec_time << " ms\t" 
			 << setiosflags(ios::basefield) << setiosflags(ios::fixed) << setprecision(1)
			 << (1000/exec_time) << " FPS" << endl << endl;
		
		#ifdef GUI
		displayStats();
		imshow( wndNameFace, imgProcessed );
		handleKeyboard( waitKey(0) );
		#endif
	}
	// Save validation data to the CSV file
	#ifdef VALIDATION
	saveEyePosValidationData( featuresIMM );
	saveMouthCornPosValidationData( featuresIMM );
	#endif

	ExitNicely(0);
}