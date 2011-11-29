#ifndef GLOBAL_H
#define GLOBAL_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

// ********************************** CASCADE FILES ******************************************
extern const char* cascadeFNameEye;
extern const char* cascadeFNameEyeRightSplit;
extern const char* cascadeFNameEyeLeftSplit;
extern const char* cascadeFNameFace;
extern const char* cascadeFNameMouth;

// ********************************** IMAGE FILES *******************************************
extern const char* IMMFaceDBFile;
extern const char* ColorFeretDBFile;
extern const char* ColorFeretDBFile_fa;
extern const char* eyeTemplateFile;

// *********************************** VIDEO FILES ******************************************
extern const char* VideoSequences;
extern const char* VideoSequence1;

// ******************************** GROUND TRUTH FILES **************************************
extern const string groundTruthsFeret;
extern const string groundTruthsBioID;
extern const string groundTruthsIMM;

// ****************************** GLOBALS ***************************************************
extern const int PROGRAM_MODE;

extern const double K_EXP_OPERATOR;

extern VideoCapture videoCapture;

extern CascadeClassifier
	cascadeFace,
	cascadeMouth,
	cascadeEye,
	cascadeEye2,
	cascadeEyeRight,
	cascadeEyeLeft;


extern cv::Point	point1, point2;
extern CvFont font;

extern CvRect
	foundFaceROI,
	foundMouthROI;

extern Scalar mouthHueAvg;

extern Mat        
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

extern Mat lookUpTable;

extern vector<Mat> rgb_planes,
			hls_planes,
			hsv_planes;


extern int        
	selectedFaceDb,
	counter,
	posRes,
	object_size,
	face_size,
	scale,
	imIt,				// image list literator
	z,
	Hough_dp,
	HoughMinDist,
	TemplMatchMet,
	eyebrowMorph,
	maxCorners,


	mouthThreshold,
	bilatBlurVal,
	eyeThreshold,
	eyebrowThreshold,

	TrackbarLoVal,
	TrackbarHiVal,
	TrackbarMaxVal;

extern bool finish,
	 pause;

extern double          
	fps, 
	sec,
	exec_time;


extern char text[255];

extern const char      
	* wndNameSrc,
	* wndNameFace,
	* wndNameMouth,
	* wndNameBlur,
	* wndNameLeftEye,
	* wndNameRightEye,
	* wndNameBilateral,
	* wndNameEyesExpTrans,
	* wndNameEyesThresh,
	* wndNameTemplRes,
	* wndNameBlobs,
	* wndNameCorners,

	* trckbarMouthThresh,
	* trckbarbilateralBlur,
	* trckbarZ,
	* trckbarEyeThreshold;

extern vector <string> 	imgFileList;
extern vector <Rect>	faces,
						eyes,
						mouths;
// *********************************** ENUMS ************************************
struct EyebrowCandidateFlags
{
	enum
	{
		LEFT = 1,
		RIGHT = 2
	};
};
struct FaceDbFlags
{
	enum
	{
		IMM = 1,
		BioID = 2,
		COLOR_FERET = 3
	};
};
struct IMMDbAnnotationPoints
{
	enum // IMM ground truth annotation points
	{
		LEFT_EYE_LC			= 21,		// left eye - left corner
		LEFT_EYE_RC			= 25,		// left eye - right corner
		LEFT_EYE_UMID		= 23,		// left eye - upper mid point
		LEFT_EYE_LMID		= 27,		// left eye - lower mid point

		RIGHT_EYE_LC		= 17,		// right eye - right corner
		RIGHT_EYE_RC		= 13,		// right eye - right corner
		RIGHT_EYE_UMID		= 15,		// right eye - upper mid point
		RIGHT_EYE_LMID		= 19,		// right eye - lower mid point

		MOUTH_LEFT_COR		= 39,		// mouth - left corner
		MOUTH_RIGHT_COR		= 43,		// mouth - right corner
		MOUTH_UP_MID		= 41,		// mouth - upper mid point
		MOUTH_LO_MID		= 45,		// mouth - lower mid point

		LEFT_EYEBROW		= 36,		// left eyebrow - centre point
		RIGHT_EYEBROW		= 31		// right eyebrow - centre point
	};
};
struct BioIDAnnotationPoints
{
	enum	// BioID ground truth annotation points
	{
		LEFT_EYE_CENTR		= 1,		// left eye centre point
		RIGHT_EYE_CENTR		= 2,		// right eye centre point

		LEFT_MOUTH_COR		= 3,		// left mouth corner
		RIGHT_MOUTH_COR		= 4,		// right mouth corner

		LEFT_EYEBROW_LEFT	= 5,		// left eyebrow - left point
		LEFT_EYEBROW_RIGHT	= 6,		// left eyebrow - right point
		RIGHT_EYEBROW_LEFT	= 7,		// right eyebrow - left point
		RIGHT_EYEBROW_RIGHT	= 8			// right eyebrow - right point
	};
};
struct FacialFeaturesValidation
{
	struct Eye
	{
		vector <Point> left,			// Left eye coordinates from GT file
					   right,			// Right eye coordinates from GT file
					   left_det,		// Detected left eye coordinates from GT file
					   right_det;		// Detected right eye coordinates from GT file

		vector <double>left_err,		// Detected left eye error
					   right_err;		// Detected right eye error
		
		vector <double>	IOD;			// Inter ocular distance from GT file
		
		int size;						// number of items
	} eyes;

	struct Mouth
	{
		vector <Point> leftCorner,		// Left mouth corner coordinates from GT file
					   rightCorner,		// Right mouth corner coordinates from GT file
					   leftCorner_det,	// Detected left mouth corner coordinates from GT file
					   rightCorner_det;	// Detected right mouth corner coordinates from GT file

		vector <double>leftCorner_err,	// Detected left mouth corner error
					   rightCorner_err;	// Detected right mouth corner error

		vector <double> MCD;			// Mouth corners distance from GT file
		
		int size;						// number of items
	} mouth;

	struct Eyebrows
	{
		vector <Point> left,
					   right,
					   left_err,
					   right_err;
		int size;
	} eyebrow;
};

extern FacialFeaturesValidation 
	featuresBioID,
	featuresIMM,
	featuresFeret;
#endif