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
extern const char* eyeTemplateFile;

// *********************************** VIDEO FILES ******************************************
extern const char* VideoSequences;
extern const char* VideoSequence1;

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

#endif