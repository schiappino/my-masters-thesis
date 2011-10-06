#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

#define HUE_PLANE 0
#define COLOR_FERET_DB_SIZE 4000
#define IMM_DB_SIZE	250

#define FACE_DETECT_DEBUG
//#define _MOUTH_ROI_DEBUG
#define EYES_DETECT_SINGLE_CASCADE
//#define EYES_DETECT_MULTI_CASCADE
//#define EYES_DETECT_HOUGH_TRANSFORM
//#define EYES_DETECT_CONNECTED_COMP
//#define EYES_TEMPLATE_MATCH_DEBUG


// ********************* FUNCTION DECLARATIONS ******************************

// **************************** HELPERS ************************************* 
inline void putTextWithShadow(Mat& img, const char *str, Point org, CvScalar color);
inline string getCurentFileName( string filePath );
double startTime(void);
double calcExecTime( double* time );
bool loadFileList( const char* fileName );
void displayStats();
// ****************************** CORE *************************************
void exponentialOperator( Mat src, Mat dst );
int Init(void);
int ExitNicely(int code);
bool DetectFaces(void);
void ColorSegment( vector<Mat> color_planes, Rect roi );
void DetectMouth(void);
void ProcessAlgorithm(void);
// ********************** GUI ***********************************************
void onThresholdTrackbar( int val, void* );
void onEyeThresholdTrackbar( int val, void* );
void onZTrackbar( int val, void* );
void onBilateralBlur( int val, void* );
void onHough_dp( int val, void* );
void onTemplateMatchingMet( int val, void* );
void onEyebrowThresh( int val, void* );
void onEyebrowMorph( int val, void* );
void InitGUI();
void handleKeyboard( char c );
