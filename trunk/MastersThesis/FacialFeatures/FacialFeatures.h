#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
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
//#define EYE_DETECT_ROI_DEBUG
//#define MOUTH_ROI_DEBUG
//#define EYEBROWS_ROI_DEBUG
//#define EYES_DETECT_DEBUG
//#define EYES_DETECT_SINGLE_CASCADE
#define EYES_DETECT_MULTI_CASCADE
//#define EYES_TEMPLATE_MATCH_DEBUG
//#define EYES_VALIDATION
#define VALIDATION
#define GUI


// ********************* FUNCTION DECLARATIONS ******************************

// **************************** HELPERS ************************************* 
double startTime(void);
double calcExecTime( double* time );
bool loadFileList( const char* fileName, vector <string>& list );
string convertInt(int number);
double square_distance( double a, double b );
// ****************************** CORE *************************************
void exponentialOperator( Mat src, Mat dst );
int Init(void);
int ExitNicely(int code);
bool DetectFaces(void);
void ColorSegment( vector<Mat> color_planes, Rect roi );
void ProcessAlgorithm(void);