#include "Globals.h"
#include "FacialFeatures.h"

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
inline void putTextWithShadow(Mat& img, const char *str, Point org, CvScalar color = CV_RGB(0, 255, 100) );
inline string getCurentFileName( string filePath );
void displayStats();