#include "GUI.h"

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
void onMaxCorners( int val, void* )
{
	maxCorners = val;
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
	//namedWindow( wndNameBlobs, flags );
	namedWindow( wndNameCorners, flags );

	createTrackbar( trckbarMouthThresh, wndNameMouth, &mouthThreshold, 255, onThresholdTrackbar );
	createTrackbar( trckbarbilateralBlur, "", &bilatBlurVal, 20, onBilateralBlur );
	createTrackbar( "Hough dp", "", &Hough_dp, 20, onHough_dp );
	createTrackbar( trckbarEyeThreshold, "", &eyeThreshold, 255, onEyeThresholdTrackbar );
	createTrackbar( trckbarZ, "", &z, 50, onZTrackbar );
	createTrackbar( "Templ Match Met", "", &TemplMatchMet, 5, onTemplateMatchingMet );
	createTrackbar( "Eyebrow THR", "", &eyebrowThreshold, 255, onEyebrowThresh );
	createTrackbar( "Eyebrow Morphology", "", &eyebrowMorph, 10, onEyebrowMorph );
	createTrackbar( "Mouth Corners", "", &maxCorners, 100, onMaxCorners);
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
void putTextWithShadow(Mat& img, const char *str, Point org )
{
	putText( img, str, Point(org.x -1, org.y-1), FONT_HERSHEY_PLAIN, 1, CV_RGB(50, 50, 50), 2 );
	putText( img, str, org, FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 255, 100) );
};
inline string getCurentFileName( string filePath )
{
	size_t found = filePath.find_last_of("/");
	return filePath.substr( found + 1 );
};
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