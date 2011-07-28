// OpenCV_Helloworld.cpp : Defines the entry point for the console application.
// Created for build/install tutorial, Microsoft Visual Studio and OpenCV 2.2.0

#include "stdafx.h"

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

using namespace cv;


// GLOBALS

int		hdims = 50;   
float	hranges_arr[] = {0,255};   
float*	hranges = hranges_arr;   
float	max_val = 0.f;   
int		i, bin_w;  

// END GLOBALS

int _tmain(int argc, _TCHAR* argv[])
{
	const char*	window_name = "OpenCV";
	const char* window_name_hist = "Histogram";

	cvNamedWindow( window_name, CV_WINDOW_AUTOSIZE );
	cvNamedWindow( window_name_hist, CV_WINDOW_AUTOSIZE );

	CvCapture* capture = NULL;
	capture = cvCreateCameraCapture(CV_CAP_ANY);
	

	//assert( capture != NULL );
    if(!capture)
    {
        std::cout << "Dupa" << std::endl;
        return -1;
    }	
	IplImage* frame = NULL,
			* histImage = NULL,
			* gray = NULL;
	CvHistogram *histogram = NULL;
	
	while( true )
	{
		frame = cvQueryFrame( capture );
		if( !frame ) break;

		cvShowImage( window_name, frame );

		gray = cvCreateImage(cvGetSize(frame), 8, 1);   
		cvCvtColor(frame, gray, CV_RGBA2GRAY); 
		histogram = cvCreateHist(1, &hdims, CV_HIST_ARRAY, &hranges, 1);
		histImage = cvCreateImage(cvSize(320,200),8,3);   
		cvZero(histImage);

		cvCalcHist(&gray,histogram,0,0);   
		cvGetMinMaxHistValue(histogram,0,&max_val,0,0);   
		cvConvertScale(histogram->bins,histogram->bins,max_val?255. /max_val:0.,0);   
		cvZero(histImage);   
		bin_w = histImage->width / hdims;   

		for(i=0; i < hdims; ++i)   
		{   
			int val = cvRound(cvGetReal1D(histogram->bins,i)*histImage->height/255);   
			CvScalar color = CV_RGB(255,255,255);   
			cvRectangle(histImage,
				cvPoint(i*bin_w,histImage->height),   
				cvPoint((i+1)*bin_w,histImage->height-val),
				color,-1,8,0);   
		}   

		cvShowImage(window_name_hist,histImage);


		char c = cvWaitKey(33);
		if( c == 27 ) break;
	}
	cvReleaseCapture( &capture );
	cvDestroyAllWindows();
}

