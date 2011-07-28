// HistogramOpenCV.cpp : Defines the entry point for the console application.
/* The histograms are drawn with 256 bins.  Each bin corresponds to an intensity level; 0 being completely dark and 255, completely light. This is done because the images being captured have a depth of 8 bits, or a range of 256 intensity levels. The number of pixels that fall into each category are represented by the height of the corresponding bar. No scale is given on the y-axis because the image is scaled such that the highest bar is always at the top of the screen. Since the histogram updates in real time, this y-axis scale would change too fast to see. Therefore the graph shows the relative number of pixels that fall into each category. Only three histograms are shown on the graph, one for the red, green and blue channel. The other colours: cyan, yellow, magenta and white represent overlap. It can be confusing to see all the histograms at the same time so the option to toggle each channel on or off is made available.*/

#include "stdafx.h"
#include <highgui.h>
#include <cv.h>

using namespace cv;

int main(int argc, char *argv[])
{
	// Variable to store the keyboard input
	char d;

	// Initialize flags to false. These flags are used for keyboard input (keys q, r, g and b)
	bool Q = false;
	bool R = false;
	bool G = false;
	bool B = false;

	CvCapture* capture = cvCaptureFromCAM(0);

	// Allocate memory for all images
	IplImage *src_img;
	IplImage *histogram;
	IplImage *disp_img;
	IplImage *gray_img;
	IplImage *red_img, *red_histogram;
	IplImage *green_img, *green_histogram;
	IplImage *blue_img, *blue_histogram;

	// Initialize historgrams
	int hist_size = 256;
	float range[] = {0,256};
	float *ranges[] = {range};

	CvHistogram* hist_red = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);
	CvHistogram* hist_blue = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);
	CvHistogram* hist_green = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);
	double max_value = 0;
	double max_value_red = 0;
	double max_value_green = 0;
	double max_value_blue = 0;
	double find_max = 0;

	// Create the windows
	// "mainWin"  shows the actual captured image
	cvNamedWindow("mainWin", CV_WINDOW_AUTOSIZE);
	cvMoveWindow("mainWin", 5, 5);
	// "histogramWin" shows the histogram
	cvNamedWindow("histogramWin", CV_WINDOW_AUTOSIZE);
	cvMoveWindow("histogramWin", 435, 5);

	// Print instructions for keyboard input
	printf("RGB Histogram\n\n");
	printf("To toggle red channel ON/OFF press:	'r'\n");
	printf("To toggle green channel ON/OFF press: 'g'\n");
	printf("To toggle blue channel ON/OFF press: 'b'\n");
	printf("To quit press: 'q'\n");

	// Do the following inside while loop forever
	while(1)
	{
		// Clear all max values to 0
		max_value = 0; max_value_red = 0; max_value_green = 0; max_value_blue = 0;

		histogram = cvLoadImage( "histogram_scale.png" );

		// Initialize three images that will show each histogram
		red_histogram = cvCreateImage( cvGetSize(histogram), IPL_DEPTH_8U, 3 );
		green_histogram = cvCreateImage( cvGetSize(histogram), IPL_DEPTH_8U, 3 );
		blue_histogram = cvCreateImage( cvGetSize(histogram), IPL_DEPTH_8U, 3 );

		// Get the source frame by querying the capture and resize it for display
		src_img=cvQueryFrame(capture);
		disp_img = cvCreateImage(cvSize((src_img->width)/1.6,(src_img->height)/1.6),IPL_DEPTH_8U,3);
		cvResize(src_img,disp_img,CV_INTER_LINEAR);

		// Create 3 single channel images to store each channels data and split the source image into the RGB channels.
		blue_img = cvCreateImage( cvGetSize(src_img), IPL_DEPTH_8U, 1 );
		green_img = cvCreateImage( cvGetSize(src_img), IPL_DEPTH_8U, 1 );
		red_img = cvCreateImage( cvGetSize(src_img), IPL_DEPTH_8U, 1 );
		cvCvtPixToPlane( src_img, blue_img, green_img, red_img, 0 );

		// Calculate a histogram for each channel.
		cvCalcHist( &red_img, hist_red, 0, NULL );
		cvCalcHist( &blue_img, hist_blue, 0, NULL );
		cvCalcHist( &green_img, hist_green, 0, NULL );

		// Search through the histograms for their maximum value and store it.
		for( int i = 0; i < hist_size; i++ )
		{
			find_max = cvQueryHistValue_1D(hist_red,i);
			if (find_max > max_value_red)
			{
				max_value_red = find_max;
			}
		}
		for( int i = 0; i < hist_size; i++ )
		{
			find_max = cvQueryHistValue_1D(hist_green,i);
			if (find_max > max_value_green)
			{
				max_value_green = find_max;
			}
		}
		for( int i = 0; i < hist_size; i++ )
		{
			find_max = cvQueryHistValue_1D(hist_blue,i);
			if (find_max > max_value_blue)
			{
				max_value_blue = find_max;
			}
		}
		// The largest value in all the histograms is found.
		max_value = max(max(max_value_red,max_value_green),max_value_blue);

		// Draw the histogram for each channel, if the flag for that channel is set
		if (R)
		{
			cvScale( hist_red->bins, hist_red->bins, 438/max_value);
			for( int i= 0; i < hist_size; i++ )
			{
				cvRectangle( red_histogram, cvPoint(i*3+ 15, 448),cvPoint(i*3+16, 448 - cvRound(cvQueryHistValue_1D(hist_red,i))),cvScalar(0x00,0x00,0xff,0), -1);
			}
			cvAdd(histogram,red_histogram,histogram,0);
		}
		if (G)
		{
			cvScale( hist_green->bins, hist_green->bins, 438/max_value);
			for( int i= 0; i < hist_size; i++ )
			{
				cvRectangle( green_histogram, cvPoint(i*3+ 15, 448),cvPoint(i*3+16, 448 - cvRound(cvQueryHistValue_1D(hist_green,i))),cvScalar(0x00,0xff,0x00,0), -1);
			}
			cvAdd(histogram,green_histogram,histogram,0);
		}
		if (B)
		{
			cvScale( hist_blue->bins, hist_blue->bins, 438/max_value);
			for( int i= 0; i < hist_size; i++ )
			{
				cvRectangle( blue_histogram, cvPoint(i*3+ 15, 448),cvPoint(i*3+16, 448 - cvRound(cvQueryHistValue_1D(hist_blue,i))),cvScalar(0xff,0x00,0x00,0), -1);
			}
			cvAdd(histogram,blue_histogram,histogram,0);
		}

		// Show the images in the windows
		cvShowImage("mainWin", disp_img);
		cvShowImage("histogramWin", histogram);

		// Set flags
		d=cvWaitKey(15);
		/* A simple case statement takes the input from the keyboard and sets the flags accordingly. The R, G and B flags are XOR’ed with 1 to change state each time r, g, or b is pressed. This makes r g and b into toggle switches.*/
		switch (d)
		{
			case 'r':	R = R^1;	break;
			case 'g':	G = G^1;	break;
			case 'b':	B = B^1;	break;
			case 'q':	Q = true;	break;
			default:	break;
		}
		if(Q)break;		//quit program

		// Release the images that we created
		cvReleaseImage(&disp_img );
		cvReleaseImage(&red_img );
		cvReleaseImage(&green_img );
		cvReleaseImage(&blue_img );
		cvReleaseImage(&red_histogram );
		cvReleaseImage(&green_histogram );
		cvReleaseImage(&blue_histogram );
		cvReleaseImage(&histogram );
	}
	return 0;
}