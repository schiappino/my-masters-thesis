// Sample 3.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <time.h>

void detect_and_draw( IplImage* img );

const char*	window_name = "OpenCV";
//const char* cascade_name = "C:/Program Files/OpenCV/data/haarcascades/haarcascade_eye.xml";
const char* cascade_name = "C:/Program Files/OpenCV/data/haarcascades/haarcascade_frontalface_alt.xml";
//const char* cascade_name = "C:/Program Files/OpenCV/data/haarcascades/haarcascade_mcs_nose.xml";


static CvMemStorage* storage = NULL;
static CvHaarClassifierCascade* cascade = NULL;

int scale = 1;
// Create two points to represent the face locations
CvPoint pt1, pt2;

CvSeq* faces = NULL;

time_t start, end;
double fps;
int counter = 0,
	posRes = 0;
double sec;
char fpsCstring[3];



using namespace cv;

int _tmain(int argc, _TCHAR* argv[])
{
	int percent = 50;

	cvNamedWindow( window_name, CV_WINDOW_AUTOSIZE );

	CvCapture* capture = NULL;
	//capture = cvCreateCameraCapture(CV_CAP_ANY);
	capture = cvCreateFileCapture( "C:/Users/netholik/Videos/240/Video_00006.avi" );
    if(!capture)
    {
        std::cout << "Dupa" << std::endl;
        return -1;
    }	

	/*cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH,  240);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 180);*/


	IplImage* frame = NULL,
			*RawFrame = NULL,
			*resizedFrame = NULL,
			*grayFrame = NULL;

	// Allocate the memory storage
	storage = cvCreateMemStorage(0);
	cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );
	
	if( !cascade )
	{
		fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
		return -1;
	}
	


	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.0, 1.0, 0, 1, CV_AA);



	char *cvLibraries; 
	char *cvModules; 
	cvGetModuleInfo(0, (const char**) &cvLibraries, (const char**) &cvModules); 
	printf("OpenCV Libraries: %s\nOpenCV Modules: %s\n", cvLibraries, cvModules); 
 


	frame = cvQueryFrame( capture );
	if( !frame ) return -1;



	CvSize frameSize = cvSize(frame->width, frame->height);
	CvSize orgFrameSize = frameSize;

	frameSize.height *= percent / 100;
	frameSize.height *= percent / 100;
	grayFrame = cvCreateImage ( orgFrameSize, IPL_DEPTH_8U, 1);



	time(&start);
	while( true )
	{
		RawFrame = cvQueryFrame( capture );
		if( !RawFrame ) break;

		cvCvtColor( RawFrame, grayFrame, CV_BGR2GRAY);
		//cvResize(tmp, RawFrame);

		cvShowImage( window_name, grayFrame );
		detect_and_draw( grayFrame );
		

		// FPS counter
		time(&end);
		++counter;
		sec = difftime (end, start);     
		fps = counter / sec;
		//printf("FPS = %.1f\n", fps);
		_gcvt(fps, 3, fpsCstring);
		cvPutText(grayFrame, fpsCstring, cvPoint(10, 20), &font, cvScalar(255, 255, 255, 0));
		cvShowImage( window_name, grayFrame);


		char c = cvWaitKey(1);
		if( c == 27 ) break;
	}

	time(&end);
	std::cout	<< "Total effectiveness estemated " << ((double) posRes / (double) counter) * 100 << std::endl
				<< "Total exec time was " << end - start <<  std::endl;

	cvReleaseMemStorage(&storage);
	cvReleaseCapture( &capture );
	cvDestroyAllWindows();
}

// Function to detect and draw any faces that is present in an image
void detect_and_draw( IplImage* img )
{
	int i;
	
	// Clear the memory storage which was used before
	cvClearMemStorage( storage );

	// There can be more than one face in an image. So create a growable sequence of faces.
	// Detect the objects and store them in the sequence
	faces = cvHaarDetectObjects( img,
								cascade, 
								storage,
								1.7, 2, 
								CV_HAAR_FIND_BIGGEST_OBJECT,
								cvSize(40, 40) );

	// Loop the number of faces found.
	for( i = 0; i < (faces ? faces->total : 0); i++ )
	{
		// Create a new rectangle for drawing the face
		CvRect* r = (CvRect*)cvGetSeqElem( faces, i );

		// Find the dimensions of the face,and scale it if necessary
		pt1.x = r->x*scale;
		pt2.x = (r->x+r->width)*scale;
		pt1.y = r->y*scale;
		pt2.y = (r->y+r->height)*scale;

		printf("Frame %d\t\tFace coords: %d\t %d\t %d\t %d\n", counter, pt1.x, pt1.y, pt2.x, pt2.y);

		// Draw the rectangle in the input image
		cvRectangle( img, pt1, pt2, CV_RGB(255,255,0), 2, 8, 0 );

		// increamet found objects count
		++posRes;
	}

	faces = NULL;
}

