#include "stdafx.h"
#include <iostream>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <time.h>

#define PROGRAM_MODE 10

void detect_and_draw( IplImage* img );


// ********************************** CASCADE FILES ******************************************
const char* cascade_fname_eye	= "../data/haarcascade_eye_alt.xml";
const char* cascade_fname_face	= "../data/haarcascade_frontalface_alt.xml";
const char* cascade_fname_mouth = "../data/haarcascade_mcs_mouth.xml";

// *********************************** VIDEO FILES ******************************************
const char* video_fname = "C:/Users/netholik/Videos/240/Video_00006.avi";


// ****************************** GLOBALS ***************************************************
static CvCapture				*capture		= NULL;
static CvMemStorage				*storage		= NULL,
								*storage2		= NULL,
								*storage3		= NULL;
static CvHaarClassifierCascade	*cascade_face	= NULL,
								*cascade_eye	= NULL,
								*cascade_mouth	= NULL;


CvPoint		point1, 
			point2;
CvFont		font;
static CvRect		*face_rect = NULL;

CvSeq		*faces			= NULL,
			*eyes			= NULL,
			*mouths			= NULL;

IplImage	*frame			= NULL,
			*RawFrame		= NULL,
			*ROIFrame		= NULL,
			*grayFrame		= NULL,
			*normFrame		= NULL,
			*tmpFrame		= NULL;

time_t		start;
time_t		end;

int			counter		= 0,
			posRes		= 0,
			object_size = 0,
			scale		= 1;

double		fps = 0, 
			sec = 0;

char		text[3];
const char	*window_name		= "Final frame",
			*window_name_norma	= "Normalized frame",
			*window_name_roi	= "ROI frame";



using namespace cv;

//****************** MAIN ENTRY HERE ***************************
int _tmain(int argc, _TCHAR* argv[])
{
	cvNamedWindow( window_name, CV_WINDOW_AUTOSIZE );
	cvNamedWindow( window_name_norma, CV_WINDOW_AUTOSIZE );

	if( PROGRAM_MODE == 1 )
	{
		capture = cvCreateCameraCapture(CV_CAP_ANY);
		
		cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH,  240);
		cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 180);
	}
	else
	{
		capture = cvCreateFileCapture( video_fname );
	}

    
	if(!capture)
    {
        std::cout << "Capture could not be initialized properly" << std::endl;
        return -1;
    }	

	storage			= cvCreateMemStorage();
	storage2		= cvCreateMemStorage();
	storage3		= cvCreateMemStorage();
	cascade_face	= (CvHaarClassifierCascade*)cvLoad( cascade_fname_face, 0, 0, 0 );
	cascade_eye		= (CvHaarClassifierCascade*)cvLoad( cascade_fname_eye, 0, 0, 0 );
	cascade_mouth	= (CvHaarClassifierCascade*)cvLoad( cascade_fname_mouth, 0, 0, 0 );
	
	if( !cascade_face || !cascade_eye || !cascade_mouth )
	{
		fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
		return -1;
	}
	

	cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.0, 1.0, 0, 1, CV_AA);



	frame = cvQueryFrame( capture );
	if( !frame ) return -1;



	CvSize frameSize = cvSize(frame->width, frame->height);
	CvSize orgFrameSize = frameSize;
	grayFrame = cvCreateImage ( orgFrameSize, IPL_DEPTH_8U, 1);
	normFrame = cvCreateImage ( orgFrameSize, IPL_DEPTH_8U, 1);
	tmpFrame = cvCreateImage ( orgFrameSize, IPL_DEPTH_8U, 1);



	time(&start);
	while( true )
	{
		RawFrame = cvQueryFrame( capture );
		if( !RawFrame ) break;

		cvCvtColor( RawFrame, grayFrame, CV_BGR2GRAY);
		cvEqualizeHist( grayFrame, normFrame );

		cvShowImage( window_name_norma, normFrame );
		cvShowImage( window_name, grayFrame );
		detect_and_draw( normFrame );
		

		// FPS counter
		time(&end);
		++counter;
		sec = difftime (end, start);     
		fps = counter / sec;
		_gcvt(fps, 3, text);
		cvPutText(normFrame, text, cvPoint(10, 20), &font, cvScalar(255, 255, 255, 0));
		
		_gcvt(object_size, 3, text);
		cvPutText(normFrame, text, cvPoint(10, 40), &font, cvScalar(255, 255, 255, 0));

		cvShowImage( window_name_norma, normFrame);


		char c = cvWaitKey(1);
		if( c == 27 ) break;
	}
	time(&end);

	std::cout	<< "Total effectiveness estemated " << ((double) posRes / (double) counter) * 100 << std::endl
				<< "Total exec time was " << end - start <<  std::endl
				<< "Avarage FPS was " << (double)counter / (double)(end - start) << std::endl;

	cvReleaseMemStorage( &storage );
	cvReleaseMemStorage( &storage2 );
	cvReleaseMemStorage( &storage3 );
	cvReleaseCapture( &capture );
	cvDestroyAllWindows();
}

// Function to detect and draw any faces that is present in an image
void detect_and_draw( IplImage* img )
{
	int i;
	
	// Clear the memory storage which was used before
	cvClearMemStorage( storage );
	cvClearMemStorage( storage2 );
	cvClearMemStorage( storage3 );

	// There can be more than one face in an image. So create a growable sequence of faces.
	// Detect the objects and store them in the sequence
	faces = cvHaarDetectObjects( img,
								cascade_face, 
								storage,
								1.3, 2, 
								CV_HAAR_FIND_BIGGEST_OBJECT,
								cvSize(30, 30) );
	if ( faces->total > 0 )
	{
		CvRect* r = (CvRect*)cvGetSeqElem( faces, 0 );
		face_rect = (CvRect*)cvGetSeqElem( faces, 0 );

		// Find the dimensions of the face,and scale it if necessary
		point1.x = r->x;	point2.x = r->x + r->width;
		point1.y = r->y;	point2.y = r->y + r->height;

		object_size = point2.x - point1.x;
		printf("Frame %d\tsize %d\tcoords: %d\t %d\t %d\t %d\n", counter, object_size, point1.x, point1.y, point2.x, point2.y);

		// Draw the rectangle in the input image
		cvRectangle( img, point1, point2, CV_RGB(0,0,0), 2, 8, 0 );

		// increamet found objects count
		++posRes;

		// Clear mem for next detection
		cvClearMemStorage(storage);

		cvSetImageROI(img, cvRect(r->x, r->y + (r->height/5.5), r->width, r->height/3.0));
		eyes = cvHaarDetectObjects(	img, 
									cascade_eye, 
									storage2,
									1.15, 3, 
									CV_HAAR_DO_CANNY_PRUNING,
									cvSize(25, 15)
		);
		for( i = 0; i < (eyes ? eyes->total : 0); i++ ) 
		{
			r = (CvRect*)cvGetSeqElem( eyes, i );
			cvRectangle(img, 
						cvPoint(r->x, r->y), 
						cvPoint(r->x + r->width, r->y + r->height),
						CV_RGB(0, 0, 0), 1, 8, 0);
		}
		cvResetImageROI(img);

		face_rect = (CvRect*)cvGetSeqElem( faces, 0 );
		r = (CvRect*)face_rect;	
		/*point1.x = r->x + (r->width)/6.0;		point2.x = r->x + 5*(r->width)/6.0;
		point1.y = r->y + 4*(r->height)/6.0;	point2.y = r->y + 7*(r->height)/6.0;
		cvRectangle( img, point1, point2, CV_RGB(255,255,255), 1, 8, 0 );*/


		cvSetImageROI(img, cvRect(	r->x + (r->width)/6.0,
									r->y + 2*(r->height)/3.0,
									5*(r->width)/6.0,
									7*(r->height)/6.0));
		mouths = cvHaarDetectObjects(	img, 
										cascade_mouth, 
										storage3,
										1.4, 3, 
										CV_HAAR_FIND_BIGGEST_OBJECT,
										cvSize(25, 15)
		);
		if ( mouths->total > 0 )
		{
		CvRect *m = (CvRect*)cvGetSeqElem(mouths, 0);
		cvRectangle(img, 
					cvPoint(m->x, m->y), 
					cvPoint(m->x + m->width, m->y + m->height),
					CV_RGB(0, 0, 0), 1, 8, 0);
		}

		cvResetImageROI(img);
	}

	mouths	= NULL;
	eyes	= NULL;
	faces	= NULL;
}