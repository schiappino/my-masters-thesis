#include "Mouth.h"

void DetectMouth()
{
	// Create ROI for mouth detection
	Rect mouthROI = Rect(
		(int) (faces[0].x + 0.2*faces[0].width), 
		(int) (faces[0].y + 0.65*faces[0].height),
		(int) (0.6*faces[0].width), 
		(int) (0.45*faces[0].height));

	#ifdef MOUTH_ROI_DEBUG
	// Show up ROI on source image
	rectangle( 
		imgProcessed, 
		Point( mouthROI.x, mouthROI.y ),
		Point( mouthROI.x + mouthROI.width, mouthROI.y + mouthROI.height ),
		CV_RGB(0,255,0)
	);
	putTextWithShadow(
		imgProcessed,
		"Mouth detection ROI",
		Point( mouthROI.x, mouthROI.y )
	);
	#endif
	
	// Setup ROI on image where detection will be done
	Mat imgMouthROI( imgGray, mouthROI );

	cascadeMouth.detectMultiScale(
		imgMouthROI,					// image to search
		mouths,							// found objects container
		1.2,							// window increase param
		3,								// min neighbors to accept object
		CV_HAAR_FIND_BIGGEST_OBJECT		// search method
	);

	// Check if detector found anything; if yes draw it
	if( mouths.size() )
	{
		// Adjust found mouth region
		foundMouthROI = Rect(
			(int) (mouthROI.x + mouths[0].x - 0.1*mouths[0].width), (int) (mouthROI.y + mouths[0].y - 0.1*mouths[0].height),
			(int) (1.2*mouths[0].width), mouths[0].height 
		);
		
		#ifdef MOUTH_ROI_DEBUG
		// Setup ROI on output image so that object 
		// coordinates compliy with those on search image
		Mat imgProcessedROI( imgProcessed, mouthROI );

		// ..and draw it
		rectangle( imgProcessedROI, 
			Point( foundMouthROI.x, foundMouthROI.y ),
			Point( foundMouthROI.x + foundMouthROI.width, foundMouthROI.y + foundMouthROI.height ),
			CV_RGB(0,0,0) 
		);

		putTextWithShadow(
			imgProcessed,
			"Found mouth",
			Point( mouths[0].x, mouths[0].y )
		);
		#endif

		Mat imgMouthHue( hls_planes[HUE_PLANE], foundMouthROI );
		Mat imgMouthThresh ( imgMouthHue.size(), imgMouthHue.type() );
		Mat imgBlurredMouth;
		bilateralFilter( imgMouthHue, imgBlurredMouth, bilatBlurVal, bilatBlurVal*2, bilatBlurVal/2 );
		imshow( wndNameBlur, imgBlurredMouth );
		mouthHueAvg = mean( imgMouthHue );

		mouthThreshold = (int)mouthHueAvg.val[0];
		threshold( imgBlurredMouth, imgMouthThresh, (double) mouthThreshold, 255, THRESH_BINARY_INV );
		imshow( wndNameMouth, imgMouthThresh );
	}
};