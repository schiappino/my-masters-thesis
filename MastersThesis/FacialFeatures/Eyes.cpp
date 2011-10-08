#include "Eyes.h"

void DetectEyes()
{
	// Start detecting only if face is found
	if( faces.size() )
	{
		// Iris is typically 7% of face size
		int irisRadiusMax = cvRound(face_size*0.03);

		Rect eyesROI	 = Rect( faces[0].x,							(int)(faces[0].y + 0.2*faces[0].height), 
								 faces[0].width,						(int)(0.4*faces[0].height) );

		Rect eyeLeftROI	 = Rect( (int)(faces[0].x + 0.1*faces[0].width),(int)(faces[0].y + 0.2*faces[0].height), 
								 (int)(0.4*faces[0].width),				(int)(0.4*faces[0].height) );

		Rect eyeRightROI = Rect( (int)(faces[0].x + 0.5*faces[0].width),(int)(faces[0].y + 0.2*faces[0].height), 
								 (int)(0.4*faces[0].width),				(int)(0.4*faces[0].height) );
		
		// Normalize histogram to improve all shit
		Mat imgGrayEyes ( imgGray, eyesROI );
		equalizeHist( imgGrayEyes, imgGrayEyes );

		#ifdef EYES_DETECT_SINGLE_CASCADE		
		// Here both eyes are found at the same time by single pass
		cascadeEye.detectMultiScale(
			imgGrayEyes,
			eyes,
			1.3,
			3,
			CV_HAAR_DO_CANNY_PRUNING
		);
		
		// Setup roi on image
		Mat imgProcessedROI (imgProcessed, eyesROI );
		
		// draw all found eyes
		for( int i = 0; i < (int)eyes.size(); ++i )
		{
			rectangle( imgProcessedROI,
				Point( eyes[i].x, eyes[i].y),
				Point( eyes[i].x + eyes[i].width, eyes[i].y + eyes[i].height),
				CV_RGB(100, 100, 255)
			);
		}
		#endif
		#ifdef EYES_DETECT_MULTI_CASCADE
		// TEMPORARY APPROACH: detecting eye pos in two passes		
		vector<Rect> eyesLeft,
					 eyesRight;
		Mat imgEyeLeftGray	( imgGray, eyeLeftROI ),
			imgEyeRightGray ( imgGray, eyeRightROI );

		cascadeEyeLeft.detectMultiScale( imgEyeLeftGray,	eyesLeft, 1.2, 2, CV_HAAR_FIND_BIGGEST_OBJECT );
		cascadeEyeRight.detectMultiScale( imgEyeRightGray, eyesRight, 1.2, 2, CV_HAAR_FIND_BIGGEST_OBJECT );
		
		Mat imgProcessedWithRightEye ( imgProcessed, eyeRightROI ),
			imgProcessedWithLeftEye	 ( imgProcessed, eyeLeftROI );
		
		#ifdef EYE_DETECT_ROI_DEBUG
			for( int i = 0; i < (int)eyesRight.size(); ++i )
			{
				rectangle( imgProcessedWithRightEye,
					Point( eyesRight[i].x, eyesRight[i].y),
					Point( eyesRight[i].x + eyesRight[i].width, eyesRight[i].y + eyesRight[i].height),
					CV_RGB(0, 0, 0)
				);
			}
			for( int i = 0; i < (int)eyesLeft.size(); ++i )
			{
				rectangle( imgProcessedWithLeftEye,
					Point( eyesLeft[i].x, eyesLeft[i].y),
					Point( eyesLeft[i].x + eyesLeft[i].width, eyesLeft[i].y + eyesLeft[i].height),
					CV_RGB(0, 0, 0)
				);
			}
			// Show on processed parts of image where eyes have been found
			imshow( "Left", imgProcessedWithLeftEye );
			imshow( "Right", imgProcessedWithRightEye );
		#endif

		Mat imgRedCopy;
		rgb_planes[0].copyTo( imgRedCopy );
		Mat imgEyes ( imgRedCopy, eyesROI );

		Mat imgEyeLeft ( imgRedCopy, eyeLeftROI );
		Mat imgEyeRight (imgRedCopy, eyeRightROI );

		equalizeHist( imgEyeLeft, imgEyeLeft );
		equalizeHist( imgEyeRight, imgEyeRight );
		bitwise_not( imgEyes, imgEyes );
		exponentialOperator( imgEyes, imgEyes);

		if( eyesRight.size() )
		{
			Rect foundROI = eyesRight[0];
			Mat imgFoundRightEye ( imgEyeRight, foundROI );
			imshow( "Found eye right", imgFoundRightEye );
			Mat imgProcessedFoundRightEye ( imgProcessedWithRightEye, foundROI );
			EyeTemplateMatching( imgFoundRightEye, imgProcessedFoundRightEye, imgTempl, irisRadiusMax );
		} else {
			// if right eye is not found then try templ match on bigger ROI
			Mat imgProcessedRightEye ( imgProcessed, eyeRightROI );
			EyeTemplateMatching( imgEyeRight, imgProcessedRightEye, imgTempl, irisRadiusMax );
		}
		if( eyesLeft.size() )
		{
			Rect foundROI = eyesLeft[0];
			Mat imgFoundLeftEye ( imgEyeLeft, foundROI );
			imshow( "Found eye left", imgFoundLeftEye );
			Mat imgProcessedFoundLeftEye ( imgProcessedWithLeftEye, foundROI );
			EyeTemplateMatching( imgFoundLeftEye, imgProcessedFoundLeftEye, imgTempl, irisRadiusMax );
		} else { 
			// if left eye is not found then try templ match on bigger ROI
			Mat imgProcessedLeftEye ( imgProcessed, eyeLeftROI );
			EyeTemplateMatching( imgEyeLeft, imgProcessedLeftEye, imgTempl, irisRadiusMax );
		}

		#endif
	}
};
void EyeTemplateMatching( Mat src, Mat disp, Mat templ, int irisRadius)
{
	Mat result;
	/// Create the result matrix
	int result_cols =  src.cols - templ.cols + 1;
	int result_rows = src.rows - templ.rows + 1;   

	result.create( result_cols, result_rows, CV_32FC1 );

	/// Do the Matching and Normalize
	matchTemplate( src, templ, result, TemplMatchMet );
	normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

	/// Localizing the best match with minMaxLoc
	double minVal, maxVal; 
	Point minLoc, maxLoc, matchLoc;

	minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. 
	/// For all the other methods, the higher the better
	if( TemplMatchMet  == CV_TM_SQDIFF || TemplMatchMet == CV_TM_SQDIFF_NORMED )
	{ matchLoc = minLoc; }
	else  
	{ matchLoc = maxLoc; }
	
	Point center = Point( matchLoc.x + cvRound(templ.cols/2.0), matchLoc.y + cvRound(templ.rows/2.0));

	/// Show me what you got
	circle( disp, center, irisRadius, CV_RGB(0,100,255), 2 );
	circle( result, center, irisRadius, CV_RGB(0,100,255), 2 );

	#ifdef EYES_TEMPLATE_MATCH_DEBUG
	imshow( wndNameTemplRes, result );
	#endif
};
