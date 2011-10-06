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
		Mat imgEyeLeft	( imgGray, eyeLeftROI ),
			imgEyeRight ( imgGray, eyeRightROI );
		
		cascadeEyeLeft.detectMultiScale( imgEyeLeft,	eyesLeft, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT );
		cascadeEyeRight.detectMultiScale( imgEyeRight, eyesRight, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT );
		
		Mat imgProcessedWithRightEye ( imgGray, eyeRightROI ),
			imgProcessedWithLeftEye	 ( imgGray, eyeLeftROI );
		
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

		imshow( "Left", imgProcessedWithLeftEye );
		imshow( "Right", imgProcessedWithRightEye );
		#endif
		
		Mat imgEyes		( rgb_planes[0], eyesROI );
		Mat imgEyeLeft	( rgb_planes[0], eyeLeftROI ),
			imgEyeRight ( rgb_planes[0], eyeRightROI );

		equalizeHist( imgEyeRight, imgEyeRight );
		equalizeHist( imgEyeLeft, imgEyeLeft );
		bitwise_not( imgEyes, imgEyes );
		exponentialOperator( imgEyes, imgEyes );
		imshow( wndNameEyesExpTrans, imgEyes );

		Mat imgProcessedLeftEye ( imgProcessed, eyeLeftROI ),
			imgProcessedRightEye ( imgProcessed, eyeRightROI );
		EyeTemplateMatching( imgEyeLeft, imgProcessedLeftEye, imgTempl, irisRadiusMax );
		EyeTemplateMatching( imgEyeRight, imgProcessedRightEye, imgTempl, irisRadiusMax );

		#ifdef EYES_DETECT_HOUGH_TRANSFORM
		// --> Hough Circle transform for iris detection
		HoughMinDist = cvRound(face_size/3.0);
		vector<Vec3f> iris;
		Mat imgEyesFiltered;
		bilateralFilter( imgEyes, imgEyesFiltered, bilatBlurVal, bilatBlurVal*2, bilatBlurVal/2 );
		imshow( wndNameBilateral, imgEyesFiltered );

		HoughCircles( imgEyesFiltered, iris, CV_HOUGH_GRADIENT,
			Hough_dp, HoughMinDist, 100, 200, 3, irisRadiusMax );
		for( int i = 0; i < iris.size(); ++i )
		{
			Point center( cvRound(iris[i][0]), cvRound(iris[i][1]) );
			int radius = cvRound(iris[i][2]);

			Mat imgEyesIris ( imgProcessed, eyesROI );
			circle( imgEyesIris, center, radius, CV_RGB(250,0,0) );
		}
		// <-- Hough Circle transform for iris detection
		#endif

		#ifdef EYES_DETECT_CONNECTED_COMP
		Scalar avgIntensityLeftEye = mean( imgEyeLeft );
		Scalar avgIntensityRightEye = mean( imgEyeRight );
		Scalar stdDev, avgIntensity;
		meanStdDev( imgEyeLeft, avgIntensity, stdDev );
		eyeThreshold = (int)(avgIntensity.val[0] + stdDev.val[0]*z/10);

		threshold( imgEyes, imgEyes, eyeThreshold, 255, THRESH_BINARY );
		Mat kernel = getStructuringElement( MORPH_ELLIPSE, Size(3, 3) );
		morphologyEx( imgEyes, imgEyes, MORPH_CLOSE, kernel, Point(1,1), 1 );
		morphologyEx( imgEyes, imgEyes, MORPH_OPEN, kernel, Point(1,1), 1 );

		Mat imgEyeBinaryCopy;
		imgEyeLeft.copyTo( imgEyeBinaryCopy );
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours( imgEyeBinaryCopy, contours, hierarchy,
					  CV_RETR_LIST, CV_CHAIN_APPROX_TC89_KCOS );
		Mat imgProcessedLeftEye ( imgProcessed, eyeLeftROI );
		for( int i = 0; i < contours.size(); ++i )
			drawContours( imgProcessedLeftEye, contours, i, CV_RGB(0,100,255) );
		
		imshow( wndNameLeftEye, imgEyeLeft );
		imshow( wndNameRightEye, imgEyeRight );
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
