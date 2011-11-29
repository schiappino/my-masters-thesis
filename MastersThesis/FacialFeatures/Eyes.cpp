#include "Eyes.h"

void DetectEyes()
{
	// Start detecting only if face is found
	if( faces.size() )
	{
		// Left and right eye coordinates
		Point leftEyeCoords, 
			rightEyeCoords;

		// Iris is typically 7% of face size
		int irisRadiusMax = cvRound(face_size*0.035);

		Rect eyesROI	 = Rect( faces[0].x,							(int)(faces[0].y + 0.2*faces[0].height), 
								 faces[0].width,						(int)(0.4*faces[0].height) );

		Rect eyeLeftROI	 = Rect( (int)(faces[0].x + 0.1*faces[0].width),(int)(faces[0].y + 0.3*faces[0].height), 
								 (int)(0.4*faces[0].width),				(int)(0.3*faces[0].height) );

		Rect eyeRightROI = Rect( (int)(faces[0].x + 0.5*faces[0].width),(int)(faces[0].y + 0.3*faces[0].height), 
								 (int)(0.4*faces[0].width),				(int)(0.3*faces[0].height) );
		
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
		// Vectors which will store results of Haar Detector
		vector<Rect> eyesLeft,
					 eyesRight;

		// Gray images for each eye used in Haar Detector
		Mat imgEyeLeftGray	( imgGray, eyeLeftROI ),
			imgEyeRightGray ( imgGray, eyeRightROI );

		// Runs Haar Detector
		cascadeEyeLeft.detectMultiScale( imgEyeLeftGray,	eyesLeft, 1.2, 2, CV_HAAR_FIND_BIGGEST_OBJECT );
		cascadeEyeRight.detectMultiScale( imgEyeRightGray, eyesRight, 1.2, 2, CV_HAAR_FIND_BIGGEST_OBJECT );
		
		// RGB images for each eye used for output drawing
		Mat imgProcessedWithRightEye ( imgProcessed, eyeRightROI ),
			imgProcessedWithLeftEye	 ( imgProcessed, eyeLeftROI );
		
		// Draw found ROIs by Haar
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
			Mat imgProcessedFoundRightEye ( imgProcessedWithRightEye, foundROI );
			rightEyeCoords = EyeTemplateMatching( imgFoundRightEye, imgProcessedFoundRightEye, imgTempl, irisRadiusMax );
			// Add the offset of ROI to obtain global location of the eye
			rightEyeCoords = Point( rightEyeCoords.x + eyeRightROI.x + foundROI.x, rightEyeCoords.y + eyeRightROI.y + foundROI.y );

			#ifdef EYES_DETECT_DEBUG
				imshow( "Found eye right", imgFoundRightEye );
			#endif
		} else { // if right eye is not found then try templ match on bigger ROI
			Mat imgProcessedRightEye ( imgProcessed, eyeRightROI );
			rightEyeCoords = EyeTemplateMatching( imgEyeRight, imgProcessedRightEye, imgTempl, irisRadiusMax );
			// Add the offset of ROI to obtain global location of the eye
			rightEyeCoords = Point( rightEyeCoords.x + eyeRightROI.x, rightEyeCoords.y + eyeRightROI.y );
		}
		if( eyesLeft.size() )
		{
			Rect foundROI = eyesLeft[0];
			Mat imgFoundLeftEye ( imgEyeLeft, foundROI );
			Mat imgProcessedFoundLeftEye ( imgProcessedWithLeftEye, foundROI );
			leftEyeCoords = EyeTemplateMatching( imgFoundLeftEye, imgProcessedFoundLeftEye, imgTempl, irisRadiusMax );
			// Add the offset of ROI to obtain global location of the eye
			leftEyeCoords = Point( leftEyeCoords.x + eyeLeftROI.x + foundROI.x, leftEyeCoords.y + eyeLeftROI.y + foundROI.y );

			#ifdef EYES_DETECT_DEBUG
				imshow( "Found eye left", imgFoundLeftEye );
			#endif
		} else { // if left eye is not found then try templ match on bigger ROI
			Mat imgProcessedLeftEye ( imgProcessed, eyeLeftROI );
			leftEyeCoords = EyeTemplateMatching( imgEyeLeft, imgProcessedLeftEye, imgTempl, irisRadiusMax );
			// Add the offset of ROI to obtain global location of the eye
			leftEyeCoords = Point( leftEyeCoords.x + eyeLeftROI.x, leftEyeCoords.y + eyeLeftROI.y );
		}
		
		// Validation: Draw ground truth eye centres
		#ifdef VALIDATION
			#ifdef GUI
			DrawGroundTruthEyePos( getCurrentFaceDbFatures(selectedFaceDb) );
			#endif
			eyePositionsMetric( leftEyeCoords, rightEyeCoords, getCurrentFaceDbFatures(selectedFaceDb) );
		#endif

		#endif
	}
};
Point EyeTemplateMatching( Mat src, Mat disp, Mat templ, int irisRadius)
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

	return center;
};
void DrawGroundTruthEyePos( FacialFeaturesValidation& features )
{
	Point ptx1, ptx2, pty1, pty2;
	Scalar colour = Scalar::all(255);
	const int offset = 10,
			thickness = 1;

	ptx1 = Point( features.eyes.left.at( imIt ).x - offset, 
				  features.eyes.left.at( imIt ).y);
	ptx2 = Point( features.eyes.left.at( imIt ).x + offset, 
				  features.eyes.left.at( imIt ).y);
	pty1 = Point( features.eyes.left.at( imIt ).x, 
				  features.eyes.left.at( imIt ).y - offset);
	pty2 = Point( features.eyes.left.at( imIt ).x, 
				  features.eyes.left.at( imIt ).y + offset);
	line( imgProcessed, ptx1, ptx2, colour, thickness );
	line( imgProcessed, pty1, pty2, colour, thickness );

	ptx1 = Point( features.eyes.right.at( imIt ).x - offset, 
				  features.eyes.right.at( imIt ).y);
	ptx2 = Point( features.eyes.right.at( imIt ).x + offset, 
				  features.eyes.right.at( imIt ).y);
	pty1 = Point( features.eyes.right.at( imIt ).x, 
				  features.eyes.right.at( imIt ).y - offset);
	pty2 = Point( features.eyes.right.at( imIt ).x, 
				  features.eyes.right.at( imIt ).y + offset);
	line( imgProcessed, ptx1, ptx2, colour, thickness );
	line( imgProcessed, pty1, pty2, colour, thickness );
}
double getInterocularDist( FacialFeaturesValidation& features )
{
	double dx = (features.eyes.right.at( imIt ).x - features.eyes.left.at( imIt ).x); 
	double dy = (features.eyes.right.at( imIt ).y - features.eyes.left.at( imIt ).y); 
	return (sqrt( dy*dy + dx*dx ));
}
void eyePositionsMetric( Point& left, Point& right, FacialFeaturesValidation& features )
{
	double interOcularDist	= getInterocularDist( features );

	double leftEye_Xerr		= (abs(features.eyes.left.at( imIt ).x - left.x ) / interOcularDist) * 100;
	double leftEye_Yerr		= (abs(features.eyes.left.at( imIt ).y - left.y ) / interOcularDist) * 100;
	double rightEye_Xerr	= (abs(features.eyes.right.at( imIt ).x - right.x ) / interOcularDist) * 100;
	double rightEye_Yerr	= (abs(features.eyes.right.at( imIt ).y - right.y ) / interOcularDist) * 100;

	// Calculate detection error using Pitagoras equetion
	double leftEye_err		= square_distance( leftEye_Xerr, leftEye_Yerr );
	double rightEye_err		= square_distance( rightEye_Xerr, rightEye_Yerr );

	features.eyes.IOD.at( imIt ) = interOcularDist;
	features.eyes.right_det.at( imIt ) = right;
	features.eyes.left_det.at( imIt ) = left;
	features.eyes.right_err.at( imIt ) = rightEye_err;
	features.eyes.left_err.at( imIt ) = leftEye_err;

	cout << "Eyes validation\t\tIOD " << setw(4) << interOcularDist
		 << "\tLeft " << setiosflags(ios::basefield) << setiosflags(ios::fixed) << setprecision(1)
		 << setw(4) << leftEye_err << "%"
		 << "\tRight" << setiosflags(ios::basefield) << setiosflags(ios::fixed) << setprecision(1)
		 << setw(5) << rightEye_err << "%" 
		 << endl;
}
bool saveEyePosValidationData( FacialFeaturesValidation& features )
{
	string output_file_name = "Results Eye Validation ";
	output_file_name += convertInt(abs((int)getTickCount()));
	output_file_name += ".csv";
	const string next_cell = ";";
	ofstream out;

	out.open( output_file_name );
	if( !out.good() ){ cerr << "An error occured when saving Eye Validation data" << endl; return false; }

	out << "Results for Eye Validation" << endl
		<< "Item;" << "IOD;" << "TREL x;" << "TREL y;" << "TLEL x;" << "TLEL y;" 
		<< "DREL x;" << "DREL y;" << "DLEL x;" << "DLEL y;" << "REE %;" << "LEE %;" << endl;

	int cnt = features.eyes.left.size();
	for( int i = 0; i < cnt; ++i )
	{
		out << i << next_cell 
			<< features.eyes.IOD.at(i)			<< next_cell 

			<< features.eyes.right.at(i).x		<< next_cell
			<< features.eyes.right.at(i).y		<< next_cell
			
			<< features.eyes.left.at(i).x		<< next_cell
			<< features.eyes.left.at(i).y		<< next_cell
			
			<< features.eyes.right_det.at(i).x	<< next_cell
			<< features.eyes.right_det.at(i).y	<< next_cell
			
			<< features.eyes.left_det.at(i).x	<< next_cell
			<< features.eyes.left_det.at(i).y	<< next_cell

			<< features.eyes.right_err.at(i)	<< next_cell
			<< features.eyes.left_err.at(i)		<< next_cell << endl;
	}
}