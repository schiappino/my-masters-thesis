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
	Mat imgMouthGray( imgGray, mouthROI );
	equalizeHist( imgMouthGray, imgMouthGray );

	cascadeMouth.detectMultiScale(
		imgMouthGray,					// image to search
		mouths,							// found objects container
		1.2,							// window increase param
		3,								// min neighbors to accept object
		CV_HAAR_FIND_BIGGEST_OBJECT		// search method
	);

	// Check if detector found anything; if yes draw it
	if( mouths.size() )
	{
		#ifdef MOUTH_ROI_DEBUG
			// Setup ROI on output image so that object 
			// coordinates compliy with those on search image
			Rect roi = Rect ( mouthROI.x + mouths[0].x, mouthROI.y + mouths[0].y,
							mouths[0].width, mouths[0].height );

			Mat imgProcessedMouth( imgProcessed, mouthROI );
			// ..and draw it
			rectangle( imgProcessed, 
				Point( roi.x, roi.y ),
				Point( roi.x + roi.width, roi.y + roi.height ),
				CV_RGB(0,0,0) 
			);

			putTextWithShadow(
				imgProcessed,
				"Found mouth",
				Point( foundMouthROI.x, foundMouthROI.y )
			);
		#endif
		
		// Adjust found mouth region
		foundMouthROI = Rect(
			(int) (mouthROI.x + mouths[0].x - 0.2*mouths[0].width), (int) (mouthROI.y + mouths[0].y - 0.2*mouths[0].height),
			(int) (1.4*mouths[0].width), (int) (1.4*mouths[0].height)
		);

		rectangle( imgProcessed, 
				Point( foundMouthROI.x, foundMouthROI.y ),
				Point( foundMouthROI.x + foundMouthROI.width, foundMouthROI.y + foundMouthROI.height ),
				CV_RGB(0,0,255),
				2
			);


		// Get Hue, Saturation and Greyscale images of found mouth region
		Mat imgMouthHue( hls_planes[0], foundMouthROI );
		Mat imgMouthSat( hls_planes[2], foundMouthROI );
		Mat imgMouthGray( imgGray,		foundMouthROI );

		Mat imgMouthSatBlurred;
		int bilatValSat = 12; // Bilateral filter value for saturation channel
		bilateralFilter( imgMouthSat, imgMouthSatBlurred, bilatValSat, bilatValSat*2, bilatValSat/2 );
		
		// Set the mouth threshold to the experimentally found value
		Scalar _meanSatLevel = mean( imgMouthSatBlurred );
		double mouthSatThr = 1.2 * _meanSatLevel.val[0];
		Mat imgMouthSatThr;
		threshold( imgMouthSatBlurred, imgMouthSatThr, mouthSatThr, 255, CV_THRESH_BINARY );

		// Preprocess mouth grayscale image
		Mat imgMouthGrayBlur;
		GaussianBlur( imgMouthGray, imgMouthGrayBlur, Size(3,3), 0 );
		
		Rect leftCornerROI	= Rect (0,						 0, foundMouthROI.width/2.0, foundMouthROI.height );
		Rect rightCornerROI = Rect (foundMouthROI.width/2.0, 0, foundMouthROI.width/2.0, foundMouthROI.height );
		Mat imgMouthGrayBlurLeft ( imgMouthGrayBlur, leftCornerROI );
		Mat imgMouthGrayBlurRight ( imgMouthGrayBlur, rightCornerROI );

		Scalar _meanGrayLevelLeft = mean( imgMouthGrayBlurLeft );
		Scalar _meanGrayLevelRight = mean( imgMouthGrayBlurRight );
		double threshGrayCoeff = 0.25;
		double mouthGrayThrLeft = threshGrayCoeff * _meanGrayLevelLeft.val[0];
		double mouthGrayThrRight = threshGrayCoeff * _meanGrayLevelRight.val[0];
		cout << ">> Grayscale mean value left: " << mouthGrayThrLeft << endl;
		cout << ">> Grayscale mean value right: " << mouthGrayThrRight << endl;
		
		Mat imgMouthGrayThrLeft, imgMouthGrayThrRight;
		threshold( imgMouthGrayBlur, imgMouthGrayThrLeft, mouthGrayThrLeft, 255, CV_THRESH_BINARY );
		threshold( imgMouthGrayBlur, imgMouthGrayThrRight, mouthGrayThrRight, 255, CV_THRESH_BINARY );
		
		
		// Use Shi-Tomasi corner detector
		vector <Point2f> cornersCandidatesSat, 
						 cornersCandidatesGrayLeft,
						 cornersCandidatesGrayRight;
		Mat imgMouthCornersSat, imgMouthCornersGray;
		imgMouthCornersSat = imgMouthSatBlurred.clone();

		RNG rng(startTime());
		cornerDetector( imgMouthSatThr, cornersCandidatesSat );
		cornerDetector( imgMouthGrayThrLeft, cornersCandidatesGrayLeft );
		cornerDetector( imgMouthGrayThrRight, cornersCandidatesGrayRight );

		// Find best corner candidates from saturation channel image
		Point2f leftCorner, rightCorner;
		getBestMouthCornerCandidateLeft( leftCorner, cornersCandidatesSat );
		getBestMouthCornerCandidateRight( rightCorner, cornersCandidatesSat );

		// Find best corner candidates from grayscale image
		Point2f leftCornGray, rightCornGray;
		getBestMouthCornerCandidateLeft( leftCornGray, cornersCandidatesGrayLeft );
		getBestMouthCornerCandidateRight( rightCornGray, cornersCandidatesGrayRight );


		#ifdef MOUTH_DETECT_DEBUG
		for( int i = 0; i < cornersCandidatesSat.size(); i++ )
		{
			circle( imgMouthCornersSat, cornersCandidatesSat[i], 4, 
					Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255)), -1, 8, 0 ); 
		}
		imshow( wndNameMouth, imgMouthCornersSat );

		// Draw detected corner points
		Mat imgProcessedMouthCorners ( imgProcessed, foundMouthROI );
		// from Saturation channel image
		circle( imgProcessedMouthCorners, leftCorner, 4, CV_RGB(0,0,255), -1, 8, 0 ); 
		circle( imgProcessedMouthCorners, rightCorner, 4, CV_RGB(0,0,255), -1, 8, 0 ); 
		// from grayscale image
		circle( imgProcessedMouthCorners, leftCornGray, 4, CV_RGB(255,0,0), -1, 8, 0 ); 
		circle( imgProcessedMouthCorners, rightCornGray, 4, CV_RGB(255,0,0), -1, 8, 0 ); 
		imshow( wndNameCorners, imgMouthCornersSat );
		#endif

		// Calculate average corner position
		if( mouthSatThr > 0 ) // combine results from grayscale and saturation channels
		{
			leftCorner = Point2f( (leftCorner.x + leftCornGray.x)/2, (leftCorner.y + leftCornGray.y)/2 );
			rightCorner = Point2f( (rightCorner.x + rightCornGray.x)/2, (rightCorner.y + rightCornGray.y)/2 );
		}
		else // mouthSatThr is 0 so copy the results from grayscale search
		{
			leftCorner = leftCornGray;
			rightCorner = rightCornGray;
		}

		// Adjust corner points with respect to whole image
		leftCorner = Point2f( leftCorner.x + foundMouthROI.x, leftCorner.y + foundMouthROI.y );
		rightCorner = Point2f( rightCorner.x + foundMouthROI.x, rightCorner.y + foundMouthROI.y );


		circle( imgProcessed, leftCorner, 4, CV_RGB(0,255,0), -1, 8, 0 ); 
		circle( imgProcessed, rightCorner, 4, CV_RGB(0,255,0), -1, 8, 0 ); 
		
		#ifdef VALIDATION
		mouthCornersPositionsMetric( leftCorner, rightCorner, getCurrentFaceDbFatures(selectedFaceDb) );
		DrawGroundTruthMouthConerPos( getCurrentFaceDbFatures(selectedFaceDb) );
		#endif
	}
};
void cornerDetector( Mat img, vector<Point2f>& corners )
{
	if ( maxCorners == 0 ) { maxCorners = 1; }

	double	qualityLevel = 0.01;
	double	minDistance = 10;
	int		blockSize = 3;
	bool	useHarrisDetector = false;
	double	k = 0.04;

	goodFeaturesToTrack( img,
						 corners,
						 maxCorners,
						 qualityLevel,
						 minDistance,
						 Mat(),
						 3);
};
void getBestMouthCornerCandidateLeft( Point2f& left, vector <Point2f>& candidates )
{
	size_t ptsCnt = candidates.size();
	size_t i;
	float avg = 0;

	if( ptsCnt <= 1 )
		return;

	// Calculate avarage y coord
	for( i = 0; i < ptsCnt; ++i )	
		avg += candidates[i].y;

	float tolerance = stdev_vertical( candidates );

	avg /= ptsCnt;
	float avgUp = avg + tolerance;
	float avgDown = avg - tolerance;

	float min = numeric_limits<float>::max();
	size_t leftIdx;
	for( i = 0; i < ptsCnt; ++i )
	{
		if( candidates[i].x < min )
		{
			// Look for left corner
			if( candidates[i].y <= avgUp && candidates[i].y >= avgDown )
			{
				min = candidates[i].x;
				leftIdx = i;
			}
		}
	}
	left = candidates[leftIdx];
};
void getBestMouthCornerCandidateRight( Point2f& right, vector <Point2f>& candidates )
{
	size_t ptsCnt = candidates.size();
	size_t i;
	float avg = 0;

	if( ptsCnt <= 2 )
		return;

	// Calculate avarage y coord
	for( i = 0; i < ptsCnt; ++i )	
		avg += candidates[i].y;

	float tolerance = stdev_vertical( candidates );

	avg /= ptsCnt;
	float avgUp = avg + tolerance;
	float avgDown = avg - tolerance;

	float max = 0;
	size_t rightIdx;
	for( i = 0; i < ptsCnt; ++i )
	{
		if( candidates[i].x > max )
		{
			// Look for right corner
			if( candidates[i].y < avgUp && candidates[i].y > avgDown )
			{
				max = candidates[i].x;
				rightIdx = i;
			}
		}
	}
	right = candidates[rightIdx];
};
double stdev_vertical( vector <Point2f>& points )
{
	size_t i;
	double avg = 0;
	const int size = points.size();

	for( i = 0; i < size; ++i )
		avg += points[i].y;

	avg /= size;
	double stdev = 0;
	for( i = 0; i < size; ++i )
		stdev += (points[i].y - avg) * (points[i].y - avg);

	return sqrt( stdev/size );
}
void DrawGroundTruthMouthConerPos( FacialFeaturesValidation& features )
{
	Point ptx1, ptx2, pty1, pty2;
	Scalar colour = Scalar::all(255);
	const int offset = 10,
			thickness = 1;

	/// Draw left mouth corner ground truth data
	ptx1 = Point( features.mouth.leftCorner.at( imIt ).x - offset, 
				  features.mouth.leftCorner.at( imIt ).y);
	ptx2 = Point( features.mouth.leftCorner.at( imIt ).x + offset, 
				  features.mouth.leftCorner.at( imIt ).y);
	pty1 = Point( features.mouth.leftCorner.at( imIt ).x, 
				  features.mouth.leftCorner.at( imIt ).y - offset);
	pty2 = Point( features.mouth.leftCorner.at( imIt ).x, 
				  features.mouth.leftCorner.at( imIt ).y + offset);

	line( imgProcessed, ptx1, ptx2, colour, thickness );
	line( imgProcessed, pty1, pty2, colour, thickness );


	/// Draw right mouth corner ground truth data
	ptx1 = Point( features.mouth.rightCorner.at( imIt ).x - offset, 
				  features.mouth.rightCorner.at( imIt ).y);
	ptx2 = Point( features.mouth.rightCorner.at( imIt ).x + offset, 
				  features.mouth.rightCorner.at( imIt ).y);
	pty1 = Point( features.mouth.rightCorner.at( imIt ).x, 
				  features.mouth.rightCorner.at( imIt ).y - offset);
	pty2 = Point( features.mouth.rightCorner.at( imIt ).x, 
				  features.mouth.rightCorner.at( imIt ).y + offset);

	line( imgProcessed, ptx1, ptx2, colour, thickness );
	line( imgProcessed, pty1, pty2, colour, thickness );
}
double getMouthDist( FacialFeaturesValidation& features )
{
	double dx = (features.mouth.leftCorner.at( imIt ).x - features.mouth.rightCorner.at( imIt ).x); 
	double dy = (features.mouth.leftCorner.at( imIt ).y - features.mouth.rightCorner.at( imIt ).y); 
	return (sqrt( dy*dy + dx*dx ));
}
void mouthCornersPositionsMetric( Point2f& left, Point2f& right, FacialFeaturesValidation& features )
{
	double interCornerDist		= features.eyes.IOD.at( imIt );

	double leftCorner_Xerr		= (abs(features.mouth.leftCorner.at( imIt ).x - left.x ) / interCornerDist) * 100;
	double leftCorner_Yerr		= (abs(features.mouth.leftCorner.at( imIt ).y - left.y ) / interCornerDist) * 100;
	double rightCorner_Xerr		= (abs(features.mouth.rightCorner.at( imIt ).x - right.x ) / interCornerDist) * 100;
	double rightCorner_Yerr		= (abs(features.mouth.rightCorner.at( imIt ).y - right.y ) / interCornerDist) * 100;

	// Calculate detection error using Pitagoras equetion
	double leftCorner_err		= square_distance( leftCorner_Xerr, leftCorner_Yerr );
	double rightCorner_err		= square_distance( rightCorner_Xerr, rightCorner_Yerr );

	// Copy validation data to the container
	features.mouth.MCD.at( imIt )				= interCornerDist;
	features.mouth.rightCorner_det.at( imIt )	= right;
	features.mouth.rightCorner_err.at( imIt )	= rightCorner_err;
	features.mouth.leftCorner_det.at( imIt )	= left;
	features.mouth.leftCorner_err.at( imIt )	= leftCorner_err;

	cout << "Mouth validation\tMCD " << setw(4) << interCornerDist
		 << "\tLeft " << setiosflags(ios::basefield) << setiosflags(ios::fixed) << setprecision(1)
		 << setw(4) << leftCorner_err << "%"
		 << "\tRight" << setiosflags(ios::basefield) << setiosflags(ios::fixed) << setprecision(1)
		 << setw(5) << rightCorner_err << "%" 
		 << endl;
}
bool saveMouthCornPosValidationData( FacialFeaturesValidation& features )
{
	string output_file_name = "Results Mouth Validation ";
	output_file_name += convertInt(abs((int)getTickCount()));
	output_file_name += ".csv";
	const string next_cell = ";";
	ofstream out;

	out.open( output_file_name );
	if( !out.good() ){ cerr << "An error occured when saving Mouth Validation data" << endl; return false; }

	out << "Results for Mouth Validation" << endl
		<< "Item;" << "MD;" << "TRLC x;" << "TRLC y;" << "TMLC x;" << "TMLC y;" 
		<< "DMRC x;" << "DMRC y;" << "DMLC x;" << "DMLC y;" << "RCE %;" << "LCE %;" << endl;

	int cnt = features.mouth.size;
	for( int i = 0; i < cnt; ++i )
	{
		out << i << next_cell 
			<< features.mouth.MCD.at(i)					<< next_cell 

			<< features.mouth.rightCorner.at(i).x		<< next_cell
			<< features.mouth.rightCorner.at(i).y		<< next_cell
			
			<< features.mouth.leftCorner.at(i).x		<< next_cell
			<< features.mouth.leftCorner.at(i).y		<< next_cell
			
			<< features.mouth.rightCorner_det.at(i).x	<< next_cell
			<< features.mouth.rightCorner_det.at(i).y	<< next_cell
			
			<< features.mouth.leftCorner_det.at(i).x	<< next_cell
			<< features.mouth.leftCorner_det.at(i).y	<< next_cell

			<< features.mouth.rightCorner_err.at(i)		<< next_cell
			<< features.mouth.leftCorner_err.at(i)		<< next_cell << endl;
	}
	return true;
}
