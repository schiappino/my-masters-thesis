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

		Mat imgMouthHue( hls_planes[HUE_PLANE], foundMouthROI );
		Mat imgMouthThresh ( imgMouthHue.size(), imgMouthHue.type() );
		Mat imgMouthBlurred;
		bilateralFilter( imgMouthHue, imgMouthBlurred, bilatBlurVal, bilatBlurVal*2, bilatBlurVal/2 );
		imshow( wndNameBlur, imgMouthBlurred );
		mouthHueAvg = mean( imgMouthHue );

		mouthThreshold = (int)mouthHueAvg.val[0];
		threshold( imgMouthBlurred, imgMouthThresh, (double) mouthThreshold, 255, THRESH_BINARY_INV );
		imshow( wndNameMouth, imgMouthThresh );

		// User Shi-Tomasi corner detector
		vector <Point2f> cornersCandidates;
		Mat imgMouthCorners;
		imgMouthCorners = imgMouthBlurred.clone();

		RNG rng(startTime());
		cornerDetector( imgMouthBlurred, cornersCandidates );

		Point2f leftCorner, rightCorner;
		getBestMouthCornerCadidates( leftCorner, rightCorner, cornersCandidates, imgMouthBlurred.rows/4.0 );

		for( int i = 0; i < cornersCandidates.size(); i++ )
		{
			circle( imgMouthCorners, cornersCandidates[i], 4, 
			Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255)), -1, 8, 0 ); 
		}
		Mat imgProcessedMouthCorners ( imgProcessed, foundMouthROI );
		circle( imgProcessedMouthCorners, leftCorner, 4, CV_RGB(0,0,255), -1, 8, 0 ); 
		circle( imgProcessedMouthCorners, rightCorner, 4, CV_RGB(0,0,255), -1, 8, 0 ); 
		imshow( wndNameCorners, imgMouthCorners );
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
void getBestMouthCornerCadidates( Point2f& left, Point2f& right, vector <Point2f>& candidates, float tolerance )
{
	size_t ptsCnt = candidates.size();
	size_t i;
	float avg = 0;

	if( ptsCnt < 2 )
		return;

	// Calculate avarage y coord
	for( i = 0; i < ptsCnt; ++i )	
		avg += candidates[i].y;

	avg /= ptsCnt;
	float avgUp = avg + tolerance;
	float avgDown = avg - tolerance;

	float min = numeric_limits<float>::max();
	float max = 0;
	size_t leftIdx, rightIdx;
	for( i = 0; i < ptsCnt; ++i )
	{
		if( candidates[i].x < min )
		{
			// Look for left corner
			if( candidates[i].y < avgUp && candidates[i].y > avgDown )
			{
				min = candidates[i].x;
				leftIdx = i;
			}
		}
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
	left = candidates[leftIdx];
	right = candidates[rightIdx];
};