#include "Eyebrows.h"

void DetectEyebrows()
{
	if( faces.size() )
	{
		Rect eyesbrowsROI	 = Rect( faces[0].x,							(int)(faces[0].y + 0.2*faces[0].height), 
									 faces[0].width,						(int)(0.2*faces[0].height) );

		Rect eyebrowLeftROI	 = Rect( (int)(faces[0].x + 0.1*faces[0].width),(int)(faces[0].y + 0.2*faces[0].height), 
									 (int)(0.4*faces[0].width),				(int)(0.2*faces[0].height) );

		Rect eyebrowRightROI = Rect( (int)(faces[0].x + 0.5*faces[0].width),(int)(faces[0].y + 0.2*faces[0].height), 
									 (int)(0.4*faces[0].width),				(int)(0.2*faces[0].height) );

		Mat imgEyebrows		( rgb_planes[0], eyesbrowsROI ),
			imgEyebrowLeft	( rgb_planes[0], eyebrowLeftROI ),
			imgEyebrowRight	( rgb_planes[0], eyebrowRightROI ),
			imgGrayEyebrows ( imgGray,		 eyesbrowsROI );

		Mat imgEyebrowsRed;

		equalizeHist( imgEyebrowLeft, imgEyebrowLeft );
		equalizeHist( imgEyebrowRight, imgEyebrowRight );

		imgEyebrows.copyTo( imgEyebrowsRed );

		bitwise_not( imgEyebrowLeft, imgEyebrowLeft );
		bitwise_not( imgEyebrowRight, imgEyebrowRight );

		exponentialOperator( imgEyebrowLeft, imgEyebrowLeft );
		exponentialOperator( imgEyebrowRight, imgEyebrowRight );

		Mat leftSmoothed, rightSmoothed, redSmoothed,
			leftThresh, rightThresh, grayThresh, redThresh;
		int b = bilatBlurVal;

		bilateralFilter( imgEyebrowLeft, leftSmoothed, b, 2*b, b/2. );
		bilateralFilter( imgEyebrowRight, rightSmoothed, b, 2*b, b/2. );
		bilateralFilter( imgEyebrowsRed, redSmoothed, b, 2*b, b/2. );

		imshow( "Eyebrows Red channel smooth", redSmoothed );
		imshow( "Left eyebrow smoothed", leftSmoothed );
		imshow( "Right eyebrow smoothed", rightSmoothed );

		Mat leftEybrowSmoothedRed ( redSmoothed, Rect(redSmoothed.size().width*0.1, 0, redSmoothed.size().width*0.4, redSmoothed.size().height ));
		Mat rightEybrowSmoothedRed ( redSmoothed, Rect(redSmoothed.size().width*0.5, 0, redSmoothed.size().width*0.4, redSmoothed.size().height ));
		
		vector < vector <Point>> eyebrowCandidates;
		vector<KeyPoint> keyPoints;
		Point offset;
		
		offset.x = eyesbrowsROI.x + eyesbrowsROI.width*0.1;
		offset.y = eyesbrowsROI.y;
		blobDetector( leftEybrowSmoothedRed, eyebrowCandidates, keyPoints );
		drawEyebrow( imgProcessed, eyebrowCandidates, keyPoints, offset );

		offset.x = eyesbrowsROI.x + eyesbrowsROI.width*0.5;
		blobDetector( rightEybrowSmoothedRed, eyebrowCandidates, keyPoints );
		drawEyebrow( imgProcessed, eyebrowCandidates, keyPoints, offset );

		imshow( wndNameFace, imgProcessed );
	}
};
void drawEyebrow( Mat img, vector <vector <Point>>& eyebrowCandidates, vector<KeyPoint>& keyPoints, Point offset = Point() )
{
	// Assume that longest candidate is best match for eyebrow
	// Search for best match candidate Index
	size_t bestMatchCandidateIdx;
	if( eyebrowCandidates.size() > 1 )
	{
		float maxArcLen = 0;
		for( size_t i = 0; i < eyebrowCandidates.size(); ++i )
		{
			float curArcLen = arcLength( Mat(eyebrowCandidates[i]), false );
			if( curArcLen > maxArcLen )
			{
				maxArcLen = curArcLen;
				bestMatchCandidateIdx = i;
			}
		}
	} else 
		bestMatchCandidateIdx = 0;

	// Make candidate copy just for convienance
	vector <Point> candidate (eyebrowCandidates[bestMatchCandidateIdx]);

	// Search for min and max X-axis coord of candidate
	int pointsCnt = candidate.size();
	int left_tmp = numeric_limits<int>::max(),
		right_tmp = 0;
	size_t leftIdx,
		   rightIdx;;
	for( size_t i = 0; i < pointsCnt; ++i )
	{
		if( candidate[i].x < left_tmp )
		{
			leftIdx = i;
			left_tmp = candidate[i].x;
		}
		if( candidate[i].x > right_tmp )
		{
			rightIdx = i;
			right_tmp = candidate[i].x;
		}
	}
	Point leftMostPoint (candidate[leftIdx]),
		 rightMostPoint (candidate[rightIdx]);


	// Having left- and righ-most X-axis coords look for middple coords
	// Start by creating copy of candidate
	int middlePointX = cvRound( (leftMostPoint.x + rightMostPoint.x) / 2 );
	vector <int> y_coords;
	for( size_t i = 0; i < pointsCnt; ++i )
	{
		int cur_x = candidate[i].x;
		if( cur_x > 0. *middlePointX && cur_x < 1.2*middlePointX )
			y_coords.push_back( candidate[i].y);
	}
	int y_tmp_coord = 0;
	for( size_t i = 0; i < y_coords.size(); ++i )
		y_tmp_coord += y_coords[i];

	int middlePointY = cvRound( y_tmp_coord/y_coords.size() );
	Point midllePoint (middlePointX, middlePointY);

	// Update interest point by offset
	if ( offset.x || offset.y )
	{
		leftMostPoint.x += offset.x;
		leftMostPoint.y += offset.y;
		rightMostPoint.x += offset.x;
		rightMostPoint.y += offset.y;
		midllePoint.x += offset.x;
		midllePoint.y += offset.y;
	}

	// draw eyebrow lines
	line( img, leftMostPoint, midllePoint, CV_RGB(0,0,255), 4 );
	line( img, rightMostPoint, midllePoint, CV_RGB(0,0,255), 4 );
}
void blobDetector( Mat src, vector <vector <Point>>& candidates, vector<KeyPoint>& keyPoints )
{
	candidates.clear();
	keyPoints.clear();

	Mat out;
	vector <vector <Point>> contours;

	SimpleBlobDetector::Params params;
	params.minThreshold = 50;
	params.maxThreshold = 100;
	params.thresholdStep = 5;

	params.minArea = 100; 
	params.minConvexity = 0.3;
	params.minInertiaRatio = 0.01;

	params.maxArea = 8000;
	params.maxConvexity = 10;

	params.filterByArcLength = true;
	params.minArcLen = 100;
	params.maxArcLen = 400;

	params.filterByColor = false;
	params.filterByCircularity = false;

	SimpleBlobDetector blobs( params );
	blobs.create("SimpleBlob");

 	blobs.detectEx( src, keyPoints, contours, Mat() );
	drawKeypoints( src, keyPoints, out, CV_RGB(0,255,0), DrawMatchesFlags::DEFAULT );
	candidates.resize( contours.size() );

	for( int i = 0; i < contours.size(); ++i )
	{
		approxPolyDP( Mat(contours[i]), candidates[i], 3, 1 );
		drawContours( out, contours, i, CV_RGB(rand()&255, rand()&255, rand()&255) );
		drawContours( out, candidates, i, CV_RGB(rand()&255, rand()&255, rand()&255) );
	}
	cout << "DEBUG Keypoints " << keyPoints.size() << endl;

	imshow( wndNameBlobs, out );
};