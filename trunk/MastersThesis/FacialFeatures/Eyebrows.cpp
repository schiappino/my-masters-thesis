#include "Eyebrows.h"
#include <iterator>

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
		
		vector <vector <Point>> eyebrowCandidates;
		vector <KeyPoint> keyPoints;
		vector <Point> bestMatch;
		Point offset,
			  bestMatchCenter;
		
		offset.x = eyesbrowsROI.x + eyesbrowsROI.width*0.1;
		offset.y = eyesbrowsROI.y;
		blobDetector( leftEybrowSmoothedRed, eyebrowCandidates, keyPoints );
		getBestEyebrowCadidate( leftEybrowSmoothedRed, eyebrowCandidates, keyPoints, 
			bestMatch, bestMatchCenter, EyebrowCandidateFlags::LEFT );
		drawEyebrow( imgProcessed, bestMatch, bestMatchCenter, offset );

		offset.x = eyesbrowsROI.x + eyesbrowsROI.width*0.5;
		blobDetector( rightEybrowSmoothedRed, eyebrowCandidates, keyPoints );
		getBestEyebrowCadidate( leftEybrowSmoothedRed, eyebrowCandidates, keyPoints, 
			bestMatch, bestMatchCenter, EyebrowCandidateFlags::RIGHT );
		drawEyebrow( imgProcessed, bestMatch, bestMatchCenter, offset );

		imshow( wndNameFace, imgProcessed );
	}
};
void drawEyebrow( Mat img, vector <Point>& eyebrowContour, Point& eyebrowCenter, Point offset = Point() )
{	
	// Assume that eyebrow is some % of face size
	int ellipseX = 0.15 * face_size,
		ellipseY = 0.04 * face_size;
	Point center = eyebrowCenter;

	// Update interest point by offset
	if ( offset.x || offset.y )
	{
		center.x += offset.x;
		center.y += offset.y;
	}
	ellipse( img, center, Size(ellipseX,ellipseY), 0, 220, 310, CV_RGB(0,0,255), 4 );
}
void blobDetector( Mat src, vector <vector <Point>>& candidates, vector<KeyPoint>& keyPoints )
{
	candidates.clear();
	keyPoints.clear();

	Mat out;
	vector <vector <Point>> contours;

	SimpleBlobDetector::Params params;
	params.minThreshold = 30;
	params.maxThreshold = 60;
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
void getBestEyebrowCadidate( Mat img, vector <vector <Point>>& candidates, 
							 vector<KeyPoint>& keyPoints, vector <Point>& bestMatch,
							 Point& center, int flag )
{
	bestMatch.clear();
	int candCnt = candidates.size();

	// When only on candidate is avaliable we simply return it
	if ( candCnt == 1 )
	{
		copy( candidates[0].begin(), candidates[0].end(), back_inserter(bestMatch));
		center = keyPoints[0].pt;
		return;
	}

	// Case when there is more than one candidate
	int regionHalfWidth = img.size().width/2.0,
		regionHalfHeight = img.size().height/2.0;
	float dy_tmp,
		  dy = numeric_limits<float>::max();
	size_t idx;
	size_t yBestMatchCandidateIdx;
	for( idx = 0; idx < candCnt; ++idx )
	{
		dy_tmp = abs( regionHalfHeight - keyPoints[idx].pt.y );
		if ( dy_tmp < dy )
		{
			dy = dy_tmp;
			yBestMatchCandidateIdx = idx;
		}
	}

	float dx = 0;
	size_t xBestMatchCandidateIdx;
	// We assume that left eyebrow will be to the most right
	// posiotion as on most left there will be shadows of hair and etc.
	if ( flag == EyebrowCandidateFlags::LEFT )
	{
		for( idx = 0; idx < candCnt; ++idx )
		{
			if ( dx < keyPoints[idx].pt.x )
			{
				dx = keyPoints[idx].pt.x;
				xBestMatchCandidateIdx = idx;
			}
		}
	}
	else if ( flag == EyebrowCandidateFlags::RIGHT )
	{
		dx = numeric_limits<float>::max();
		for( idx = 0; idx < candCnt; ++idx )
		{
			if ( dx > keyPoints[idx].pt.x )
			{
				dx = keyPoints[idx].pt.x;
				xBestMatchCandidateIdx = idx;
			}
		}
	}
	else { return; }

	if ( xBestMatchCandidateIdx != yBestMatchCandidateIdx )
	{
		copy( candidates[xBestMatchCandidateIdx].begin(), candidates[xBestMatchCandidateIdx].end(), back_inserter(bestMatch));
		center = keyPoints[xBestMatchCandidateIdx].pt;
	}
	else
	{
		copy( candidates[xBestMatchCandidateIdx].begin(), candidates[xBestMatchCandidateIdx].end(), back_inserter(bestMatch));
		center = keyPoints[xBestMatchCandidateIdx].pt;
	}

	return;
}