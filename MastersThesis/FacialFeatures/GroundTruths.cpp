#include "GroundTruths.h"

void getGroundTruthsData( FacialFeaturesValidation& features, const string files, int flag )
{
	if		( flag == FaceDbFlags::IMM )			{ getGroundTruthsIMM( features, files ); }
	else if ( flag == FaceDbFlags::BioID )		{ getGroundTruthsBioID( features, files ); }
	else if ( flag == FaceDbFlags::COLOR_FERET ) { getGroundTruthsColorFeret( features, files ); }
};
void getGroundTruthsIMM( FacialFeaturesValidation& features, const string files )
{
	vector <string> fileList;
	ifstream file;

	bool isFileListLoaded = loadFileList( files.c_str(), fileList );
	if( !isFileListLoaded ) { cerr << "Cannot open input file list: " << files << endl; }

	Point tmp1, tmp2, feature_point;
	for( size_t i = 0; i < fileList.size(); ++i )
	{
		// Left eye coordinates
		tmp1 = getPointFromASFFile( fileList[i], IMMDbAnnotationPoints::LEFT_EYE_LMID );
		tmp2 = getPointFromASFFile( fileList[i], IMMDbAnnotationPoints::LEFT_EYE_UMID );
		feature_point = Point( (tmp1.x + tmp2.x)/2, (tmp1.y + tmp2.y)/2 );
		features.eyes.left.push_back( feature_point );

		// Right eye coordinates
		tmp1 = getPointFromASFFile( fileList[i], IMMDbAnnotationPoints::RIGHT_EYE_LMID );
		tmp2 = getPointFromASFFile( fileList[i], IMMDbAnnotationPoints::RIGHT_EYE_UMID);
		feature_point = Point( (tmp1.x + tmp2.x)/2, (tmp1.y + tmp2.y)/2 );
		features.eyes.right.push_back( feature_point );

		// Left mouth corner coordinates
		feature_point = getPointFromASFFile( fileList[i], IMMDbAnnotationPoints::MOUTH_LEFT_COR );
		features.mouth.leftCorner.push_back( feature_point );

		// Right mouth corner coordinates
		feature_point = getPointFromASFFile( fileList[i], IMMDbAnnotationPoints::MOUTH_RIGHT_COR );
		features.mouth.rightCorner.push_back( feature_point );

		// Left eyebrow centre point coordinates
		feature_point = getPointFromASFFile( fileList[i], IMMDbAnnotationPoints::LEFT_EYEBROW );
		features.eyebrow.left.push_back( feature_point );

		// Right eyebrow centre point coordinates
		feature_point = getPointFromASFFile( fileList[i], IMMDbAnnotationPoints::RIGHT_EYEBROW );
		features.eyebrow.right.push_back( feature_point );
	}
	// Asserts for eyes
	if (features.eyes.left.size() == features.eyes.right.size() )
	{ 
		features.eyes.size = features.eyes.left.size(); 
		features.eyes.IOD.resize( features.eyes.size );
		features.eyes.left_det.resize( features.eyes.size );
		features.eyes.left_err.resize( features.eyes.size );
		features.eyes.right_det.resize( features.eyes.size );
		features.eyes.right_err.resize( features.eyes.size );
	}
	else { cerr << "IMM ground truth parser: number of left & right eyes does not match" << endl; };

	if (features.eyebrow.left.size() == features.eyebrow.right.size() )
	{ features.eyebrow.size = features.eyebrow.left.size();	}
	else { cerr << "IMM ground truth parser: number of left & right eyebrows does not match" << endl; };

	if (features.mouth.leftCorner.size() == features.mouth.rightCorner.size() )
	{ features.mouth.size = features.mouth.leftCorner.size();	}
	else { cerr << "IMM ground truth parser: number of left & right mouth corners does not match" << endl; };
};
void getGroundTruthsBioID( FacialFeaturesValidation& features, const string files ){};
void getGroundTruthsColorFeret( FacialFeaturesValidation& features, const string files )
{
	const string RIGHT_EYE_PARAM_NAME = "left_eye_coordinates=",
				 LEFT_EYE_PARAM_NAME = "right_eye_coordinates=";
	Point leftEyePoint, rightEyePoint;
	bool leftEyeCoordsFound = false,
		rightEyeCoordsFound = false;
	vector <string> fileList;
	ifstream file;

	bool isFileListLoaded = loadFileList( files.c_str(), fileList );
	if( !isFileListLoaded ) { cerr << "Cannot open input file list: " << files << endl; }

	for( size_t i = 0; i < fileList.size(); ++i )
	{
		leftEyeCoordsFound = false;
		rightEyeCoordsFound = false;

		file.open( fileList[i] );
		if( !file.is_open() ){ cerr << "Cannot open input file: " << fileList[i] << endl; }

		while( file.good() )
		{
			string line;
			getline( file, line );

			size_t found_idx = line.find( LEFT_EYE_PARAM_NAME );
			if ( !found_idx )
			{
				leftEyeCoordsFound = true;
				found_idx = numeric_limits<int>::max();

				size_t pos1 = line.find( "=" ) + 1;
				size_t pos2 = line.find( " " );
				string s_x = line.substr( pos1, line.size() - pos2 - 1 );
				string s_y = line.substr( pos2+1 );

				istringstream iss_x(s_x);
				istringstream iss_y(s_y);
				int x_coord, y_coord;
				iss_x >> x_coord;
				iss_y >> y_coord;

				leftEyePoint = Point ( x_coord, y_coord );
				features.eyes.left.push_back( leftEyePoint );
			}


			found_idx = line.find( RIGHT_EYE_PARAM_NAME );
			if ( !found_idx )
			{
				rightEyeCoordsFound = true;

				size_t pos1 = line.find( "=" ) + 1;
				size_t pos2 = line.find( " " );
				string s_x = line.substr( pos1, line.size() - pos2 - 1 );
				string s_y = line.substr( pos2+1 );

				istringstream iss_x(s_x);
				istringstream iss_y(s_y);
				int x_coord, y_coord;
				iss_x >> x_coord;
				iss_y >> y_coord;

				rightEyePoint = Point ( x_coord, y_coord );
				features.eyes.right.push_back( rightEyePoint );
			}
		}
		file.close();
		
		// If there were no eye coordinate in ground truth file
		// add zero valued points to the list (0,0)
		if( !leftEyeCoordsFound ) 
		{ 
			rightEyePoint = Point();
			features.eyes.right.push_back( rightEyePoint ); 
		}
		if( !rightEyeCoordsFound ) 
		{ 
			leftEyePoint = Point();
			features.eyes.left.push_back( leftEyePoint ); 
		}
	}

	// Resize all other fields in container
	features.eyes.size = imgFileList.size();
	features.eyes.IOD.resize( features.eyes.size );
	features.eyes.left_det.resize( features.eyes.size );
	features.eyes.right_det.resize( features.eyes.size );
	features.eyes.left_err.resize( features.eyes.size );
	features.eyes.right_err.resize( features.eyes.size );
};

// Low level parsing function for IMM db ASF files
Point getPointFromASFFile( const string file, int point_no )
{
	ifstream gt;
	string line;
	string sx, sy;
	int line_no = 17 + point_no;
	const int IMG_WIDTH = 640,
			 IMG_HEIGHT = 480;

	gt.open( file, ifstream::in );
	if( gt.is_open() )
	{
		for( int i = 0; (i < line_no) && gt.good(); ++i )
			getline( gt, line );

		gt.close();
		sx = line.substr(6,10); 
		sy = line.substr(18,10);

		return Point( cvRound(atof(sx.c_str())*IMG_WIDTH), cvRound(atof(sy.c_str())*IMG_HEIGHT) );
	}
	else
		return Point(-1,-1);
}