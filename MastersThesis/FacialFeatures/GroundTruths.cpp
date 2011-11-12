#include "GroundTruths.h"

void getGroundTruthsData( FacialFeaturesValidation& features, const string files, int flag )
{
	if		( flag == FaceDbFlags::IMM )			{ getGroundTruthsIMM( features, files ); }
	else if ( flag == FaceDbFlags::BioID )		{ getGroundTruthsBioID( features, files ); }
	else if ( flag == FaceDbFlags::COLOR_FERET ) { getGroundTruthsColorFeret( features, files ); }
};
void getGroundTruthsIMM( FacialFeaturesValidation& features, const string files ){};
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
};
