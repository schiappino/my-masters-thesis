#include "Globals.h"
#include "FacialFeatures.h"

void getGroundTruthsData( FacialFeaturesValidation& features, const string files, int flag )
{
	if		( flag == FaceDbFlags.IMM )			{ getGroundTruthsIMM( features, files ); }
	else if ( flag == FaceDbFlags.BioID )		{ getGroundTruthsBioID( features, files ); }
	else if ( flag == FaceDbFlags.COLOR_FERET ) { getGroundTruthsColorFeret( features, files ); }
};
void getGroundTruthsIMM( FacialFeaturesValidation& features, const string files ){};
void getGroundTruthsBioID( FacialFeaturesValidation& features, const string files ){};
void getGroundTruthsColorFeret( FacialFeaturesValidation& features, const string files )
{
	string	file_name = "ground truth.txt",
			file_list = "N:/data/colorferet/data/ground_truths/name_value/filelist.txt",
			output_file = "coordinates output.txt";

	ifstream file;
	ofstream output( output_file, ios::out );

	bool is_file_list_loaded = load_file_list( file_list );
	if( !is_file_list_loaded ) { cerr << "Cannot open input file list" << endl; return 1; }

	for( size_t i = 0; i < list_of_files.size(); ++i )
	{
		file.open( list_of_files[i] );
		if( !file.is_open() ){ cerr << "Cannot open input file" << endl; return 1; }
		if( !output.is_open() ){ cerr << "Cannot open output file" << endl; return 1; }

		while( file.good() && output.good() )
		{
			string line;
			getline( file, line );

			size_t found_idx = line.find( "left_eye_coordinates=" );
			if ( !found_idx )
			{
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

				output << "Right eye: " << x_coord << " " << y_coord << endl;
			}

			found_idx = line.find( "right_eye_coordinates=" );
			if ( !found_idx )
			{
				size_t pos1 = line.find( "=" ) + 1;
				size_t pos2 = line.find( " " );
				string s_x = line.substr( pos1, line.size() - pos2 - 1 );
				string s_y = line.substr( pos2+1 );

				istringstream iss_x(s_x);
				istringstream iss_y(s_y);
				int x_coord, y_coord;
				iss_x >> x_coord;
				iss_y >> y_coord;

				output << "Left eye: " << x_coord << " " << y_coord << endl;
			}
		}
		file.close();
	}
	output.close();
};
