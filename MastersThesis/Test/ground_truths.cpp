#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

using namespace std;

vector <string> list_of_files;

bool load_file_list( const string fileName )
{
	ifstream in;
	string line;

	in.open( fileName );
	if( !in )
	{
		cout << "--(!) Cannot read input file list" << endl;
		return false;
	}

	while( !in.eof() )
	{
		getline(in, line );
		list_of_files.push_back( line );
	}

	if( list_of_files.size() > 0 )
		return true;
	else
	{
		cout << "--(!) Cannot read input file list" << endl;
		return false;
	}
}

int main()
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
	return 0;
};