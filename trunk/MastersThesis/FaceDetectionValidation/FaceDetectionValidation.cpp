#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

const string cascadeFNameFace		= "../data/cascades/haarcascade_frontalface_alt2.xml";
const string IMMFaceDBFile			= "../data/facedb/imm/filelist.txt";
const string ColorFeretDBFile		= "../data/facedb/color feret/filelist.txt";
const string ColorFeretDBFile_fa	= "../data/facedb/color feret/filelist_fa.txt";
const string BioIDDBDile			= "../data/facedb/BioID/filelist.txt";

vector <string> imgFileList;

bool loadFileList( const string fileName )
{
	ifstream in;
	string line;
	imgFileList.clear();

	in.open( fileName );
	if( !in )
	{
		cout << "--(!) Cannot read input file list" << endl;
		return false;
	}

	while( !in.eof() )
	{
		getline(in, line );
		imgFileList.push_back( line );
	}

	if( imgFileList.size() > 0 )
		return true;
	else
	{
		cout << "--(!) Cannot read input file list" << endl;
		return false;
	}
}
double startTime(void)
{
	return (double) getTickCount();
}
double calcExecTime( double* time )
{
	*time = 1000 * ((double)getTickCount() - *time)/ getTickFrequency(); 
	return *time;
}


int main( int argc, char** argv )
{
	bool finish = false;
	bool use_resize = false;
	bool extended_mode = false;
	int found_cnt = 0;
	const string sep = ";";
	char name[255];
	double exec_time,
		   res_time;
	vector <double> execTimes,
					res_times;
	vector <vector <string>> db_files;
	vector <Rect> faces;
	CascadeClassifier cascade;
	Size dst_size;
	Mat imsrc,
		imgray,
		imout,
		imface;
	ofstream csv;

	if( argc == 3 || argc == 4 )
	{
		dst_size.width = atoi( argv[1] );
		dst_size.height = atoi( argv[2] );
		bool(atoi( argv[3] )) ? extended_mode = true : extended_mode = false;
		use_resize = true;
	}
	else if( argc == 1 ) { use_resize = false; }
	else{ std::cout << "Wrong arguments!" << endl; return -1; }

	if( !cascade.load( cascadeFNameFace) ){ printf("--(!)Error loading\n"); return -1; };
	useOptimized() ? (cout << "Optimization enabled") : (cout << "Optimization disabled");
	if( useOptimized() ) cout << endl << "Threads " << getNumThreads() << endl << endl;
	
	loadFileList( IMMFaceDBFile );
	db_files.push_back( imgFileList );

	loadFileList( BioIDDBDile );
	db_files.push_back( imgFileList );
	
	loadFileList( ColorFeretDBFile );
	db_files.push_back( imgFileList );

	sprintf_s( name, 255, "Results %dx%d %d.csv", dst_size.width, dst_size.height, (int)getTickCount() );
	csv.open( name );

	csv  << "Results for " << dst_size.width << "x" << dst_size.height << endl
		<< "Scale" <<  sep << "Images processed" << sep << "Faces found" 
		<< "Percent" << sep
		<< sep << "Avg detection time"
		<< sep << "Avg resize time" << sep << "FPS" << sep << endl;


	for( double scale = 1.05; scale < 1.60; scale += 0.05 )
	{
		std::cout << endl
			 << ">>>>>>>>>>>>>>>>>>>>>>>> Window Scale Factor: " << scale
			 << " <<<<<<<<<<<<<<<<<<<<<<<" << endl;

		for( size_t k = 0; k < db_files.size(); ++k )
		{
			int db_size = db_files[k].size();
			for( size_t i = 0; i < db_size; ++i )
			{
				// show progress
				cout << "\r" << setiosflags(ios::basefield) << setiosflags(ios::fixed) << setprecision(1) 
					 << setw(7) << (float)i*100 / (float)db_size << " %";

				imsrc = imread( db_files[k][i], CV_LOAD_IMAGE_UNCHANGED );
				if( imsrc.channels() > 1 ) cvtColor( imsrc, imgray, CV_RGB2GRAY );
				else imgray = imsrc;
				equalizeHist( imgray, imgray );

				if( use_resize )
				{
					res_time = startTime();
					float dx = (float)dst_size.width / (float)imsrc.cols;
					float dy = (float)dst_size.height / (float)imsrc.rows;
					float resscale;
					( dx >= dy ) ? resscale = dx : resscale = dy;
					resize( imgray, imout, Size(), resscale, resscale, CV_INTER_AREA );
					res_time = calcExecTime( &res_time );
					res_times.push_back( res_time );
				} 
				else imout = imgray;

				faces.clear();
				exec_time = startTime();
		
				cascade.detectMultiScale( imout, faces, scale, 3, CASCADE_FIND_BIGGEST_OBJECT );
		
				exec_time = calcExecTime( &exec_time );
				execTimes.push_back( exec_time );

				if( faces.size() )
				{
					found_cnt++;
					
					if( extended_mode )
					{
						sprintf_s( name, "results\\db-%d im-%d ws-%1.2f.jpg", k+1, i+1, scale );
						Mat imface ( imout, faces[0] );
						imwrite( name, imface );
					}
				}
			}

			// Calculate average Haar time
			double total_time = 0,
				   total_res = 0;
			for( size_t i = 0; i < execTimes.size(); ++i )
			{
				total_time += execTimes[i];
				if( use_resize ) total_res += res_times[i];
			}

			double avg_time = total_time / execTimes.size();
			double avg_res = total_res / res_times.size();


			std::cout << "\r" << setw(10) << db_files[k].size() << setw(5) << found_cnt
				 << setw(10) << setiosflags(ios::basefield) << setiosflags(ios::fixed) << setprecision(1)
				 << ((float)found_cnt / (float)db_files[k].size()) * 100 << "%"
				 << setw(10) << setiosflags(ios::basefield) << setiosflags(ios::fixed) << setprecision(1)
				 << avg_time << " ms"
				 << setw(10) << setiosflags(ios::basefield) << setiosflags(ios::fixed) << setprecision(1)
				 << avg_res << " ms"
				 << setw(10) << setiosflags(ios::basefield) << setiosflags(ios::fixed) << setprecision(1)
				 << (1000/avg_time) << " FPS" << endl;

			csv  << scale << sep
				 << db_files[k].size() << sep
				 << found_cnt << sep
				 << setiosflags(ios::basefield) << setiosflags(ios::fixed) << setprecision(1)
				 << ((float)found_cnt / (float)db_files[k].size()) * 100 << "%" << sep
				 << setiosflags(ios::basefield) << setiosflags(ios::fixed) << setprecision(1)
				 << avg_time << sep
				 << setiosflags(ios::basefield) << setiosflags(ios::fixed) << setprecision(1)
				 << avg_res << sep
				 << setiosflags(ios::basefield) << setiosflags(ios::fixed) << setprecision(1)
				 << (1000/avg_time) << sep << endl;

			execTimes.clear();
			res_times.clear();
			found_cnt = 0;
		}
	}

	csv.close();
	return 0;
}