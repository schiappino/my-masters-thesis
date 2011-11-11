#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

int main()
{
	for( int i = 0; i < 100; ++i )
		cout << "\r" << i;
	return 0;
}