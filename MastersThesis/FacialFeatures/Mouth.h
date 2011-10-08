#include "Globals.h"
#include "GUI.h"
#include "FacialFeatures.h"

void DetectMouth(void);
void cornerDetector( Mat img, vector<Point2f>& corners );
void getBestMouthCornerCadidates( Point2f& left, Point2f& righ, vector <Point2f>& candidates, float tolerance );