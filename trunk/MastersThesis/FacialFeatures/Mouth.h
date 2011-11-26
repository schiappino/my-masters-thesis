#include "Globals.h"
#include "GUI.h"
#include "FacialFeatures.h"

void DetectMouth(void);
void cornerDetector( Mat img, vector<Point2f>& corners );
void getBestMouthCornerCadidates( Point2f& left, Point2f& righ, vector <Point2f>& candidates );
double stdev_vertical( vector <Point2f>& points );
void DrawGroundTruthMouthConerPos( FacialFeaturesValidation& features );
double getMouthDist( FacialFeaturesValidation& features );
void mouthCornersPositionsMetric( Point2f& left, Point2f& right, FacialFeaturesValidation& features );
bool saveMouthCornPosValidationData( FacialFeaturesValidation& features );