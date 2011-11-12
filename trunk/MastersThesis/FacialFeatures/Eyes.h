#include "Globals.h"
#include "FacialFeatures.h"

void DetectEyes(void);
Point EyeTemplateMatching( Mat src, Mat disp, Mat templ, int irisRadius);
void DrawGroundTruthEyePos();
inline double getInterocularDist( FacialFeaturesValidation& features );
void eyePositionsMetric( Point& left, Point& right, FacialFeaturesValidation& features );