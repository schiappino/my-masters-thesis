#include "Globals.h"
#include "FacialFeatures.h"

void DetectEyebrows(void);
void drawEyebrow( Mat img, vector <vector <Point>>& eyebrowCandidates, Point offset );
void BlobDetector( Mat src, vector <vector <Point>>& candidates );