#include "Globals.h"
#include "FacialFeatures.h"

void DetectEyebrows(void);
void drawEyebrow( Mat img, vector <vector <Point>>& eyebrowCandidates, vector<KeyPoint>& keyPoints, Point offset );
void blobDetector( Mat src, vector <vector <Point>>& candidates, vector<KeyPoint>& keyPoints );
void getBestEyebrowCadidate( Mat img, vector <vector <Point>>& candidates, vector<KeyPoint>& keyPoints, vector <Point>& bestMatch, int flag );