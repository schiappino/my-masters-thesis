#include "Globals.h"
#include "GUI.h"
#include "FacialFeatures.h"

void DetectEyebrows(void);
void drawEyebrow( Mat img, vector <Point>& eyebrowContour, Point& eyebrowCenter, Point offset );
void blobDetector( Mat src, vector <vector <Point>>& candidates, vector<KeyPoint>& keyPoints );
void getBestEyebrowCadidate( Mat img, vector <vector <Point>>& candidates, vector<KeyPoint>& keyPoints, vector <Point>& bestMatch, Point& center, int flag );