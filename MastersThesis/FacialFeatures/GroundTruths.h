#include "Globals.h"
#include "FacialFeatures.h"

void getGroundTruthsData( FacialFeaturesValidation& features, const string files, int flag );
void getGroundTruthsIMM( FacialFeaturesValidation& features, const string files );
void getGroundTruthsBioID( FacialFeaturesValidation& features, const string files );
void getGroundTruthsColorFeret( FacialFeaturesValidation& features, const string files );
Point getPointFromASFFile( const string file, int point_no );
Point getPointFromPtsFile( const string file, int point_no );