#ifndef OPENSEES_POSTPROCESSOR_H
#define OPENSEES_POSTPROCESSOR_H
class json_t;
#include <fstream>
using namespace::std;

class OpenSeesPostprocessor {

 public:
  OpenSeesPostprocessor();
  ~OpenSeesPostprocessor();

  int processResults(const char *BIM, const char *SAM, const char *EDP);

  int processEDPs();

 private:
  char *filenameEDP;
  char *filenameBIM;
  char *filenameSAM;  

  json_t *rootEDP;
  json_t *rootSAM;
  json_t *rootBIM;

  double unitConversionFactorLength;
  double unitConversionFactorForce;
  double unitConversionFactorAcceleration;
};

#endif // OPENSEES_POSTPROCESSOR_H
