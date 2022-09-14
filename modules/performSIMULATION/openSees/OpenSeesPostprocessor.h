#ifndef OPENSEES_POSTPROCESSOR_H
#define OPENSEES_POSTPROCESSOR_H
class json_t;
#include <fstream>
using namespace::std;

class OpenSeesPostprocessor {

 public:
  OpenSeesPostprocessor();
  ~OpenSeesPostprocessor();

  int processResults(const char *AIM, const char *SAM, const char *EDP);

  int processEDPs();

 private:
  char *filenameEDP;
  char *filenameAIM;
  char *filenameSAM;  

  json_t *rootEDP;
  json_t *rootSAM;
  json_t *rootAIM;

  double unitConversionFactorLength;
  double unitConversionFactorForce;
  double unitConversionFactorAcceleration;
};

#endif // OPENSEES_POSTPROCESSOR_H
