#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <cctype>
#include "HazusSAM_Generator.h"
#include "Building.h"
#include <cstring>
#include "whereami.h"
#include <jansson.h>


int main(int argc, char **argv)
{
  char *filenameAIM = 0;
  char *filenameEVENT = 0;
  char *filenameSAM = 0;

  // get path to exe
  int dirname_length = 0;
  int length = wai_getExecutablePath(NULL, 0, NULL);
  char* filenameHazusData = (char*)malloc(length + 20);
  wai_getExecutablePath(filenameHazusData, length, &dirname_length);
  strncpy(&filenameHazusData[dirname_length],"/data/HazusData.txt",19);
  filenameHazusData[dirname_length+19] = '\0';

  double stdStiffness = 0.1;
  double stdDamping = 0.1;
  bool getRV = false;

  int arg = 1;
  while (arg < argc) {

    if ((strcmp(argv[arg], "-filenameAIM") ==0) || 
	(strcmp(argv[arg], "--filenameAIM") ==0) ||
	(strcmp(argv[arg], "-filenameAIM") ==0) ||
	(strcmp(argv[arg], "--filenameAIM") ==0)) {	
      arg++;
      filenameAIM = argv[arg];      
    }
    else if ((strcmp(argv[arg], "-filenameEVENT") == 0) ||
	     (strcmp(argv[arg], "--filenameEVENT") == 0)) {
      arg++;
      filenameEVENT = argv[arg];
    }
    else if ((strcmp(argv[arg], "-filenameSAM") == 0) ||
	     (strcmp(argv[arg], "--filenameSAM") == 0)) {
      arg++;
      filenameSAM = argv[arg];
    }
    else if ((strcmp(argv[arg], "-hazusData") == 0) ||
	     (strcmp(argv[arg], "--hazusData") == 0)) {
      arg++;
      filenameHazusData = argv[arg];
    }
    else if ((strcmp(argv[arg], "-stdStiffness") == 0) ||
	     (strcmp(argv[arg], "--stdStiffness") == 0)) {
      arg++;
      stdStiffness = atof(argv[arg]);
    }
    else if ((strcmp(argv[arg], "-stdDamping") == 0) ||
	     (strcmp(argv[arg], "--stdDamping") == 0)) {
      arg++;
      stdDamping = atof(argv[arg]);
    }
    else if ((strcmp(argv[arg], "-getRV") == 0) ||
	     (strcmp(argv[arg], "--getRV") == 0)) {
      getRV = true;
    }
    arg++;
  }
  json_error_t error;
  json_t *rootBIM = json_load_file(filenameAIM, 0, &error);  
  json_t *modType = json_object_get(rootBIM,"Modeling");
  if (modType != NULL) {
    json_t *hType = json_object_get(modType,"hazusData");
    json_t *dType = json_object_get(modType,"stdDamping");
    json_t *kType = json_object_get(modType,"stdStiffness");      
    if (dType != NULL) {
      stdDamping = json_number_value(dType);
    }
    if (kType != NULL) {
      stdStiffness = json_number_value(kType);
    }
    if (hType != NULL) {
      const char *filename = json_string_value(hType);
      strcpy(filenameHazusData, filename);
    }
  }
    
  HazusSAM_Generator* aim = new HazusSAM_Generator(filenameHazusData);
  Building *theBuilding = new Building();
  
  if(getRV == true) {
    theBuilding->readBIM(filenameEVENT, filenameAIM);
    theBuilding->writeRV(filenameSAM, stdStiffness, stdDamping);
  } else {
    theBuilding->readBIM(filenameEVENT, filenameAIM, filenameSAM);
    aim->CalcBldgPara(theBuilding);
    theBuilding->writeSAM(filenameSAM);
  }

  delete aim;
  return 0;
}

