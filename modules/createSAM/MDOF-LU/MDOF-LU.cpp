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


int main(int argc, char **argv)
{
  char *filenameBIM = 0;
  char *filenameEVENT = 0;
  char *filenameSAM = 0;

  // get path to exe
  int dirname_length = 0;
  int length = wai_getExecutablePath(NULL, 0, NULL);
  char* filenameHazusData = (char*)malloc(length + 20);
  wai_getExecutablePath(filenameHazusData, length, &dirname_length);
  strncpy(&filenameHazusData[dirname_length],"/data/HazusData.txt",19);
  filenameHazusData[dirname_length+19] = '\0';

  //  std::cerr << "My path: " <<  filenameHazusData << "\n";
  
  double stdStiffness = 0.1;
  double stdDamping = 0.1;
  bool getRV = false;

  int arg = 1;
  while (arg < argc) {

    if ((strcmp(argv[arg], "-filenameBIM") ==0) || 
	(strcmp(argv[arg], "--filenameBIM") ==0) ||
	(strcmp(argv[arg], "-filenameAIM") ==0) ||
	(strcmp(argv[arg], "--filenameAIM") ==0)) {	
      arg++;
      filenameBIM = argv[arg];      
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
  
  
  HazusSAM_Generator* aim = new HazusSAM_Generator(filenameHazusData);
  Building *theBuilding = new Building();
  
  
  if(getRV == true) {
    theBuilding->readBIM(filenameEVENT, filenameBIM);
    theBuilding->writeRV(filenameSAM, stdStiffness, stdDamping);
  } else {
    theBuilding->readBIM(filenameEVENT, filenameBIM, filenameSAM);
    aim->CalcBldgPara(theBuilding);
    theBuilding->writeSAM(filenameSAM);
  }

  delete aim;
  return 0;
}

