#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>

#include <jansson.h>  // for Json
#include <Units.h>


#include <python2.7/Python.h>

typedef struct tapData {
  double locX;
  double locY;
  int face;
  int id;
  double *forces;
  double *moments;
} TAP;


//
// some functions define at end of file
//

TAP *findNearestTAP(TAP *, int numTaps,  double locX, double locY, int face);
int addForcesFace(TAP *theTAPS, int numTaps, double height, double length, int numDivisonX, int numDivisonY, int face, int numFloors);
int addEvent(json_t *input, json_t *currentEvent, json_t *outputEvent, bool getRV);

int 
callLowRise_TPU(const char *shape,   
		const char *heightBreadth, 
		const char *depthBreadth, 
		const char *pitch,
		int incidenceAngle,
		const char *outputFilename);

int
main(int argc, char **argv) {

  //
  // parse input args for filenames
  //

  char *filenameBIM = NULL;   // inputfile
  char *filenameEVENT = NULL; // outputfile

  bool doRV = false;

  int arg = 1;
  while(arg < argc) {
    if (strcmp(argv[arg], "--filenameBIM") == 0) {
      arg++;
      filenameBIM = argv[arg];
    }
    else if (strcmp(argv[arg], "--filenameEVENT") == 0) {
      arg++;
      filenameEVENT = argv[arg];
    }
    else if (strcmp(argv[arg], "--getRV") == 0) {
      doRV = true;
    }
    arg++;
  }

  if (filenameBIM == NULL || filenameEVENT == NULL) {
    std::cerr << "FATAL ERROR - no bim or sam file provided\n";
  }

  //
  // create output JSON object for EVENTs and RVs
  //

  json_t *outputEventsArray = json_array(); 
  json_t *rvArray = json_array(); 

  //
  // load INPUT file, loop over events
  //

  json_error_t error;
  json_t *input = json_load_file(filenameBIM, 0, &error);
  if (input == NULL) {
    std::cerr << "FATAL ERROR - input file does not exist or not a JSON file\n";
    std::cerr << filenameBIM;
    exit(-1);
  }
  
  json_t *generalInformation = json_object_get(input, "GeneralInformation");  
  json_t *inputEventsArray = json_object_get(input, "Events");  
  if (generalInformation == NULL || inputEventsArray == NULL) {
    std::cerr << "FATAL ERROR - input file conatins no Events key-pair\n";
    exit(-1);
  }
  
  // parse each event in input:
  int index;
  json_t *inputEvent;
  
  int numEDP = 0;

  json_array_foreach(inputEventsArray, index, inputEvent) {
    
    json_t *type = json_object_get(inputEvent,"type");
    const char *eventType = json_string_value(type);

    if (strcmp(eventType,"LowRiseTPU") != 0) {
      
      json_array_append(outputEventsArray, inputEvent); 
      
    } else {

      // create output event
      json_t *outputEvent = json_object();
      json_object_set(outputEvent,"type", json_string("Wind"));
      json_object_set(outputEvent, "subtype", json_string("LowRiseTPU"));
      
      /* set metric units for event */
      // set units to be same as BIM 
      json_t *units = json_object();
      json_object_set(units,"force",json_string("KN"));
      json_object_set(units,"length",json_string("m"));
      json_object_set(units,"time",json_string("sec"));
      json_object_set(outputEvent,"units",units);

      // call function to fill in event details .. depends on getRV flag what is acually done
      addEvent(generalInformation, inputEvent, outputEvent, doRV);

      json_array_append(outputEventsArray, outputEvent);
    }
  }

  // write the variables & events                                             
  json_t *rootEvent = json_object();
  json_object_set(rootEvent,"randomVariables",rvArray);
  json_object_set(rootEvent,"Events",outputEventsArray);

  // dump the event file
  json_dump_file(rootEvent,filenameEVENT,0);

  // done
  return 0;
}

int addEvent(json_t *generalInfo, json_t *currentEvent, json_t *outputEvent, bool getRV) {

    if (getRV == false) {

      //First let's read units from bim
      json_t* bimUnitsJson = json_object_get(generalInfo, "units");
      json_t* bimLengthJson = json_object_get(bimUnitsJson, "length");
      if (bimUnitsJson == 0 || bimLengthJson == 0) {
	std::cerr << "ERROR no Length Units in GeneralInformation\n";
	return -1;
      }

      Units::UnitSystem bimUnits;
      bimUnits.lengthUnit = Units::ParseLengthUnit(json_string_value(bimLengthJson));

      /*
      Units::UnitSystem eventUnits;
      eventUnits.lengthUnit = Units::ParseLengthUnit("m");
      double lengthUnitConversion = Units::GetLengthFactor(bimUnits, eventUnits);
      */

      double lengthUnitConversion = 1.0;
      
      // get info needed form general info
      
      json_t *heightJO = json_object_get(generalInfo,"height");
      json_t *widthJO = json_object_get(generalInfo,"width");
      json_t *depthJO = json_object_get(generalInfo,"depth");
      json_t *storiesJO = json_object_get(generalInfo,"stories");
      
      if (heightJO == NULL ||
	  widthJO == NULL  ||
	  depthJO == NULL  ||
	  storiesJO == NULL ) {
	std::cerr << "ERROR missing Information from GeneralInformation (height, width, stories all neeed)\n";
	return -2;        
      }
      
      // convert to metric as that is what TPU data is in
      
      int numFloors = json_integer_value(storiesJO);
      double height = json_number_value(heightJO) * lengthUnitConversion;
      double breadth = json_number_value(widthJO) * lengthUnitConversion;
      double depth = json_number_value(depthJO)   * lengthUnitConversion;
      
      //
      // load the json file obtained from TPU with the --getRV flag
      //
      
      json_error_t error;
      json_t *tapData = json_load_file("tmpSimCenterLowRiseTPU.json", 0, &error);
      
      if (tapData == NULL) {
	std::cerr << "FATAL ERROR - json file  does not exist\n";
	return -3;
      }

      //First let's read units from bim
      json_t* modelUnitsJson = json_object_get(tapData, "units");
      json_t* modelLengthJson = json_object_get(modelUnitsJson, "length");

      Units::UnitSystem modelUnits;
      modelUnits.lengthUnit = Units::ParseLengthUnit(json_string_value(bimLengthJson));
      double lengthModelUnitConversion = Units::GetLengthFactor(modelUnits, bimUnits);

      json_t *roofTypeT = json_object_get(tapData,"roofType");
      json_t *modelHeightT = json_object_get(tapData,"height");
      json_t *modelDepthT = json_object_get(tapData,"depth");
      json_t *modelBreadthT = json_object_get(tapData,"breadth");
      if (roofTypeT == NULL || modelHeightT == NULL || modelDepthT == NULL || modelBreadthT == NULL) {
	std::cerr << "FATAL ERROR - json file does not contain roofType, height, depth or breadth data \n";
	return -3;
      }

      const char *roofType = json_string_value(roofTypeT);
      double modelHeight = json_number_value(modelHeightT) * lengthModelUnitConversion;
      double modelDepth = json_number_value(modelDepthT) * lengthModelUnitConversion;
      double modelBreadth = json_number_value(modelBreadthT) * lengthModelUnitConversion;

      //
      // for each tap we want to calculate some factors for applying loads at the floors
      //

      // load tap data from the file
      json_t *tapLocations = json_object_get(tapData,"tapLocations");

      
      int numTaps = json_array_size(tapLocations);
      if (numTaps == 0) {
	std::cerr << "FATAL ERROR - no tapLocations or tapLocations empty\n";
	exit(-1);
      }

      TAP *theTAPS = new TAP[numTaps];
      for (int i=0; i<numTaps; i++) {
	json_t *jsonTap = json_array_get(tapLocations, i);
	int face = json_integer_value(json_object_get(jsonTap,"face"));
	theTAPS[i].face = face;
	theTAPS[i].id = json_integer_value(json_object_get(jsonTap,"id"));
	double locX = json_number_value(json_object_get(jsonTap,"xLoc"));
	double locY = json_number_value(json_object_get(jsonTap,"yLoc"));

	// want to scale tap position to building(prototype) dimensions
	if ((strcmp(roofType,"Flat") == 0) || (strcmp(roofType,"Gable") == 0)) { 
	  if (face == 1 || face == 3) {
	    locX = locX*breadth/modelBreadth;
	    locY = locY*height/modelHeight;
	  } else if (face == 2 || face == 4) {
	    locX = locX*depth/modelDepth;
	    locY = locY*height/modelHeight;
	  } else {
	    locX = locX*breadth/modelBreadth;
	    locY = locY*height/modelDepth;
	  }
	  theTAPS[i].locX = locX;
	  theTAPS[i].locY = locY;
	}

	double *forces = new double[numFloors];
	double *moments = new double[numFloors];
	for (int i=0; i<numFloors; i++) {
	  forces[i]=0.0;
	  moments[i]=0.0;
	}
	  
	theTAPS[i].forces = forces;
	theTAPS[i].moments = moments;
      }


      //
      // for each tap determine factors fr moments and forces for the buiding asuming a mesh discretization
      //

      int numDivisionX = 10;
      int numDivisionY = 10;

      addForcesFace(theTAPS, numTaps, height, breadth, numDivisionX, numDivisionY, 1, numFloors);
      addForcesFace(theTAPS, numTaps, height, depth, numDivisionX, numDivisionY, 2, numFloors);
      addForcesFace(theTAPS, numTaps, height, breadth, numDivisionX, numDivisionY, 3, numFloors);
      addForcesFace(theTAPS, numTaps, height, depth, numDivisionX, numDivisionY, 4, numFloors);

      //
      // write out the forces to the event file
      //

    } else {

      //
      // this is where we call the TPU Website to get the data & then process it into a json format
      // 

      // 
      // parse event data, and make call to TPU database
      //

      int incidenceAngle = json_integer_value(json_object_get(currentEvent,"incidenceAngle"));
      const char *roofType = json_string_value(json_object_get(currentEvent,"roofType"));
      const char *pitch = json_string_value(json_object_get(currentEvent,"pitch"));
      const char *depthBreadth = json_string_value(json_object_get(currentEvent,"depthBreadth"));
      const char *heightBreadth = json_string_value(json_object_get(currentEvent,"heightBreadth"));

      
      callLowRise_TPU(roofType, heightBreadth, depthBreadth, pitch, incidenceAngle, "tmpSimCenterLowRiseTPU.mat");
      
      //
      // invoke python and LowRiseTPU.py script to process the .mat file into json file
      //

      int pyArgc = 3;
      char *pyArgv[3];
      pyArgv[0] = (char *)"LowRiseTPU.py";
      pyArgv[1] = (char *)"tmpSimCenterLowRiseTPU.mat";
      pyArgv[2] = (char *)"tmpSimCenterLowRiseTPU.json";

      Py_SetProgramName(pyArgv[0]);
      Py_Initialize();
      PySys_SetArgv(pyArgc, pyArgv);
      FILE *file = fopen(pyArgv[0],"r");
      PyRun_SimpleFile(file, pyArgv[0]);
      Py_Finalize();

      /*
      std::string str = "python3 LowRiseTPU.py tmpSimCenterLowRiseTPU.mat tmpSimCenterLowRiseTPU.json"; 
      const char *command = str.c_str(); 
      system(command); 
      */

    }
    return 0;
}


//
// function to add factors for forces and moment contribution coefficients for taps to building floor
// determine coeffiecients for each tap for a building face. This is done by going over each story of 
// For each story break into numDiv X numDiv segments. For each segment assume point load at center 
// segment and equal in mag to area of segment and using simply supported beam formula determine force 
// at floor below and floor above. based on distance from center line of story determine actibg moments 
// on floors.
//
// inputs: height: height of building
//         length: length of building face (width, depth) depending on side
//         numDivisionX: number of elements in mesh along length
//         numDivisionY: number of elements in mesh over building height
//         face: faceID 
//         numFloors: number of floors
// output: 0 ok, -1 error
// 

int addForcesFace(TAP *theTaps, int numTaps, 
		  double height, double length, 
		  int numDivisionX, int numDivisionY, 
		  int face, 
		  int numFloors) {
  


  double heightStory = height/(1.0*numFloors);
  double dY = heightStory/numDivisionY;
  double dX = length/numDivisionX;
  double centerLine = length/2.0;
  double A = dY*dX;
  double locY = dY/2.0;
  
  for (int i=0; i<numDivisionY; i++) {
    double locX = dX/2.0;
    double Rbelow = locY*A/heightStory;
    double Rabove = (heightStory-locY)*A/heightStory;
    
    for (int i=0; i<numDivisionX; i++) {
      double Mabove = Rabove*(locX-centerLine);
      double Mbelow = Rbelow*(locX-centerLine);
      
      // find nearestTAP
      TAP *theTap = findNearestTAP(theTaps, numTaps, locX, locY, face);

      // add force coefficients
      if (theTap != NULL) {
	if (i != 0) { // don;t add to ground floor
	  theTap->forces[i-1] += Rbelow;
	  theTap->moments[i-1] += Mbelow;
	}
	theTap->forces[i] += Rabove;
	theTap->moments[i] += Mabove;
      }	    
      locX += dX;
    }
    locY += dY;
  }
  return 0;
}

//
// function to fnd nearest tap
// inputs: theTAPS: array of Taps, 
//         numTaps: number of taps in array
//         xLoc, yLoc: is location of inut point
//         face: if of face
// output: pinter to nearest TAp in the array, NULL if no taps with face
// 

TAP *findNearestTAP(TAP *theTAPS, int numTaps, double locX, double locY, int face) {
  TAP *theRes = NULL;
  double nearestDistance = 1e100; // something large!
  for (int i=0; i<numTaps; i++) {
    TAP *theTap = &theTAPS[i];
    if (theTap->face == face) {
      double deltaX = theTap->locX - locX;
      double deltaY = theTap->locY - locY;
      double dist = deltaX*deltaX + deltaY*deltaY;
      if (dist < nearestDistance) {
	theRes = theTap;
	nearestDistance = dist;
      }
    }
  }
  
  return theRes;
}
