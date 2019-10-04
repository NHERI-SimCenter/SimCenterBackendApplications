#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>

#include <jansson.h>
#include <common/Units.h>

typedef struct tapData {
  double locX;
  double locY;
  int face;
  int id;
  double *forces;
  double *moments;
  double *data;
} TAP;


//
// some functions define at end of file
//

TAP *findNearestTAP(TAP *, int numTaps,  double locX, double locY, int face);
int addForcesFace(TAP *theTAPS, int numTaps, double height, double length, int numDivisonX, int numDivisonY, int face, int numFloors);
int addEvent(json_t *input, json_t *currentEvent, json_t *outputEvent, bool getRV);

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

    if (strcmp(eventType,"WindTunnelExperiment") != 0) {
      
      json_array_append(outputEventsArray, inputEvent); 
      
    } else {

      // create output event
      json_t *outputEvent = json_object();
      json_object_set(outputEvent,"type", json_string("Wind"));
      json_object_set(outputEvent, "subtype", json_string("WindTunnelExperiment"));
      
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

      Units::UnitSystem eventUnits;
      eventUnits.lengthUnit = Units::ParseLengthUnit("m");

      //
      // Read building information and scale to event units
      // 

      double lengthUnitConversion = Units::GetLengthFactor(bimUnits, eventUnits);
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
      
      int    numFloors = json_integer_value(storiesJO);
      double height = json_number_value(heightJO) * lengthUnitConversion;
      double breadth = json_number_value(widthJO) * lengthUnitConversion;
      double depth = json_number_value(depthJO)   * lengthUnitConversion;

      //
      // get wind speed from event object
      //

      json_t *windSpeedJO = json_object_get(currentEvent,"windSpeed");      
      if (windSpeedJO == NULL) {
	std::cerr << "ERROR missing windSpeed from event)\n";
	return -2;        
      }
      double windSpeed = json_number_value(windSpeedJO); // no unit conversion as m/sec

      json_t *filenameJO = json_object_get(currentEvent,"filename");      
      if (filenameJO == NULL) {
	std::cerr << "ERROR missing inputFile from event)\n";
	return -2;        
      }
      const char *testFile = json_string_value(filenameJO); // no unit conversion as m/sec

      //
      // load the json file containing experiment data
      //
      
      json_error_t error;
      json_t *tapData = json_load_file(testFile, 0, &error);
      
      if (tapData == NULL) {
	std::cerr << "FATAL ERROR - json file  does not exist\n";
	return -3;
      }

      //
      // Read model data from experimental file
      // 

      json_t* modelUnitsJson = json_object_get(tapData, "units");
      json_t* modelLengthJson = json_object_get(modelUnitsJson, "length");

      Units::UnitSystem modelUnits;
      modelUnits.lengthUnit = Units::ParseLengthUnit(json_string_value(bimLengthJson));
      double lengthModelUnitConversion = Units::GetLengthFactor(modelUnits, eventUnits);

      json_t *roofTypeT = json_object_get(tapData,"roofType");
      json_t *modelHeightT = json_object_get(tapData,"height");
      json_t *modelDepthT = json_object_get(tapData,"depth");
      json_t *modelBreadthT = json_object_get(tapData,"breadth");
      json_t *windSpeedT = json_object_get(tapData,"windSpeed");
      json_t *frequencyT = json_object_get(tapData,"frequency");
      json_t *periodT = json_object_get(tapData,"period");

      if (roofTypeT == NULL     || 
	  modelHeightT == NULL  || 
	  modelDepthT == NULL   || 
	  modelBreadthT == NULL || 
	  periodT == NULL       || 
	  frequencyT == NULL) {

	std::cerr << "FATAL ERROR - json file does not contain roofType, height, depth or breadth data \n";
	return -3;
      }

      const char *roofType = json_string_value(roofTypeT);
      double modelHeight = json_number_value(modelHeightT) * lengthModelUnitConversion;
      double modelDepth = json_number_value(modelDepthT) * lengthModelUnitConversion;
      double modelBreadth = json_number_value(modelBreadthT) * lengthModelUnitConversion;
      double modelWindSpeed = json_number_value(windSpeedT) * lengthModelUnitConversion;
      double modelPeriod = json_number_value(periodT);
      double modelFrequency = json_number_value(frequencyT);
      int numSteps = round(modelPeriod*modelFrequency);

      //      double airDensity = 1.225 * 9.81 / 1000.0;  // 1.225kg/m^3 to kN/m^3
      double airDensity = 1.225;  // 1.225kg/m^3
      double lambdaL = modelHeight/height;
      double lambdaU = modelWindSpeed/windSpeed;
      std::cerr << "modelWs, wS: " << modelWindSpeed << " " << windSpeed << "\n";
      double lambdaT = lambdaL/lambdaU;
      std::cerr << "lambdaL, lambdaU: " << lambdaL << " " << lambdaU << "\n";
      double dT = 1.0/(modelFrequency*lambdaT);
      std::cerr << "dT, freq, lambdaT: " << dT << " " << modelFrequency << " " << lambdaT << "\n";

      double loadFactor = airDensity*0.5*windSpeed*windSpeed / 1000.; // N to KN


      //
      // for each tap we want to calculate some factors for applying loads at the floors
      //

      // load tap data from the file
      json_t *tapLocations = json_object_get(tapData,"tapLocations");
      json_t *tapCp = json_object_get(tapData,"pressureCoefficients");

      
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
	double *data = new double[numSteps];
	for (int j=0; j<numFloors; j++) {
	  forces[j]=0.0;
	  moments[j]=0.0;
	  data[j]=0.0;
	}

	for (int j=0; j<numTaps; j++) {
	  json_t *jsonCP = json_array_get(tapCp, j);
	  int tapID = json_integer_value(json_object_get(jsonCP,"id"));
	  if (theTAPS[i].id == tapID) {
	    json_t *arrayData = json_object_get(jsonCP,"data");
	    for (int k=0; k<numSteps; k++) {
	      data[k] = json_real_value(json_array_get(arrayData, k));
	    }	    
	    j = numTaps;
	  }
	}
	  
	theTAPS[i].forces = forces;
	theTAPS[i].moments = moments;
	theTAPS[i].data = data;
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


      // fill in a blank event for floor loads
      json_t *timeSeriesArray = json_array();
      json_t *patternArray = json_array();
      json_t *pressureArray = json_array();

      for (int i = 0; i < numFloors; i++) {
	
	// create and fill in a time series object
	char floor[10];
	char name[50];

	sprintf(floor,"%d",i+1);

	//
	// forces in x direction
	//

	sprintf(name,"Fx_%d",i+1);
	json_t *timeSeriesX = json_object();     
	json_object_set(timeSeriesX,"name",json_string(name));    
	json_object_set(timeSeriesX,"dT",json_real(dT));
	json_object_set(timeSeriesX,"type",json_string("Value"));
	json_t *dataFloorX = json_array();   
	double maxPressureX = 0.0;
	double minPressureX = 0.0;

	for (int j=0; j<numSteps; j++) {
	  double value = 0.0;
	  for (int k=0; k<numTaps; k++) {
	    if (theTAPS[k].face == 1 || theTAPS[k].face == 3)
	      value = value + theTAPS[k].forces[i] * theTAPS[k].data[j];
	  }
	  value = loadFactor * value;
	  json_array_append(dataFloorX,json_real(value));
	}
	json_object_set(timeSeriesX,"data",dataFloorX);

	json_t *patternX = json_object();
	json_object_set(patternX,"name",json_string(name));        
	json_object_set(patternX,"timeSeries",json_string(name));        
	json_object_set(patternX,"type",json_string("WindFloorLoad"));        

	json_object_set(patternX,"floor",json_string(floor));        
	json_object_set(patternX,"dof",json_integer(1));        
	json_array_append(patternArray,patternX);

	//
	// forces y direction
	//

	sprintf(name,"Fy_%d",i+1);
	json_t *timeSeriesY = json_object();     
	json_object_set(timeSeriesY,"name",json_string(name));    
	json_object_set(timeSeriesY,"dT",json_real(dT));
	json_object_set(timeSeriesY,"type",json_string("Value"));
	json_t *dataFloorY = json_array();   
	double maxPressureY = 0.0;
	double minPressureY = 0.0;

	for (int j=0; j<numSteps; j++) {
	  double value = 0.0;
	  for (int k=0; k<numTaps; k++) {
	    if (theTAPS[k].face == 2 || theTAPS[k].face == 4)
	      value = value + theTAPS[k].forces[i] * theTAPS[k].data[j];
	  }
	  value = loadFactor * value;
	  json_array_append(dataFloorY,json_real(value));
	}

	json_object_set(timeSeriesY,"data",dataFloorY);

	json_t *patternY = json_object();
	json_object_set(patternY,"name",json_string(name));        
	json_object_set(patternY,"timeSeries",json_string(name));        
	json_object_set(patternY,"type",json_string("WindFloorLoad"));        
	json_object_set(patternY,"floor",json_string(floor));        
	json_object_set(patternY,"dof",json_integer(2));        
	json_array_append(patternArray,patternY);

	//
	// moments about z
	//

	sprintf(name,"Mz_%d",i+1);
	json_t *timeSeriesRZ = json_object();     
	json_object_set(timeSeriesRZ,"name",json_string(name));    
	json_object_set(timeSeriesRZ,"dT",json_real(dT));
	json_object_set(timeSeriesRZ,"type",json_string("Value"));
	json_t *dataFloorRZ = json_array();   

	for (int j=0; j<numSteps; j++) {
	  double value = 0.0;
	  for (int k=0; k<numTaps; k++) {
	    value = value + theTAPS[k].moments[i] * theTAPS[k].data[j];
	  }
	  value = loadFactor * value;
	  json_array_append(dataFloorRZ,json_real(value));
	}
	json_object_set(timeSeriesRZ,"data",dataFloorRZ);

	json_t *patternRZ = json_object();
	json_object_set(patternRZ,"name",json_string(name));        
	json_object_set(patternRZ,"timeSeries",json_string(name));        
	json_object_set(patternRZ,"type",json_string("WindFloorLoad"));        
	json_object_set(patternRZ,"floor",json_string(floor));        
	json_object_set(patternRZ,"dof",json_integer(6));        
	json_array_append(patternArray,patternRZ);
	
	json_t *pressureObject = json_object();
	json_t *pressureStoryArray = json_array();
	
	json_array_append(pressureStoryArray, json_real(minPressureX));
	json_array_append(pressureStoryArray, json_real(maxPressureX));
	json_object_set(pressureObject,"pressure",pressureStoryArray);
	json_object_set(pressureObject,"story",json_string(name));
	
	json_array_append(pressureArray, pressureObject);
	
	// add object to timeSeries array
	json_array_append(timeSeriesArray,timeSeriesX);
	json_array_append(timeSeriesArray,timeSeriesY);
	json_array_append(timeSeriesArray,timeSeriesRZ);
      }
      
      json_t *units = json_object();
      json_object_set(units,"force",json_string("KN"));
      json_object_set(units,"length",json_string("m"));
      json_object_set(units,"time",json_string("sec"));
      json_object_set(outputEvent,"units",units);
      
      json_object_set(outputEvent,"timeSeries",timeSeriesArray);
      json_object_set(outputEvent,"pattern",patternArray);
      json_object_set(outputEvent,"pressure",pressureArray);
      json_object_set(outputEvent,"dT",json_real(dT));
      json_object_set(outputEvent,"numSteps",json_integer(numSteps));

    } else {

      // need to write empty event for EDP

      json_t *storiesJO = json_object_get(generalInfo,"stories");
      if (storiesJO == NULL ) {
	std::cerr << "ERROR missing Information from GeneralInformation (height, width, stories all neeed)\n";
	return -2;        
      }
      
      int numFloors = json_integer_value(storiesJO);

      // fill in a blank event for floor loads
      json_t *timeSeriesArray = json_array();
      json_t *patternArray = json_array();
      json_t *pressureArray = json_array();

      for (int i = 0; i < numFloors; i++) {
	
	// create and fill in a time series object
	char floor[10];
	char name[50];

	sprintf(floor,"%d",i+1);

	sprintf(name,"Fx_%d",i+1);
	json_t *timeSeriesX = json_object();     
	json_object_set(timeSeriesX,"name",json_string(name));    
	json_object_set(timeSeriesX,"dT",json_real(0.01));
	json_object_set(timeSeriesX,"type",json_string("Value"));
	json_t *dataFloorX = json_array();   
	json_object_set(timeSeriesX,"data",dataFloorX);

	json_t *patternX = json_object();
	json_object_set(patternX,"name",json_string(name));        
	json_object_set(patternX,"timeSeries",json_string(name));        
	json_object_set(patternX,"type",json_string("WindFloorLoad"));        

	json_object_set(patternX,"floor",json_string(floor));        
	json_object_set(patternX,"dof",json_integer(1));        
	json_array_append(patternArray,patternX);

	sprintf(name,"Fy_%d",i+1);
	json_t *timeSeriesY = json_object();     
	json_object_set(timeSeriesY,"name",json_string(name));    
	json_object_set(timeSeriesY,"dT",json_real(0.01));
	json_object_set(timeSeriesY,"type",json_string("Value"));
	json_t *dataFloorY = json_array();   
	json_object_set(timeSeriesY,"data",dataFloorY);

	json_t *patternY = json_object();
	json_object_set(patternY,"name",json_string(name));        
	json_object_set(patternY,"timeSeries",json_string(name));        
	json_object_set(patternY,"type",json_string("WindFloorLoad"));        
	json_object_set(patternY,"floor",json_string(floor));        
	json_object_set(patternY,"dof",json_integer(2));        
	json_array_append(patternArray,patternY);

	sprintf(name,"Mz_%d",i+1);
	json_t *timeSeriesRZ = json_object();     
	json_object_set(timeSeriesRZ,"name",json_string(name));    
	json_object_set(timeSeriesRZ,"dT",json_real(0.01));
	json_object_set(timeSeriesRZ,"type",json_string("Value"));
	json_t *dataFloorRZ = json_array();   
	json_object_set(timeSeriesRZ,"data",dataFloorRZ);

	json_t *patternRZ = json_object();
	json_object_set(patternRZ,"name",json_string(name));        
	json_object_set(patternRZ,"timeSeries",json_string(name));        
	json_object_set(patternRZ,"type",json_string("WindFloorLoad"));        
	json_object_set(patternRZ,"floor",json_string(floor));        
	json_object_set(patternRZ,"dof",json_integer(6));        
	json_array_append(patternArray,patternRZ);
	
	json_t *pressureObject = json_object();
	json_t *pressureStoryArray = json_array();

	json_array_append(pressureStoryArray, json_real(0.0));
	json_array_append(pressureStoryArray, json_real(0.0));
	json_object_set(pressureObject,"pressure",pressureStoryArray);
	json_object_set(pressureObject,"story",json_string(name));
	
	json_array_append(pressureArray, pressureObject);
	
	// add object to timeSeries array
	json_array_append(timeSeriesArray,timeSeriesX);
	json_array_append(timeSeriesArray,timeSeriesY);
	json_array_append(timeSeriesArray,timeSeriesRZ);
      }
      
      json_t *units = json_object();
      json_object_set(units,"force",json_string("KN"));
      json_object_set(units,"length",json_string("m"));
      json_object_set(units,"time",json_string("sec"));
      json_object_set(outputEvent,"units",units);
      
      json_object_set(outputEvent,"timeSeries",timeSeriesArray);
      json_object_set(outputEvent,"pattern",patternArray);
      json_object_set(outputEvent,"pressure",pressureArray);
      json_object_set(outputEvent,"dT",json_real(0.01));
      json_object_set(outputEvent,"numSteps",json_integer(0));

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
	  if (face == 1 || face == 2) // pressure on face 3 and 4 are negative to pos x & y dirn
	    theTap->forces[i-1] += Rbelow;
	  else
	    theTap->forces[i-1] -= Rbelow;

	  theTap->moments[i-1] += Mbelow;
	}
	if (face == 1 || face == 2) // pressure on face 3 and 4 are negative to pos x & y dirn

	  theTap->forces[i] += Rabove;
	else
	  theTap->forces[i] -= Rabove;

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
