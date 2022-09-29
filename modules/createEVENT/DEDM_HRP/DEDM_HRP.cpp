#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>

#include <jansson.h>  // for Json
#include "common/Units.h"

int addEvent(json_t *input, json_t *currentEvent, json_t *outputEvent,
             bool getRV, int incidenceAngle);

int 
callDEDM_HRP(double shpValue,   // Cross-sectional shape (model):  1.00, 0.50, 0.33
	     double hValue,     // Model height (1,2,3,4,5): 0.1,0.2,0.3,0.4,0.5
	     int expCond,       // Exposure condition: 4,6
	     double timeValue,  // Averaging time Time_value 3600
	     double uH,         // Mean wind velocity at top (Ultimate limit state 700yr) m/s
	     double uHHab,      // Mean wind velocity at top (Habitability 10 yr) m/s
	     double B, double D, double H, // building width, depth, and height in METRE
	     int nFloor,        // number of floors
	     const char *outputFilename);

int
main(int argc, char **argv) {

  //
  // parse input args for filenames
  //

  char *filenameAIM = NULL;   // inputfile
  char *filenameEVENT = NULL; // outputfile

  bool doRV = false;

  int arg = 1;
  while(arg < argc) {
    if (strcmp(argv[arg], "--filenameAIM") == 0) {
      arg++;
      filenameAIM = argv[arg];
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

  if (filenameAIM == NULL || filenameEVENT == NULL) {
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
  json_t *input = json_load_file(filenameAIM, 0, &error);
  if (input == NULL) {
    std::cerr << "FATAL ERROR - input file does not exist\n";
    exit(-1);
  }
  
  json_t *inputEventsArray = json_object_get(input, "Events");  
  if (inputEventsArray == NULL) {
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

    if (strcmp(eventType,"DEDM_HRP") != 0) {
      
      json_array_append(outputEventsArray, inputEvent); 
      
    } else {

      // check subtype when have more stochastic models
      int planShape = json_integer_value(json_object_get(inputEvent,"checkedPlan"));
      json_t * angle = json_object_get(inputEvent, "incidenceAngle");
      int incidenceAngle = json_integer_value(angle);
      std::string dataBaseFile;

      // If plan shape factor is 1, need to mirror for incidence angle greater
      // than 45 degrees
      if (planShape == 1 && incidenceAngle > 45) {
        dataBaseFile =
            "EVENT.json." + std::to_string(90 - incidenceAngle) + ".json";
      } else {
        dataBaseFile = "EVENT.json." + std::to_string(incidenceAngle) + ".json";
      }

      json_t *outputEvent = json_object();
      json_object_set(outputEvent,"type", json_string("Wind"));
      json_object_set(outputEvent, "subtype", json_string("DEDM_HRP"));
      json_object_set(outputEvent, "eventFile", json_string(dataBaseFile.c_str()));

      json_t *units = json_object();
      json_object_set(units,"force",json_string("KN"));
      json_object_set(units,"length",json_string("m"));
      json_object_set(units,"time",json_string("sec"));
      json_object_set(outputEvent,"units",units);

      //      json_object_set(outputEvent, "units", units);      

      addEvent(input, inputEvent, outputEvent, doRV, incidenceAngle);

      json_array_append(outputEventsArray, outputEvent);
    }
  }

  // write the variables & events                                             
  json_t *rootEvent = json_object();
  json_object_set(rootEvent,"randomVariables",rvArray);
  json_object_set(rootEvent,"Events",outputEventsArray);

  // dump the event file
  json_dump_file(rootEvent,filenameEVENT,0);

  std::cerr << "JSON DUMP: " << filenameEVENT << "\n";

  // done
  return 0;
}

int addEvent(json_t *input, json_t *currentEvent, json_t *outputEvent, 
	     bool getRV, int incidenceAngle) {

  int planShape = json_integer_value(json_object_get(currentEvent,"checkedPlan"));

  if (getRV == false) {

    //
    // get the name of json file to load, load it and place in outputEvent
    //

    json_error_t error;
    std::string eventFile;

    // If plan shape factor is 1, need to mirror for incidence angle greater
    // than 45 degrees
    if (planShape == 1 && incidenceAngle > 45) {
      eventFile = "EVENT.json." + std::to_string(90 - incidenceAngle) + ".json";
    } else {
      eventFile = "EVENT.json." + std::to_string(incidenceAngle) + ".json";
    }

    json_t *event = json_load_file(eventFile.c_str(), 0, &error);

    //
    // get wind speed and number of floors
    //

    json_t *generalInfo = json_object_get(input, "GeneralInformation");      
    if (generalInfo == NULL) {
      std::cerr << "ERROR no GeneralInformation in input\n";
      return -1;
    }

    json_t *storiesJO = json_object_get(generalInfo,"stories");
    int numFloors = json_integer_value(storiesJO);

    json_t *windSpeedJO = json_object_get(currentEvent,"windSpeed");
    double windSpeed = json_number_value(windSpeedJO); 

    //
    // for each floor we need to modify the time step and load factor
    // to reflect time step and forces .. data obtained for U=100m/s
    // forces factor = windSpeed^2/100^2, time step factor = 100/windSpeed
    //  and if shape == 1 (square), we need to get forces from other file and swicth as no angle > 45
    //

    json_t *dtJO = json_object_get(event, "dT");

    double dT_100 = json_number_value(dtJO); 
    double dT = dT_100 * 100.0/windSpeed;
    double loadFactor = windSpeed * windSpeed / 1e4;
    std::cerr << "loadFactor: " << loadFactor << " dT: " << dT << " dt_100: " << dT_100 << " windSpeed: " << windSpeed;

    json_t *newDtJO = json_object();
    int res = json_object_set(event, "dT", json_real(dT));

    //    int res = json_object_update(dtJO, newDtJO);

    json_t *timeSeriesArray = json_object_get(event, "timeSeries");
    json_t *patternArray = json_object_get(event, "pattern");

    int index = 0;
    for (int i=0; i<numFloors; i++) {

      for (int i=0; i<3; i++) {
	json_t *pattern = json_array_get(patternArray, index+i);
	res = json_object_set(pattern, "factor", json_real(loadFactor));	

	json_t *timeSeries = json_array_get(timeSeriesArray, index+i);
	//	json_t *dtJO = json_object_get(timeSeries, "dT");
	//	json_object_update(dtJO, newDtJO);
	res = json_object_set(timeSeries, "dT", json_real(dT));
      }

      // flip time series associated with a pettern
      if (planShape == 1 && incidenceAngle > 45) {

	json_t *FxPattern = json_array_get(patternArray, index);      
	//json_t *MzPattern = json_array_get(patternArray, index+1);      
	json_t *FyPattern = json_array_get(patternArray, index+2);            
	
	//json_t *FxSeries = json_array_get(timeSeriesArray, index);
	//json_t *MzSeries = json_array_get(timeSeriesArray, index+1);      
	//json_t *FySeries = json_array_get(timeSeriesArray, index+2);            
	char nameX[50];	
	char nameY[50];	
	// flip translational forces
	sprintf(nameX,"%d_Fx",i+1);
	sprintf(nameY,"%d_Fy",i+1);
	json_object_set(FxPattern, "timeSeries", json_string(nameY));
	json_object_set(FyPattern, "timeSeries", json_string(nameX));
	
	/*
	// flip translational forces
	int index;
	json_t *forceX;
	json_t *forceY;

	json_t *xComponent = json_array_get(timeSeriesArray, 0);
	json_t *yComponent = json_array_get(timeSeriesArray, 1);      
	forceX = json_object_get(xComponent, "data");
	forceY = json_object_get(yComponent, "data");
	json_t *tempX = json_copy(forceX);
	forceX = forceY;
	forceY = tempX;
	json_object_set(xComponent, "data", forceX);
	json_object_set(yComponent, "data", forceY);      
	*/
      }

      index += 3;
    }

    if (event == NULL) {
      std::cerr << "FATAL ERROR - event file " << eventFile << " does not exist\n";
      exit(-1);
    }

    json_object_update(outputEvent, event);

  } else {

    //
    // this is where we call the DEDM_HRP website to get the data for the building
    // it involves 2 calls .. one to get DEDM_HRP to create wind file & second to go get the wind file
    //

    json_t *generalInfo = json_object_get(input, "GeneralInformation");      
    if (generalInfo == NULL) {
      std::cerr << "ERROR no GeneralInformation in input\n";
      return -1;
    }
    
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
    double lengthUnitConversion = Units::GetLengthFactor(bimUnits, eventUnits);
    
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
      return -1;        
    }
    
    int numFloors = json_integer_value(storiesJO);
    double height = json_number_value(heightJO) * lengthUnitConversion;
    double width = json_number_value(widthJO) * lengthUnitConversion;
    double depth = json_number_value(depthJO) * lengthUnitConversion;


    // fill in a blank event for floor loads
    json_t *timeSeriesArray = json_array();
    json_t *patternArray = json_array();
    json_t *pressureArray = json_array();

    for (int i = 0; i < numFloors; i++) {
      
      // create and fill in a time series object
      char floor[10];
      char name[50];
      
      sprintf(floor,"%d",i+1);
      
      sprintf(name,"%d_Fx",i+1);
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
      
      sprintf(name,"%d_Fy",i+1);
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
      
      sprintf(name,"%d_Mz",i+1);
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
    
    // get info from event
    
    json_t *modelPlanJO = json_object_get(currentEvent,"checkedPlan");
    json_t *modelHeightJO = json_object_get(currentEvent,"checkedHeight");
    json_t *modelExposureJO = json_object_get(currentEvent,"checkedExposure");
    json_t *windSpeedJO = json_object_get(currentEvent,"windSpeed");
    
    if (modelPlanJO == NULL || 
	modelHeightJO == NULL ||
	modelExposureJO == NULL ||
	windSpeedJO == NULL) {
      std::cerr << "ERROR missing Information from Event (modelPlan, modelHeight, exposure, windSpeed all neeed)\n";
      return -1;        
    }
    
    //
    // check inputs
    //
    
    if (numFloors == 0) {
      std::cerr << "Invalid numFloors must be greater than 0\n";
      exit(-1);
    }
    
    if (height <= 0.0 || width <= 0.0 || depth <= 0.0) {  
      std::cerr << "Invalid height, width and/or width, both must be greater than 0\n";
      exit(-1);
    }
    
    int selection= json_integer_value(modelPlanJO);
    double shpValue = 1.0;  
    if (selection == 2)
      shpValue = 0.5;
    else if (selection == 3)
      shpValue = 0.33;
    
    selection= json_integer_value(modelHeightJO);      
    double hValue = 0.1*selection;
    
    int expCondition = 4;
    selection= json_integer_value(modelHeightJO);      
    if (selection == 2)
      expCondition = 6;
    
    double timeValue = 3600;
    // double windSpeed = json_number_value(windSpeedJO); // allowing windSpeed to be a random variable
    double windSpeed = 100;
    
    callDEDM_HRP(shpValue, hValue, expCondition, timeValue, windSpeed, windSpeed*.5, width, depth, height, numFloors, "tmpSimCenterDEDM.mat"); 
  }
  return 0;
}
