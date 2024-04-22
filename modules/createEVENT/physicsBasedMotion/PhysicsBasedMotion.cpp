
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <filesystem>

using namespace std;

#include <jansson.h>  // for Json

int addEvent(const char *fileNameEvent, json_t *obj);
int createSimCenterEvent(const char *motionName, const char *fileType); // fileType: AT2 or json

int main(int argc, char **argv)
{
  // StandardEarthquakeEDP --filenameAIM file --filenameEVENT file? <--getRV>

  char *filenameAIM;     
  char *filenameEVENT; 
  bool getRV = false;  
  int arg = 1;

  while(arg < argc) {
    if (strcmp(argv[arg], "--filenameAIM") == 0)
      {
        arg++;
        filenameAIM = argv[arg];
      }
    else if (strcmp(argv[arg], "--filenameEVENT") == 0)
      {
        arg++;
        filenameEVENT = argv[arg];
      }
    else if (strcmp(argv[arg], "--getRV") == 0)
      {
	getRV = true;
      }
    arg++;
  }

  if (filenameAIM == NULL || filenameEVENT == NULL) {
    std::cerr << "FATAL ERROR - no bim or sam file provided to MultiplePEER_Event\n";
    exit(-1);
  }

  // create output JSON object for EVENT file and create events array
  json_t *rootEvent = json_object();
  json_t *newEventArray = json_array(); 

  // load INPUT file
  json_error_t error;
  json_t *rootINPUT = json_load_file(filenameAIM, 0, &error);
  json_t *eventsArray = json_object_get(rootINPUT,"Events");  

  if (getRV) { 

    //
    // if --getRV:
    //   1) we create Event Files
    //   2) we want to set EVENT file with random variables and events that just contain name
    //

    // output jsonObject for any Random Variables
    json_t *rvArray=json_array();   

    // 
    // parse each event in input:
    //  1. make sure earthquake
    //  2. add responses
    //
  
    int index;
    json_t *value;

    //    int numEDP = 0;

    std::cerr << json_array_size(eventsArray) << "\n";
    
    json_array_foreach(eventsArray, index, value) {
      
      // check earthquake

      json_t *type = json_object_get(value,"type");
      if (type == NULL) {
	std::cerr << "no type in object \n";
	std::cerr << json_dumps(value, JSON_INDENT(4));
	exit(-1);
      }
      const char *eventType = json_string_value(type);

      if (strcmp(eventType,"PhysicsBasedMotion") != 0) {
	
	json_array_append(newEventArray, value); // copy event for next event app to parse

      } else {

	json_t *eventObj = json_object();
	json_object_set(eventObj,"type", json_string("Seismic"));
	json_object_set(eventObj,"subtype", json_string("PhysicsBasedMotion"));
	
	json_t *motionsArray = json_object_get(value,"motions");
	const char *fileType = json_string_value(json_object_get(value,"fileType"));
	int numExisting = json_array_size(motionsArray);      
	
	if (numExisting > 1) {

	  json_t *randomVar = json_object();
	  json_object_set(randomVar, "distribution",json_string("discrete_design_set_string"));
	  json_object_set(randomVar, "name",json_string("MultipleEvent"));
	  json_object_set(randomVar, "value",json_string("RV.MultipleEvent"));
	  json_t *theMultipleEvents = json_array();
	  
	  json_t *currentMotion = 0;
	  json_array_foreach(motionsArray, index, currentMotion) { 
	    createSimCenterEvent(json_string_value(currentMotion), fileType);
	    json_array_append(theMultipleEvents, currentMotion);
	  }
	  json_object_set(randomVar, "elements", theMultipleEvents);
	  json_array_append(rvArray, randomVar);
	  json_object_set(eventObj, "index", json_string("RV.MultipleEvent"));
	
	} else {

	  json_t *currentMotion = json_array_get(motionsArray,0);
	  createSimCenterEvent(json_string_value(currentMotion), fileType);
	  json_object_set(eventObj, "index", json_integer(0));

	}

	//add first event to event
	json_t *firstEvent = json_array_get(motionsArray, 0);
	const char *fileName = json_string_value(firstEvent);
	addEvent(fileName, eventObj);
	json_array_append(newEventArray, eventObj);
	
      }
    }

    // write the variables & events
    json_object_set(rootEvent,"randomVariables",rvArray);
    json_object_set(rootEvent,"Events",newEventArray);

    // dump the event file
    json_dump_file(rootEvent,filenameEVENT,0);
    
    //    json_dump_file(rootEvent,filenameEVENT,JSON_INDENT(1));   

  }  else { // if not --getRV we want to copy file to EVENT fileName

    //
    // need to open up EVENT file and process to see which of EVENTS to use
    // need to open up INPUT file to see the name of this file (it should be in dir, 
    // then copy file to replace EVENT
    // 

    json_t *rootINPUT = json_load_file(filenameAIM, 0, &error);
    json_t *rootEVENT = json_load_file(filenameEVENT, 0, &error);
    
    // load INPUT file
    //    json_error_t error;
    json_t *inputEventsArray = json_object_get(rootINPUT,"Events");  
    json_t *eventsEventsArray = json_object_get(rootEVENT,"Events");  
    
    int count;
    json_t *value;
    
    // int numEDP = 0;
    
    json_array_foreach(eventsEventsArray, count, value) {
      
      // check earthquake
      json_t *type = json_object_get(value,"type");  
      const char *eventType = json_string_value(type);
      
      if (strcmp(eventType,"Seismic") == 0) {
	json_t *subType = json_object_get(value,"subtype");  
	if ((subType != NULL) && (strcmp("PhysicsBasedMotion",json_string_value(subType)) ==0)) {

	  json_t *index = json_object_get(value,"index"); 

	  if (json_is_integer(index) == false) {
	    
	    const char *eventName = json_string_value(index);

	    // we need to replace the EVENT with another event
	    json_t *inputEvent = json_array_get(inputEventsArray,count);
	    json_t *events = json_object_get(inputEvent,"motions");
	    for (int i=0; i<json_array_size(events); i++) {
	      json_t *theEvent = json_array_get(events, i);
	      const char * name = json_string_value(theEvent);

	      if (strcmp(eventName, name) == 0) {
		addEvent(name, value);
		
		i = json_array_size(events);
	      }
	    }
	  }
	  json_t *eventObj = json_object();
	}	  
      }
    }
    // write rootEvent
    json_dump_file(rootEVENT,filenameEVENT,0);
    
  }  

  return 0;
}


//
// procedure to open an existing event file, take first event and "UPDATE"
// the passed obj with the contents of this first event
//

int addEvent(const char *eventName, json_t *obj) {

  // open file and get the first event
  json_error_t error;
  json_t *rootEVENT = 0;  

  std::filesystem::path currentDir; currentDir = std::filesystem::current_path();  
  std::filesystem::path newFilePath;
  newFilePath = currentDir.parent_path();
  newFilePath.append("input_data");
  
  if (std::filesystem::exists(newFilePath)) {
    
    std::string fileName("sc"); fileName.append(eventName);
    newFilePath.append(fileName);
    std::string stringPath; stringPath = newFilePath.string();
    const char *charPath = stringPath.c_str();
    std::cerr << "NEW_FILE_PATH" << charPath;
    rootEVENT = json_load_file(charPath, 0, &error);      
    
  } else {
    
    std::string fileName("sc"); fileName.append(eventName);
    newFilePath.append(fileName);
    std::string newFilePath_str; newFilePath_str = newFilePath.string();
    const char *charPath = newFilePath_str.c_str();
    rootEVENT = json_load_file(charPath, 0, &error);      
    
  }

  json_t *eventsArray = json_object_get(rootEVENT,"Events");  
  json_t *eventToCopy = json_array_get(eventsArray,0);

  // update the object with object just read
  json_object_update(obj, eventToCopy); 

  return 0;
}

//
// procedure to create a SimCenter Event given the record in the peerEVENT
//  - the SimCenter Event will be written to a file given by name value


int
createSimCenterEvent(const char *eventName, const char *fileType) {

  //
  // get name and type 
  //
  // arrays for timeSeries and pattern
  json_t *timeSeriesArray = json_array();
  json_t *patternArray = json_array();
  double dT =0.0;
  int numSteps =0;
 
  if (strcmp(fileType,"JSON") == 0) {

    std::filesystem::path currentDir; currentDir = std::filesystem::current_path();
    std::filesystem::path jsonFilePath; jsonFilePath = currentDir.parent_path();
    jsonFilePath.append("input_data");
    jsonFilePath.append(eventName);
    std::string jsonFilePath_str; jsonFilePath_str = jsonFilePath.string();    
    const char *filePath = jsonFilePath_str.c_str();
    std::cerr << "jsonFilePath: " << jsonFilePath << "\n";
    
    //
    // open JSON file
    //
    
    json_error_t error;    
    json_t *rootJSON = json_load_file(filePath, 0, &error);
    json_t *dtObject = json_object_get(rootJSON,"dT");
    json_t *numStepsObject = json_object_get(rootJSON,"numSteps");
    json_t *accel_x  = json_object_get(rootJSON,"accel_x");
    json_t *accel_y  = json_object_get(rootJSON,"accel_y");
    json_t *accel_z  = json_object_get(rootJSON,"accel_z");
    
    if (dtObject != NULL)
      dT = json_number_value(dtObject);

    if (numStepsObject != NULL)
      numSteps = json_integer_value(numStepsObject);    

    //
    // for each dirn, if accel found
    //    create a series and pattern object
    //    fill in the data for both
    //    append the timeSeries and pattern objects to bigger arrays
    //
    
    if (accel_x != NULL) {
      json_t *dirn1Series = json_object();
      json_t *dirn1Pattern = json_object();    
      json_object_set(dirn1Series,"name",json_string("dirn1"));
      json_object_set(dirn1Series,"type",json_string("Value"));
      json_object_set(dirn1Series,"factor",json_real(1.0));
      json_object_set(dirn1Series,"dT",json_real(dT));
      json_object_set(dirn1Series,"numSteps",json_integer(numSteps));
      json_object_set(dirn1Series,"data",accel_x);
      json_object_set(dirn1Pattern,"timeSeries",json_string("dirn1"));
      json_object_set(dirn1Pattern,"type",json_string("UniformAcceleration"));
      json_object_set(dirn1Pattern,"dof",json_integer(1));
      
      json_array_append(timeSeriesArray, dirn1Series);
      json_array_append(patternArray, dirn1Pattern);
      
    }
    if (accel_y != NULL) {
      json_t *dirn2Series = json_object();
      json_t *dirn2Pattern = json_object();
      
      json_object_set(dirn2Series,"name",json_string("dirn2"));
      json_object_set(dirn2Series,"type",json_string("Value"));
      json_object_set(dirn2Series,"factor",json_real(1.0));
      json_object_set(dirn2Series,"dT",json_real(dT));
      json_object_set(dirn2Series,"numSteps",json_integer(numSteps));
      json_object_set(dirn2Series,"data",accel_y);
      json_object_set(dirn2Pattern,"timeSeries",json_string("dirn2"));
      json_object_set(dirn2Pattern,"type",json_string("UniformAcceleration"));
      json_object_set(dirn2Pattern,"dof",json_integer(2));
      
      json_array_append(timeSeriesArray, dirn2Series);
      json_array_append(patternArray, dirn2Pattern);      
    }
    if (accel_z != NULL) {
      json_t *dirn3Series = json_object();
      json_t *dirn3Pattern = json_object();
      
      json_object_set(dirn3Series,"name",json_string("dirn3"));
      json_object_set(dirn3Series,"type",json_string("Value"));
      json_object_set(dirn3Series,"factor",json_real(1.0));
      json_object_set(dirn3Series,"dT",json_real(dT));
      json_object_set(dirn3Series,"numSteps",json_integer(numSteps));
      json_object_set(dirn3Series,"data",accel_z);
      json_object_set(dirn3Pattern,"timeSeries",json_string("dirn3"));
      json_object_set(dirn3Pattern,"type",json_string("UniformAcceleration"));
      json_object_set(dirn3Pattern,"dof",json_integer(3));
      
      json_array_append(timeSeriesArray, dirn3Series);
      json_array_append(patternArray, dirn3Pattern);          
    }
    
    //
    // create and fill in the SimCenter EVENT
    //

    json_t *outputObj = json_object();
    json_t *eventObj = json_object();    
    json_object_set(eventObj,"type",json_string("Seismic"));
    json_object_set(eventObj,"dT",json_real(dT));
    json_object_set(eventObj,"numSteps",json_integer(numSteps));
    json_object_set(eventObj,"timeSeries",timeSeriesArray);
    json_object_set(eventObj,"pattern", patternArray);

    //
    // now the EVENT needs to be added to an Events array in the actual SimCenterEvent
    //  as each SimCenterEvent can be used by itself, we create this Events array  and place in the output obj
    //

    json_t *eventsArray = json_array();
    json_array_append(eventsArray, eventObj);
    json_object_set(outputObj,"Events",eventsArray);

    //
    // write a SimCenter Event file by dumping outputObj JSON to a file
    //

    std::filesystem::path newFilePath; newFilePath = currentDir.parent_path();
    newFilePath.append("input_data");
    
    if (std::filesystem::exists(newFilePath)) {

      std::string fileName("sc"); fileName.append(eventName);
      newFilePath.append(fileName);
      std::string stringPath; stringPath = newFilePath.string();
      const char *charPath = stringPath.c_str();
      json_dump_file(outputObj, charPath, JSON_COMPACT);
      
    } else {
      
      std::string fileName("sc"); fileName.append(eventName);
      newFilePath.append(fileName);
      std::string newFilePath_str; newFilePath_str = newFilePath.string();
      const char *charPath = newFilePath_str.c_str();
      
      json_dump_file(outputObj, charPath, JSON_COMPACT);
    }
  }
  
  // done!
  return 0;
}
