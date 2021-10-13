#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
using namespace std;

#include <jansson.h>  // for Json

int main(int argc, char **argv)
{
  if (argc == 10) { // only do if --getRV is passed

    char *filenameBIM = argv[2];     
    char *filenameEVENT = argv[4]; 
    char *filenameSAM   = argv[6];
    char *filenameEDP   = argv[8];

    // create output JSON object
    json_t *rootEDP = json_object();
    
    // place an empty random variable field
    json_t *rvArray=json_array();   
    json_object_set(rootEDP,"RandomVariables",rvArray);
    
    //
    // for each event we create the edp's
    //
    
    json_t *eventArray = json_array(); // for each analysis event
    
    // load SAM and EVENT files
    json_error_t error;
    
    json_t *rootEVENT = json_load_file(filenameEVENT, 0, &error);
    json_t *eventsArray = json_object_get(rootEVENT,"Events");  
    
    // 
    // parse each event:
    //  1. make sure earthquake
    //  2. add responses
    //
  
    int index;
    json_t *value;
    
    int numEDP = 0;
    
    json_array_foreach(eventsArray, index, value) {
      
      // add the EDP for the event
      json_t *eventObj = json_object();
      
      json_t *name = json_object_get(value,"name"); 
      const char *eventName = json_string_value(name);
      json_object_set(eventObj,"name",json_string(eventName));

      //    json_dump_file(eventObj,"TEST",0);
      json_t *responsesArray = json_array(); // for each analysis event
      json_t *response = json_object();
      json_object_set(response,"type",json_string("dummy"));
      json_t *dataArray = json_array();
      json_array_append(dataArray, json_integer(0));
      json_object_set(response,"scalar_data",dataArray);
      
      json_array_append(responsesArray,response);
      
      json_object_set(eventObj,"responses",responsesArray);
      json_array_append(eventArray,eventObj);
    }
    
    json_object_set(rootEDP,"total_number_edp",json_integer(1));  
    json_object_set(rootEDP,"EngineeringDemandParameters",eventArray);  

    json_dump_file(rootEDP,filenameEDP,0);
    
    return 0;
  }

  
  return -1;
}
