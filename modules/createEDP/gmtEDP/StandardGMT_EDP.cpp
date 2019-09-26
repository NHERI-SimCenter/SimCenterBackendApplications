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

  char *filenameEVENT = NULL;
  char *filenameEDP = NULL;
  bool getRV = false;

  int arg = 1;
  while (arg < argc) {
      if (strcmp(argv[arg], "--filenameEVENT") ==0) {
	arg++;
	filenameEVENT = argv[arg];
      }
      else if (strcmp(argv[arg], "--filenameEDP") ==0) {
	arg++;
	filenameEDP = argv[arg];
      }
      else if (strcmp(argv[arg], "--getRV") ==0) {
	getRV = true;
      }
      
      arg++;
    }

    //
    // if not all args present, exit with error
    //

    if (filenameEVENT == 0) {
      std::cerr << "ERROR - missing input args\n";
      exit(-1);
    }

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
    
    int index;
    json_t *value;
    
    int numEDP = 0;
    
    json_array_foreach(eventsArray, index, value) {
      
      // check earthquake
      json_t *type = json_object_get(value,"type");  
      const char *eventType = json_string_value(type);
      
      if (strcmp(eventType,"Seismic") != 0) {
	json_object_clear(rootEVENT);
	printf("WARNING event type %s not Seismic NO OUTPUT", eventType);
      }
      
      // add the EDP for the event
      json_t *eventObj = json_object();
      
      json_t *name = json_object_get(value,"name"); 
      const char *eventName = json_string_value(name);
      json_object_set(eventObj,"name",json_string(eventName));

      int numDOF = 0;
      json_t *theDOFs = json_array();
      int *tDOFs = 0;
      json_t *patternArray = json_object_get(value,"pattern");
      int numPattern = json_array_size(patternArray);
      tDOFs = new int[numPattern];

      if (numPattern != 0) {
	for (int ii=0; ii<numPattern; ii++) {
	  json_t *thePattern = json_array_get(patternArray, ii);
	  json_t *theDof = json_object_get(thePattern, "dof");
	  tDOFs[ii] = json_integer_value(theDof);
	  if (theDof != 0) {
	    json_array_append(theDOFs, theDof);
	    numDOF++;
	  } else {
	    printf("ERROR no dof with Seismic event pattern %d", ii);
	    exit(-1);
	  }
	}
      } else {
	printf("ERROR no patterns with Seismic event");
	exit(-1);
      }
      
      //    json_dump_file(eventObj,"TEST",0);

      json_t *responsesArray = json_array(); // for each analysis event
      
      // max ground acceleration
      json_t *responseA = json_object();
      json_object_set(responseA,"type",json_string("PGA"));      
      json_object_set(responseA,"dofs",theDOFs);
      json_t *dataArrayA = json_array(); 
      json_object_set(responseA,"scalar_data",dataArrayA);
      json_array_append(responsesArray,responseA);
      numEDP += numDOF;

      json_object_set(eventObj,"responses",responsesArray);
      
      json_array_append(eventArray,eventObj);

      if (tDOFs != 0)
	delete [] tDOFs;
    }

    json_object_set(rootEDP,"total_number_edp",json_integer(numEDP));  
    json_object_set(rootEDP,"EngineeringDemandParameters",eventArray);  

    json_dump_file(rootEDP,filenameEDP,0);   

    return 0;
}
