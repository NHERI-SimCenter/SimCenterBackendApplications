#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
using namespace std;
#include <jansson.h>  // for Json

int main(int argc, char ** argv) {

  //
  // parse input args for filenames
  //

  char * filenameAIM = NULL;
  char * filenameEVENT = NULL;
  char * filenameSAM = NULL;
  char * filenameEDP = NULL;

  bool doRV = false;

  int arg = 1;
  while (arg < argc) {
    if (strcmp(argv[arg], "--filenameAIM") == 0) {
      arg++;
      filenameAIM = argv[arg];
    } else if (strcmp(argv[arg], "--filenameEVENT") == 0) {
      arg++;
      filenameEVENT = argv[arg];
    } else if (strcmp(argv[arg], "--filenameSAM") == 0) {
      arg++;
      filenameSAM = argv[arg];
    } else if (strcmp(argv[arg], "--filenameEDP") == 0) {
      arg++;
      filenameEDP = argv[arg];
    } else if (strcmp(argv[arg], "--getRV") == 0) {
      doRV = true;
    }
    arg++;
  }

  if (filenameAIM == NULL || filenameEVENT == NULL || filenameSAM == NULL || filenameEDP == NULL) {
    std::cerr << "FATAL ERROR - no aim, sam, evt, or edp filefile provided\n";
  }

  if (doRV == true) { // only do if --getRV is passed

    // create output JSON object
    json_t * rootEDP = json_object();

    // place an empty random variable field
    json_t * rvArray = json_array();
    json_object_set(rootEDP, "RandomVariables", rvArray);

    //
    // for each event we create the edp's
    //

    json_t * eventArray = json_array(); // for each analysis event

    // load SAM and EVENT files
    json_error_t error;

    json_t * rootEVENT = json_load_file(filenameEVENT, 0, & error);
    json_t * eventsArray = json_object_get(rootEVENT, "Events");

    json_t * rootSAM = json_load_file(filenameSAM, 0, & error);
    json_t * mappingArray = json_object_get(rootSAM, "NodeMapping");
    json_t * theNDM = json_object_get(rootSAM, "ndm");
    int ndm = json_integer_value(theNDM);

    int index;
    json_t * value;

    int numEDP = 0;


    json_t* mainJsonAIM  = json_load_file(filenameAIM, 0, & error);
    json_t* edpObj = json_object_get(mainJsonAIM, "EDP");
    json_t* comptArray = json_object_get(edpObj, "components");

    json_array_foreach(eventsArray, index, value) {

      // check for wind
      json_t * type = json_object_get(value, "type");
      bool ok = true;
      if (type == NULL) {
        printf("WARNING no event type in event object %d\n", index);
        ok = false;
      }
      const char * eventType = json_string_value(type);
      if (ok != true && eventType == NULL) {
        printf("WARNING event type is not a string for event %d\n", index);
        ok = false;
      }

      if (ok != true && strcmp(eventType, "Wind") != 0) {
        json_object_clear(rootEVENT);
        printf("WARNING event type %s not Wind NO OUTPUT", eventType);
        ok = false;
      }

      // add the EDP for the event
      json_t * eventObj = json_object();

      json_t * name = json_object_get(value, "name");
      const char * eventName = json_string_value(name);
      json_object_set(eventObj, "name", json_string(eventName));

      //
      // create a json_array of ints indicating what dof the event patterns apply to
      //  -- needed in EDP

      int numDOF = 0;
      json_t * theDOFs = json_array();
      int * tDOFs = 0;
      json_t * patternArray = json_object_get(value, "pattern");
      int numPattern = json_array_size(patternArray);
      tDOFs = new int[numPattern];

      if (numPattern != 0) {
        for (int ii = 0; ii < numPattern; ii++){
          json_t * thePattern = json_array_get(patternArray, ii);
          json_t * theDof = json_object_get(thePattern, "dof");
          int theDOF = json_integer_value(theDof);
          if (theDOF < 3) {
            bool dofIncluded = false;
            for (int j = 0; j < numDOF; j++){
              if(tDOFs[j] == theDOF){
                dofIncluded = true;
                }
            }
            if (!dofIncluded) {
              tDOFs[numDOF] = theDOF;
              json_array_append(theDOFs, theDof);
              numDOF++;
            }
          }
        }
      } 
      else {
        printf("ERROR no patterns");
        exit(-1);
      }

      for (int ii = 0; ii < numDOF; ii++) {
        std::cerr << tDOFs[ii] << " ";
      }

      //    json_dump_file(eventObj,"TEST",0);

      json_t * responsesArray = json_array(); // for each analysis event

      // create responses for floor accel and story drift 

      int mapIndex1;
      json_t * value1;

      int count = 0;
      const char * floor1 = 0;
      const char * cline = 0;
      const char * floor = 0;

      json_array_foreach(mappingArray, mapIndex1, value1){
        cline = json_string_value(json_object_get(value1, "cline"));
        floor = json_string_value(json_object_get(value1, "floor"));
        int node = json_integer_value(json_object_get(value1, "node"));

        if (strcmp(cline, "response") == 0) {

          if (count > 0) {

            // floor abs acceleration
            json_t * responseA = json_object();
            json_object_set(responseA, "type", json_string("max_abs_acceleration"));
            json_object_set(responseA, "cline", json_string(cline));
            json_object_set(responseA, "floor", json_string(floor));
            json_object_set(responseA, "dofs", theDOFs);
            json_t * dataArrayA = json_array();
            json_object_set(responseA, "scalar_data", dataArrayA);
            json_array_append(responsesArray, responseA);
            numEDP += numDOF;

            // floor abs acceleration
            json_t * responseRMS = json_object();
            json_object_set(responseRMS, "type", json_string("rms_acceleration"));
            json_object_set(responseRMS, "cline", json_string(cline));
            json_object_set(responseRMS, "floor", json_string(floor));
            json_object_set(responseRMS, "dofs", theDOFs);
            json_t * dataArrayRMS = json_array();
            json_object_set(responseRMS, "scalar_data", dataArrayRMS);
            json_array_append(responsesArray, responseRMS);
            numEDP += numDOF;

            // floor relative disp
            json_t * responseD = json_object();
            json_object_set(responseD, "type", json_string("max_rel_disp"));
            json_object_set(responseD, "cline", json_string(cline));
            json_object_set(responseD, "floor", json_string(floor));
            json_object_set(responseD, "dofs", theDOFs);
            json_t * dataArrayD = json_array();
            json_object_set(responseD, "scalar_data", dataArrayD);
            json_array_append(responsesArray, responseD);
            numEDP += numDOF;

            // drift
            for (int i = 0; i < numDOF; i++) {
              json_t * response = json_object();
              json_object_set(response, "type", json_string("max_drift"));
              json_object_set(response, "cline", json_string(cline));
              json_object_set(response, "floor1", json_string(floor1));
              json_object_set(response, "floor2", json_string(floor));

              // we cannot just add dof's as before in case vertical
              json_t * dofArray = json_array();
              json_array_append(dofArray, json_integer(tDOFs[i]));

              //	    if (tDOFs[i] != ndm) {
              numEDP += 1; // drift and pressure
              //	    }
              json_object_set(response, "dofs", dofArray);

              json_t * dataArray = json_array();
              json_object_set(response, "scalar_data", dataArray);
              json_array_append(responsesArray, response);
            }
          }

          floor1 = floor;
          count++;
        }
      }

      // Load data, peak pressure and forces 
      json_t *loadsArray = json_array();
      const char* compName = 0;
      int comp_idx;
      json_t * comp_value;

      json_array_foreach(comptArray, comp_idx, comp_value) 
      {
        compName = json_string_value(json_object_get(comp_value, "componentName"));

        //Peak pressure EDP
        json_t *peak_pressure = json_object();
        json_object_set(peak_pressure, "type", json_string("peak_pressure"));      
        json_object_set(peak_pressure, "cline", json_string("load"));      
        json_object_set(peak_pressure, "name", json_string(compName));
        json_t *peak_pressure_data = json_array(); 
        json_object_set(peak_pressure,"scalar_data", peak_pressure_data);     

        //Mean pressure EDP
        json_t *mean_pressure = json_object();
        json_object_set(mean_pressure, "type", json_string("mean_pressure"));      
        json_object_set(mean_pressure, "cline", json_string("load"));      
        json_object_set(mean_pressure, "name", json_string(compName));
        json_t *mean_pressure_data = json_array(); 
        json_object_set(mean_pressure,"scalar_data", mean_pressure_data);     

        //RMS pressure EDP
        json_t *rms_pressure = json_object();
        json_object_set(rms_pressure, "type", json_string("rms_pressure"));      
        json_object_set(rms_pressure, "cline", json_string("load"));      
        json_object_set(rms_pressure, "name", json_string(compName));
        json_t *rms_pressure_data = json_array(); 
        json_object_set(rms_pressure,"scalar_data", rms_pressure_data);     


        /*===========================================================*/
        /*================= For the Forces ==========================*/
        /*===========================================================*/

        // Peak forces 
        json_t *peak_force = json_object();
        json_object_set(peak_force, "type", json_string("peak_force"));      
        json_object_set(peak_force, "cline", json_string("load"));      
        json_object_set(peak_force, "name", json_string(compName));
        json_t *peak_force_data = json_array(); 
        json_object_set(peak_force,"scalar_data", peak_force_data);     
        
        // Mean forces 
        json_t *mean_force = json_object();
        json_object_set(mean_force, "type", json_string("mean_force"));      
        json_object_set(mean_force, "cline", json_string("load"));      
        json_object_set(mean_force, "name", json_string(compName));
        json_t *mean_force_data = json_array(); 
        json_object_set(mean_force,"scalar_data", mean_force_data);     

        // RMS forces 
        json_t *rms_force = json_object();
        json_object_set(rms_force, "type", json_string("rms_force"));      
        json_object_set(rms_force, "cline", json_string("load"));      
        json_object_set(rms_force, "name", json_string(compName));
        json_t *rms_force_data = json_array(); 
        json_object_set(rms_force,"scalar_data", rms_force_data);  

        json_array_append(loadsArray, peak_pressure);
        json_array_append(loadsArray, mean_pressure);
        json_array_append(loadsArray, rms_pressure);
        json_array_append(loadsArray, peak_force);
        json_array_append(loadsArray, mean_force);
        json_array_append(loadsArray, rms_force);
      }
      
      json_object_set(eventObj, "responses", responsesArray);
      json_object_set(eventObj, "loads", loadsArray);
      json_array_append(eventArray, eventObj);

      if (tDOFs != 0)
        delete[] tDOFs;
    }

    json_object_set(rootEDP, "total_number_edp", json_integer(numEDP));
    json_object_set(rootEDP, "EngineeringDemandParameters", eventArray);

    json_dump_file(rootEDP, filenameEDP, 0);
    
  } 
  else 
  {
    
    // fill in EDP with the data

    // 
    // Open the EDP JSON file
    //
    
    FILE *file = fopen(filenameEDP, "r");
    if (!file) {
        std::cerr << "Error: Could not open file " << filenameEDP << std::endl;
        return -1;
    }

    // Parse the JSON file
    json_error_t error;
    json_t *rootEDP = json_loadf(file, 0, &error);
    fclose(file); // Close the file after parsing

    if (!rootEDP) {
        std::cerr << "Error: Failed to parse JSON - " << error.text << " at line " << error.line << std::endl;
        return -1;
    }

    // Check if the root is a JSON object
    if (!json_is_object(rootEDP)) {
        std::cerr << "Error: Root element is not a JSON object!" << std::endl;
        json_decref(rootEDP);
        return -1;
    }
    
    json_t* mainJsonAIM  = json_load_file(filenameAIM, 0, & error);
    json_t* evtObj = json_object_get(mainJsonAIM, "Event");

    // std::string casePath = std::string(json_string_value(json_object_get(evtObj, "caseDirectoryPath"))) + "/constant/simCenter/output/windLoads";


   // Reading data from files
    std::vector<std::string> fileNames = {"peakP.csv", "peakP.csv", "peakP.csv", "peakP.csv", "peakP.csv", "peakP.csv"};
    std::vector<std::vector<double>> dataArrays(6);
    
    for (size_t i = 0; i < fileNames.size(); i++) {
        std::ifstream file(fileNames[i]);
        if (!file) {
            std::cerr << "Error: Could not open file " << fileNames[i] << std::endl;
            continue;
        }
        std::string line;
	std::string token;
        while (std::getline(file, line)) {
	  std::stringstream ss(line);
	  while (std::getline(ss, token, ',')) {
	    dataArrays[i].push_back(std::stod(token)); // Convert to double
	  }
        }
        file.close();
    }

    // get loads array in first array element of EngineeringDemandParameters
    json_t *edpArray = json_object_get(rootEDP, "EngineeringDemandParameters");
    if (!json_is_array(edpArray)) {
        std::cerr << "Error: EngineeringDemandParameters is not an array!" << std::endl;
        json_decref(rootEDP);
        return -1;
    }
    
    json_t *firstEDP = json_array_get(edpArray, 0);
    json_t *loads = json_object_get(firstEDP, "loads");
    if (!json_is_array(loads)) {
        std::cerr << "Error: responses is not an array!" << std::endl;
        json_decref(rootEDP);
        return -1;
    }

    // now loop over load entries putting in values

    int count  = 0;
    for (size_t j = 0; j < json_array_size(loads); j++) {
      
      json_t *load = json_array_get(loads, j);
      json_t *scalarData = json_object_get(load, "scalar_data");
      json_t *loadType = json_object_get(load, "type");

      if (!json_is_array(scalarData)) {
        std::cerr << "Error: scalar_data is not an array!" << std::endl;
        continue;
      }

      // Add 1 to the "scalar_data" array

      if(strcmp(json_string_value(loadType),"mean_pressure")){
        json_array_append_new(scalarData, json_real(dataArrays[0][count]));
      }
      else if(strcmp(json_string_value(loadType),"rms_pressure")){
        json_array_append_new(scalarData, json_real(dataArrays[1][count]));
      }
	else if(strcmp(json_string_value(loadType),"peak_pressure")){
        json_array_append_new(scalarData, json_real(dataArrays[2][count]));
      }
	else if(strcmp(json_string_value(loadType),"mean_force")){
        json_array_append_new(scalarData, json_real(dataArrays[3][count]));
      }
	else if(strcmp(json_string_value(loadType),"rms_force")){
        json_array_append_new(scalarData, json_real(dataArrays[4][count]));
      }
	else if(strcmp(json_string_value(loadType),"peak_force")){
        json_array_append_new(scalarData, json_real(dataArrays[5][count]));
      }

      count ++ ;
    }

    json_dump_file(rootEDP, filenameEDP, 0);
  }

  return 0;
}
