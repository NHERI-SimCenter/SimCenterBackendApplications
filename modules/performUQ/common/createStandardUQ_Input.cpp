#include <iostream>
#include <fstream>
#include <jansson.h>
#include <string>
#include <string.h>
#include <list>
#include <algorithm>
#include <vector>
#include <algorithm>
#include <set>
#include <string.h>
#include <cmath>
using namespace std;
// parses JSON for random variables & returns number found


int
gatherRV(json_t *rootINPUT, std::set<std::string> &rvFiles){

  // std::cerr << "creareStandardUQINput:: gatherRV\n";
  
  json_error_t error;
  json_t *rootRVs = json_object_get(rootINPUT,"randomVariables");
  if (rootRVs == 0) {
    rootRVs = json_object_get(rootINPUT,"RandomVariables");
    if (rootRVs == 0) {    
      std::cerr << "createStandardInput::gatherRV() - no randomVariables in rootINPUT\n";
      return -1;
    }
    json_t *newRootRVs = json_copy(rootRVs);
    json_object_del(rootINPUT, "RandomVariables");
    json_object_set(rootINPUT, "randomVariables", newRootRVs);
    rootRVs = newRootRVs;
    // std::cerr << "createStandardInput::gatherRV() - renaming rootRVs\n";    
  } 

  // vector of names so don't add duplicates
  std::vector<std::string> randomVariableNames;

  if ((rootRVs != NULL) && json_is_array(rootRVs)) {
      size_t index;
      json_t *value;
      
      json_array_foreach(rootRVs, index, value) {
	json_t *nameValue = json_object_get(value, "name");
	const char *name = json_string_value(nameValue);
	randomVariableNames.push_back(std::string(name));
      }
  }

  //
  // look for DefaultValues section & if rvFiles go parse each
  // adding any rv found to rootINPUTS randomVariables section
  //
  
  json_t *rootDefault =  json_object_get(rootINPUT, "DefaultValues");
  if (rootDefault != NULL) {
    json_t *defaultRVs =  json_object_get(rootDefault, "rvFiles");
    if ((defaultRVs != NULL) && json_is_array(defaultRVs)) {
      size_t index;
      json_t *value;
      
      json_array_foreach(defaultRVs, index, value) {
	
	const char *fName = json_string_value(value);
	// std::cerr << "rvFILE: " << fName << "\n";
	json_t *rootOther = json_load_file(fName, 0, &error);
	json_t *fileRandomVariables =  json_object_get(rootOther, "randomVariables");
	if (fileRandomVariables == NULL) {
	  fileRandomVariables =  json_object_get(rootOther, "RandomVariables");
	}
	if (fileRandomVariables != NULL) {
	  // std::cerr << "RANDOM VARIABLES: " << json_dumps(fileRandomVariables, JSON_ENCODE_ANY) << "\n\n";	  
	  int numRVs = json_array_size(fileRandomVariables);
	  //std::cerr << "numRVs=" <<numRVs <<std::endl;
	  for (int i=0; i<numRVs; i++) {
	    json_t *fileRandomVariable = json_array_get(fileRandomVariables,i);
	    json_t *nameValue = json_object_get(fileRandomVariable, "name");
	    const char *name = json_string_value(nameValue);
	    std::string nameS(name);

	    std::vector<std::string>::iterator it;

	    it = std::find(randomVariableNames.begin(), 
			   randomVariableNames.end(), 
			   nameS);

	    if (it == randomVariableNames.end() ) {
	      randomVariableNames.push_back(nameS);
	      json_array_append(rootRVs, fileRandomVariable);
	    }


	  }

    json_t * corrMatJson =  json_object_get(rootINPUT,"correlationMatrix");
    json_t * newCorrMatJson = json_array();
    
    // if correlation matrix exists
    if (corrMatJson != NULL) {

      int numCorrs = json_array_size(corrMatJson);
      int numOrgRVs = std::sqrt(numCorrs);
      int numTotRVs = json_array_size(rootRVs);

      std::cerr << "RANDOM VARIABLES: numORIG: " << numOrgRVs << " new: " << numTotRVs << "\n";

      if (numOrgRVs<numTotRVs){

	double val;
	for (int i=0;i<numTotRVs;i++) {
	  for (int j=0; j<numTotRVs; j++) {
	    if ((i<numOrgRVs) && (j<numOrgRVs)) {
	      val = json_number_value(json_array_get(corrMatJson,i + numOrgRVs*j)); // symmetry so i,j order does not matter
	    } else {
	      if (i == j)
		val = 1.0;
	      else
		val = 0.0;
	    }
	    json_array_append(newCorrMatJson,json_real(val));
	  }
	}
      }
      //json_object_update_existing(json_object_get(rootINPUT,"correlationMatrix"),newCorrMatJson);

      
      json_object_set(rootINPUT, "correlationMatrix", newCorrMatJson);

      //std::cerr << "OLD CORRELATION " << json_dumps(corrMatJson, JSON_ENCODE_ANY) << "\n\n";      
      //std::cerr << "NEW CORRELATION " << json_dumps(newCorrMatJson, JSON_ENCODE_ANY) << "\n\n";
      
    }


	  // KZ: commented for fixing RVs not in BIM.json which is not passed to individual json files (i.e., EVENT.json, SIM.json, SAM.json)
	  //if (numRVs != 0) {
	  rvFiles.insert(std::string(fName));
	  //}
	} else {
	  ; //	std::cerr << "NO RANDOM VARIABLES: " << json_dumps(rootOther, JSON_ENCODE_ANY) << "\n\n";
	}
      }
    }
  } 

  return 0;  
}

int
gatherEDP(json_t *rootINPUT, std::string &edpFile){


  // std::cerr << "creareStandardUQINput:: gatherRV\n";
  
  //
  // get rootEPDs
  //
  
  json_error_t error;
  json_t *rootEDPs = json_object_get(rootINPUT,"EDP");
  
  if (rootEDPs == 0) {
    std::cerr << "createStandardInput::gatherRV() - no EDPs in rootINPUT\n";
  }

  // if not an array, we are going to create new one and update
  bool createdNew = false;
  if ((rootEDPs == NULL) || (!json_is_array(rootEDPs))) {
    rootEDPs = json_array();
    createdNew = true;
  }
  
  //
  // checkfor edpFiles in DefaultValues section
  //  - if present open each file, parse for EDP and add them
  //  - if these edp are from engineering workflows we have to provide names
  //
  
  json_t *rootDefault =  json_object_get(rootINPUT, "DefaultValues");
  if (rootDefault != NULL) {
    json_t *defaultEDPs =  json_object_get(rootDefault, "edpFiles");
    if ((defaultEDPs != NULL) && json_is_array(defaultEDPs)) {
      size_t index;
      json_t *value;
      
      json_array_foreach(defaultEDPs, index, value) {
	const char *fName = json_string_value(value);
	// std::cerr << "Parsing file: " << fName << "\n";
	json_t *rootOther = json_load_file(fName, 0, &error);
	json_t *fileEDPs =  json_object_get(rootOther, "EngineeringDemandParameters");
	if (fileEDPs != NULL) {

	  //std::string fNameString(fName);
	  edpFile = std::string(fName);
	  
	  // for each event write the edps
	  int numEvents = json_array_size(fileEDPs);

	  //
	  // loop over all events
	  //
	  
	  for (int i=0; i<numEvents; i++) {
	    
	    json_t *event = json_array_get(fileEDPs,i);
	    json_t *eventEDPs = json_object_get(event,"responses");
	    int numResponses = json_array_size(eventEDPs);

	    //
	    // loop over all edp for the event
	    //
	    
	    for (int j=0; j<numResponses; j++) {
	      
	      json_t *eventEDP = json_array_get(eventEDPs,j);
	      const char *edpEngType = json_string_value(json_object_get(eventEDP,"type"));
	      bool known = false;
	      std::string edpAcronym("");
	      std::string edpName("");
	      const char *floor = NULL;
	      // std::cerr << "writeResponse: type: " << edpEngType << " " << j << " " << numResponses << "\n";
	      // based on edp do something
	      if (strcmp(edpEngType,"max_abs_acceleration") == 0) {
		edpAcronym = "PFA";
		floor = json_string_value(json_object_get(eventEDP,"floor"));
		known = true;
	      } else if (strcmp(edpEngType,"rms_acceleration") == 0) {
		edpAcronym = "RMSA";
		floor = json_string_value(json_object_get(eventEDP,"floor"));
		known = true;		
	      } else if	(strcmp(edpEngType,"max_drift") == 0) {
		edpAcronym = "PID";
		floor = json_string_value(json_object_get(eventEDP,"floor2"));
		known = true;
	      } else if	(strcmp(edpEngType,"residual_disp") == 0) {
		edpAcronym = "RD";
		floor = json_string_value(json_object_get(eventEDP,"floor"));
		known = true;
	      } else if (strcmp(edpEngType,"max_pressure") == 0) {
		edpAcronym = "PSP";
		floor = json_string_value(json_object_get(eventEDP,"floor2"));
		known = true;
	      } else if (strcmp(edpEngType,"max_rel_disp") == 0) {
		edpAcronym = "PFD";
		floor = json_string_value(json_object_get(eventEDP,"floor"));
		known = true;
	      } else if (strcmp(edpEngType,"max_roof_drift") == 0) {
		edpAcronym = "PRD";
		floor = "1";
		known = true;		
	      } else if (strcmp(edpEngType,"peak_wind_gust_speed") == 0) {
		edpAcronym = "PWS";
		floor = json_string_value(json_object_get(eventEDP,"floor"));
		known = true;
	      } else {
		edpName = edpEngType;
		// edpList.push_back(newEDP);
	      }
	      
	      if (known == true) {
		json_t *dofs = json_object_get(eventEDP,"dofs");
		int numDOF = json_array_size(dofs);
		
		// loop over all edp for the event
		for (int k=0; k<numDOF; k++) {
		  int dof = json_integer_value(json_array_get(dofs,k));
		  edpName = std::string(std::to_string(i+1)) + std::string("-")
		    + edpAcronym
		    + std::string("-")
		    + std::string(floor) +
		    std::string("-") + std::string(std::to_string(dof));
		  
		  // edpList.push_back(newEDP);
		  // add new EDP
		  json_t *newEDP = json_object();
		  json_object_set(newEDP, "length", json_integer(1));
		  json_object_set(newEDP, "type", json_string("scalar"));
		  json_object_set(newEDP, "name", json_string(edpName.c_str()));
		  
		  json_array_append(rootEDPs, newEDP);	      
		}		  
	      }

	      // KZ: for not standard edp (known==false)
	      else {
		// add new EDP as defined by users
		json_t *newEDP = json_object();
		json_object_set(newEDP, "length", json_integer(1));
		json_object_set(newEDP, "type", json_string("scalar"));
		json_object_set(newEDP, "name", json_string(edpName.c_str()));
		json_array_append(rootEDPs, newEDP);	   
		
	      }
	    }
	  }
	} else {
	  
	  // standard EDP
	  
	}	    
      }
    }
  }

  // std::cerr << "DONE PARSING EDP\n";
  
  if (createdNew == true)
    json_object_set(rootINPUT,"EDP", rootEDPs);
      

  return 0;
}

int main(int argc, char **argv) {

  
  const char *inputFile = argv[1];
  const char *outputFile = argv[2];
  const char *workflowOld = argv[3];
  const char *workflowNew = argv[4];  
  const char *runType = argv[5];
  const char *osType = argv[6];  

  //
  // open file & read JSON contents into rootINPUT
  //
  
  json_error_t error;
  json_t *rootINPUT = json_load_file(inputFile, 0, &error);
  if (rootINPUT == NULL) {
    std::cerr << "createStandardUQ_INput could not open file" << inputFile << "\n";
    exit(801); // no random variables is allowed
  }

  //
  // gather other RV and EDP & place in rootINPUT
  //

  
  std::set<std::string> rvFiles;
  std::string edpFile;

  gatherRV(rootINPUT, rvFiles);
  gatherEDP(rootINPUT, edpFile);

  //
  // create new workflow file
  //

  std::string dpreproCommand;
  std::string edpCommand;
  std::string callOldWorkflow;
  std::string localDir(json_string_value(json_object_get(rootINPUT, "localAppDir")));
  std::string remoteDir(json_string_value(json_object_get(rootINPUT, "remoteAppDir")));

  // KZ: for different os system
  std::string workflowNew_os = std::string(workflowNew);
  if ((strcmp(runType, "runningLocal")==0) && strcmp(osType,"Windows") == 0)
  {
    workflowNew_os = std::string(workflowNew)+std::string(".bat");
  }

  std::ofstream workflowDriverFile(workflowNew_os, std::ios::binary);
  //std::ofstream workflowDriverFile(workflowNew, std::ios::binary);
  if (!workflowDriverFile.is_open()) {
    //std::cerr << "parseFileForRV:: could not create workflow driver file: " << workflowNew << "\n";
    std::cerr << "parseFileForRV:: could not create workflow driver file: " << workflowNew_os << "\n";
    exit(802); // no random variables is allowed
  }  

  if ((strcmp(runType,"runningLocal") == 0)
      && strcmp(osType,"Windows") == 0) {
    
    dpreproCommand = std::string("\"") + localDir + std::string("/applications/performUQ/templateSub/simCenterSub.exe\"");
    edpCommand = std::string("\"") + localDir + std::string("/applications/performUQ/common/extractEDP\"");
    
    callOldWorkflow = std::string("call ./") + std::string(workflowOld) + std::string(".bat\n");    
    
  } else {

    workflowDriverFile << "#!/bin/bash\n";
    
    if (strcmp(runType,"runningLocal") == 0) {
      dpreproCommand = std::string("\"") + localDir + std::string("/applications/performUQ/templateSub/simCenterSub\"");
      edpCommand = std::string("\"") + localDir + std::string("/applications/performUQ/common/extractEDP\"");
    } 
    else {
      
      dpreproCommand = std::string("\"") + remoteDir + std::string("/applications/performUQ/templateSub/simCenterSub\"");
      edpCommand = std::string("\"") + remoteDir + std::string("/applications/performUQ/common/extractEDP\"");
      
    }

    callOldWorkflow = std::string("source ./") + std::string(workflowOld);
  }

  if (rvFiles.size() != 0) {

    std::set<std::string>::iterator iter = rvFiles.begin();
    while (iter != rvFiles.end()) {

      std::string fNameCopy  = *iter + std::string(".sc");
      rename (iter->c_str(), fNameCopy.c_str());
      workflowDriverFile << dpreproCommand << " params.in " << fNameCopy << " " << *iter << "\n";
      std::cerr << dpreproCommand << " params.in " << fNameCopy << " " << *iter << "\n";

      //Increment the iterator
      iter++;
    }
  }
  
  // commands to source the workflow driver, extract the edp's into results.ot & close file
  workflowDriverFile << callOldWorkflow;
  workflowDriverFile << "\n" << edpCommand << " " << edpFile << " results.out\n";
  workflowDriverFile.close();

  
  //
  // write rootINPUT to new file
  //
  
  size_t flags = 0;
  return json_dump_file(rootINPUT, outputFile, flags);
}
