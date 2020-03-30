#include <iostream>
#include <fstream>
#include <jansson.h> 
#include <string.h>
#include <string>
#include <sstream>
#include <list>


struct normalRV {
  std::string name;
  double mean;
  double stdDev;
};

struct lognormalRV {
  std::string name;
  double mean;
  double stdDev;
};

struct constantRV {
  std::string name;
  double value;
};

struct uniformRV {
  std::string name;
  double lowerBound;
  double upperBound;
};

struct continuousDesignRV {
  std::string name;
  double lowerBound;
  double upperBound;
  double initialPoint;
};

struct weibullRV {
  std::string name;
  double scaleParam;
  double shapeParam;
};

struct gammaRV {
  std::string name;
  double alphas;
  double betas;
};

struct gumbellRV {
  std::string name;
  double alphas;
  double betas;
};

struct betaRV {
  std::string name;
  double alphas;
  double betas;
  double lowerBound;
  double upperBound;
};

struct discreteDesignSetRV {
  std::string name;
  std::list<std::string> elements;
};

struct randomVariables {
  int numRandomVariables;
  std::list<struct normalRV> normalRVs;
  std::list<struct lognormalRV> lognormalRVs;
  std::list<struct constantRV> constantRVs;
  std::list<struct uniformRV> uniformRVs;
  std::list<struct continuousDesignRV> continuousDesignRVs;
  std::list<struct weibullRV> weibullRVs;
  std::list<struct gammaRV> gammaRVs;
  std::list<struct gumbellRV> gumbellRVs;
  std::list<struct betaRV> betaRVs;
  std::list<struct discreteDesignSetRV> discreteDesignSetRVs;
};
  

// parses JSON for random variables & returns number found
  
int
parseForRV(json_t *root, struct randomVariables &theRandomVariables){ 

  int numberRVs = 0;

  json_t *fileRandomVariables =  json_object_get(root, "randomVariables");
  if (fileRandomVariables == NULL) {
    return 0; // no random variables is allowed
  }
  
  int numRVs = json_array_size(fileRandomVariables);
  for (int i=0; i<numRVs; i++) {
    json_t *fileRandomVariable = json_array_get(fileRandomVariables,i);
    const char *variableType = json_string_value(json_object_get(fileRandomVariable,"distribution"));
    
    if ((strcmp(variableType, "Normal") == 0) || (strcmp(variableType, "normal")==0)) {
      
      struct normalRV theRV;
      
      theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));
      theRV.mean = json_number_value(json_object_get(fileRandomVariable,"mean"));
      theRV.stdDev = json_number_value(json_object_get(fileRandomVariable,"stdDev"));
      
      theRandomVariables.normalRVs.push_back(theRV);
      theRandomVariables.numRandomVariables += 1;
      numberRVs++;

    }

    else if ((strcmp(variableType, "Lognormal") == 0) || (strcmp(variableType, "lognormal") == 0)) {

      struct lognormalRV theRV;

      theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));
      theRV.mean = json_number_value(json_object_get(fileRandomVariable,"mean"));
      theRV.stdDev = json_number_value(json_object_get(fileRandomVariable,"stdDev"));

      theRandomVariables.lognormalRVs.push_back(theRV);
      theRandomVariables.numRandomVariables += 1;
      numberRVs++;

    }

    else if (strcmp(variableType, "Constant") == 0) {

      struct constantRV theRV;

      theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));
      theRV.value = json_number_value(json_object_get(fileRandomVariable,"value"));

      theRandomVariables.constantRVs.push_back(theRV);
      theRandomVariables.numRandomVariables += 1;
      numberRVs++;

    }

    else if (strcmp(variableType, "Uniform") == 0) {

      struct uniformRV theRV;

      theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));
      theRV.lowerBound = json_number_value(json_object_get(fileRandomVariable,"lowerbound"));
      theRV.upperBound = json_number_value(json_object_get(fileRandomVariable,"upperbound"));

      theRandomVariables.uniformRVs.push_back(theRV);
      theRandomVariables.numRandomVariables += 1;
      numberRVs++;

    }

    else if (strcmp(variableType, "ContinuousDesign") == 0) {
      struct continuousDesignRV theRV;

      theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));
      theRV.lowerBound = json_number_value(json_object_get(fileRandomVariable,"lowerbound"));
      theRV.upperBound = json_number_value(json_object_get(fileRandomVariable,"upperbound"));
      theRV.initialPoint = json_number_value(json_object_get(fileRandomVariable,"initialpoint"));

      theRandomVariables.continuousDesignRVs.push_back(theRV);
      theRandomVariables.numRandomVariables += 1;
      numberRVs++;
    }

    else if (strcmp(variableType, "Weibull") == 0) {

      struct weibullRV theRV;

      theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));
      theRV.shapeParam = json_number_value(json_object_get(fileRandomVariable,"shapeparam"));
      theRV.scaleParam = json_number_value(json_object_get(fileRandomVariable,"scaleparam"));

      theRandomVariables.weibullRVs.push_back(theRV);
      theRandomVariables.numRandomVariables += 1;
      numberRVs++;
    }

    else if (strcmp(variableType, "Gamma") == 0) {

      struct gammaRV theRV;

      theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));
      theRV.alphas = json_number_value(json_object_get(fileRandomVariable,"alphas"));
      theRV.betas = json_number_value(json_object_get(fileRandomVariable,"betas"));

      theRandomVariables.gammaRVs.push_back(theRV);
      theRandomVariables.numRandomVariables += 1;
      numberRVs++;
    }

    else if (strcmp(variableType, "Gumbell") == 0) {

      struct gumbellRV theRV;

      theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));
      theRV.alphas = json_number_value(json_object_get(fileRandomVariable,"alphas"));
      theRV.betas = json_number_value(json_object_get(fileRandomVariable,"betas"));

      theRandomVariables.gumbellRVs.push_back(theRV);
      theRandomVariables.numRandomVariables += 1;
      numberRVs++;
    }


    else if (strcmp(variableType, "Beta") == 0) {

      struct betaRV theRV;

      theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));
      theRV.alphas = json_number_value(json_object_get(fileRandomVariable,"alphas"));
      theRV.betas = json_number_value(json_object_get(fileRandomVariable,"betas"));
      theRV.lowerBound = json_number_value(json_object_get(fileRandomVariable,"lowerbound"));
      theRV.upperBound = json_number_value(json_object_get(fileRandomVariable,"upperbound"));
      std::cerr << theRV.name << " " << theRV.upperBound << " " << theRV.lowerBound << " " << theRV.alphas << " " << theRV.betas;
      theRandomVariables.betaRVs.push_back(theRV);
      theRandomVariables.numRandomVariables += 1;
      numberRVs++;
    }

    else if (strcmp(variableType, "discrete_design_set_string") == 0) {

      struct discreteDesignSetRV theRV;

      theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));
      std::list<std::string> theValues;
      json_t *elementsSet =  json_object_get(fileRandomVariable, "elements");
      if (elementsSet != NULL) {

	int numValues = json_array_size(elementsSet);
	for (int j=0; j<numValues; j++) {
	  json_t *element = json_array_get(elementsSet,j);
	  std::string value = json_string_value(element);
	    theValues.push_back(value);
	}

	theRV.elements = theValues;

	theRandomVariables.discreteDesignSetRVs.push_back(theRV);
	theRandomVariables.numRandomVariables += 1;
	numberRVs++;
      }
    }

  } // end loop over random variables

  return numRVs;
}


int
writeRV(std::ofstream &dakotaFile, struct randomVariables &theRandomVariables, std::string idVariables){ 


  if (idVariables.empty())
    dakotaFile << "variables \n active uncertain \n";
  else
    dakotaFile << "variables \n id_variables =  '" << idVariables << "'\n active uncertain \n";    

    int numNormalUncertain = theRandomVariables.normalRVs.size();

    int numNormal = theRandomVariables.normalRVs.size();
    if (theRandomVariables.normalRVs.size() > 0) {
      dakotaFile << "  normal_uncertain = " << numNormal << "\n    means = ";
      // std::list<struct normalRV>::iterator it;
      for (auto it = theRandomVariables.normalRVs.begin(); it != theRandomVariables.normalRVs.end(); it++)
	dakotaFile << it->mean << " ";
      dakotaFile << "\n    std_deviations = ";
      for (auto it = theRandomVariables.normalRVs.begin(); it != theRandomVariables.normalRVs.end(); it++)
	dakotaFile << it->stdDev << " ";
      dakotaFile << "\n    descriptors = ";
      for (auto it = theRandomVariables.normalRVs.begin(); it != theRandomVariables.normalRVs.end(); it++)
	dakotaFile << "\'" << it->name << "\' ";
      dakotaFile << "\n";
    }

    int numLognormal = theRandomVariables.lognormalRVs.size();
    if (numLognormal > 0) {
      dakotaFile << "  lognormal_uncertain = " << numLognormal << "\n    means = ";
      //      std::list<struct lognormalRV>::iterator it;
      for (auto it = theRandomVariables.lognormalRVs.begin(); it != theRandomVariables.lognormalRVs.end(); it++)
	dakotaFile << it->mean << " ";
      dakotaFile << "\n    std_deviations = ";
      for (auto it = theRandomVariables.lognormalRVs.begin(); it != theRandomVariables.lognormalRVs.end(); it++)
	dakotaFile << it->stdDev << " ";
      dakotaFile << "\n    descriptors = ";
      for (auto it = theRandomVariables.lognormalRVs.begin(); it != theRandomVariables.lognormalRVs.end(); it++)
	dakotaFile << "\'" << it->name << "\' ";
      dakotaFile << "\n";
    }

    int numUniform = theRandomVariables.uniformRVs.size();
    if (numLognormal > 0) {
      dakotaFile << "  uniform_uncertain = " << numUniform << "\n    lower_bounds = ";
      // std::list<struct uniformRV>::iterator it;
      for (auto it = theRandomVariables.uniformRVs.begin(); it != theRandomVariables.uniformRVs.end(); it++)
	dakotaFile << it->lowerBound << " ";
      dakotaFile << "\n    upper_bound = ";
      for (auto it = theRandomVariables.uniformRVs.begin(); it != theRandomVariables.uniformRVs.end(); it++)
	dakotaFile << it->upperBound << " ";
      dakotaFile << "\n    descriptors = ";
      for (auto it = theRandomVariables.uniformRVs.begin(); it != theRandomVariables.uniformRVs.end(); it++)
	dakotaFile << "\'" << it->name << "\' ";
      dakotaFile << "\n";
    }

    int numContinuousDesign = theRandomVariables.continuousDesignRVs.size();
    if (numContinuousDesign > 0) {
      dakotaFile << "  continuous_design = " << numContinuousDesign << "\n    initial_point = ";
      // std::list<struct continuousDesignRV>::iterator it;
      for (auto it = theRandomVariables.continuousDesignRVs.begin(); it != theRandomVariables.continuousDesignRVs.end(); it++)
	dakotaFile << it->initialPoint << " ";
      dakotaFile << "\n    lower_bounds = ";
      for (auto it = theRandomVariables.continuousDesignRVs.begin(); it != theRandomVariables.continuousDesignRVs.end(); it++)
	dakotaFile << it->lowerBound << " ";
      dakotaFile << "\n    upper_bounds = ";
      for (auto it = theRandomVariables.continuousDesignRVs.begin(); it != theRandomVariables.continuousDesignRVs.end(); it++)
	dakotaFile << it->upperBound << " ";
      dakotaFile << "\n    descriptors = ";
      for (auto it = theRandomVariables.continuousDesignRVs.begin(); it != theRandomVariables.continuousDesignRVs.end(); it++)
	dakotaFile << "\'" << it->name << "\' ";
      dakotaFile << "\n";
    }

    int numWeibull = theRandomVariables.weibullRVs.size();
    if (numWeibull > 0) {
      dakotaFile << "  weibull_uncertain = " << numWeibull << "\n    alphas = ";
      // std::list<struct weibullRV>::iterator it;
      for (auto it = theRandomVariables.weibullRVs.begin(); it != theRandomVariables.weibullRVs.end(); it++)
	dakotaFile << it->shapeParam << " ";
      dakotaFile << "\n    betas = ";
      for (auto it = theRandomVariables.weibullRVs.begin(); it != theRandomVariables.weibullRVs.end(); it++)
	dakotaFile << it->scaleParam << " ";
      dakotaFile << "\n    descriptors = ";
      for (auto it = theRandomVariables.weibullRVs.begin(); it != theRandomVariables.weibullRVs.end(); it++)
	dakotaFile << "\'" << it->name << "\' ";
      dakotaFile << "\n";
    }

    int numGumbell = theRandomVariables.gumbellRVs.size();
    if (numGumbell > 0) {
      dakotaFile << "  gumbell_uncertain = " << numGumbell << "\n    alphas = ";
      // std::list<struct gumbellRV>::iterator it;
      for (auto it = theRandomVariables.gumbellRVs.begin(); it != theRandomVariables.gumbellRVs.end(); it++)
	dakotaFile << it->alphas << " ";
      dakotaFile << "\n    betas = ";
      for (auto it = theRandomVariables.gumbellRVs.begin(); it != theRandomVariables.gumbellRVs.end(); it++)
	dakotaFile << it->betas << " ";
      dakotaFile << "\n    descriptors = ";
      for (auto it = theRandomVariables.gumbellRVs.begin(); it != theRandomVariables.gumbellRVs.end(); it++)
	dakotaFile << "\'" << it->name << "\' ";
      dakotaFile << "\n";
    }


    int numGamma = theRandomVariables.gammaRVs.size();
    if (numGamma > 0) {
      dakotaFile << "  gamma_uncertain = " << numGamma << "\n    alphas = ";
      std::list<struct gammaRV>::iterator it;
      for (auto it = theRandomVariables.gammaRVs.begin(); it != theRandomVariables.gammaRVs.end(); it++)
	dakotaFile << it->alphas << " ";
      dakotaFile << "\n    betas = ";
      for (auto it = theRandomVariables.gammaRVs.begin(); it != theRandomVariables.gammaRVs.end(); it++)
	dakotaFile << it->betas << " ";
      dakotaFile << "\n    descriptors = ";
      for (auto it = theRandomVariables.gammaRVs.begin(); it != theRandomVariables.gammaRVs.end(); it++)
	dakotaFile << "\'" << it->name << "\' ";
      dakotaFile << "\n";
    }

    int numBeta = theRandomVariables.betaRVs.size();
    if (numBeta > 0) {
      dakotaFile << "  beta_uncertain = " << numBeta << "\n    alphas = ";
      //std::list<struct betaRV>::iterator it;
      for (auto it = theRandomVariables.betaRVs.begin(); it != theRandomVariables.betaRVs.end(); it++)
	dakotaFile << it->alphas << " ";
      dakotaFile << "\n    betas = ";
      for (auto it = theRandomVariables.betaRVs.begin(); it != theRandomVariables.betaRVs.end(); it++)
	dakotaFile << it->betas << " ";
      dakotaFile << "\n    lower_bounds = ";
      for (auto it = theRandomVariables.betaRVs.begin(); it != theRandomVariables.betaRVs.end(); it++)
	dakotaFile << it->lowerBound << " ";
      dakotaFile << "\n    upper_bounds = ";
      for (auto it = theRandomVariables.betaRVs.begin(); it != theRandomVariables.betaRVs.end(); it++)
	dakotaFile << it->upperBound << " ";
      dakotaFile << "\n    descriptors = ";
      for (auto it = theRandomVariables.betaRVs.begin(); it != theRandomVariables.betaRVs.end(); it++)
	dakotaFile << "\'" << it->name << "\' ";
      dakotaFile << "\n";
    }
            
    int numDiscreteDesignSet = theRandomVariables.discreteDesignSetRVs.size();
    if (numDiscreteDesignSet > 0) {
      dakotaFile << "    discrete_uncertain_set\n    string " << numDiscreteDesignSet << "\n    num_set_values = ";
      std::list<struct discreteDesignSetRV>::iterator it;
      for (it = theRandomVariables.discreteDesignSetRVs.begin(); it != theRandomVariables.discreteDesignSetRVs.end(); it++)
	dakotaFile << it->elements.size() << " ";
      dakotaFile << "\n    set_values ";
      for (it = theRandomVariables.discreteDesignSetRVs.begin(); it != theRandomVariables.discreteDesignSetRVs.end(); it++) {
	std::list<std::string>::iterator element;
	for (element = it->elements.begin(); element != it->elements.end(); element++) 
	  dakotaFile << " \'" << *element << "\'";
      }
      dakotaFile << "\n    descriptors = ";
      for (auto it = theRandomVariables.discreteDesignSetRVs.begin(); it != theRandomVariables.discreteDesignSetRVs.end(); it++)
	dakotaFile << "\'" << it->name << "\' ";
      dakotaFile << "\n";
    }

    // if no random variables .. create 1 call & call it dummy!
    int numRV = theRandomVariables.numRandomVariables;
    if (numRV == 0) {
      dakotaFile << "   discrete_uncertain_set\n    string 1 \n    num_set_values = 2";      
      dakotaFile << "\n    set_values  '1' '2'";
      dakotaFile << "\n    descriptors = dummy\n";
    }

    dakotaFile << "\n\n";

    return 0;
}

int
writeInterface(std::ostream &dakotaFile, json_t *uqData, std::string &workflowDriver, std::string idInterface) {

  dakotaFile << "interface \n";
  if (!idInterface.empty())
    dakotaFile << "  id_interface = '" << idInterface << "'\n";

  dakotaFile << "  analysis_driver = '" << workflowDriver << "'\n";

  dakotaFile << " fork\n";  

  dakotaFile << "   parameters_file = 'params.in'\n";
  dakotaFile << "   results_file = 'results.out' \n";
  dakotaFile << "   aprepro \n";
  dakotaFile << "   work_directory\n";
  dakotaFile << "     named \'workdir\' \n";
  dakotaFile << "     directory_tag\n";
  dakotaFile << "     directory_save\n";

  /*
    if uqData['keepSamples']:
        dakota_input += ('        directory_save\n')    
  */

  dakotaFile << "     copy_files = 'templatedir/*' \n";
  dakotaFile << "  asynchronous\n\n";

  /*
  if (runType == "local") {
    uqData['concurrency'] = uqData.get('concurrency', 4)
  }    
  if uqData['concurrency'] == None:
     dakota_input += "  asynchronous\n"
  elif uqData['concurrency'] > 1:
     dakota_input += "  asynchronous evaluation_concurrency = {}\n".format(uqData['concurrency'])
  }
  */
}

int
writeResponse(std::ostream &dakotaFile, json_t *rootEDP, std::string idResponse) {

  int numResponses = json_integer_value(json_object_get(rootEDP,"total_number_edp"));

  dakotaFile << "responses\n";

  if (!idResponse.empty())
    dakotaFile << "  id_responses = '" << idResponse << "'\n";

  dakotaFile << "  response_functions = " << numResponses << "\nresponse_descriptors = ";

  json_t *EDPs = json_object_get(rootEDP,"EngineeringDemandParameters");
  if (EDPs == NULL) {
    dakotaFile << 0 << "\n";
    return 803;
  }
  
  int numEvents = json_array_size(EDPs);

  // loop over all events
  for (int i=0; i<numEvents; i++) {

    json_t *event = json_array_get(EDPs,i);
    json_t *eventEDPs = json_object_get(event,"responses");
    int numResponses = json_array_size(eventEDPs);  

    // loop over all edp for the event
    for (int j=0; j<numResponses; j++) {

      json_t *eventEDP = json_array_get(eventEDPs,j);
      const char *eventType = json_string_value(json_object_get(eventEDP,"type"));
      bool known = false;
      std::string edpAcronym("");
      const char *floor = NULL;
      std::cerr << "writeResponse: type: " << eventType;
      // based on edp do something 
      if (strcmp(eventType,"max_abs_acceleration") == 0) {
	edpAcronym = "PFA";
	floor = json_string_value(json_object_get(eventEDP,"floor"));
	known = true;
      } else if	(strcmp(eventType,"max_drift") == 0) {
	edpAcronym = "PID";
	floor = json_string_value(json_object_get(eventEDP,"floor2"));
	known = true;
      } else if	(strcmp(eventType,"residual_disp") == 0) {
	edpAcronym = "RD";
	floor = json_string_value(json_object_get(eventEDP,"floor"));
	known = true;
      } else if (strcmp(eventType,"max_pressure") == 0) {
	edpAcronym = "PSP";
	floor = json_string_value(json_object_get(eventEDP,"floor2"));
	known = true;
      } else if (strcmp(eventType,"max_rel_disp") == 0) {
	edpAcronym = "PFD";
	floor = json_string_value(json_object_get(eventEDP,"floor"));
	known = true;
      } else if (strcmp(eventType,"peak_wind_gust_speed") == 0) {
	edpAcronym = "PWS";
	floor = json_string_value(json_object_get(eventEDP,"floor"));
	known = true;
      } else {
	dakotaFile << "'" << eventType << "' ";
      }

      if (known == true) {
	json_t *dofs = json_object_get(eventEDP,"dofs");
	int numDOF = json_array_size(dofs);

	// loop over all edp for the event
	for (int k=0; k<numDOF; k++) {
	  int dof = json_integer_value(json_array_get(dofs,k));
	  dakotaFile << "'" << i+1 << "-" << edpAcronym << "-" << floor << "-" << dof << "' ";
	}
      }
    }
  }

  dakotaFile << "\nno_gradients\nno_hessians\n\n";

  return numResponses;
}


int main(int argc, const char **argv) {

  const char *bimName = argv[1];
  const char *evtName = argv[2];
  const char *samName = argv[3];
  const char *edpName = argv[4];
  const char *simName = argv[5];
  const char *driver = argv[6];
  const char *runType = argv[7];
  const char *osType = argv[8];

  struct randomVariables theRandomVariables;
  theRandomVariables.numRandomVariables = 0;

  std::string workflowDriver = "workflow_driver";
  if ((strcmp(osType,"Windows") == 0) && (strcmp(runType,"Local") == 0))
    workflowDriver = "workflow_driver.bat";


  //
  // open workflow driver 
  //

  std::ofstream workflowDriverFile(workflowDriver);

  if (!workflowDriverFile.is_open()) {
    std::cerr << "parseFileForRV:: could not create dakota input file: dakota.in\n";
    exit(802); // no random variables is allowed
  }

  //
  // open files & parse for RV, if RV found rename file, and put text in workflow driver for dprepro
  // 

  // open bim

  json_error_t error;
  json_t *rootINPUT = json_load_file(bimName, 0, &error);
  if (rootINPUT == NULL) {
    std::cerr << "parseFileForRV:: could not open BIM file with name: " << bimName << "\n";
    exit(801); // no random variables is allowed
  } 

  std::string dpreproCommand;
  const char *localDir = json_string_value(json_object_get(rootINPUT,"localAppDir"));
  const char *remoteDir = json_string_value(json_object_get(rootINPUT,"remoteAppDir"));

  if (strcmp(runType, "local")) {
    dpreproCommand = localDir + std::string("/applications/performUQ/dakota/simCenterDprepro");
  } else {
    dpreproCommand = remoteDir + std::string("/applications/performUQ/dakota/simCenterDprepro");
  }

  int numRV = parseForRV(rootINPUT, theRandomVariables);
  if (numRV > 0) {
    if (rename(bimName, "bim.j") != 0) {
      std::cerr << "preprocessDakota - cound not rename bim file\n";
      exit(802);
    }
    workflowDriverFile << dpreproCommand << " params.in bim.j " << bimName << "\n";
  } 

  // load event read random variables, if any rename file and add a dprepro line to workflow

  json_t *rootEVT = json_load_file(evtName, 0, &error);
  if (rootEVT == NULL) {
    std::cerr << "parseFileForRV:: could not open EVT file with name: " << evtName << "\n";
    exit(801); // no random variables is allowed
  } 
  numRV = parseForRV(rootEVT, theRandomVariables);
  if (numRV > 0) {
    if (rename(evtName, "evt.j") != 0) {
      std::cerr << "preprocessDakota - cound not rename event file\n";
      exit(802);
    }
    workflowDriverFile << dpreproCommand << " params.in evt.j " << evtName << "\n";
  }

  // load sam, read random variables, if any rename file and add a dprepro line to workflow

  json_t *rootSAM = json_load_file(samName, 0, &error);
  if (rootSAM == NULL) {
  std::cerr << "parseFileForRV:: could not open SAM file with name: " << samName << "\n";
    exit(801); // no random variables is allowed
  } 
  numRV = parseForRV(rootSAM, theRandomVariables);
  if (numRV > 0) {
    if (rename(samName, "sam.j") != 0) {
      std::cerr << "preprocessDakota - cound not rename bim file\n";
      exit(802);
    }
    workflowDriverFile << dpreproCommand << " params.in sam.j " << samName << "\n";
  }

  // load sim, read random variables, if any rename file and add a dprepro to workflow

  json_t *rootSIM = json_load_file(simName, 0, &error);
  if (rootSIM == NULL) {
    std::cerr << "parseFileForRV:: could not open SIM file with name: " << simName << "\n";
    exit(801); // no random variables is allowed
  } 
  numRV = parseForRV(rootSIM, theRandomVariables);
  if (numRV > 0) {
    if (rename(simName, "sim.j") != 0) {
      std::cerr << "preprocessDakota - cound not rename sim file\n";
      exit(802);
    }
    workflowDriverFile << dpreproCommand << " params.in sim.j " << simName << "\n";
  }

  // load edp, read random variables, if any rename file and add a dprepro to workflow

  json_t *rootEDP = json_load_file(edpName, 0, &error);
  if (rootEDP == NULL) {
    std::cerr << "parseFileForRV:: could not open EDP file with name: " << edpName << "\n";
    exit(801); // no random variables is allowed
  } 
  numRV = parseForRV(rootEDP, theRandomVariables);
  if (numRV > 0) {
    if (rename(edpName, "edp.j") != 0) {
      std::cerr << "preprocessDakota - cound not rename edp file\n";
      exit(802);
    }
    workflowDriverFile << dpreproCommand << " params.in edp.j " << edpName << "\n";
  }

  //
  // open driver file, copy to workflorDriverFile and close files
  //

  std::ifstream originalDriverFile(driver);
  std::string line;
  while (std::getline(originalDriverFile, line)) {
    workflowDriverFile << line << "\n";
  }

  std::string extractEDP;
  if (strcmp(runType, "local")) {
    extractEDP = localDir + std::string("/applications/performUQ/dakota/extractEDP");
  } else {
    extractEDP = remoteDir + std::string("/applications/performUQ/dakota/extractEDP");
  }
  workflowDriverFile << extractEDP << " "  << edpName << "  results.out "  <<  bimName  <<  "\n"; //  + numR + ' ' + files + '\n')

  originalDriverFile.close();
  workflowDriverFile.close();

  //
  // open empty dakota input file
  //

  std::ofstream dakotaFile("dakota.in");

  if (!dakotaFile.is_open()) {
    std::cerr << "parseFileForRV:: could not create dakota input file: dakota.in\n";
    exit(802); // no random variables is allowed
  }

  //
  // get UQ Method data
  // 
  
  json_t *uqData =  json_object_get(rootINPUT, "UQ_Method");
  if (uqData == NULL) {
    std::cerr << "preprocessJSON - no UQ Data in inputfile\n";
    exit(-1); // no random variables is allowed
  }

  const char *type = json_string_value(json_object_get(uqData, "uqType"));
  
  bool sensitivityAnalysis = false;
  if (strcmp(type, "Sensitivity Analysis") == 0)
    sensitivityAnalysis = true;


  //
  // based on method do stuff
  // 

  if ((strcmp(type, "Forward Propagation") == 0) || sensitivityAnalysis == true) {

    json_t *samplingMethodData = json_object_get(uqData,"samplingMethodData");

    const char *method = json_string_value(json_object_get(samplingMethodData,"method"));

    if (strcmp(method,"Monte Carlo")==0) {
      int numSamples = json_integer_value(json_object_get(samplingMethodData,"samples"));
      int seed = json_integer_value(json_object_get(samplingMethodData,"seed"));

      dakotaFile << "environment \n tabular_data \n tabular_data_file = 'dakotaTab.out' \n\n";
      dakotaFile << "method, \n sampling \n sample_type = random \n samples = " << numSamples << " \n seed = " << seed << "\n\n";

      if (sensitivityAnalysis == true)
	dakotaFile << "variance_based_decomp \n\n";

      std::string emptyString;
      writeRV(dakotaFile, theRandomVariables, emptyString);
      writeInterface(dakotaFile, uqData, workflowDriver, emptyString);
      writeResponse(dakotaFile, rootEDP, emptyString);
    }

    else if (strcmp(method,"LHS")==0) {

      int numSamples = json_integer_value(json_object_get(samplingMethodData,"samples"));
      int seed = json_integer_value(json_object_get(samplingMethodData,"seed"));

      std::cerr << numSamples << " " << seed;

      dakotaFile << "environment \n tabular_data \n tabular_data_file = 'dakotaTab.out' \n\n";
      dakotaFile << "method,\n sampling\n sample_type = lhs \n samples = " << numSamples << " \n seed = " << seed << "\n\n";

      if (sensitivityAnalysis == true)
	dakotaFile << "variance_based_decomp \n\n";

      std::string emptyString;
      writeRV(dakotaFile, theRandomVariables, emptyString);
      writeInterface(dakotaFile, uqData, workflowDriver, emptyString);
      writeResponse(dakotaFile, rootEDP, emptyString);
    }

    else if (strcmp(method,"Importance Sampling")==0) {

      const char *isMethod = json_string_value(json_object_get(samplingMethodData,"ismethod"));
      int numSamples = json_integer_value(json_object_get(samplingMethodData,"samples"));
      int seed = json_integer_value(json_object_get(samplingMethodData,"seed"));

      dakotaFile << "environment \n tabular_data \n tabular_data_file = 'dakotaTab.out' \n\n";
      dakotaFile << "method, \n importance_sampling \n " << isMethod << " \n samples = " << numSamples << "\n seed = " << seed << "\n\n";

      std::string emptyString;
      writeRV(dakotaFile, theRandomVariables, emptyString);
      writeInterface(dakotaFile, uqData, workflowDriver, emptyString);
      writeResponse(dakotaFile, rootEDP, emptyString);
    }

    else if (strcmp(method,"Gaussian Process Regression")==0) {

      int trainingSamples = json_integer_value(json_object_get(samplingMethodData,"trainingSamples"));
      int trainingSeed = json_integer_value(json_object_get(samplingMethodData,"trainingSeed"));
      const char *trainMethod = json_string_value(json_object_get(samplingMethodData,"trainingMethod"));    
      int samplingSamples = json_integer_value(json_object_get(samplingMethodData,"samplingSamples"));
      int samplingSeed = json_integer_value(json_object_get(samplingMethodData,"samplingSeed"));
      const char *sampleMethod = json_string_value(json_object_get(samplingMethodData,"samplingMethod"));

      const char *surrogateMethod = json_string_value(json_object_get(samplingMethodData,"surrogateSurfaceMethod"));

      std::string trainingMethod(trainMethod);
      std::string samplingMethod(sampleMethod);
      if (strcmp(trainMethod,"Monte Carlo") == 0)
	trainingMethod = "random";
      if (strcmp(sampleMethod,"Monte Carlo") == 0) 
	samplingMethod = "random";


      dakotaFile << "environment \n method_pointer = 'SurrogateMethod' \n tabular_data \n tabular_data_file = 'dakotaTab.out'\n";
      dakotaFile << "custom_annotated header eval_id \n\n";

      dakotaFile << "method \n id_method = 'SurrogateMethod' \n model_pointer = 'SurrogateModel'\n";
      dakotaFile << " sampling \n samples = " << samplingSamples << "\n seed = " << samplingSeed << "\n sample_type = "
		 << samplingMethod << "\n\n";

      dakotaFile << "model \n id_model = 'SurrogateModel' \n surrogate global \n dace_method_pointer = 'TrainingMethod'\n "
		 << surrogateMethod << "\n\n";

      dakotaFile << "method \n id_method = 'TrainingMethod' \n model_pointer = 'TrainingModel'\n";
      dakotaFile << " sampling \n samples = " << trainingSamples << "\n seed = " << trainingSeed << "\n sample_type = "
		 << trainingMethod << "\n\n";

      dakotaFile << "model \n id_model = 'TrainingModel' \n single \n interface_pointer = 'SimulationInterface'";

      std::string emptyString;
      std::string interfaceString("SimulationInterface");
      writeRV(dakotaFile, theRandomVariables, emptyString);
      writeInterface(dakotaFile, uqData, workflowDriver, interfaceString);
      writeResponse(dakotaFile, rootEDP, emptyString);

    }

    else if (strcmp(method,"Polynomial Chaos Expansion")==0) {

      const char *dataMethod = json_string_value(json_object_get(samplingMethodData,"dataMethod"));    
      int intValue = json_integer_value(json_object_get(samplingMethodData,"level"));
      int samplingSeed = json_integer_value(json_object_get(samplingMethodData,"samplingSeed"));
      int samplingSamples = json_integer_value(json_object_get(samplingMethodData,"samplingSamples"));
      const char *sampleMethod = json_string_value(json_object_get(samplingMethodData,"samplingMethod"));

      std::string pceMethod;
      if (strcmp(dataMethod,"Quadrature") == 0)
	pceMethod = "quadrature_order = ";
      else if (strcmp(dataMethod,"Smolyak Sparse_Grid") == 0)
	pceMethod = "sparse_grid_level = ";
      else if (strcmp(dataMethod,"Stroud Curbature") == 0)
	pceMethod = "cubature_integrand = ";
      else if (strcmp(dataMethod,"Orthogonal Least_Interpolation") == 0)
	pceMethod = "orthogonal_least_squares collocation_points = ";
      else
	pceMethod = "quadrature_order = ";

      std::string samplingMethod(sampleMethod);
      if (strcmp(sampleMethod,"Monte Carlo") == 0) 
	samplingMethod = "random";

      dakotaFile << "environment \n  tabular_data \n tabular_data_file = 'a.out'\n\n"; // a.out for trial data

      std::string emptyString;
      std::string interfaceString("SimulationInterface");
      writeRV(dakotaFile, theRandomVariables, emptyString);
      writeInterface(dakotaFile, uqData, workflowDriver, interfaceString);
      int numResponse = writeResponse(dakotaFile, rootEDP, emptyString);

      dakotaFile << "method \n polynomial_chaos \n " << pceMethod << intValue;
      dakotaFile << "\n samples_on_emulator = " << samplingSamples << "\n seed = " << samplingSeed << "\n sample_type = "
		 << samplingMethod << "\n";
      dakotaFile << " probability_levels = ";
      for (int i=0; i<numResponse; i++)
	dakotaFile << " .1 .5 .9 ";
      dakotaFile << "\n export_approx_points_file = 'dakotaTab.out'\n\n"; // dakotaTab.out for surrogate evaluations
    }

  } else {
    std::cerr << "uqType: NOT KNOWN\n";
    exit(800);
  }

  dakotaFile.close();
  std::cerr << "NUM RV: " << theRandomVariables.numRandomVariables << "\n";
  std::cerr << "DONE PREPROCESSOR .. DONE\n";
  exit(0);
}


