#include <iostream>
#include <fstream>
#include <jansson.h>
#include <string.h>
#include <string>
#include <sstream>
#include <list>
#include <vector>
#include <set>

#include "../common/parseWorkflowInput.h"

int
writeRV(std::ostream &dakotaFile, struct randomVariables &theRandomVariables, std::string idVariables, std::vector<std::string> &rvList, bool includeActiveText = true){


    int numContinuousDesign = theRandomVariables.continuousDesignRVs.size();

    if (numContinuousDesign != 0) {

      if (idVariables.empty())
	dakotaFile << "variables \n ";
      else
	dakotaFile << "variables \n id_variables =  '" << idVariables << "'\n";

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
	for (auto it = theRandomVariables.continuousDesignRVs.begin(); it != theRandomVariables.continuousDesignRVs.end(); it++) {
	  dakotaFile << "\'" << it->name << "\' ";
	  rvList.push_back(it->name);
	}
	dakotaFile << "\n\n";
      }


    int numConstant = theRandomVariables.constantRVs.size();
    if (numConstant > 0) {
      dakotaFile << "  discrete_state_set  \n    real = " << numConstant;
      dakotaFile << "\n    elements_per_variable = ";
      for (auto it = theRandomVariables.constantRVs.begin(); it != theRandomVariables.constantRVs.end(); it++)
        dakotaFile << "1 ";     //std::list<struct betaRV>::iterator it;
      dakotaFile << "\n    elements = ";
      for (auto it = theRandomVariables.constantRVs.begin(); it != theRandomVariables.constantRVs.end(); it++)
        dakotaFile << it->value << " ";
      dakotaFile << "\n    descriptors = ";
      for (auto it = theRandomVariables.constantRVs.begin(); it != theRandomVariables.constantRVs.end(); it++) {
        dakotaFile << "\'" << it->name << "\' ";
        rvList.push_back(it->name);
      }
      dakotaFile << "\n";
    }

      
      return 0;
    }

    if (includeActiveText == true) {
      if (idVariables.empty())
	dakotaFile << "variables \n active uncertain \n";
      else
	dakotaFile << "variables \n id_variables =  '" << idVariables << "'\n active uncertain \n";
    } else {
	dakotaFile << "variables \n";
    }

      // int numNormalUncertain = theRandomVariables.normalRVs.size();

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
      for (auto it = theRandomVariables.normalRVs.begin(); it != theRandomVariables.normalRVs.end(); it++) {
	dakotaFile << "\'" << it->name << "\' ";
	rvList.push_back(it->name);
      }

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
      for (auto it = theRandomVariables.lognormalRVs.begin(); it != theRandomVariables.lognormalRVs.end(); it++) {
	dakotaFile << "\'" << it->name << "\' ";
	rvList.push_back(it->name);
      }
      dakotaFile << "\n";
    }

    int numUniform = theRandomVariables.uniformRVs.size();
    if (numUniform > 0) {
      dakotaFile << "  uniform_uncertain = " << numUniform << "\n    lower_bounds = ";
      // std::list<struct uniformRV>::iterator it;
      for (auto it = theRandomVariables.uniformRVs.begin(); it != theRandomVariables.uniformRVs.end(); it++)
	dakotaFile << it->lowerBound << " ";
      dakotaFile << "\n    upper_bound = ";
      for (auto it = theRandomVariables.uniformRVs.begin(); it != theRandomVariables.uniformRVs.end(); it++)
	dakotaFile << it->upperBound << " ";
      dakotaFile << "\n    descriptors = ";
      for (auto it = theRandomVariables.uniformRVs.begin(); it != theRandomVariables.uniformRVs.end(); it++) {
	dakotaFile << "\'" << it->name << "\' ";
	rvList.push_back(it->name);
      }
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
      for (auto it = theRandomVariables.weibullRVs.begin(); it != theRandomVariables.weibullRVs.end(); it++) {
	dakotaFile << "\'" << it->name << "\' ";
	rvList.push_back(it->name);
      }
      dakotaFile << "\n";
    }

    int numGumbell = theRandomVariables.gumbellRVs.size();
    if (numGumbell > 0) {
      dakotaFile << "  gumbel_uncertain = " << numGumbell << "\n    alphas = ";
      // std::list<struct gumbellRV>::iterator it;
      for (auto it = theRandomVariables.gumbellRVs.begin(); it != theRandomVariables.gumbellRVs.end(); it++)
	dakotaFile << it->alphas << " ";
      dakotaFile << "\n    betas = ";
      for (auto it = theRandomVariables.gumbellRVs.begin(); it != theRandomVariables.gumbellRVs.end(); it++)
	dakotaFile << it->betas << " ";
      dakotaFile << "\n    descriptors = ";
      for (auto it = theRandomVariables.gumbellRVs.begin(); it != theRandomVariables.gumbellRVs.end(); it++) {
	dakotaFile << "\'" << it->name << "\' ";
	rvList.push_back(it->name);
      }
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
      for (auto it = theRandomVariables.gammaRVs.begin(); it != theRandomVariables.gammaRVs.end(); it++) {
	dakotaFile << "\'" << it->name << "\' ";
	rvList.push_back(it->name);
      }
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
      for (auto it = theRandomVariables.betaRVs.begin(); it != theRandomVariables.betaRVs.end(); it++) {
	dakotaFile << "\'" << it->name << "\' ";
	rvList.push_back(it->name);
      }
      dakotaFile << "\n";
    }

    int numConstant = theRandomVariables.constantRVs.size();
    if (numConstant > 0) {
      dakotaFile << "  discrete_state_set  \n    real = " << numConstant;
      dakotaFile << "\n    elements_per_variable = ";
      for (auto it = theRandomVariables.constantRVs.begin(); it != theRandomVariables.constantRVs.end(); it++)
        dakotaFile << "1 ";     //std::list<struct betaRV>::iterator it;
      dakotaFile << "\n    elements = ";
      for (auto it = theRandomVariables.constantRVs.begin(); it != theRandomVariables.constantRVs.end(); it++)
        dakotaFile << it->value << " ";
      dakotaFile << "\n    descriptors = ";
      for (auto it = theRandomVariables.constantRVs.begin(); it != theRandomVariables.constantRVs.end(); it++) {
        dakotaFile << "\'" << it->name << "\' ";
        rvList.push_back(it->name);
      }
      dakotaFile << "\n";
    }

    //nt numConstant = theRandomVariables.constantRVs.size();
    //#if (numConstant > 0) {
    //  for (auto it = theRandomVariables.constantRVs.begin(); it != theRandomVariables.constantRVs.end(); it++) {
    //    rvList.push_back(it->name);
    //  }
    //}    

    int numDiscreteDesignSet = theRandomVariables.discreteDesignSetRVs.size();
    if (numDiscreteDesignSet > 0) {
      dakotaFile << "    discrete_uncertain_set\n    string " << numDiscreteDesignSet << "\n    num_set_values = ";
      std::list<struct discreteDesignSetRV>::iterator it;
      for (it = theRandomVariables.discreteDesignSetRVs.begin(); it != theRandomVariables.discreteDesignSetRVs.end(); it++)
	dakotaFile << it->elements.size() << " ";
      dakotaFile << "\n    set_values ";
      for (it = theRandomVariables.discreteDesignSetRVs.begin(); it != theRandomVariables.discreteDesignSetRVs.end(); it++) {
	it->elements.sort(); // sort the elements NEEDED THOUGH NOT IN DAKOTA DOC!
	std::list<std::string>::iterator element;
	for (element = it->elements.begin(); element != it->elements.end(); element++)
	  dakotaFile << " \'" << *element << "\'";
      }
      dakotaFile << "\n    descriptors = ";
      for (auto it = theRandomVariables.discreteDesignSetRVs.begin(); it != theRandomVariables.discreteDesignSetRVs.end(); it++) {
	dakotaFile << "\'" << it->name << "\' ";
	rvList.push_back(it->name);
      }
      dakotaFile << "\n";
    }

    // if no random variables .. create 1 call & call it dummy!
    int numRV = theRandomVariables.numRandomVariables;
    if (numRV == 0) {
      dakotaFile << "   discrete_uncertain_set\n    string 1 \n    num_set_values = 2";
      dakotaFile << "\n    set_values  '1' '2'";
      dakotaFile << "\n    descriptors = dummy\n";
      rvList.push_back(std::string("dummy"));
    }
    dakotaFile << "\n";

    // if correlations, (sy)
     //if (theRandomVariables.corrMat[0] != 0) {

    int corrSize = theRandomVariables.ordering.size();
    if (!theRandomVariables.corrMat.empty()) {
      
      if (theRandomVariables.corrMat[0]!=0) {

        std::vector<int> newOrder;
        for (int i=0; i<18; i++) {
           for (int j=0; j<corrSize; j++) {
             if (i==theRandomVariables.ordering[j]) {
                newOrder.push_back(j);
             }
          }         
        }


        dakotaFile<<"uncertain_correlation_matrix\n";
        for (int i : newOrder) {
          dakotaFile << "    ";
          for (int j : newOrder) {
            double corrval = theRandomVariables.corrMat[i*corrSize+j];
            dakotaFile << corrval << " ";
          }
          dakotaFile << "\n";
        }
      }
    }
    dakotaFile << "\n\n";

    return 0;
}

int
writeInterface(std::ostream &dakotaFile, json_t *uqData, std::string &workflowDriver, std::string idInterface, int evalConcurrency, bool saveWorkDirs) {

  dakotaFile << "interface \n";
  if (!idInterface.empty())
    dakotaFile << "  id_interface = '" << idInterface << "'\n";

  dakotaFile << "  analysis_driver = '" << workflowDriver << "'\n";

  dakotaFile << "  fork\n";

  dakotaFile << "   parameters_file = 'paramsDakota.in'\n";
  dakotaFile << "   results_file = 'results.out' \n";
  dakotaFile << "   aprepro \n";
  dakotaFile << "   work_directory\n";
  dakotaFile << "     named \'workdir\' \n";
  if (saveWorkDirs) {
    dakotaFile << "     directory_tag\n";
    dakotaFile << "     directory_save\n";
  }
  dakotaFile << "     file_save\n";

  /*
    if uqData['keepSamples']:
        dakota_input += ('        directory_save\n')
  */

  dakotaFile << "     copy_files = 'templatedir/*' \n";
  if (evalConcurrency > 0)
    dakotaFile << "  asynchronous evaluation_concurrency = " << evalConcurrency << "\n\n";
  else
    dakotaFile << "  asynchronous \n\n";

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

  return 0;
}

int
writeResponse(std::ostream &dakotaFile,
	      json_t *rootEDP,
	      std::string idResponse,
	      bool numericalGradients,
	      bool numericalHessians,
              std::vector<std::string> &edpList,
	      const char *calFileName,
	      std::vector<double> &scaleFactors) {
  
  int numResponses = 0;

  dakotaFile << "responses\n";

  if (!idResponse.empty() && (idResponse.compare("calibration") != 0 || idResponse.compare("BayesCalibration") != 0))
    dakotaFile << " id_responses = '" << idResponse << "'\n";

  //
  // quoFEM .. just a list of straight EDP
  //
  
  numResponses = json_array_size(rootEDP);
  
  std::vector<int> lenList(numResponses, 1);
  
  int numFieldResponses = 0;
  int numScalarResponses = 0;
  
  if (!(idResponse.compare("calibration") == 0 || idResponse.compare("BayesCalibration") == 0 || idResponse.compare("optimization") == 0)){
    dakotaFile << " response_functions = " << numResponses << "\n response_descriptors = ";
  } else if (idResponse.compare("optimization") == 0) {
    dakotaFile << " objective_functions = " << numResponses << "\n descriptors = ";
  } else
    dakotaFile << " calibration_terms = " << numResponses << "\n response_descriptors = ";
  
  for (int j=0; j<numResponses; j++) {
    json_t *theEDP_Item = json_array_get(rootEDP,j);
    const char *theEDP = json_string_value(json_object_get(theEDP_Item,"name"));
    dakotaFile << "'" << theEDP << "' ";
    std::string newEDP(theEDP);
    edpList.push_back(newEDP);
    
    if (json_object_get(theEDP_Item,"type")) {
      std::string varType = json_string_value(json_object_get(theEDP_Item,"type"));
      if (varType.compare("field") == 0) {
	numFieldResponses++;
      }
      else {
	numScalarResponses++;
      }
    }
  }

  if (numFieldResponses > 0) {
    if (!(idResponse.compare("calibration") == 0 || idResponse.compare("BayesCalibration") == 0)) {
      if (numScalarResponses > 0) {
	dakotaFile << "\n  scalar_responses = " << numScalarResponses;
      }
      dakotaFile << "\n  field_responses = " << numFieldResponses << "\n  lengths = ";
    }
    else {
      if (numScalarResponses > 0) {
	dakotaFile << "\n  scalar_calibration_terms = " << numScalarResponses;
      }
      dakotaFile << "\n  field_calibration_terms = " << numFieldResponses << "\n   lengths = ";
    }
    for (int j = 0; j < numResponses; j++) {
      json_t *theEDP_Item = json_array_get(rootEDP, j);
      std::string varType = json_string_value(json_object_get(theEDP_Item, "type"));
      if (varType.compare("field") == 0) {
	int len = json_integer_value(json_object_get(theEDP_Item, "length"));
	dakotaFile << len << " ";
	lenList[j] = len;
      }
    }
    
    //          bool readFieldCoords = true;
    //          if (readFieldCoords) {
    //              dakotaFile << "\n  read_field_coordinates" << "\n  num_coordinates_per_field = ";
    //              for (int j = 0; j < numResponses; j++) {
    //                  json_t *theEDP_Item = json_array_get(rootEDP, j);
    //                  int numCoords = json_integer_value(json_object_get(theEDP_Item, "numIndCoords"));
    //                  dakotaFile << numCoords << " ";
    //              }
    //          }
    //
    //      }
  }
  
  if ((idResponse.compare("calibration") == 0) || (idResponse.compare("BayesCalibration") == 0)) {
    std::vector<std::string> errFilenameList = {};
    std::stringstream errTypeStringStream;
    
    int numExp = processDataFiles(calFileName, edpList, lenList, numResponses, numFieldResponses, errFilenameList,
				  errTypeStringStream, idResponse, scaleFactors);
    
    bool readCalibrationData = true;
    if (readCalibrationData) {
      if (numFieldResponses > 0) {
	int nExp = numExp;
	if (nExp < 1) {
	  nExp = 1;
	}
	dakotaFile << "\n  calibration_data";
	dakotaFile << "\n   num_experiments = " << nExp;
	if (idResponse.compare("BayesCalibration") == 0) {
	  dakotaFile << "\n   experiment_variance_type = ";
	  dakotaFile << errTypeStringStream.str();
	}
      }
      else {
	int nExp = numExp;
	if (nExp < 1) {
	  nExp = 1;
	}
	dakotaFile << "\n  calibration_data_file = 'quoFEMScalarCalibrationData.cal'";
	dakotaFile << "\n    freeform";
	dakotaFile << "\n    num_experiments = " << nExp;
	if (idResponse.compare("BayesCalibration") == 0) {
	  dakotaFile << "\n    experiment_variance_type = ";
	  dakotaFile << errTypeStringStream.str();
	}
      }
    }
  }


  if (numericalGradients == true)
    dakotaFile << "\n numerical_gradients";
  else
    dakotaFile << "\n no_gradients";

  if (numericalHessians == true)
    dakotaFile << "\n numerical_hessians\n\n";
  else
    dakotaFile << "\n no_hessians\n\n";

  return numResponses;
}


int
writeDakotaInputFile(std::ostream &dakotaFile,
		     json_t *uqData,
		     json_t *rootEDP,
		     struct randomVariables &theRandomVariables,
		     std::string &workflowDriver,
		     std::vector<std::string> &rvList,
		     std::vector<std::string> &edpList,
		     int evalConcurrency) {


  int evaluationConcurrency = evalConcurrency;

  // test if parallelExe is false, if so set evalConcurrency = 1;
  json_t *parallelExe = json_object_get(uqData, "parallelExecution");
  if (parallelExe != NULL) {
    if (json_is_false(parallelExe))
      evaluationConcurrency = 1;
  }
  // Save all the working dirs?
  bool saveWorkDirs = true;
  json_t *saveDirs = json_object_get(uqData,"saveWorkDir");
  if (saveDirs != NULL) {
    if (json_is_false(saveDirs))
      saveWorkDirs = false;
  }
  const char *type = json_string_value(json_object_get(uqData, "uqType"));

  bool sensitivityAnalysis = false;
  if (strcmp(type, "Sensitivity Analysis") == 0)
    sensitivityAnalysis = true;

  json_t *EDPs = json_object_get(rootEDP,"EngineeringDemandParameters");
  int numResponses = 0;
  if (EDPs != NULL) {
    numResponses = json_integer_value(json_object_get(rootEDP,"total_number_edp"));
  } else {
    numResponses = json_array_size(rootEDP);
  }

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

      const char * calFileName = new char[1];
      std::string emptyString;
      std::vector<double> scaleFactors;
      writeRV(dakotaFile, theRandomVariables, emptyString, rvList, true);
      writeInterface(dakotaFile, uqData, workflowDriver, emptyString, evaluationConcurrency, saveWorkDirs);
      writeResponse(dakotaFile, rootEDP, emptyString, false, false, edpList, calFileName, scaleFactors);
    }

    else if (strcmp(method,"LHS")==0) {

      int numSamples = json_integer_value(json_object_get(samplingMethodData,"samples"));
      int seed = json_integer_value(json_object_get(samplingMethodData,"seed"));

      //std::cerr << numSamples << " " << seed;

      dakotaFile << "environment \n tabular_data \n tabular_data_file = 'dakotaTab.out' \n\n";
      dakotaFile << "method,\n sampling\n sample_type = lhs \n samples = " << numSamples << " \n seed = " << seed << "\n\n";

      if (sensitivityAnalysis == true)
	  dakotaFile << "variance_based_decomp \n\n";


      const char * calFileName = new char[1];
      std::string emptyString;
      std::vector<double> scaleFactors;
      writeRV(dakotaFile, theRandomVariables, emptyString, rvList);
      writeInterface(dakotaFile, uqData, workflowDriver, emptyString, evaluationConcurrency, saveWorkDirs);

      writeResponse(dakotaFile, rootEDP, emptyString, false, false, edpList, calFileName, scaleFactors);
    }

    /*
    else if (strcmp(method,"Importance Sampling")==0) {

      const char *isMethod = json_string_value(json_object_get(samplingMethodData,"ismethod"));
      int numSamples = json_integer_value(json_object_get(samplingMethodData,"samples"));
      int seed = json_integer_value(json_object_get(samplingMethodData,"seed"));

      dakotaFile << "environment \n tabular_data \n tabular_data_file = 'dakotaTab.out' \n\n";
      dakotaFile << "method, \n importance_sampling \n " << isMethod << " \n samples = " << numSamples << "\n seed = " << seed << "\n\n";
      const char *calFileName;
      std::string emptyString;
      writeRV(dakotaFile, theRandomVariables, emptyString, rvList);
      writeInterface(dakotaFile, uqData, workflowDriver, emptyString, evaluationConcurrency);
      writeResponse(dakotaFile, rootEDP, emptyString, false, false, edpList, calFileName);
    }
    */
//    }
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
      const char * calFileName = new char[1];
      std::string emptyString;
      std::vector<double> scaleFactors;
      std::string interfaceString("SimulationInterface");
      writeRV(dakotaFile, theRandomVariables, emptyString, rvList);
      writeInterface(dakotaFile, uqData, workflowDriver, interfaceString, evaluationConcurrency, saveWorkDirs);
      writeResponse(dakotaFile, rootEDP, emptyString, false, false, edpList, calFileName, scaleFactors);

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
      else if (strcmp(dataMethod,"Stroud Cubature") == 0)
	pceMethod = "cubature_integrand = ";
      else if (strcmp(dataMethod,"Orthogonal Least_Interpolation") == 0)
	pceMethod = "orthogonal_least_squares collocation_points = ";
      else
	pceMethod = "quadrature_order = ";

      std::string samplingMethod(sampleMethod);
      if (strcmp(sampleMethod,"Monte Carlo") == 0)
	samplingMethod = "random";

      dakotaFile << "environment \n  tabular_data \n tabular_data_file = 'a.out'\n\n"; // a.out for trial data
      const char * calFileName = new char[1];
      std::string emptyString;
      std::vector<double> scaleFactors;
      std::string interfaceString("SimulationInterface");
      writeRV(dakotaFile, theRandomVariables, emptyString, rvList);
      writeInterface(dakotaFile, uqData, workflowDriver, interfaceString, evaluationConcurrency, saveWorkDirs);
      int numResponse = writeResponse(dakotaFile, rootEDP, emptyString, false, false, edpList, calFileName, scaleFactors);

      dakotaFile << "method \n polynomial_chaos \n " << pceMethod << intValue;
      dakotaFile << "\n samples_on_emulator = " << samplingSamples << "\n seed = " << samplingSeed << "\n sample_type = "
		 << samplingMethod << "\n";
      dakotaFile << " probability_levels = ";
      for (int i=0; i<numResponse; i++)
	dakotaFile << " .1 .5 .9 ";
      dakotaFile << "\n export_approx_points_file = 'dakotaTab.out'\n\n"; // dakotaTab.out for surrogate evaluations
    }

  }

  else if ((strcmp(type, "Reliability Analysis") == 0)) {

    json_t *reliabilityMethodData = json_object_get(uqData,"reliabilityMethodData");

    const char *method = json_string_value(json_object_get(reliabilityMethodData,"method"));

    if (strcmp(method,"Local Reliability")==0) {

      const char *localMethod = json_string_value(json_object_get(reliabilityMethodData,"localMethod"));
      const char *mppMethod = json_string_value(json_object_get(reliabilityMethodData,"mpp_Method"));
      const char *levelType = json_string_value(json_object_get(reliabilityMethodData,"levelType"));
      const char *integrationMethod = json_string_value(json_object_get(reliabilityMethodData,"integrationMethod"));

      std::string intMethod;
      if (strcmp(integrationMethod,"First Order") == 0)
	intMethod = "first_order";
      else
	intMethod = "second_order";

      dakotaFile << "environment \n tabular_data \n tabular_data_file = 'dakotaTab.out' \n\n";
      if (strcmp(localMethod,"Mean Value") == 0) {
	dakotaFile << "method, \n local_reliability \n";
      } else {
	dakotaFile << "method, \n local_reliability \n mpp_search " << mppMethod
		   << " \n integration " << intMethod << " \n";
      }

      json_t *levels =  json_object_get(reliabilityMethodData, "probabilityLevel");
      if (levels == NULL) {
	return 0;
      }

      int numLevels = json_array_size(levels);
      if (strcmp(levelType, "Probability Levels") == 0)
	dakotaFile << " \n num_probability_levels = ";
      else
	dakotaFile << " \n num_response_levels = ";

      for (int i=0; i<numResponses; i++)
	dakotaFile << numLevels << " ";

      if (strcmp(levelType, "Probability Levels") == 0)
	dakotaFile << " \n probability_levels = " ;
      else
	dakotaFile << " \n response_levels = " ;

      for (int j=0; j<numResponses; j++) {
	for (int i=0; i<numLevels; i++) {
	    json_t *responseLevel = json_array_get(levels,i);
	    double val = json_number_value(responseLevel);
	    dakotaFile << val << " ";
	  }
	dakotaFile << "\n\t";
      }
      dakotaFile << "\n\n";
      const char * calFileName = new char[1];
      std::string emptyString;
      std::vector<double> scaleFactors;
      writeRV(dakotaFile, theRandomVariables, emptyString, rvList);
      writeInterface(dakotaFile, uqData, workflowDriver, emptyString, evaluationConcurrency, saveWorkDirs);
      writeResponse(dakotaFile, rootEDP, emptyString, true, true, edpList, calFileName, scaleFactors);
    }

    else if (strcmp(method,"Global Reliability")==0) {

      const char *gp = json_string_value(json_object_get(reliabilityMethodData,"gpApproximation"));
      std::string gpMethod;
      if (strcmp(gp,"x-space") == 0)
	gpMethod = "x_gaussian_process";
      else
	gpMethod = "u_gaussian_process";


      json_t *levels =  json_object_get(reliabilityMethodData, "responseLevel");
      if (levels == NULL) {
	return 0;
      }
      int numLevels = json_array_size(levels);

      dakotaFile << "environment \n tabular_data \n tabular_data_file = 'dakotaTab.out' \n\n";
      dakotaFile << "method, \n global_reliability " << gpMethod << " \n"; // seed " << seed;

      dakotaFile << " \n num_response_levels = ";
      for (int i=0; i<numResponses; i++)
	dakotaFile << numLevels << " ";

      dakotaFile << " \n response_levels = " ;
      for (int j=0; j<numResponses; j++) {
	for (int i=0; i<numLevels; i++) {
	  json_t *responseLevel = json_array_get(levels,i);
	  double val = json_number_value(responseLevel);
	  dakotaFile << val << " ";
	}
	dakotaFile << "\n\t";
      }
      dakotaFile << "\n\n";
      const char * calFileName = new char[1];
      std::string emptyString;
      std::vector<double> scaleFactors;
      writeRV(dakotaFile, theRandomVariables, emptyString, rvList);
      writeInterface(dakotaFile, uqData, workflowDriver, emptyString, evaluationConcurrency, saveWorkDirs);
      writeResponse(dakotaFile, rootEDP, emptyString, true, false, edpList, calFileName, scaleFactors);
    }

    else if (strcmp(method,"Importance Sampling")==0) {

      const char *isMethod = json_string_value(json_object_get(reliabilityMethodData,"ismethod"));
      int numSamples = json_integer_value(json_object_get(reliabilityMethodData,"samples"));
      int seed = json_integer_value(json_object_get(reliabilityMethodData,"seed"));

      json_t *levels =  json_object_get(reliabilityMethodData, "responseLevel");
      if (levels == NULL) {
        return 0;
      }

      int numLevels = json_array_size(levels);

       dakotaFile << "environment \n tabular_data \n tabular_data_file = 'dakotaTab.out' \n\n";
       dakotaFile << "method, \n importance_sampling \n " << isMethod << " \n samples = " << numSamples << "\n seed = " << seed << "\n\n";

    //   std::string emptyString;
    //   writeRV(dakotaFile, theRandomVariables, emptyString, rvList);
    //   writeInterface(dakotaFile, uqData, workflowDriver, emptyString, evaluationConcurrency);
    //   writeResponse(dakotaFile, rootEDP, emptyString, false, false, edpList);
      dakotaFile << " \n num_response_levels = ";
      for (int i=0; i<numResponses; i++)
        dakotaFile << numLevels << " ";

      dakotaFile << " \n response_levels = " ;
      for (int j=0; j<numResponses; j++) {
        for (int i=0; i<numLevels; i++) {
          json_t *responseLevel = json_array_get(levels,i);
          double val = json_number_value(responseLevel);
          dakotaFile << val << " ";
        }
        dakotaFile << "\n\t";
      }
      dakotaFile << "\n\n";

      const char * calFileName = new char[1];;
      std::string emptyString;
      std::vector<double> scaleFactors;
      writeRV(dakotaFile, theRandomVariables, emptyString, rvList);
      writeInterface(dakotaFile, uqData, workflowDriver, emptyString, evaluationConcurrency, saveWorkDirs);
      writeResponse(dakotaFile, rootEDP, emptyString, true, false, edpList, calFileName, scaleFactors);
    }

  } else if ((strcmp(type, "Parameters Estimation") == 0)) {

    json_t *methodData = json_object_get(uqData,"calibrationMethodData");

    const char *method = json_string_value(json_object_get(methodData,"method"));

    std::string methodString("nl2sol");
    if (strcmp(method,"OPT++GaussNewton")==0)
      methodString = "optpp_g_newton";

    int maxIterations = json_integer_value(json_object_get(methodData,"maxIterations"));
    double tol = json_number_value(json_object_get(methodData,"convergenceTol"));
//    const char *factors = json_string_value(json_object_get(methodData,"factors"));
    const char *calFileName = json_string_value(json_object_get(methodData, "calibrationDataFile"));

    dakotaFile << "environment \n tabular_data \n tabular_data_file = 'dakotaTab.out' \n\n";

    dakotaFile << "method, \n " << methodString << "\n  convergence_tolerance = " << tol
	       << " \n   max_iterations = " << maxIterations;

//    if (strcmp(factors,"") != 0)
    dakotaFile << "\n  scaling\n";

    dakotaFile << "\n\n";

    std::string calibrationString("calibration");
    std::string emptyString;
    std::vector<double> scaleFactors;
    writeRV(dakotaFile, theRandomVariables, emptyString, rvList);
    writeInterface(dakotaFile, uqData, workflowDriver, emptyString, evaluationConcurrency, saveWorkDirs);
    writeResponse(dakotaFile, rootEDP, calibrationString, true, false, edpList, calFileName, scaleFactors);

//    dakotaFile << "\n  primary_scales = 1052.69 1.53\n";
//    if (strcmp(factors,"") != 0) {
//      dakotaFile << "\n  primary_scale_types = \"value\" \n  primary_scales = ";
//      std::string factorString(factors);
//      std::stringstream factors_stream(factorString);
//      std::string tmp;
//      while (factors_stream >> tmp) {
//	// maybe some checks, i.e. ,
//	dakotaFile << tmp << " ";
//      }
//      dakotaFile << "\n";
//    }

  } else if ((strcmp(type, "Inverse Problem") == 0)) {

    json_t *methodData = json_object_get(uqData,"bayesianCalibrationMethodData");

    const char *method = json_string_value(json_object_get(methodData,"method"));

    /*
    const char *emulator = json_string_value(json_object_get(methodData,"emulator"));
    std::string emulatorString("gaussian_process");
    if (strcmp(emulator,"Polynomial Chaos")==0)
      emulatorString = "pce";
    else if (strcmp(emulator,"Multilevel Polynomial Chaos")==0)
      emulatorString = "ml_pce";
    else if (strcmp(emulator,"Multifidelity Polynomial Chaos")==0)
      emulatorString = "mf_pce";
    else if (strcmp(emulator,"Stochastic Collocation")==0)
      emulatorString = "sc";
    */

    int chainSamples = json_integer_value(json_object_get(methodData,"chainSamples"));
    int seed = json_integer_value(json_object_get(methodData,"seed"));
    int burnInSamples = json_integer_value(json_object_get(methodData,"burnInSamples"));
    int jumpStep = json_integer_value(json_object_get(methodData,"jumpStep"));
    //    int maxIterations = json_integer_value(json_object_get(methodData,"maxIter"));
    //    double tol = json_number_value(json_object_get(methodData,"tol"));
    const char *calFileName = json_string_value(json_object_get(methodData, "calibrationDataFile"));

      if (strcmp(method,"DREAM")==0) {

      int chains = json_integer_value(json_object_get(methodData,"chains"));

      dakotaFile << "environment \n tabular_data \n tabular_data_file = 'dakotaTab.out' \n\n";
      dakotaFile << "method \n bayes_calibration dream "
		 << "\n  chain_samples = " << chainSamples
		 << "\n  chains = " << chains
		 << "\n  jump_step = " << jumpStep
		 << "\n  burn_in_samples = " << burnInSamples
     << "\n  seed = " << seed
		 << "\n  calibrate_error_multipliers per_response";

	  dakotaFile << "\n  scaling\n" << "\n";

    } else {

      const char *mcmc = json_string_value(json_object_get(methodData,"mcmcMethod"));
      std::string mcmcString("dram");
      if (strcmp(mcmc,"Delayed Rejection")==0)
	mcmcString = "delayed_rejection";
      else if (strcmp(mcmc,"Adaptive Metropolis")==0)
	mcmcString = "adaptive_metropolis";
      else if (strcmp(mcmc,"Metropolis Hastings")==0)
	mcmcString = "metropolis_hastings";
      else if (strcmp(mcmc,"Multilevel")==0)
	mcmcString = "multilevel";

      dakotaFile << "environment \n tabular_data \n tabular_data_file = 'dakotaTab.out' \n\n";
      dakotaFile << "method \n bayes_calibration queso\n  " << mcmc
		 << "\n  chain_samples = " << chainSamples
		 << "\n  burn_in_samples = " << burnInSamples << "\n\n";
    }

    std::string calibrationString("BayesCalibration");
    std::string emptyString;
    std::vector<double> scaleFactors;
    writeRV(dakotaFile, theRandomVariables, emptyString, rvList, false);
    writeInterface(dakotaFile, uqData, workflowDriver, emptyString, evaluationConcurrency, saveWorkDirs);
    writeResponse(dakotaFile, rootEDP, calibrationString, false, false, edpList, calFileName, scaleFactors);
//    calDataFile.close();

  } else if ((strcmp(type, "Optimization") == 0)) {

    int numRVs = theRandomVariables.numRandomVariables;

    json_t *methodData = json_object_get(uqData,"optimizationMethodData");

    const char *method = json_string_value(json_object_get(methodData,"method"));

    std::string methodString("coliny_pattern_search");
    bool gradientBool = false;
    bool hessianBool = false;
    if (strcmp(method,"Derivative-Free Local Search")==0)
      methodString = "coliny_pattern_search";
      gradientBool = false;
      hessianBool = false;

    int maxIterations = json_integer_value(json_object_get(methodData,"maxIterations"));
    double tol = json_number_value(json_object_get(methodData,"convergenceTol"));
    double contractionFactor = json_number_value(json_object_get(methodData, "contractionFactor"));
    double initialDelta = json_number_value(json_object_get(methodData, "initialDelta"));
    int maxFunEvals = json_integer_value(json_object_get(methodData, "maxFunEvals"));
    double thresholdDelta = json_number_value(json_object_get(methodData, "thresholdDelta"));
    // double solutionTarget = json_number_value(json_object_get(methodData, "solutionTarget"));
//    const char *factors = json_string_value(json_object_get(methodData,"factors"));

    dakotaFile << "environment \n tabular_data \n tabular_data_file = 'dakotaTab.out' \n\n";

    dakotaFile << "method, \n " << methodString 
      << "\n  contraction_factor = " << contractionFactor
      << "\n  convergence_tolerance = " << tol 
      << "\n  initial_delta = " << initialDelta 
      << "\n  max_function_evaluations = " << maxFunEvals 
      << "\n  max_iterations = " << maxIterations 
      << "\n  total_pattern_size = " << 2*numRVs
      << "\n  variable_tolerance = " << thresholdDelta;
      // << "\n  solution_target = " << solutionTarget

//    if (strcmp(factors,"") != 0)
    dakotaFile << "\n  scaling\n";

    dakotaFile << "\n\n";

    std::string optimizationString("optimization");
    std::string emptyString;
    std::vector<double> scaleFactors;
    writeRV(dakotaFile, theRandomVariables, emptyString, rvList);
    writeInterface(dakotaFile, uqData, workflowDriver, emptyString, evaluationConcurrency, saveWorkDirs);
    writeResponse(dakotaFile, rootEDP, optimizationString, gradientBool, hessianBool, edpList, emptyString.c_str(), scaleFactors);

//    dakotaFile << "\n  primary_scales = 1052.69 1.53\n";
//    if (strcmp(factors,"") != 0) {
//      dakotaFile << "\n  primary_scale_types = \"value\" \n  primary_scales = ";
//      std::string factorString(factors);
//      std::stringstream factors_stream(factorString);
//      std::string tmp;
//      while (factors_stream >> tmp) {
//	// maybe some checks, i.e. ,
//	dakotaFile << tmp << " ";
//      }
//      dakotaFile << "\n";
//    }

  } else {
    std::cerr << "uqType: NOT KNOWN\n";
    return -1;
  }
  return 0;
}


