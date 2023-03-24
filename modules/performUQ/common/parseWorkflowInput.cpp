#include <iostream>
#include <fstream>
#include <jansson.h>
#include <string.h>
#include <string>
#include <sstream>
#include <list>
#include <vector>
#include <set>

#include "parseWorkflowInput.h"

/*
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
  std::set<std::string>theNames;
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
  std::vector<int> ordering;
  std::vector<double> corrMat;
};
*/

// parses JSON for random variables & returns number found

int
parseInputForRV(json_t *root, struct randomVariables &theRandomVariables){
  
  int numberRVs = 0;

  json_t *fileRandomVariables =  json_object_get(root, "randomVariables");
  if (fileRandomVariables == NULL) {
    return 0; // no random variables is allowed
  }

  int numRVs = json_array_size(fileRandomVariables);

  for (int i=0; i<numRVs; i++) {
    
    json_t *fileRandomVariable = json_array_get(fileRandomVariables,i);
    const char *variableType = json_string_value(json_object_get(fileRandomVariable,"distribution"));
    std::string rvName = json_string_value(json_object_get(fileRandomVariable,"name"));

    if (theRandomVariables.theNames.find(rvName) == theRandomVariables.theNames.end()) {

      if ((strcmp(variableType, "Normal") == 0) || (strcmp(variableType, "normal")==0)) {

	struct normalRV theRV;
	//theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));

	theRandomVariables.theNames.insert(rvName);
	theRV.name = rvName;
	theRV.mean = json_number_value(json_object_get(fileRandomVariable,"mean"));
	theRV.stdDev = json_number_value(json_object_get(fileRandomVariable,"stdDev"));
	
	theRandomVariables.normalRVs.push_back(theRV);
	theRandomVariables.numRandomVariables += 1;
	theRandomVariables.ordering.push_back(1);
	numberRVs++;
	
      }
      
      else if ((strcmp(variableType, "Lognormal") == 0) || (strcmp(variableType, "lognormal") == 0)) {
	
      struct lognormalRV theRV;

      //      theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));

      theRandomVariables.theNames.insert(rvName);
      theRV.name = rvName;      
      theRV.mean = json_number_value(json_object_get(fileRandomVariable,"mean"));
      theRV.stdDev = json_number_value(json_object_get(fileRandomVariable,"stdDev"));

      theRandomVariables.lognormalRVs.push_back(theRV);
      theRandomVariables.numRandomVariables += 1;
      theRandomVariables.ordering.push_back(2);
      numberRVs++;

    }


    else if (strcmp(variableType, "Uniform") == 0) {

      struct uniformRV theRV;

      //theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));

      theRandomVariables.theNames.insert(rvName);
      theRV.name = rvName;      
      theRV.lowerBound = json_number_value(json_object_get(fileRandomVariable,"lowerbound"));
      theRV.upperBound = json_number_value(json_object_get(fileRandomVariable,"upperbound"));

      theRandomVariables.uniformRVs.push_back(theRV);
      theRandomVariables.numRandomVariables += 1;
      theRandomVariables.ordering.push_back(3);
      numberRVs++;

    }


    else if (strcmp(variableType, "Constant") == 0) {

      struct constantRV theRV;
      //theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));

      theRandomVariables.theNames.insert(rvName);
      theRV.name = rvName;      
      theRV.value = json_number_value(json_object_get(fileRandomVariable,"value"));

      theRandomVariables.constantRVs.push_back(theRV);
      theRandomVariables.numRandomVariables += 1;
      numberRVs++;

    }



    else if (strcmp(variableType, "ContinuousDesign") == 0) {
      struct continuousDesignRV theRV;

      //theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));
      theRandomVariables.theNames.insert(rvName);
      theRV.name = rvName;      
      theRV.lowerBound = json_number_value(json_object_get(fileRandomVariable,"lowerbound"));
      theRV.upperBound = json_number_value(json_object_get(fileRandomVariable,"upperbound"));
      theRV.initialPoint = json_number_value(json_object_get(fileRandomVariable,"initialpoint"));

      theRandomVariables.continuousDesignRVs.push_back(theRV);
      theRandomVariables.numRandomVariables += 1;
      numberRVs++;
    }

    else if (strcmp(variableType, "Weibull") == 0) {

      struct weibullRV theRV;

      //theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));
      theRandomVariables.theNames.insert(rvName);
      theRV.name = rvName;      
      theRV.shapeParam = json_number_value(json_object_get(fileRandomVariable,"shapeparam"));
      theRV.scaleParam = json_number_value(json_object_get(fileRandomVariable,"scaleparam"));

      theRandomVariables.weibullRVs.push_back(theRV);
      theRandomVariables.numRandomVariables += 1;
      theRandomVariables.ordering.push_back(11);
      numberRVs++;
    }

    else if (strcmp(variableType, "Gamma") == 0) {

      struct gammaRV theRV;

      //theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));
      theRandomVariables.theNames.insert(rvName);
      theRV.name = rvName;      
      theRV.alphas = json_number_value(json_object_get(fileRandomVariable,"alphas"));
      theRV.betas = json_number_value(json_object_get(fileRandomVariable,"betas"));

      theRandomVariables.gammaRVs.push_back(theRV);
      theRandomVariables.numRandomVariables += 1;
      theRandomVariables.ordering.push_back(8);
      numberRVs++;
    }

    else if (strcmp(variableType, "Gumbel") == 0) {

      struct gumbellRV theRV;

      //theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));

      theRandomVariables.theNames.insert(rvName);
      theRV.name = rvName;      
      theRV.alphas = json_number_value(json_object_get(fileRandomVariable,"alphaparam"));
      theRV.betas = json_number_value(json_object_get(fileRandomVariable,"betaparam"));

      theRandomVariables.gumbellRVs.push_back(theRV);
      theRandomVariables.numRandomVariables += 1;
      theRandomVariables.ordering.push_back(9);
      numberRVs++;
    }


    else if (strcmp(variableType, "Beta") == 0) {

      struct betaRV theRV;

      //theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));
      theRandomVariables.theNames.insert(rvName);
      theRV.name = rvName;      
      theRV.alphas = json_number_value(json_object_get(fileRandomVariable,"alphas"));
      theRV.betas = json_number_value(json_object_get(fileRandomVariable,"betas"));
      theRV.lowerBound = json_number_value(json_object_get(fileRandomVariable,"lowerbound"));
      theRV.upperBound = json_number_value(json_object_get(fileRandomVariable,"upperbound"));
      std::cerr << theRV.name << " " << theRV.upperBound << " " << theRV.lowerBound << " " << theRV.alphas << " " << theRV.betas;
      theRandomVariables.betaRVs.push_back(theRV);
      theRandomVariables.numRandomVariables += 1;
      theRandomVariables.ordering.push_back(7);
      numberRVs++;
    }

    else if (strcmp(variableType, "discrete_design_set_string") == 0) {

      struct discreteDesignSetRV theRV;

      // theRV.name = json_string_value(json_object_get(fileRandomVariable,"name"));
      theRandomVariables.theNames.insert(rvName);
      theRV.name = rvName;      
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
      //theRandomVariables.ordering.push_back(-1);
    	numberRVs++;
      }
    }

    else if (strcmp(variableType, "Discrete") == 0) {

      struct discreteUncertainIntegerSetRV theRV;

      theRandomVariables.theNames.insert(rvName);
      theRV.name = rvName; 

      std::list<int> theValues;
      json_t *elementsSet =  json_object_get(fileRandomVariable, "Values");
      if (elementsSet != NULL) {
        int numValues = json_array_size(elementsSet);
        for (int j=0; j<numValues; j++) {
          json_t *element = json_array_get(elementsSet, j);
          int value = json_integer_value(element);
          theValues.push_back(value);
          // if (value != 0) {
          //   theValues.push_back(value);
          // } else {
          //     std::cout << "ERROR: The value at index " << j << " of the Discrete RV " << rvName << 
          //     " must be an integer, but was " << element << std::endl;
          //     return EXIT_FAILURE;
          // }
          }
        theRV.elements = theValues;

        std::list<double> theWeights;
        json_t *weightsSet =  json_object_get(fileRandomVariable, "Weights");
        if (weightsSet != NULL) {
          int numWeights = json_array_size(weightsSet);
          for (int j=0; j<numWeights; j++) {
            json_t *wt = json_array_get(weightsSet, j);
            double weight = json_number_value(wt);
            theWeights.push_back(weight);
          }
        }
        theRV.weights = theWeights;

        theRandomVariables.discreteUncertainIntegerSetRVs.push_back(theRV);
        theRandomVariables.numRandomVariables += 1;
        //theRandomVariables.ordering.push_back(-1);
        numberRVs++;
        }
    }

    //correlation
    json_t* corrMatJson =  json_object_get(root,"correlationMatrix");
    if (corrMatJson != NULL) {
      int numCorrs = json_array_size(corrMatJson);
      for (int i=0; i<numCorrs; i++) {
        const double corrVal = json_number_value(json_array_get(corrMatJson,i));
        theRandomVariables.corrMat.push_back(corrVal);
      }
    } else {
      theRandomVariables.corrMat.push_back(0.0);
    }
    
    } // end if rvName not in theRandomVariables.theNames
    
  } // end loop over random variables

  return numRVs;
}



int processDataFiles(const char *calFileName,
                     std::vector<std::string> &edpList,
                     std::vector<int> &lengthList,
                     int numResponses, int numFieldResponses,
                     std::vector<std::string> &errFileList,
                     std::stringstream &errType, std::string idResponse,
                     std::vector<double> &scaleFactors) {

    std::ifstream calDataFile;
    calDataFile.open(calFileName, std::ios_base::in);

    // Compute the expected length of each line and cumulative length corresponding to the responses
    // within each line
    int lineLength = 0;
    std::vector<int> cumLenList(numResponses, 0);
    for (int i = 0; i < numResponses; i++) {
        lineLength += lengthList[i];
        cumLenList[i] += lineLength;
    }

    // Create an fstream to write the calibration data after removing all commas
    std::fstream spacedDataFile;
    std::string spacedFileName = "quoFEMTempCalibrationDataFile.cal";
    spacedDataFile.open(spacedFileName.c_str(), std::fstream::out);
    // Check if open succeeded
    if (!spacedDataFile) {
        std::cerr << "ERROR: unable to open file: " << spacedFileName << std::endl;
        return EXIT_FAILURE;
    }

    // Count the number of experiments, check the length of each line, remove commas and write data to a temp file
    int numExperiments = 0;
    std::string line;
    int lineNum = 0;
    // For each line in the calibration data file
    while (getline(calDataFile, line)) {
        // Get a string stream for each line
        std::stringstream lineStream(line);
        if (!line.empty()) {
            ++lineNum;
            int wordCount = 0;
            // Check length of each line
            char *word;
            word = strtok(const_cast<char *>(line.c_str()), " \t,");
            while (word != nullptr) { // while end of cstring is not reached
                ++wordCount;
                spacedDataFile << word << " ";
                word = strtok(nullptr, " \t,");
            }
            if (wordCount != lineLength) {
                std::cout << std::endl << "ERROR: The number of calibration terms expected in each line is "
                          << lineLength
                          << ", but the length of line " << lineNum << " is " << wordCount << ". Aborting."
                          << std::endl;
                spacedDataFile.close();
                return EXIT_FAILURE;
            }

            spacedDataFile << std::endl;
        }
    }
    numExperiments = lineNum;
    spacedDataFile.close();
    calDataFile.clear();
    calDataFile.close();

    std::string filename = calFileName;
    std::string calDirectory;
    const size_t last_slash_idx = filename.rfind('\\/');
    if (std::string::npos != last_slash_idx)
    {
        calDirectory = filename.substr(0, last_slash_idx);
    }
    for (int expNum = 1; expNum <= numExperiments; expNum++) {
        for (int responseNum = 0; responseNum < numResponses; responseNum++) {
            std::stringstream errFileName;
            errFileName << edpList[responseNum] << "." << expNum << ".sigma";
            std::ifstream checkFile(errFileName.str());
            if (checkFile.good()) {
                errFileList.push_back(errFileName.str());
            }
        }
    }

    // Check if there are any files describing the error covariance structure in errFileList
    bool readErrorFile = false;
    if (!errFileList.empty()) {
        readErrorFile = true;
    }

    // Start making a list of all the calibration data files to be moved to the upper directory
    std::string trackerFileName = "calibrationDataFilesToMove.cal";
    std::ofstream calFilesTracker;
    calFilesTracker.open(trackerFileName.c_str(), std::ofstream::out);

    // =============================================
    // When there are any non-scalar response terms
    // =============================================
    if (numFieldResponses > 0) {
        std::string calDataLine;
        calDataFile.open(spacedFileName.c_str());

        // For each line in the calibration data file
        while (getline(calDataFile, calDataLine)) {
            if (!calDataLine.empty()) {
                std::stringstream lineStream(calDataLine);
                int wordCount = 0;

                // For each response quantity, create a data file and write the corresponding data to it
                for (int responseNum = 0; responseNum < numResponses; responseNum++) {

                    // =============================================
                    // Create a file for each response variable data
                    std::stringstream fName;
                    fName << edpList[responseNum] << "." << lineNum << ".dat";
                    std::ofstream outfile;
                    outfile.open(fName.str().c_str());
                    // Get the calibration terms corresponding to length of each response quantity
                    while (wordCount < cumLenList[responseNum]) {
                        std::string word;
                        lineStream >> word;
                        outfile << word << std::endl;
                        wordCount++;
                    }
                    // Close the file
                    outfile.close();
                    // Add this filename to the file containing the list of files to move to upper directory
                    calFilesTracker << fName.str() << std::endl;
                    // =============================================

                    // Only for Bayesian calibration - create error covariance files per response per experiment
                    if (idResponse.compare("BayesCalibration") == 0) {
                        // =============================================
                        // Create filename for error variance
                        std::stringstream errFileName;
                        errFileName << edpList[responseNum] << "." << lineNum << ".sigma";
                        std::ofstream errFile;

                        // Check if an error variance file with the created name is in the list of error files
                        bool createErrFile = true;
                        std::string errFileToProcess;
                        if (readErrorFile) {
                            for (const auto &path : errFileList) {
                                std::string base_filename = path.substr(path.find_last_of("/\\") + 1);
                                if (base_filename == errFileName.str()) {
                                    createErrFile = false;
                                    errFileToProcess = path;
                                    break;
                                }
                            }
                        }
                        // If there is no user-supplied error covariance file with the created name, create error file
                        if (createErrFile) {
                            errFile.open(errFileName.str().c_str());
                            // Write the default error covariance - scalar with variance 1
                            errFile << "1" << std::endl;
                            errType << "'scalar' ";
                            // Add the filename to the file containing the list of files to move to upper directory
                            calFilesTracker << errFileName.str() << std::endl;
                            // Close the file
                            errFile.close();
                        }
                        else { // There is a user supplied error covariance file
                            // Process the user supplied error covariance file
                            // Get the dimensions of the contents of this file
                            std::ifstream errFileUser(errFileToProcess);
                            int nrow = 0;
                            int ncol = 0;
                            std::string fileLine;

                            // Get the first line
                            getline(errFileUser, fileLine);
                            while (fileLine.empty()) {
                                getline(errFileUser, fileLine);
                            }
//                            // Check if there were any commas
//                            bool commaSeparated = false;
//                            if (fileLine.find(',') != std::string::npos) {
//                                commaSeparated = true;
//                            }
                            // Get the number of columns of the first row
                            char *entry;
                            entry = strtok(const_cast<char *>(fileLine.c_str()), " \t,");
                            while (entry != nullptr) { // while end of cstring is not reached
                                ++ncol;
                                entry = strtok(nullptr, " \t,");
                            }
                            // Create temporary file to hold the space separated error data. This file will be moved to the
                            // upper directory, and then the extension '.tmpFile' will be removed from the filename
                            std::string tmpErrorFileName = errFileToProcess + ".tmpFile";
                            std::ofstream tmpErrorFile(tmpErrorFileName);

                            // Now, loop over each line of the user supplied error covariance file
                            errFileUser.close();
                            errFileUser.open(errFileToProcess);
                            while (getline(errFileUser, fileLine)) {
                                if (!fileLine.empty()) {
                                    nrow++;
                                    // Get the number of columns of each row
                                    int numCols = 0;
                                    char *word;
                                    word = strtok(const_cast<char *>(fileLine.c_str()), " \t,");
                                    while (word != nullptr) { // while end of cstring is not reached
                                        ++numCols;
                                        tmpErrorFile << word << " ";
//                                        if (commaSeparated) {
//                                            tmpErrorFile << word << " ";
//                                        }
                                        word = strtok(nullptr, " \t,");
                                    }
                                    if (numCols != ncol) {
                                        std::cout << "ERROR: The number of columns must be the same for each row in the "
                                                     "error covariance file " << errFileToProcess << ". \nThe expected"
                                                  << " length is " << ncol << ", but the length of row " << nrow
                                                  << " is " << numCols << "." << std::endl;
                                        return EXIT_FAILURE;
                                    }
                                }
                            }
                            errFileUser.close();
                            tmpErrorFile.close();

                            if (nrow == 1) {
                                if (ncol == 1) {
                                    errType << "'scalar' ";
                                } else if (ncol == lengthList[responseNum]) {
                                    errType << "'diagonal' ";
                                } else {
                                    std::cout << "ERROR: The number of columns does not match the expected number of error "
                                                 "covariance terms. Expected " << lengthList[responseNum] << "terms but "
                                              << "got " << ncol << " terms." << std::endl;
                                    return EXIT_FAILURE;
                                }
                            } else if (nrow == lengthList[responseNum]) {
                                if (ncol == 1) {
                                    errType << "'diagonal' ";
                                } else if (ncol == lengthList[responseNum]) {
                                    errType << "'matrix' ";
                                } else {
                                    std::cout << "ERROR: The number of columns does not match the expected number of error "
                                                 "covariance terms. Expected " << lengthList[responseNum] << "terms but "
                                              << "got " << ncol << " terms." << std::endl;
                                    return EXIT_FAILURE;
                                }
                            } else {
                                std::cout << "ERROR: The number of rows does not match the expected number of error "
                                             "covariance terms. Expected " << lengthList[responseNum] << "terms but "
                                          << "got " << nrow << " terms." << std::endl;
                                return EXIT_FAILURE;
                            }
                            // Add this filename to the list of files to be moved
                            calFilesTracker << tmpErrorFileName << std::endl;
                        }
                        // =============================================
                    }
                }
            }
        }
    }
    // =============================================
    // When there are only scalar response terms
    // =============================================
    else {
        // Create an ofstream to write the data and error variances
        std::ofstream scalarCalDataFile;
        std::string scalarCalDataFileName = "quoFEMScalarCalibrationData.cal";
        // Add the name of this file that contains the calibration data and error variance values
        calFilesTracker << scalarCalDataFileName << std::endl;

        if (idResponse.compare("BayesCalibration") != 0) {
            // This is the case of deterministic calibration
            // A single calibration data file needs to be created, which contains the data
            // Renaming the spaced data file will suffice for this case
            std::rename(spacedFileName.c_str(), scalarCalDataFileName.c_str());
        }
        else {
            // This is the Bayesian calibration case.
            // A single calibration data file needs to be created, which contains the data and the variances
            scalarCalDataFile.open(scalarCalDataFileName.c_str(), std::fstream::out);
            spacedDataFile.open(spacedFileName.c_str(), std::fstream::in);
            // Loop over the number of experiments to write data to file line by line
            for (int expNum = 1; expNum <= numExperiments; expNum++) {
                // Get the calibration data
                getline(spacedDataFile, line);
                // Write this data to the scalar data file
                scalarCalDataFile << line;

                // For each response quantity, check if there is an error variance file
                for (int responseNum = 0; responseNum < numResponses; responseNum++) {
                    // Create filename for error variance
                    std::stringstream errFileName;
                    errFileName << edpList[responseNum] << "." << lineNum << ".sigma";
                    // Check if an error variance file with the created name is in the list of error files
                    std::string errFileToProcess;
                    if (readErrorFile) {// If any user supplied error variance files exist
                        for (const auto &path : errFileList) {
                            std::string base_filename = path.substr(path.find_last_of("/\\") + 1);
//                            std::cout << "Base filename: " << base_filename << std::endl;
                            if (base_filename == errFileName.str()) {
                                errFileToProcess = path;
                                break;
                            }
                        }
                        std::ifstream errFileUser(errFileToProcess);
                        std::string fileLine;
                        // Get the first line
                        getline(errFileUser, fileLine);
                        while (fileLine.empty()) {
                            getline(errFileUser, fileLine);
                        }
                        // Get the first word from the file
                        char *entry;
                        entry = strtok(const_cast<char *>(fileLine.c_str()), " \t,");
                        scalarCalDataFile << " " << entry;
                    }
                    else {// If user supplied error variance files do not exist
                        scalarCalDataFile << " " << "1";
                    }
                }
                scalarCalDataFile << std::endl;
            }
            errType << "'scalar' ";
            scalarCalDataFile.close();
            spacedDataFile.close();
        }
    }
    calFilesTracker.close();
    return numExperiments;
}

int
parseForRV(json_t *rootINPUT, struct randomVariables &theRandomVariables){

  int result = 0;
  result = parseInputForRV(rootINPUT, theRandomVariables);

  if (result < 0) {
    std::cerr << "parseForRV - failure\n";
  }

  /*
  json_error_t error;
  json_t *rootDefault =  json_object_get(rootINPUT, "DefaultValues");
  if (rootDefault != NULL) {
    json_t *defaultRVs =  json_object_get(rootDefault, "rvFiles");
    if ((defaultRVs != NULL) && json_is_array(defaultRVs)) {
	size_t index;
	json_t *value;
	
	json_array_foreach(defaultRVs, index, value) {
	  const char *fName = json_string_value(value);
	  std::cerr << "Parsing file: " << fName << "\n";
	  json_t *rootOther = json_load_file(fName, 0, &error);	  
	  if (rootOther != NULL) {
	    int numLocalRV = parseInputForRV(rootOther, theRandomVariables);
	    result += numLocalRV;
	    std::cerr << "num new RV in file: " << numLocalRV << "\n";      
	  }  else {
	    std::cerr << "no new randomVariables in file " << fName << "\n";      
	  }	  
	}
    }
  } 
  */
  
  return result;
  
}
