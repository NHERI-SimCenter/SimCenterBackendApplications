#include "../common/parseWorkflowInput.h"

int writeRV(std::ostream &dakotaFile, 
	    struct randomVariables &theRandomVariables, 
	    std::string idVariables,
	    std::vector<std::string> &rvList,
	    bool includeActiveUncertainText = true);

int writeInterface(std::ostream &dakotaFile, 
		   json_t *uqData, 
		   std::string &workflowDriver, 
		   std::string idInterface,
		   int evalConcurrency);

int writeResponse(std::ostream &dakotaFile, 
		  json_t *rootEDP,  
		  std::string idResponse, 
		  bool numericalGradients, 
		  bool numericalHessians,
		  std::vector<std::string> &edpList,
		  std::istream &calDataFile);

int writeDakotaInputFile(std::ostream &dakotaFile, 
			 json_t *uqData, 
			 json_t *rootEDP, 
			 struct randomVariables &theRandomVariables, 
			 std::string &workflowDriver,
			 std::vector<std::string> &rvList,
			 std::vector<std::string> &edpList,
			 int evalConcurrency);

int processDataFiles(const char *calFileName,
                     std::vector<std::string> &edpList,
                     std::vector<int> &lengthList,
                     int numResponses, int numFieldResponses,
                     std::vector<std::string> &errFileList,
                     std::stringstream &errType, std::string idResponse);
