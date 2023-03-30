#ifndef _PARSE_WORKFLOW_INPUT
#define _PARSE_WORKFLOW_INPUT

#define OVERSUBSCRIBE_CORE_MULTIPLIER 1

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
  // std::list<double> weights;
};

struct discreteUncertainIntegerSetRV {
  std::string name;
  std::list<int> elements;
  std::list<double> weights;
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
  std::list<struct discreteUncertainIntegerSetRV> discreteUncertainIntegerSetRVs;
  std::vector<int> ordering;
  std::vector<double> corrMat;
};


struct qoi {
  std::string name;
  std::string type;  
  int length;
};

struct quantatiesOfInterest {
  int numFieldResponses;
  int numScalarResponse;  
  std::set<std::string>theNames;
  std::list<struct qoi> theQOIs;
};


// parses JSON for random variables & returns number found

extern "C" int parseForRV(json_t *root, struct randomVariables &theRandomVariables);

extern "C" int processDataFiles(const char *calFileName,
                     std::vector<std::string> &edpList,
                     std::vector<int> &lengthList,
                     int numResponses, int numFieldResponses,
                     std::vector<std::string> &errFileList,
                     std::stringstream &errType, std::string idResponse,
                     std::vector<double> &scaleFactors);

#endif
