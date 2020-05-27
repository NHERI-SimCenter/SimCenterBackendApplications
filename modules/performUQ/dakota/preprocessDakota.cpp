#include <iostream>
#include <fstream>
#include <jansson.h> 
#include <string.h>
#include <string>
#include <sstream>
#include <list>

#include "dakotaProcedures.h"

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

for (int i=0; i<=8; i++)
    std::cerr << i << " " << argv[i] << "\n";

  std::string workflowDriver = "workflow_driver";
  if ((strcmp(osType,"Windows") == 0) && (strcmp(runType,"run") == 0))
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

  if ((strcmp(runType, "local") == 0) || (strcmp(runType,"run") == 0)) {
    dpreproCommand = localDir + std::string("/applications/performUQ/dakota/simCenterDprepro");
  } else {
    dpreproCommand = remoteDir + std::string("/applications/performUQ/dakota/simCenterDprepro");
  }

  int numRV = parseForRV(rootINPUT, theRandomVariables);
  //  if (numRV > 0) {
  if (rename(bimName, "bim.j") != 0) {
    std::cerr << "preprocessDakota - cound not rename bim file\n";
    exit(802);
  }
  workflowDriverFile << dpreproCommand << " params.in bim.j " << bimName << "\n";
  //  } 

  // load event read random variables, if any rename file and add a dprepro line to workflow

  json_t *rootEVT = json_load_file(evtName, 0, &error);
  if (rootEVT == NULL) {
    std::cerr << "parseFileForRV:: could not open EVT file with name: " << evtName << "\n";
    exit(801); // no random variables is allowed
  } 
  numRV = parseForRV(rootEVT, theRandomVariables);
  //if (numRV > 0) {
  if (rename(evtName, "evt.j") != 0) {
    std::cerr << "preprocessDakota - cound not rename event file\n";
    exit(802);
  }
  workflowDriverFile << dpreproCommand << " params.in evt.j " << evtName << "\n";
  //  }

  // load sam, read random variables, if any rename file and add a dprepro line to workflow

  json_t *rootSAM = json_load_file(samName, 0, &error);
  if (rootSAM == NULL) {
  std::cerr << "parseFileForRV:: could not open SAM file with name: " << samName << "\n";
    exit(801); // no random variables is allowed
  } 
  numRV = parseForRV(rootSAM, theRandomVariables);
  //if (numRV > 0) {
  if (rename(samName, "sam.j") != 0) {
    std::cerr << "preprocessDakota - cound not rename bim file\n";
    exit(802);
  }
  workflowDriverFile << dpreproCommand << " params.in sam.j " << samName << "\n";
  //}

  // load sim, read random variables, if any rename file and add a dprepro to workflow

  json_t *rootSIM = json_load_file(simName, 0, &error);
  if (rootSIM == NULL) {
    std::cerr << "parseFileForRV:: could not open SIM file with name: " << simName << "\n";
    exit(801); // no random variables is allowed
  } 
  numRV = parseForRV(rootSIM, theRandomVariables);
  //if (numRV > 0) {
  if (rename(simName, "sim.j") != 0) {
    std::cerr << "preprocessDakota - cound not rename sim file\n";
    exit(802);
  }
  workflowDriverFile << dpreproCommand << " params.in sim.j " << simName << "\n";
  //}

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
    if ((strcmp(runType, "local") == 0) || (strcmp(runType,"run") == 0)) {
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
  // write dakota file
  // 
  
  json_t *uqData =  json_object_get(rootINPUT, "UQ_Method");
  if (uqData == NULL) {
    std::cerr << "preprocessJSON - no UQ Data in inputfile\n";
    exit(-1); // no random variables is allowed
  }

  int errorWrite = writeDakotaInputFile(dakotaFile, uqData, rootEDP, theRandomVariables, workflowDriver);

  dakotaFile.close();
  std::cerr << "NUM RV: " << theRandomVariables.numRandomVariables << "\n";
  std::cerr << "DONE PREPROCESSOR .. DONE\n";

  exit(errorWrite);
}


