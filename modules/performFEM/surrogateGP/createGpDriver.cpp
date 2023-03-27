#include <iostream>
#include <fstream>
#include <jansson.h> 
#include <string.h>
#include <string>
#include <sstream>
#include <list>
#include <vector>
#include <thread>
#include <filesystem>
#include <algorithm>


int getEDP(json_t *edp, std::vector<std::string> &edpList);
int getRV(json_t *edp, std::vector<std::string> &rvList);

void eraseAllSubstring(std::string & mainStr, const std::string & toErase)
{
    size_t pos = std::string::npos;
    // Search for the substring in string in a loop until nothing is found
    while ((pos  = mainStr.find(toErase) )!= std::string::npos)
    {
        // If found then erase it from string
        mainStr.erase(pos, toErase.length());
    }
}

std::string appendModelIndexToStem(int modelIndex, std::string filename) {
    std::filesystem::path templateFilePath(filename);
    std::string newStem = templateFilePath.stem().string() + "_" + std::to_string(modelIndex);
    std::string extension = templateFilePath.extension().string();
    filename = newStem + extension;
    return filename;
}

int main(int argc, const char **argv) {
  
  if (argc < 5) {
    std::cerr << "createGpDriver:: expecting 4 inputs\n";
    exit(-1);
  }

  std::string thisProgram(argv[0]);
  std::string inputFile(argv[1]);
  std::string runType(argv[2]);
  std::string osType(argv[3]);
  std::string workflowDriver(argv[4]);
  int modelIndex = 0;

  // if case not simple defaults
  for (int i=1; i<argc; i+=2) {
    if (strcmp(argv[i],"--driverFile") == 0) {
      workflowDriver = argv[i+1];
    } else if (strcmp(argv[i],"--workflowInput") == 0) {
      inputFile = argv[i+1];
    } else if (strcmp(argv[i],"--runType") == 0) {
      runType = argv[i+1];
    } else if (strcmp(argv[i],"--osType") == 0) {
      osType = argv[i+1];
    } else if (strcmp(argv[i], "--modelIndex") == 0) {
      modelIndex = std::stoi(argv[i+1]);
    }
  }
  
  eraseAllSubstring(thisProgram,"\"");
  eraseAllSubstring(runType,"\"");
  eraseAllSubstring(osType,"\"");
  eraseAllSubstring(workflowDriver,"\"");

  if (!std::filesystem::exists(inputFile)) {
    std::cerr << "createGpDriver:: input file: " << inputFile << " does not exist\n";
    exit(801);
  }
  
  //
  // open input file to get RV's and EDP names
  //

  std::vector<std::string> rvList;  
  std::vector<std::string> edpList;
  
  json_error_t error;
  json_t *rootInput = json_load_file(inputFile.c_str(), 0, &error);
  if (rootInput == NULL) {
    std::cerr << "createGpDriver:: input file " << inputFile << " is not valid JSON\n";
    exit(801); 
  } 

  json_t *rootRV =  json_object_get(rootInput, "randomVariables");
  if (rootRV == NULL) {
    std::cerr << "createGpDriver:: no randomVariables found\n";
    return 0; // no random variables is allowed
  }
  
  json_t *rootEDP =  json_object_get(rootInput, "EDP");
  if (rootEDP == NULL) {
    std::cerr << "createGpDriver:: no EDP found\n";    
    return 0; // no random variables is allowed
  }

  int numRV = getEDP(rootRV, rvList);  
  int numEDP = getEDP(rootEDP, edpList);
  
  //
  // open workflow_driver 
  //
  
  if ((osType.compare("Windows") == 0) && (runType.compare("runningLocal") == 0))
    workflowDriver.append(std::string(".bat"));

  std::ofstream workflowDriverFile(workflowDriver, std::ios::binary);
  
  if (!workflowDriverFile.is_open()) {
    std::cerr << "createGpDriver:: could not create workflow driver file: " << workflowDriver << "\n";
    exit(802); // no random variables is allowed
  }

  std::string dpreproCommand;
  std::string openSeesCommand;
  std::string gpCommand;

  std::string pythonCommand;
  std::string feapCommand;
  std::string moveCommand;

  const char *localDir = json_string_value(json_object_get(rootInput,"localAppDir"));
  const char *remoteDir = json_string_value(json_object_get(rootInput,"remoteAppDir"));

  if (runType.compare("runningLocal") == 0) {

    if (osType.compare("Windows") == 0) {
      dpreproCommand = std::string("\"") + localDir + std::string("/applications/performUQ/templateSub/simCenterSub.exe\"");
      pythonCommand = std::string("python");            
    } else {
      dpreproCommand = std::string("\"") + localDir + std::string("/applications/performUQ/templateSub/simCenterSub\"");
      pythonCommand = std::string("python3");      
    }
    
    openSeesCommand = std::string("OpenSees");

    if (json_object_get(rootInput, "python") != NULL)
      pythonCommand = std::string("\"") + json_string_value(json_object_get(rootInput,"python")) + std::string("\"");
    
  } else {

    dpreproCommand = remoteDir + std::string("/applications/performUQ/templateSub/simCenterSub");
    openSeesCommand = std::string("/home1/00477/tg457427/bin/OpenSees");
    pythonCommand = std::string("python3");

  }

  gpCommand = pythonCommand + std::string(" \"") + localDir + std::string("/applications/performFEM/surrogateGP/gpPredict.py\"");


  json_t *rootAPPs =  json_object_get(rootInput, "Applications");
  if (rootAPPs == NULL) {
    std::cerr << "createGPDriver:: no Applications found\n";
    return 0; // no random variables is allowed
  }  
  

  json_t *femApp =  json_object_get(rootAPPs, "FEM");
  if (femApp == NULL) {
    std::cerr << "createGPDriver:: no FEM application in rootAPPs\n";    
    return -2; 
  }

    json_t *fem =  json_object_get(femApp, "ApplicationData");  
  if (fem == NULL) {
    std::cerr << "createGPDriver:: no ApplicationData in femApp\n";        
    return -3; 
  }
  
  std::string jsonFile =  json_string_value(json_object_get(fem, "mainScript"));
  std::string pklFile =  json_string_value(json_object_get(fem, "postprocessScript"));
  std::string ms = jsonFile;
  std::string ps = pklFile;
  if (modelIndex > 0) {
    jsonFile = appendModelIndexToStem(modelIndex, jsonFile);
    pklFile = appendModelIndexToStem(modelIndex, pklFile);
    std::filesystem::copy_file(ms, jsonFile);
    std::filesystem::copy_file(ps, pklFile);
  }

  workflowDriverFile << gpCommand <<  " params.in " << jsonFile << " " << pklFile << " " << inputFile << " 1> ops.out 2>&1\n ";
  
  workflowDriverFile.close();


  try {
    std::filesystem::permissions(workflowDriver,
				 std::filesystem::perms::owner_all |
				 std::filesystem::perms::group_all,
				 std::filesystem::perm_options::add);
  }
  catch (std::exception& e) {
    std::cerr << "createGpDriver - failed in setting permissions\n";
  }
    
  //
  // done
  //

  exit(0);
}

