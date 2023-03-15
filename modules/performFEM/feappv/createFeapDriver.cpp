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
    // Search for the substring in string in a loop untill nothing is found
    while ((pos  = mainStr.find(toErase) )!= std::string::npos)
    {
        // If found then erase it from string
        mainStr.erase(pos, toErase.length());
    }
}

int main(int argc, const char **argv) {

  if (argc < 5) {
    std::cerr << "createOpenSeesDriver:: expecting 4 inputs\n";
    exit(-1);
  }

  std::string thisProgram(argv[0]);
  std::string inputFile(argv[1]);
  std::string runType(argv[2]);
  std::string osType(argv[3]);
  std::string workflowDriver(argv[4]);
  
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
    }
  }

  
  eraseAllSubstring(thisProgram,"\"");
  eraseAllSubstring(runType,"\"");
  eraseAllSubstring(osType,"\"");
  eraseAllSubstring(workflowDriver,"\"");  

  if (!std::filesystem::exists(inputFile)) {
    std::cerr << "createOpenSeesDriver:: input file: " << inputFile << " does not exist\n";
    exit(801);
  }
  
  //current_path(currentPath);
  auto path = std::filesystem::current_path(); //getting path
  //  std::cerr << "PATH: " << path << '\n';
  
  // std::string fileName = std::filesystem::path(inputFile).filename();
  std::filesystem::path fileNameP = std::filesystem::path(inputFile).filename();
  std::string fileName = fileNameP.generic_string(); // std::filesystem::path(inputFile).filename();
  // std::cerr << "fileName: " << fileName << '\n';

  std::string fullPath = std::filesystem::path(inputFile).remove_filename().generic_string();
  //  std::cerr << "fullPath: " << fullPath << '\n';  

  //
  // open input file to get RV's and EDP names
  //

  std::vector<std::string> rvList;  
  std::vector<std::string> edpList;


  
  json_error_t error;
  json_t *rootInput = json_load_file(inputFile.c_str(), 0, &error);
  if (rootInput == NULL) {
    std::cerr << "createOpenSeesDriver:: input file " << inputFile << " is not valid JSON\n";
    exit(801); 
  } 

  json_t *rootAPPs =  json_object_get(rootInput, "Applications");
  if (rootAPPs == NULL) {
    std::cerr << "createOpenSeesDriver:: no Applications found\n";
    return 0; // no random variables is allowed
  }
  
  json_t *rootRV =  json_object_get(rootInput, "randomVariables");
  if (rootRV == NULL) {
    std::cerr << "createOpenSeesDriver:: no randomVariables found\n";
    return 0; // no random variables is allowed
  }
  
  json_t *rootEDP =  json_object_get(rootInput, "EDP");
  if (rootEDP == NULL) {
    std::cerr << "createOpenSeesDriver:: no EDP found\n";    
    return 0; // no random variables is allowed
  }

  int numRV = getRV(rootRV, rvList);  
  int numEDP = getEDP(rootEDP, edpList);
  
  //
  // open workflow_driver 
  //
  
  if ((osType.compare("Windows") == 0) && (runType.compare("runningLocal") == 0))
    workflowDriver.append(std::string(".bat"));
  
  std::ofstream workflowDriverFile(workflowDriver, std::ios::binary);
  
  if (!workflowDriverFile.is_open()) {
    std::cerr << "createOpenSeesDriver:: could not create workflow driver file: " << workflowDriver << "\n";
    exit(802); // no random variables is allowed
  }

  std::string dpreproCommand;
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

    if (json_object_get(rootInput, "python") != NULL)
      pythonCommand = std::string("\"") + json_string_value(json_object_get(rootInput,"python")) + std::string("\"");

    if (osType.compare("Windows") == 0) {
      feapCommand = std::string("Feappv41.exe");
      moveCommand = std::string("move /y ");
    }
    else {
      feapCommand = std::string("feappv");
      moveCommand = std::string("mv ");
    }
  } else {

    dpreproCommand = remoteDir + std::string("/applications/performUQ/templateSub/simCenterSub");
    pythonCommand = std::string("python3");
    feapCommand = std::string("/home1/00477/tg457427/bin/feappv");
    moveCommand = std::string("mv ");
  }


  //
  // based on fem program we do things
  //

  json_t *femApp =  json_object_get(rootAPPs, "FEM");
  if (femApp == NULL) {
    std::cerr << "createOpenSeesDriver:: no FEM application in rootAPPs\n";    
    return -2; 
  }
  
  json_t *fem =  json_object_get(femApp, "ApplicationData");  
  if (fem == NULL) {
    std::cerr << "createOpenSeesDriver:: no ApplicationData in femApp\n";        
    return -3; 
  }


  json_t *inputJSON = json_object_get(fem, "inputFile");
  if (inputJSON == NULL) {
    const char *jsonD =  json_dumps(fem, JSON_INDENT(4));    
    std::cerr << "createFeapDriver:: no inputFile in ApplicationData\n" << jsonD;
    return -4; 
  }
  const char *mainInput =  json_string_value(inputJSON);
  
  // depending on script type do something
  json_t *postScript = json_object_get(fem, "postprocessScript");
  if (postScript == NULL) {
    const char *jsonD =  json_dumps(fem, JSON_INDENT(4));    
    std::cerr << "createOpenSeesDriver:: no postprocessScript in ApplicationData\n" << jsonD;
    return -4; 
  }      
  const char *postprocessScript =  json_string_value(postScript);
  
  
  // write run file
  std::ofstream feapFile("feapname");
  feapFile << "SimCenterIn.txt   \n";
  feapFile << "SimCenterOut.txt   \n";
  feapFile << "SimCenterR.txt   \n";
  feapFile << "SimCenterR.txt   \n";
  feapFile << "NONE   \n";
  feapFile << "\n";
  feapFile.close();
  
  // write driiver file
  workflowDriverFile << dpreproCommand << "  params.in " << mainInput  << " SimCenterIn.txt --formatFixed\n";
  workflowDriverFile << "echo y | " << feapCommand << "\n";
  workflowDriverFile << pythonCommand << " " << postprocessScript;
  for(std::vector<std::string>::iterator itEDP = edpList.begin(); itEDP != edpList.end(); ++itEDP) {
    workflowDriverFile << " " << *itEDP;
  }
  workflowDriverFile << "\n";


  workflowDriverFile.close();

  //
  // done
  //

  exit(0);
}

