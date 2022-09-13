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
    std::cerr << "createOpenSeesPyDriver:: expecting 4 inputs\n";
    exit(-1);
  }

  std::string thisProgram(argv[0]);
  std::string inputFile(argv[1]);
  std::string runType(argv[2]);
  std::string osType(argv[3]);
  std::string workflowDriver(argv[4]);
  
  eraseAllSubstring(thisProgram,"\"");
  eraseAllSubstring(runType,"\"");
  eraseAllSubstring(osType,"\"");
  eraseAllSubstring(workflowDriver,"\"");  

  if (!std::filesystem::exists(inputFile)) {
    std::cerr << "createOpenSeesPyDriver:: input file: " << inputFile << " does not exist\n";
    exit(801);
  }
  
  //current_path(currentPath);
  auto path = std::filesystem::current_path(); //getting path
  //  std::cerr << "PATH: " << path << '\n';

  //std::string fileName = std::filesystem::path(inputFile).filename();
  std::filesystem::path fileNameP = std::filesystem::path(inputFile).filename();
  std::string fileName = fileNameP.generic_string();   
  //  std::cerr << "fileName: " << fileName << '\n';

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

  json_t *rootAPPs =  json_object_get(rootInput, "Applications");
  if (rootAPPs == NULL) {
    std::cerr << "createOpenSeesDriver:: no Applications found\n";
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
  std::string openSeesCommand;
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
    openSeesCommand = std::string("/home1/00477/tg457427/bin/OpenSees");
    pythonCommand = std::string("/home1/00477/tg457427/python/python-3.8.10/bin/python3");
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

  
  //  const char *mainScript =  json_string_value(json_object_get(fem, "mainInput"));
  const char *postprocessScript =  json_string_value(json_object_get(fem, "postprocessScript"));
  const char *parametersScript =  json_string_value(json_object_get(fem, "parametersScript"));

  json_t *femScript = json_object_get(fem, "mainScript");
  if (femScript == NULL) {
    const char *jsonD =  json_dumps(fem, JSON_INDENT(4));    
    std::cerr << "createOpenSeesPyDriver:: no mainScript in ApplicationData\n" << jsonD;
    return -4; 
  }  
  const char *mainScript =  json_string_value(femScript);

  if (strcmp(parametersScript,"") == 0) {	
    // workflowDriverFile << moveCommand << mainScript << " tmpSimCenter.script \n";
    workflowDriverFile << dpreproCommand << "  params.in tmpSimCenter.script " << mainScript << "\n";
  } else {
    // workflowDriverFile << moveCommand << parametersScript << " tmpSimCenter.params \n";
    workflowDriverFile << dpreproCommand << "  params.in  tmpSimCenter.params " << " " << parametersScript << "\n";
  }
  
  workflowDriverFile << pythonCommand << " " << mainScript; 
  
  if (strcmp(postprocessScript,"") != 0) {
    workflowDriverFile << "\n" << pythonCommand << " " << postprocessScript;
    for(std::vector<std::string>::iterator itEDP = edpList.begin(); itEDP != edpList.end(); ++itEDP) {
      workflowDriverFile << " " << *itEDP;
    }
  } else {
    for(std::vector<std::string>::iterator itEDP = edpList.begin(); itEDP != edpList.end(); ++itEDP) {
      workflowDriverFile << " " << *itEDP;
    }
  }

  workflowDriverFile << " 1> ops.out 2>&1\n"; 

  workflowDriverFile.close();
  
  try {
    std::filesystem::permissions(workflowDriver,
				 std::filesystem::perms::owner_all |
				 std::filesystem::perms::group_all,
				 std::filesystem::perm_options::add);
  }
  catch (std::exception& e) {
    std::cerr << "createOpenSeesPyDriver - failed to set permissions\n";
  }
  
  //
  // done
  //

  exit(0);
}

