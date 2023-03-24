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
#include <regex>

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

std::string appendModelIndexToStem(int modelIndex, std::string filename) {
    std::filesystem::path templateFilePath(filename);
    std::string newStem = templateFilePath.stem().string() + "_" + std::to_string(modelIndex);
    std::string extension = templateFilePath.extension().string();
    filename = newStem + extension;
    return filename;
}

int main(int argc, const char **argv) {

  std::cerr << "createOpenSeesPyDriver:: starting\n";
  
  if (argc < 5) {
    std::cerr << "createOpenSeesPyDriver:: expecting 4 inputs\n";
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
    std::cerr << "createOpenSeesPyDriver:: input file: " << inputFile << " does not exist\n";
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
    std::cerr << "createOpenSeesPyDriver:: input file " << inputFile << " is not valid JSON\n";
    exit(801); 
  } 

  json_t *rootRV =  json_object_get(rootInput, "randomVariables");
  if (rootRV == NULL) {
    std::cerr << "createOpenSeesPyDriver:: no randomVariables found\n";
    return 0; // no random variables is allowed
  }
  
  json_t *rootEDP =  json_object_get(rootInput, "EDP");
  if (rootEDP == NULL) {
    std::cerr << "createOpenSeesPyDriver:: no EDP found\n";    
    return 0; // no random variables is allowed
  }

  json_t *rootAPPs =  json_object_get(rootInput, "Applications");
  if (rootAPPs == NULL) {
    std::cerr << "createOpenSeesPyDriver:: no Applications found\n";
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
    std::cerr << "createOpenSeesPyDriver:: could not create workflow driver file: " << workflowDriver << "\n";
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
    std::cerr << "createOpenSeesPyDriver:: no FEM application in rootAPPs\n";    
    return -2; 
  }

  json_t *fem =  json_object_get(femApp, "ApplicationData");  
  if (fem == NULL) {
    std::cerr << "createOpenSeesPyDriver:: no ApplicationData in femApp\n";        
    return -3; 
  }

  std::string mainScriptTemplate = "tmpSimCenter.script";
  if (modelIndex > 0) {
    mainScriptTemplate = appendModelIndexToStem(modelIndex, mainScriptTemplate);
  }
  std::ofstream templateFile(mainScriptTemplate);

  const char *postprocessScript =  json_string_value(json_object_get(fem, "postprocessScript"));
  std::string parametersScript =  json_string_value(json_object_get(fem, "parametersScript"));

  std::string parametersScriptTemplate = "tmpSimCenter.params";
  if (strcmp(parametersScript.c_str(),"") != 0) { // if parametersScript name is provided
    if (modelIndex > 0) {
      parametersScriptTemplate = appendModelIndexToStem(modelIndex, parametersScriptTemplate);
    }
    std::filesystem::copy_file(parametersScript, parametersScriptTemplate);
  } else {
    for(std::vector<std::string>::iterator itRV = rvList.begin(); itRV != rvList.end(); ++itRV) {
      std::string nm = *itRV;
      if (nm.find("MultiModel") != std::string::npos) { // to deal with the "-" in "MultiModel-FEM"
        templateFile << std::regex_replace(nm, std::regex("-"), "_") << " = \"RV." << *itRV << "\"\n";
      } else {
        templateFile << *itRV << " = \"RV." << *itRV << "\"\n";
      }
    }
    templateFile << std::endl;
  }

  json_t *femScript = json_object_get(fem, "mainScript");
  if (femScript == NULL) {
    const char *jsonD =  json_dumps(fem, JSON_INDENT(4));    
    std::cerr << "createOpenSeesPyDriver:: no mainScript in ApplicationData\n" << jsonD;
    return -4; 
  }  
  std::string mainScript = json_string_value(femScript);
  std::ifstream modelFile(mainScript);
  while (!modelFile.eof()) {
    std::string line;
    std::getline(modelFile, line);
    templateFile << line << std::endl;
  }
  templateFile.close();

  if (strcmp(parametersScript.c_str(),"") == 0) {	
    // workflowDriverFile << moveCommand << mainScript << " tmpSimCenter.script \n";
    workflowDriverFile << dpreproCommand << "  params.in " << mainScriptTemplate << " " << mainScript << "\n";
  } else {
    // workflowDriverFile << moveCommand << parametersScript << " tmpSimCenter.params \n";
    workflowDriverFile << dpreproCommand << "  params.in " << parametersScriptTemplate << " " << parametersScript << "\n";
  }
  
  workflowDriverFile << pythonCommand << " " << mainScript; 
  
  if (strcmp(postprocessScript,"") != 0) {
    if (strstr(postprocessScript,".py") != NULL) {
      workflowDriverFile << "\n" << pythonCommand << " " << postprocessScript;
      for(std::vector<std::string>::iterator itEDP = edpList.begin(); itEDP != edpList.end(); ++itEDP) {
        workflowDriverFile << " " << *itEDP;
      }
    }
    else if (strstr(postprocessScript,".tcl") != NULL) {
      workflowDriverFile << "\n" << openSeesCommand << " " << postprocessScript;
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

