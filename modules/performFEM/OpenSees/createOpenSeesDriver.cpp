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
    std::cerr << "createOpenSeesDriver:: expecting 4 inputs\n";
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
    std::cerr << "createOpenSeesDriver:: input file: " << inputFile << " does not exist\n";
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
    std::cerr << "createOpenSeesDriver:: input file " << inputFile << " is not valid JSON\n";
    exit(801); 
  } 

  json_t *rootAPPs =  json_object_get(rootInput, "Applications");
  if (rootAPPs == NULL) {
    std::cerr << "createOpenSeesDriver:: no Applications found\n";
    return 0; // no random variables allowed
  }  
  
  json_t *rootEDP =  json_object_get(rootInput, "EDP");
  if (rootEDP == NULL) {
    std::cerr << "createOpenSeesDriver:: no EDP found\n";    
    return 0; // no random variables allowed
  }

  json_t *rootRV =  json_object_get(rootInput, "randomVariables");
  if (rootRV == NULL) {
    std::cerr << "createOpenSeesDriver:: no randomVariables found\n";
    return 0; // no random variables allowed
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
    exit(802); // no random variables allowed
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
    
  } else {

    dpreproCommand = remoteDir + std::string("/applications/performUQ/templateSub/simCenterSub");
    openSeesCommand = std::string("/home1/00477/tg457427/bin/OpenSees");
    pythonCommand = std::string("python3");

  }

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
  
  json_t *femScript = json_object_get(fem, "mainScript");
  if (femScript == NULL) {
    const char *jsonD =  json_dumps(fem, JSON_INDENT(4));    
    std::cerr << "createOpenSeesDriver:: no mainScript in ApplicationData\n" << jsonD;
    return -4; 
  }  
  const char *mainInput =  json_string_value(femScript);

  std::string templateFileName = "SimCenterInput.RV";
  if (modelIndex > 0) {
    templateFileName = appendModelIndexToStem(modelIndex, templateFileName);
  }
  std::ofstream templateFile(templateFileName);
  for(std::vector<std::string>::iterator itRV = rvList.begin(); itRV != rvList.end(); ++itRV) {
    templateFile << "pset " << *itRV << " \"RV." << *itRV << "\"\n";
  }
  
  templateFile << "\n set listQoI \"";
  for(std::vector<std::string>::iterator itEDP = edpList.begin(); itEDP != edpList.end(); ++itEDP) {
    templateFile << *itEDP << " ";
  }
  
  templateFile << "\"\n\n\n source " << mainInput << "\n";

  std::filesystem::path templateFilePath(templateFileName);
  std::string templateFileNameStem = templateFilePath.stem().string();
  workflowDriverFile << dpreproCommand << " params.in " << templateFileName << " " << templateFileNameStem << ".tcl\n";
  bool suppressOutput = false;
  if (suppressOutput) {
      if (osType.compare("Windows") == 0) {
          workflowDriverFile << openSeesCommand << " " << templateFileNameStem << ".tcl 1>nul 2>nul\n";
      }
      else {
          workflowDriverFile << openSeesCommand << " " << templateFileNameStem << ".tcl 1> /dev/null 2>&1\n";
      }
  }
  else {
      workflowDriverFile << openSeesCommand <<  " " << templateFileNameStem << ".tcl 1> workflow.err 2>&1\n";
  }

  // depending on script type do something
  json_t *postScript = json_object_get(fem, "postprocessScript");
  if (postScript == NULL) {
    const char *jsonD =  json_dumps(fem, JSON_INDENT(4));    
    std::cerr << "createOpenSeesDriver:: no postprocessScript in ApplicationData\n" << jsonD;
    return -4; 
  }      
  const char *postprocessScript =  json_string_value(postScript);

  std::cerr << "Postprocess Script: " << postprocessScript << "\n";
  
  int scriptType = 0;
  if (strstr(postprocessScript,".py") != NULL) 
    scriptType = 1;
  else if (strstr(postprocessScript,".tcl") != NULL) 
    scriptType = 2;
  
    
  if (scriptType == 1) { // python script
    workflowDriverFile << "\n" << pythonCommand << " " << postprocessScript;
    for(std::vector<std::string>::iterator itEDP = edpList.begin(); itEDP != edpList.end(); ++itEDP) {
      workflowDriverFile << " " << *itEDP;
    }
    workflowDriverFile << " >> workflow.err 2>&1";
  } 

  else if (scriptType == 2) { // tcl script
    
    templateFile << " remove recorders" << "\n";  
    templateFile << " source " << postprocessScript << "\n";      
  }
  
  templateFile.close();

  workflowDriverFile.close();


  try {
    std::filesystem::permissions(workflowDriver,
				 std::filesystem::perms::owner_all |
				 std::filesystem::perms::group_all,
				 std::filesystem::perm_options::add);
  }
  catch (std::exception& e) {
    std::cerr << "createOpenSeesDriver - failed in setting permissions\n";
    exit(-1);
  }
    
  //
  // done
  //

  exit(0);
}

