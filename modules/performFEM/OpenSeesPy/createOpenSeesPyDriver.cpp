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
#include <cstdio> // for freopen
#include <cstdlib> // for exit
#include <stdexcept>
#include <exception>
#include <csignal>



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

//int main(int argc, const char **argv) {
int createDriver(int argc, const char **argv) {
  
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

  // put in shebang fow linux
  bool isWindows = (osType.compare("Windows") == 0);
  bool isRunningLocal = (runType.compare("runningLocal") == 0);
  if (!(isWindows && isRunningLocal)) {
    workflowDriverFile << "#!/bin/bash\n";
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
    //pythonCommand = std::string("/home1/00477/tg457427/python/python-3.8.10/bin/python3");
    pythonCommand = std::string("python3");    
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
    try {
        std::filesystem::copy_file(parametersScript, parametersScriptTemplate);
        std::cout << "File copied successfully from " 
                  << parametersScript << " to " 
                  << parametersScriptTemplate << std::endl;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error copying file: " << e.what() << std::endl;
        std::cerr << "Source: " << e.path1() << ", Destination: " << e.path2() << std::endl;
        exit(-1);
    } catch (const std::exception& e) {
        std::cerr << "An unexpected error occurred: " << e.what() << std::endl;
        exit(-1);
    }


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
  std::string line;
  while (std::getline(modelFile, line)) {
      std::cout << line << std::endl; // Print line to console
      templateFile << line << std::endl; // Write line to template file
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

  workflowDriverFile << " 1> workflow.err 2>&1\n"; 

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
  std::cerr << "The run was successful" << std::endl;

  exit(0);
}

void signalHandler(int signal) {
    std::cerr << "Caught signal: " << signal << std::endl;
    exit(signal);
}

void setupSignalHandlers() {
    std::signal(SIGSEGV, signalHandler);
}

void redirectCerrToFile() {
    if (freopen("FEMpreprocessor.err", "w", stderr) == nullptr) {
        std::cerr << "Failed to redirect stderr to error.log" << std::endl;
        exit(EXIT_FAILURE);
    }
}


int main(int argc, const char **argv) {
   // to redirect the file
    redirectCerrToFile();
    setupSignalHandlers();

    try {
        createDriver(argc, argv); // Pass arguments to the main logic function
    } catch (const std::exception& e) {
        std::cerr << "Caught exception in main: " << e.what() << std::endl;
    }

    return 0;

}
